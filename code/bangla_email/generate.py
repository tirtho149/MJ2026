"""Synthetic Bangla-email generation on a single Nova GPU (vLLM, TP=1).

Refactor of ``Bangla_Email_Synthetic_Generation.ipynb`` with the Colab crash
fixed.  The screenshots showed an ``EngineDeadError`` raised from a
``compare_models()`` step on a Colab T4: it kept several vLLM engines resident at
once and ran the T4 out of memory.

The fix here is structural — **at most one vLLM engine is ever alive**:

  * the default path loads a single model and generates the whole dataset;
  * the optional :func:`compare_models` loads each candidate, scores it, and
    *frees the engine before loading the next* (:func:`free_engine`), so peak
    VRAM is one model regardless of how many candidates are compared.

All ``google.colab`` upload/download calls are removed; outputs are written to
``config.DATA_DIR``.
"""

from __future__ import annotations

import contextlib
import gc
import math
import random
import time
import unicodedata
import re

import numpy as np
import pandas as pd

from . import config, data

# ── Output cleaning / validation (pure-Python, GPU-free, unit-testable) ────────
_BENGALI_RE  = re.compile(r"[ঀ-৿]")
_LABEL_RE    = re.compile(r"^\s*(subject|ইমেইল|বিষয়|email|বিভাগ|category)\s*[:：]\s*", re.IGNORECASE)
_LEADING_NUM = re.compile(r"^\s*[\d০-৯]+[\.\)]\s*")
_MULTISPACE  = re.compile(r"[ \t]+")
_REFUSAL_HINTS = ("আমি একটি ভাষা মডেল", "as an ai", "i cannot", "i'm sorry",
                  "here is", "এখানে একটি", "নিচে একটি")


def clean_generation(raw: str) -> str:
    """Strip wrapping quotes, leading labels/numbering, and collapse spaces."""
    if raw is None:
        return ""
    t = raw.strip()
    if len(t) >= 2 and t[0] in "\"“'" and t[-1] in "\"”'":
        t = t[1:-1].strip()
    t = _LABEL_RE.sub("", t)
    t = _LEADING_NUM.sub("", t)
    t = "\n".join(_MULTISPACE.sub(" ", ln).strip() for ln in t.splitlines())
    return t.strip()


def is_valid(text: str, category: str) -> bool:
    """Content / length / refusal checks for one generated email."""
    if not text or not text.strip():
        return False
    low = text.lower()
    if any(h in low for h in _REFUSAL_HINTS):
        return False
    if not _BENGALI_RE.search(text):
        return False
    lo, hi = config.LEN_BOUNDS[category]
    return lo <= len(text) <= hi


def normalize(text: str) -> str:
    """Normalization key for exact-duplicate detection."""
    t = unicodedata.normalize("NFKC", str(text))
    return _MULTISPACE.sub(" ", t.replace("\n", " ")).strip().lower()


def filter_near_duplicates(existing_texts, candidate_texts, threshold=config.NEAR_DUP_THRESHOLD):
    """Drop candidates that are near-duplicates of the existing pool or each other.

    Char-ngram TF-IDF cosine (language-agnostic -> robust for Bangla).  One
    batched matmul against the existing pool, then a greedy pass over the
    candidate self-similarity matrix.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not candidate_texts:
        return []
    corpus = list(existing_texts) + list(candidate_texts)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    X = vec.fit_transform(corpus)
    n_e = len(existing_texts)
    E, C = X[:n_e], X[n_e:]

    if n_e:
        max_vs_e = np.asarray((C @ E.T).max(axis=1).todense()).ravel()
        survive = max_vs_e <= threshold
    else:
        survive = np.ones(C.shape[0], dtype=bool)
    surv_idx = np.where(survive)[0]
    if surv_idx.size == 0:
        return []

    Cs = C[surv_idx]
    S = (Cs @ Cs.T).tocsr()
    S.setdiag(0.0)
    kept_local, dropped = [], set()
    for i in range(Cs.shape[0]):
        if i in dropped:
            continue
        kept_local.append(i)
        row = S.getrow(i)
        for j, v in zip(row.indices, row.data):
            if j > i and v > threshold:
                dropped.add(j)
    return [candidate_texts[surv_idx[i]] for i in kept_local]


# ── Prompt construction ───────────────────────────────────────────────────────
def _facet_hint(category, rng):
    parts = [f"{k}: {rng.choice(v)}" for k, v in config.FACETS[category].items()]
    return " · ".join(parts)


def build_messages(category, real_by_cat, rng, k=3):
    """Chat-format prompt for one email: persona + few-shot reals + facet hint."""
    pool = real_by_cat.get(category, [])
    picks = rng.sample(pool, min(k, len(pool))) if pool else []
    ex_block = "\n".join(f"- {p.strip()}" for p in picks if p and p.strip())
    hint = _facet_hint(category, rng)
    user = (
        f"বিভাগ: {category} — {config.CATEGORY_DEF[category]}\n\n"
        f"এই বিভাগের কিছু বাস্তব উদাহরণ:\n{ex_block}\n\n"
        f"নতুন ইমেইলের প্রসঙ্গ (এই বিবরণ ব্যবহার করো): {hint}\n\n"
        f"উপরের উদাহরণগুলোর মতো বাস্তব ও স্বাভাবিক একটি নতুন বাংলা ইমেইল লেখো, "
        f"কিন্তু উদাহরণগুলোর হুবহু নকল নয় — সম্পূর্ণ নতুন বিষয়বস্তু। "
        f'শুধুমাত্র ইমেইলের লেখাটি দাও — কোনো উদ্ধৃতি চিহ্ন, শিরোনাম, ক্রমিক নম্বর বা '
        f'"Subject:" / "ইমেইল:" লেবেল ছাড়া।'
    )
    return [
        {"role": "system", "content": config.GEN_SYSTEM_PROMPT},
        {"role": "user",   "content": user},
    ]


# ── vLLM engine lifecycle (the OOM fix lives here) ────────────────────────────
def load_engine(model_id: str, tp: int = 1, max_model_len: int = 2048,
                gpu_mem_util: float = 0.90, dtype: str = "auto"):
    """Build a single vLLM engine.  Returns (llm, SamplingParams)."""
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model_id,
        dtype=dtype,                       # 'auto' -> bf16 on A100/H200; was hard fp16 for T4
        tensor_parallel_size=tp,           # TP=1: one GPU
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        seed=config.SEED,
        enforce_eager=False,
    )
    return llm, SamplingParams


def free_engine(llm):
    """Tear an engine fully down so the next model starts from a clean GPU.

    This is what prevents the EngineDeadError/OOM: peak VRAM stays at one model
    even when several candidates are compared back-to-back.
    """
    import torch
    with contextlib.suppress(Exception):
        from vllm.distributed.parallel_state import (
            destroy_model_parallel, destroy_distributed_environment,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    with contextlib.suppress(Exception):
        del llm.llm_engine
    del llm
    gc.collect()
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()


def _sampling(SamplingParams, category):
    return SamplingParams(
        temperature=0.95, top_p=0.95, frequency_penalty=0.30,
        max_tokens=config.MAX_TOKENS[category], seed=None,
    )


# ── Core generation loop ──────────────────────────────────────────────────────
def generate_for_category(llm, SamplingParams, category, n_target, real_by_cat,
                          seen_norm, rng, max_rounds=None, overgen=None):
    """Generate exactly ``n_target`` unique, valid emails for ``category``."""
    max_rounds = config.MAX_ROUNDS if max_rounds is None else max_rounds
    overgen    = config.OVERGEN    if overgen    is None else overgen
    params = _sampling(SamplingParams, category)
    kept, kept_pool_for_nd = [], list(real_by_cat.get(category, []))

    for rnd in range(max_rounds):
        deficit = n_target - len(kept)
        if deficit <= 0:
            break
        n_prompts = math.ceil(deficit * overgen)
        messages = [build_messages(category, real_by_cat, rng) for _ in range(n_prompts)]
        outputs = llm.chat(messages, params)
        raw_texts = [o.outputs[0].text for o in outputs]

        candidates = []
        for raw in raw_texts:
            t = clean_generation(raw)
            if not is_valid(t, category):
                continue
            key = normalize(t)
            if key in seen_norm:
                continue
            candidates.append((key, t))

        uniq, seen_round = [], set()
        for key, t in candidates:
            if key in seen_round:
                continue
            seen_round.add(key)
            uniq.append(t)

        fresh = filter_near_duplicates(kept_pool_for_nd, uniq)
        take = fresh[:deficit]
        for t in take:
            kept.append(t)
            seen_norm.add(normalize(t))
            kept_pool_for_nd.append(t)

        print(f"    [{category}] round {rnd+1}: prompted={n_prompts} "
              f"valid={len(uniq)} kept_new={len(take)} total={len(kept)}/{n_target}")

    if len(kept) < n_target:
        print(f"    ⚠️  [{category}] reached {len(kept)}/{n_target} after "
              f"{max_rounds} rounds (class will be SHORT — raise --max-rounds/--overgen).")
    return kept[:n_target]


def run_generation(llm, SamplingParams, df_real, deficits, rng,
                   max_rounds=None, overgen=None):
    """Generate synthetic emails for every category; returns a DataFrame."""
    t0 = time.time()
    real_by_cat = {c: df_real.loc[df_real["category"] == c, "text"].tolist()
                   for c in config.CATEGORIES}
    seen_norm = set(normalize(t) for t in df_real["text"])

    rows = []
    for category in sorted(config.CATEGORIES, key=lambda c: -deficits[c]):
        need = deficits[category]
        if need <= 0:
            print(f"  ▸ {category}: already at/above target, skipping.")
            continue
        print(f"  ▸ {category}: generating {need} synthetic emails…")
        emails = generate_for_category(llm, SamplingParams, category, need,
                                       real_by_cat, seen_norm, rng,
                                       max_rounds=max_rounds, overgen=overgen)
        if len(emails) < need:
            print(f"    ⚠️  {category}: only {len(emails)}/{need} — dataset will be IMBALANCED.")
        for e in emails:
            rows.append({"text": e, "category": category,
                         "target": config.CATEGORY_TARGET[category], "source": "synthetic"})

    df_syn = pd.DataFrame(rows)
    print(f"\n⏱️  Generation wall-clock: {(time.time()-t0)/60:.1f} min")
    return df_syn


# ── Optional model comparison (the fixed compare_models) ──────────────────────
def _bangla_scorecard(llm, SamplingParams, real_by_cat, rng, n_per_cat=8):
    """Quick quality proxy for one engine: valid-rate × bangla-rate × diversity."""
    valid = total = bangla = 0
    seen = set()
    for category in config.CATEGORIES:
        params = _sampling(SamplingParams, category)
        msgs = [build_messages(category, real_by_cat, rng) for _ in range(n_per_cat)]
        for o in llm.chat(msgs, params):
            total += 1
            t = clean_generation(o.outputs[0].text)
            if _BENGALI_RE.search(t):
                bangla += 1
            if is_valid(t, category):
                valid += 1
                seen.add(normalize(t))
    if total == 0:
        return 0.0, {}
    valid_rate = valid / total
    bangla_rate = bangla / total
    diversity = len(seen) / max(valid, 1)
    score = valid_rate * bangla_rate * diversity
    return score, {"valid_rate": round(valid_rate, 3),
                   "bangla_rate": round(bangla_rate, 3),
                   "diversity": round(diversity, 3)}


def compare_models(candidates, df_real, tp=1, n_per_cat=8):
    """Load each candidate *sequentially*, score, free, and return the winner.

    Only one engine is resident at a time (see :func:`free_engine`) — this is the
    structural fix for the Colab ``EngineDeadError`` OOM.
    """
    rng = random.Random(config.SEED)
    real_by_cat = {c: df_real.loc[df_real["category"] == c, "text"].tolist()
                   for c in config.CATEGORIES}
    scorecard = {}
    for model_id in candidates:
        print(f"\n=== scoring candidate: {model_id} ===")
        llm, SamplingParams = load_engine(model_id, tp=tp)
        try:
            score, detail = _bangla_scorecard(llm, SamplingParams, real_by_cat, rng, n_per_cat)
        finally:
            free_engine(llm)             # <-- freed before the next candidate loads
        scorecard[model_id] = {"score": round(score, 4), **detail}
        print(f"    score={score:.4f}  {detail}")

    winner = max(scorecard, key=lambda m: scorecard[m]["score"])
    print(f"\n🏆 winner: {winner}")
    return winner, scorecard


# ── Assembly + save (no google.colab) ─────────────────────────────────────────
def assemble_and_save(df_real, df_syn, out_dir=None):
    import os
    out_dir = out_dir or config.DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, "Bangla_Email_Dataset_Augmented.xlsx")
    out_csv  = os.path.join(out_dir, "Bangla_Email_Dataset_Augmented.csv")

    real = df_real.copy()
    real["source"] = "real"
    combined = pd.concat([real, df_syn], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)

    exact_dups  = combined["text"].duplicated().sum()
    real_dups   = real["text"].duplicated().sum()                # pre-exist in the source corpus
    syn_vs_real = int(df_syn["text"].isin(set(real["text"])).sum()) if len(df_syn) else 0
    bengali_ok  = df_syn["text"].str.contains(_BENGALI_RE).mean() if len(df_syn) else 1.0

    print("=" * 60)
    print("  FINAL DATASET")
    print("=" * 60)
    print(combined.groupby(["category", "source"]).size().unstack(fill_value=0))
    print("-" * 60)
    print(combined["category"].value_counts().sort_index().to_string())
    print("-" * 60)
    print(f"  Total rows               : {len(combined):,}")
    print(f"  Exact duplicate texts    : {exact_dups}  ({real_dups} pre-exist in the real corpus)")
    print(f"  Synthetic↔real collisions: {syn_vs_real}  (should be 0)")
    print(f"  Synthetic w/ Bangla      : {bengali_ok*100:.1f}%")

    combined.to_excel(out_xlsx, index=False)
    combined.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n💾 Saved: {out_xlsx}\n         {out_csv}")
    return combined
