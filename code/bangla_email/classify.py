"""Zero-shot Bangla email classification — refactor of the NLP notebook.

The notebook classified one email at a time with 4-bit transformers on a T4.
Here the same Qwen model is driven through the **single vLLM engine** in batch,
so an evaluation that took minutes per 100 emails runs in one batched call.
"""

from __future__ import annotations

import pandas as pd

from . import config, generate


def _build_classify_msgs(text: str):
    prompt = f'Email text (Bangla):\n"{text}"\n\nCategory:'
    return [
        {"role": "system", "content": config.CLASSIFY_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]


def _parse_label(raw: str) -> str:
    tok = raw.strip().lower().split()
    cand = tok[0] if tok else "unknown"
    cand = cand.strip(".,:;\"'()")
    return cand if cand in config.CATEGORIES_VALID else "unknown"


def classify_batch(llm, SamplingParams, texts, max_chars=300):
    """Classify a list of email texts; returns a list of predicted labels."""
    params = SamplingParams(temperature=0.0, max_tokens=8)
    msgs = [_build_classify_msgs(str(t)[:max_chars]) for t in texts]
    outputs = llm.chat(msgs, params)
    return [_parse_label(o.outputs[0].text) for o in outputs]


# ── Few-shot (in-context) classification ──────────────────────────────────────
import random as _random
from . import config as _cfg

DEMO_CHAR_CAP = 130          # truncate each demonstration email (Bangla is token-heavy)


def build_demo_pool(df, exclude_index=None, prefer_real=True, seed=_cfg.SEED):
    """Per-class shuffled pool of demonstration emails, with eval rows excluded."""
    pool = df.drop(index=exclude_index, errors="ignore") if exclude_index is not None else df
    if prefer_real and "source" in pool.columns:
        real = pool[pool["source"] == "real"]
        if len(real):
            pool = real
    rng = _random.Random(seed)
    by = {}
    for c in config.CATEGORIES:
        lst = pool.loc[pool["category"] == c, "text"].tolist()
        rng.shuffle(lst)
        by[c] = lst
    return by


def select_demos(demo_pool, k, cap=DEMO_CHAR_CAP):
    """Pick k demonstrations, round-robin across classes for max coverage."""
    demos, idx = [], {c: 0 for c in config.CATEGORIES}
    while len(demos) < k:
        progressed = False
        for c in config.CATEGORIES:
            if len(demos) >= k:
                break
            if idx[c] < len(demo_pool[c]):
                demos.append((demo_pool[c][idx[c]][:cap], c)); idx[c] += 1; progressed = True
        if not progressed:
            break
    return demos


def _build_fewshot_msgs(query, demos, max_query=300):
    parts = []
    if demos:
        parts.append("Here are labeled examples:\n")
        for t, lab in demos:
            parts.append(f"Email: {t}\nCategory: {lab}\n")
        parts.append("Now classify this email.")
    parts.append(f'Email: "{str(query)[:max_query]}"\nCategory:')
    return [
        {"role": "system", "content": config.CLASSIFY_SYSTEM_PROMPT},
        {"role": "user",   "content": "\n".join(parts)},
    ]


def classify_fewshot_batch(llm, SamplingParams, texts, demos):
    params = SamplingParams(temperature=0.0, max_tokens=8)
    msgs = [_build_fewshot_msgs(t, demos) for t in texts]
    return [_parse_label(o.outputs[0].text) for o in llm.chat(msgs, params)]


def evaluate_fewshot(llm, SamplingParams, df_eval, demos, text_col="text"):
    """k-shot eval; returns (metrics, df_with_preds, per_class_arrays)."""
    from sklearn.metrics import precision_recall_fscore_support
    preds = classify_fewshot_batch(llm, SamplingParams, df_eval[text_col].tolist(), demos)
    d = df_eval.copy(); d["pred"] = preds
    valid = d["pred"] != "unknown"
    acc = (d.loc[valid, "category"] == d.loc[valid, "pred"]).mean() if valid.any() else 0.0
    pr, rc, f1, sup = precision_recall_fscore_support(
        d["category"], d["pred"], labels=config.CATEGORIES, zero_division=0)
    metrics = {"k": len(demos), "accuracy": float(acc), "macro_f1": float(f1.mean()),
               "n_valid": int(valid.sum()), "n_eval": len(d)}
    return metrics, d, (pr, rc, f1, sup)


def evaluate(llm, SamplingParams, df, text_col="text", n_eval=None):
    """Zero-shot eval over (a sample of) ``df``; returns metrics + predictions."""
    eval_df = df.dropna(subset=[text_col]).copy()
    if n_eval is not None and n_eval < len(eval_df):
        eval_df = eval_df.sample(n_eval, random_state=config.SEED)

    eval_df["pred"] = classify_batch(llm, SamplingParams, eval_df[text_col].tolist())
    valid = eval_df["pred"] != "unknown"
    acc = (eval_df.loc[valid, "category"] == eval_df.loc[valid, "pred"]).mean() if valid.any() else 0.0

    print(f"  ✅ Zero-shot accuracy : {acc*100:.1f}%")
    print(f"  📋 Valid predictions  : {int(valid.sum())} / {len(eval_df)}")
    print(f"  ❓ Unknown / parse fail: {int((~valid).sum())}")

    report = None
    try:
        from sklearn.metrics import classification_report
        ev = eval_df[valid]
        report = classification_report(ev["category"], ev["pred"],
                                       labels=config.CATEGORIES, zero_division=0)
        print("\n" + report)
    except Exception as e:                       # pragma: no cover
        print(f"  (classification_report skipped: {e})")

    return {"accuracy": float(acc), "n_eval": len(eval_df),
            "n_valid": int(valid.sum())}, eval_df, report
