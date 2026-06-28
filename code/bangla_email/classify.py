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
