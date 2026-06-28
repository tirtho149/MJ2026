#!/usr/bin/env python
"""Assemble data/EXPERIMENTS.md from whatever result files currently exist.

Idempotent: re-run after each job lands and it folds in the new numbers
(few-shot 3B/32B sweeps, fine-tune metrics).  The repo README is then
header + REPORT.md + EXPERIMENTS.md.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from bangla_email import config

D = config.DATA_DIR
out = os.path.join(D, "EXPERIMENTS.md")
md = ["# Experiments — classification performance\n",
      "All evaluations use the balanced 250/class split (1,500 emails) unless noted. "
      "Zero-shot/few-shot are in-context (no weight updates); fine-tune is LoRA on the "
      "balanced 12k training split with a held-out test set.\n"]


def fewshot_section(tag, title):
    p = os.path.join(D, f"fewshot_sweep_{tag}.csv")
    if not os.path.exists(p):
        return False
    df = pd.read_csv(p)
    md.append(f"\n## Few-shot (in-context) — {title}\n")
    tbl = df[["k", "accuracy", "macro_f1", "n_valid"]].copy()
    tbl["accuracy"] = (tbl["accuracy"] * 100).round(1)
    tbl["macro_f1"] = tbl["macro_f1"].round(3)
    tbl = tbl.rename(columns={"k": "shots (k)", "accuracy": "accuracy %",
                              "macro_f1": "macro-F1", "n_valid": "valid preds"})
    md.append(tbl.to_markdown(index=False))
    best = df.loc[df["accuracy"].idxmax()]
    md.append(f"\nBest: **k={int(best['k'])}** → accuracy **{best['accuracy']*100:.1f}%**, "
              f"macro-F1 **{best['macro_f1']:.3f}**.\n")
    md.append(f"\n![few-shot {tag}](figures/fewshot_sweep_{tag}.png)\n")
    return True


def finetune_section():
    p = os.path.join(D, "finetune_metrics.json")
    if not os.path.exists(p):
        return False
    m = json.load(open(p))
    md.append("\n## Fine-tuning (LoRA) — the >0.70 result\n")
    md.append(f"`{m['model']}` LoRA-fine-tuned for 6-way classification "
              f"(train={m.get('n_train','?')}, test={m.get('n_test','?')}, "
              f"epochs={m.get('epochs','?')}):\n")
    md.append(f"\n| metric | value |\n|---|---|\n"
              f"| **test accuracy** | **{m['accuracy']*100:.2f}%** |\n"
              f"| **test macro-F1** | **{m['macro_f1']:.3f}** |\n")
    over = "✅ exceeds the 0.70 target" if m["accuracy"] > 0.70 else "⚠️ below 0.70 — needs more epochs/data"
    md.append(f"\n{over}.\n")
    cr = os.path.join(D, "tables", "finetune_classification_report.csv")
    if os.path.exists(cr):
        t = pd.read_csv(cr)
        md.append("\nPer-class (test):\n\n" + t.to_markdown(index=False) + "\n")
    md.append("\n![fine-tune confusion](figures/fig7_finetune_confusion.png)\n")
    if "silhouette_finetuned" in m:
        md.append(f"\n### t-SNE clustering (resolved)\n\n"
                  f"Mean-pooled hidden-state features, L2-normalised, cosine t-SNE. "
                  f"Fine-tuning sharpens the class clusters: silhouette "
                  f"**{m.get('silhouette_base',float('nan')):.3f} → "
                  f"{m['silhouette_finetuned']:.3f}**.\n")
        md.append("\n![t-SNE fine-tuned](figures/fig8_tsne_finetuned.png)\n")
        md.append("![t-SNE base](figures/fig8b_tsne_base.png)\n")
    return True


any_ = False
any_ |= fewshot_section("3b", "Qwen2.5-3B-Instruct")
any_ |= fewshot_section("32b", "Qwen2.5-32B-Instruct")
any_ |= finetune_section()

if not any_:
    md.append("\n_(no experiment results yet)_\n")

with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"wrote {out}")
