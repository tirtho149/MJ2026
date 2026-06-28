#!/usr/bin/env python
"""Full data-creation pipeline on ONE Nova GPU, end to end.

  load corpus → (single vLLM engine) generate balanced dataset → zero-shot
  classify a balanced eval sample with the SAME engine → free engine →
  build all thesis figures + tables + REPORT.md.

  python run_all.py --per-class 3000            # 6×3000 = 18,000 balanced
  python run_all.py --per-class 2000 --eval-per-class 250

Only one engine is ever resident (the OOM fix), and it is reused for both
generation and classification so the model loads once.
"""

from __future__ import annotations

import argparse
import random
import time

import pandas as pd

from bangla_email import config, data, generate, classify, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    ap.add_argument("--per-class", type=int, default=config.PER_CLASS_TARGET)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--out-dir", default=config.DATA_DIR)
    ap.add_argument("--max-rounds", type=int, default=15)
    ap.add_argument("--overgen", type=float, default=1.4)
    ap.add_argument("--eval-per-class", type=int, default=250,
                    help="balanced sample per class for the zero-shot eval")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = ap.parse_args()

    rng = random.Random(config.SEED)
    t0 = time.time()

    print("=" * 64)
    print("  BANGLA EMAIL — FULL DATA-CREATION PIPELINE (Nova, single GPU)")
    print("=" * 64)
    df_real = data.load_raw()
    deficits = data.balancing_plan(df_real, per_class=args.per_class)
    print("-" * 64)
    print(f"  target/class : {args.per_class}   (=> {args.per_class*len(config.CATEGORIES):,} total)")
    for c in config.CATEGORIES:
        print(f"     {c:<11} +synthetic={deficits[c]:>5}")
    print(f"  total synthetic to generate: {sum(deficits.values()):,}")
    print("-" * 64)

    print(f"\n🤖 Loading engine: {args.model} (TP={args.tp})")
    llm, SamplingParams = generate.load_engine(
        args.model, tp=args.tp, max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_mem_util)

    try:
        # ── 1. generate the balanced dataset ──────────────────────────────────
        df_syn = generate.run_generation(llm, SamplingParams, df_real, deficits, rng,
                                         max_rounds=args.max_rounds, overgen=args.overgen)
        combined = generate.assemble_and_save(df_real, df_syn, out_dir=args.out_dir)

        # hard balance check
        counts = combined["category"].value_counts().reindex(config.CATEGORIES)
        balanced = counts.nunique() == 1
        print(f"\n  balance check: per-class counts = {counts.to_dict()}")
        print(f"  → {'✅ PERFECTLY BALANCED' if balanced else '⚠️ NOT fully balanced'}")

        # ── 2. zero-shot classification on a BALANCED eval sample ─────────────
        print("\n" + "=" * 64)
        print("  ZERO-SHOT CLASSIFICATION (balanced eval, same engine)")
        print("=" * 64)
        eval_rows = []
        for c in config.CATEGORIES:
            sub = combined[combined.category == c]
            n = min(args.eval_per_class, len(sub))
            eval_rows.append(sub.sample(n, random_state=config.SEED))
        eval_df = pd.concat(eval_rows).reset_index(drop=True)
        metrics, eval_df, _ = classify.evaluate(llm, SamplingParams, eval_df,
                                               text_col="text", n_eval=None)
        eval_df[["text", "category", "pred"]].to_csv(
            f"{args.out_dir}/zeroshot_eval_predictions.csv", index=False, encoding="utf-8-sig")
        class_metrics = (eval_df["category"].tolist(), eval_df["pred"].tolist())
    finally:
        generate.free_engine(llm)

    # ── 3. figures + tables + REPORT.md ──────────────────────────────────────
    wall = (time.time() - t0) / 60
    print("\n" + "=" * 64)
    print("  BUILDING FIGURES + TABLES")
    print("=" * 64)
    report.write_tables_and_report(
        df_real, combined, out_dir=args.out_dir,
        class_metrics=class_metrics, overall_acc=metrics["accuracy"],
        model_id=args.model, wall_clock_min=wall)

    print(f"\n✅ DONE in {wall:.1f} min — dataset + report in {args.out_dir}/")


if __name__ == "__main__":
    main()
