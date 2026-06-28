#!/usr/bin/env python
"""Few-shot (in-context) classification sweep on one GPU.

Sweeps k ∈ {shots} demonstrations (balanced round-robin across the 6 classes,
drawn from real emails held out of the eval split) and reports accuracy +
macro-F1 on the same balanced 250/class eval used everywhere else.

  python run_fewshot_sweep.py --model Qwen/Qwen2.5-3B-Instruct  --tag 3b
  python run_fewshot_sweep.py --model Qwen/Qwen2.5-32B-Instruct --tag 32b --tp 1
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from bangla_email import config, generate, classify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    ap.add_argument("--tag", required=True, help="short label for output files, e.g. 3b / 32b")
    ap.add_argument("--shots", default="0,2,4,8,10,12,16")
    ap.add_argument("--data", default=os.path.join(config.DATA_DIR, "Bangla_Email_Dataset_Augmented.csv"))
    ap.add_argument("--eval-per-class", type=int, default=250)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--out-dir", default=config.DATA_DIR)
    args = ap.parse_args()

    shots = [int(s) for s in args.shots.split(",")]
    df = pd.read_csv(args.data)

    # balanced eval split (same seed everywhere) + held-out demo pool
    eval_parts = []
    for c in config.CATEGORIES:
        sub = df[df.category == c]
        eval_parts.append(sub.sample(min(args.eval_per_class, len(sub)), random_state=config.SEED))
    eval_df = pd.concat(eval_parts)
    demo_pool = classify.build_demo_pool(df, exclude_index=eval_df.index)
    eval_df = eval_df.reset_index(drop=True)
    print(f"eval rows: {len(eval_df)} ({args.eval_per_class}/class) · model={args.model}")

    print(f"\n🤖 Loading engine: {args.model} (TP={args.tp}, ctx={args.max_model_len})")
    llm, SamplingParams = generate.load_engine(
        args.model, tp=args.tp, max_model_len=args.max_model_len, gpu_mem_util=args.gpu_mem_util)

    rows = []
    try:
        for k in shots:
            demos = classify.select_demos(demo_pool, k) if k else []
            metrics, _, (pr, rc, f1, sup) = classify.evaluate_fewshot(
                llm, SamplingParams, eval_df, demos)
            perclass = {f"f1_{c}": round(float(v), 3) for c, v in zip(config.CATEGORIES, f1)}
            rows.append({**{k_: metrics[k_] for k_ in ("k", "accuracy", "macro_f1", "n_valid")},
                         **perclass})
            print(f"  k={k:>2}: acc={metrics['accuracy']*100:5.1f}%  "
                  f"macroF1={metrics['macro_f1']:.3f}  valid={metrics['n_valid']}/{metrics['n_eval']}")
    finally:
        generate.free_engine(llm)

    res = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, f"fewshot_sweep_{args.tag}.csv")
    res.to_csv(out_csv, index=False)
    print(f"\n💾 {out_csv}")

    # curve figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(res["k"], res["accuracy"] * 100, "o-", label="accuracy %", color="#1565C0", lw=2)
    ax.plot(res["k"], res["macro_f1"] * 100, "s--", label="macro-F1 ×100", color="#EF6C00", lw=2)
    for x, y in zip(res["k"], res["accuracy"] * 100):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax.set_xlabel("number of in-context demonstrations (k)")
    ax.set_ylabel("score"); ax.set_ylim(0, 100)
    ax.set_title(f"Few-shot classification — {args.model.split('/')[-1]}", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "figures", f"fewshot_sweep_{args.tag}.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"🖼️  {out_png}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
