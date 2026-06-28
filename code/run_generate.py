#!/usr/bin/env python
"""Generate the class-balanced Bangla email dataset on one Nova GPU.

  python run_generate.py                         # full run, Qwen2.5-3B, all classes -> 3000 each
  python run_generate.py --per-class 50          # quick partial run
  python run_generate.py --model Qwen/Qwen2.5-1.5B-Instruct
  python run_generate.py --compare Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import random

from bangla_email import config, data, generate


def main():
    ap = argparse.ArgumentParser(description="Single-GPU Bangla email synthetic generation")
    ap.add_argument("--model", default=config.DEFAULT_MODEL, help="generation model id")
    ap.add_argument("--per-class", type=int, default=config.PER_CLASS_TARGET,
                    help="target emails per class (real + synthetic)")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size (GPUs)")
    ap.add_argument("--out-dir", default=config.DATA_DIR)
    ap.add_argument("--compare", nargs="+", default=None,
                    help="optional list of candidate models to compare & pick the winner")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = ap.parse_args()
    config.seed_everything(config.SEED)

    rng = random.Random(config.SEED)

    print("=" * 60)
    print("  BANGLA EMAIL — SYNTHETIC GENERATION (Nova, single GPU)")
    print("=" * 60)
    df_real = data.load_raw()
    deficits = data.balancing_plan(df_real, per_class=args.per_class)
    print("-" * 60)
    print(f"  per-class target : {args.per_class}")
    for c in config.CATEGORIES:
        print(f"     {c:<11} +synthetic={deficits[c]:>5}")
    print(f"  total synthetic  : {sum(deficits.values()):,}")
    print("-" * 60)

    # Optional: pick the best model first (OOM-safe sequential comparison).
    model_id = args.model
    if args.compare:
        model_id, scorecard = generate.compare_models(args.compare, df_real, tp=args.tp)
        print("  scorecard:", scorecard)

    print(f"\n🤖 Loading engine: {model_id}  (TP={args.tp})")
    llm, SamplingParams = generate.load_engine(
        model_id, tp=args.tp, max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_mem_util,
    )
    try:
        df_syn = generate.run_generation(llm, SamplingParams, df_real, deficits, rng)
    finally:
        generate.free_engine(llm)

    print(f"\nSynthetic rows generated: {len(df_syn)}")
    if len(df_syn):
        print(df_syn.groupby("category").size().to_string())
    generate.assemble_and_save(df_real, df_syn, out_dir=args.out_dir)
    print("\n✅ done.")


if __name__ == "__main__":
    main()
