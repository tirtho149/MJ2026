#!/usr/bin/env python
"""Zero-shot Bangla email classification eval on one Nova GPU.

  python run_classify.py                       # eval on 200 raw emails
  python run_classify.py --n-eval 1000 --preprocess
  python run_classify.py --data data/Bangla_Email_Dataset_Augmented.csv
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from bangla_email import config, data, classify, generate
from bangla_email.preprocess import BanglaEmailPreprocessor


def main():
    ap = argparse.ArgumentParser(description="Zero-shot Bangla email classification (vLLM)")
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    ap.add_argument("--data", default=None, help="CSV/XLSX to evaluate; default = raw corpus")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--preprocess", action="store_true",
                    help="run the 13-step preprocessor and classify the cleaned text")
    args = ap.parse_args()
    config.seed_everything(config.SEED)

    if args.data and os.path.exists(args.data):
        df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_excel(args.data)
    else:
        df = data.load_raw()

    text_col = "text"
    if args.preprocess:
        print("🧹 Preprocessing (13-step pipeline)…")
        pre = BanglaEmailPreprocessor()
        df = df.copy()
        df["text_clean"] = pre.fit_transform(df["text"])
        df = df[df["text_clean"].str.len() > 0]
        text_col = "text_clean"

    print(f"\n🤖 Loading engine: {args.model} (TP={args.tp})")
    llm, SamplingParams = generate.load_engine(args.model, tp=args.tp)
    try:
        metrics, _, _ = classify.evaluate(llm, SamplingParams, df,
                                          text_col=text_col, n_eval=args.n_eval)
    finally:
        generate.free_engine(llm)
    print("\n📊 metrics:", metrics)


if __name__ == "__main__":
    main()
