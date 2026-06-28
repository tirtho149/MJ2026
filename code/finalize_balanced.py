#!/usr/bin/env python
"""Down-sample the 3000/class superset to a fixed balanced TARGET per class.

Keeps every real email (so no real data is wasted) and fills the rest with
synthetic up to TARGET, giving an exactly-balanced dataset.  CPU only — the GPU
already produced the 3000/class superset; this just selects from it.

  python finalize_balanced.py --target 2000
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from bangla_email import config, data, report

SEED = config.SEED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000, help="emails per class in the final set")
    ap.add_argument("--data-dir", default=config.DATA_DIR)
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    args = ap.parse_args()

    src = os.path.join(args.data_dir, "Bangla_Email_Dataset_Augmented.csv")
    full = pd.read_csv(src)
    # preserve the 3000/class superset under a _full name
    full.to_csv(os.path.join(args.data_dir, "Bangla_Email_Dataset_Augmented_full3000.csv"),
                index=False, encoding="utf-8-sig")

    parts = []
    print(f"Down-sampling to {args.target}/class (keep all real + fill synthetic):")
    for c in config.CATEGORIES:
        sub  = full[full.category == c]
        real = sub[sub.source == "real"]
        syn  = sub[sub.source == "synthetic"]
        if len(real) >= args.target:                     # plenty of real -> sample real only
            keep = real.sample(args.target, random_state=SEED)
        else:
            need = args.target - len(real)
            keep = pd.concat([real, syn.sample(min(need, len(syn)), random_state=SEED)])
        print(f"  {c:<11} real={len(real):>4} + syn={len(keep)-len(real):>4} = {len(keep)}")
        parts.append(keep)

    final = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    final.to_csv(src, index=False, encoding="utf-8-sig")
    final.to_excel(os.path.join(args.data_dir, "Bangla_Email_Dataset_Augmented.xlsx"), index=False)

    counts = final["category"].value_counts().reindex(config.CATEGORIES)
    print(f"\nFinal per-class counts: {counts.to_dict()}")
    print(f"Balanced? {'✅ YES (all equal, 0 spread)' if counts.nunique()==1 else '⚠️ NO'}  "
          f"total={len(final):,}")

    # rebuild the report on the final set (reuse the zero-shot eval predictions)
    df_real = data.load_raw(verbose=False)
    class_metrics, acc = None, None
    pred = os.path.join(args.data_dir, "zeroshot_eval_predictions.csv")
    if os.path.exists(pred):
        pe = pd.read_csv(pred)
        valid = pe["pred"] != "unknown"
        acc = (pe.loc[valid, "category"] == pe.loc[valid, "pred"]).mean()
        class_metrics = (pe["category"].tolist(), pe["pred"].tolist())
    report.write_tables_and_report(df_real, final, out_dir=args.data_dir,
                                   class_metrics=class_metrics, overall_acc=acc,
                                   model_id=args.model)
    print("✅ finalized + report rebuilt.")


if __name__ == "__main__":
    main()
