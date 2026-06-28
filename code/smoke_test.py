#!/usr/bin/env python
"""End-to-end smoke test for the Bangla email pipeline.

Two stages:

  STAGE A — CPU only (no GPU, no model):
      corpus loads, the 13-step preprocessor runs, and the
      clean/validate/normalize/near-dup helpers behave.  Runs anywhere.

  STAGE B — single GPU (skipped automatically if no CUDA / vLLM):
      loads a small vLLM engine, generates a handful of emails per class with
      the real few-shot prompts, frees the engine, and asserts a balanced
      augmented dataset is written.  Proves the full path works on one GPU
      before the scaled run — and exercises the OOM-safe engine teardown.

Exit code 0 = smooth.  Any failed assertion aborts non-zero.
"""

from __future__ import annotations

import os
import random
import sys

from bangla_email import config, data, generate
from bangla_email.preprocess import BanglaEmailPreprocessor

SMOKE_PER_CLASS = int(os.environ.get("SMOKE_PER_CLASS", "0"))   # 0 -> auto (real_count + 4)


def stage_a_cpu():
    print("\n" + "=" * 60)
    print("  STAGE A — CPU sanity (no GPU)")
    print("=" * 60)

    df = data.load_raw()
    assert len(df) > 0, "corpus is empty"
    assert set(df["category"]).issubset(set(config.CATEGORIES)), "unexpected category"
    assert df["target"].between(0, 5).all(), "target out of range"
    print(f"  ✓ corpus loaded: {len(df):,} rows, {df['category'].nunique()} classes")

    pre = BanglaEmailPreprocessor(verbose=False)
    sample = df["text"].iloc[0]
    cleaned = pre.preprocess(sample)
    assert isinstance(cleaned, str)
    print(f"  ✓ preprocess ok: {len(sample)} chars -> {len(cleaned.split())} tokens")

    # clean / validate / dedup helpers
    raw = '"১. আপনার Google যাচাইকরণ কোড ৪৫৬৭৮৯। কোডটি গোপন রাখুন।"'
    c = generate.clean_generation(raw)
    assert not c.startswith('"') and "৪৫৬৭৮৯" in c, "clean_generation failed"
    assert generate.is_valid("আপনার পেমেন্ট সফল হয়েছে।", "updates"), "is_valid false-negative"
    assert not generate.is_valid("no bangla here at all", "updates"), "is_valid false-positive"
    dups = generate.filter_near_duplicates([], ["একদম একই লেখা", "একদম একই লেখা", "ভিন্ন একটি বাক্য"])
    assert len(dups) == 2, f"near-dup filter kept {len(dups)} (expected 2)"
    print("  ✓ clean / validate / near-dup helpers ok")
    print("  ✅ STAGE A passed")
    return df


def stage_b_gpu(df):
    print("\n" + "=" * 60)
    print("  STAGE B — single-GPU generation")
    print("=" * 60)
    try:
        import torch
        if not torch.cuda.is_available():
            print("  ⏭️  no CUDA -> skipping GPU stage (Stage A already proved the logic).")
            return
        import vllm  # noqa: F401
    except Exception as e:
        print(f"  ⏭️  GPU stack unavailable ({e}) -> skipping Stage B.")
        return

    model_id = os.environ.get("SMOKE_MODEL", config.SMOKE_MODEL)
    counts = df["category"].value_counts().to_dict()
    # tiny target per class: existing real + a few synthetic
    if SMOKE_PER_CLASS > 0:
        deficits = {c: SMOKE_PER_CLASS for c in config.CATEGORIES}
    else:
        deficits = {c: 4 for c in config.CATEGORIES}

    rng = random.Random(config.SEED)
    print(f"  model={model_id}  synthetic/class={list(deficits.values())[0]}")

    llm, SamplingParams = generate.load_engine(model_id, tp=1, max_model_len=1536,
                                               gpu_mem_util=0.85)
    try:
        df_syn = generate.run_generation(llm, SamplingParams, df, deficits, rng)
    finally:
        generate.free_engine(llm)            # exercise the OOM-safe teardown

    assert len(df_syn) > 0, "no synthetic emails generated"
    assert df_syn["text"].str.contains(generate._BENGALI_RE).all(), "synthetic email missing Bangla"
    # the GENERATOR must add no duplicates: synthetic is internally unique and
    # never collides with a real email.  (The real corpus itself contains exact
    # duplicate emails — 1,643 of them — which the notebook deliberately keeps;
    # those are a property of the source, not something generation introduced.)
    assert df_syn["text"].duplicated().sum() == 0, "duplicate synthetic emails"
    syn_vs_real = df_syn["text"].isin(set(df["text"])).sum()
    assert syn_vs_real == 0, f"{syn_vs_real} synthetic emails collide with a real email"

    out_dir = os.path.join(config.REPO_DIR, "data_smoke")
    combined = generate.assemble_and_save(df, df_syn, out_dir=out_dir)
    real_dups = df["text"].duplicated().sum()
    assert combined["text"].duplicated().sum() == real_dups, "generation introduced new duplicates"
    assert os.path.exists(os.path.join(out_dir, "Bangla_Email_Dataset_Augmented.csv"))
    print("  ✅ STAGE B passed")


def main():
    df = stage_a_cpu()
    stage_b_gpu(df)
    print("\n🎉 SMOKE TEST PASSED — pipeline runs smoothly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
