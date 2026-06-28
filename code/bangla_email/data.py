"""Corpus loading — one canonical DataFrame regardless of the source format.

Resolution order (first that exists wins):
  1. ``config.RAW_CSV``   — the GitHub CSV (No. / Bangla Emails / Label)
  2. ``config.RAW_XLSX``  — the xlsx the original notebook uploaded interactively
  3. built-in seed        — :mod:`bangla_email.seed_data`, so smoke tests and
                            CPU-only checks run with zero external files.

Every path returns a DataFrame with exactly these columns:
    text (str) · category (str) · target (int)
"""

from __future__ import annotations

import os
import pandas as pd

from . import config


def _from_github_csv(path: str) -> pd.DataFrame:
    """Parse the GitHub CSV: columns ``No.``, ``Bangla Emails``, ``Label``.

    The published file has 5 rows whose ``Label`` is ``" 4"`` (leading space),
    so the column loads as a string — we strip and coerce to int defensively.
    """
    df = pd.read_csv(path)
    rename = {"Bangla Emails": "text", "Label": "target"}
    df = df.rename(columns=rename)

    df["text"] = df["text"].astype(str).str.strip()
    # tolerate stray whitespace / blanks in the integer label column
    df["target"] = pd.to_numeric(df["target"].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    df = df[df["target"].isin(config.TARGET_CATEGORY)]          # drop out-of-range labels
    df["category"] = df["target"].map(config.TARGET_CATEGORY)
    df = df[df["text"].str.len() > 0]
    return df[["text", "category", "target"]].reset_index(drop=True)


def _from_notebook_xlsx(path: str) -> pd.DataFrame:
    """Parse the original notebook xlsx: columns ``text``, ``category``, ``target``."""
    df = pd.read_excel(path)
    df = df[["text", "category", "target"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["target"] = df["target"].astype(int)
    df = df[df["category"].isin(config.CATEGORIES) & (df["text"].str.len() > 0)]
    return df[["text", "category", "target"]].reset_index(drop=True)


def _from_seed() -> pd.DataFrame:
    from . import seed_data
    return seed_data.as_dataframe()


def load_raw(verbose: bool = True) -> pd.DataFrame:
    """Load the real corpus, or fall back to the built-in seed."""
    if os.path.exists(config.RAW_CSV):
        df, src = _from_github_csv(config.RAW_CSV), config.RAW_CSV
    elif os.path.exists(config.RAW_XLSX):
        df, src = _from_notebook_xlsx(config.RAW_XLSX), config.RAW_XLSX
    else:
        df, src = _from_seed(), "<built-in seed>"

    if verbose:
        print(f"📥 Loaded corpus from: {src}")
        print(f"   rows: {len(df):,}")
        counts = df["category"].value_counts().reindex(config.CATEGORIES, fill_value=0)
        for c in config.CATEGORIES:
            print(f"     {c:<11} {counts[c]:>5}")
    return df


def balancing_plan(df_real: pd.DataFrame, per_class: int = config.PER_CLASS_TARGET):
    """Return ``{category: n_synthetic_needed}`` to reach ``per_class`` each."""
    counts = df_real["category"].value_counts().to_dict()
    return {c: max(0, per_class - counts.get(c, 0)) for c in config.CATEGORIES}
