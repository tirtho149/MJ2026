"""Thesis-ready figures + tables for the data-creation step.

Produces, into ``out_dir``:
  figures/  fig1_balance_before_after.png   raw (imbalanced) vs final (balanced)
            fig2_composition.png            real vs synthetic per class
            fig3_length_distribution.png    char-length histograms
            fig4_confusion_matrix.png       zero-shot confusion matrix
            fig5_perclass_metrics.png       precision / recall / F1 per class
  tables/   balance_table.csv               per-class real / synthetic / total
            classification_report.csv       per-class P / R / F1 / support
            sample_emails.md                a few real+synthetic samples/class
  REPORT.md                                 every number + figure, ready to paste

All figures use English class labels (matplotlib lacks Bengali glyphs); Bangla
text appears only in the markdown sample table, which renders fine on GitHub.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")                       # headless: no display on compute nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

CATEGORY_COLORS = {
    "primary": "#4CAF50", "updates": "#2196F3", "spam": "#F44336",
    "promotions": "#FF9800", "social": "#9C27B0", "important": "#00BCD4",
}
ORDER = config.CATEGORIES


def _dirs(out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    tab_dir = os.path.join(out_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    return fig_dir, tab_dir


# ── Figure 1: the headline — imbalance fixed ──────────────────────────────────
def fig_balance(df_real, df_combined, fig_dir):
    raw = df_real["category"].value_counts().reindex(ORDER, fill_value=0)
    bal = df_combined["category"].value_counts().reindex(ORDER, fill_value=0)

    x = np.arange(len(ORDER)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w/2, raw.values, w, label="Raw corpus (imbalanced)", color="#B0BEC5", edgecolor="white")
    ax.bar(x + w/2, bal.values, w, label="Final dataset (balanced)",
           color=[CATEGORY_COLORS[c] for c in ORDER], edgecolor="white")
    for i, v in enumerate(raw.values): ax.text(i - w/2, v + 30, str(v), ha="center", fontsize=8)
    for i, v in enumerate(bal.values): ax.text(i + w/2, v + 30, str(v), ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=20)
    ax.set_ylabel("Number of emails"); ax.set_title(
        "Class balance: raw corpus vs final augmented dataset", fontweight="bold")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); p = os.path.join(fig_dir, "fig1_balance_before_after.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); return p


# ── Figure 2: real vs synthetic composition ───────────────────────────────────
def fig_composition(df_combined, fig_dir):
    comp = (df_combined.groupby(["category", "source"]).size()
            .unstack(fill_value=0).reindex(ORDER))
    real = comp.get("real", pd.Series(0, index=ORDER))
    syn  = comp.get("synthetic", pd.Series(0, index=ORDER))
    x = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x, real.values, 0.6, label="real", color="#455A64", edgecolor="white")
    ax.bar(x, syn.values, 0.6, bottom=real.values, label="synthetic", color="#FFB74D", edgecolor="white")
    for i in x:
        ax.text(i, real.values[i]/2, str(int(real.values[i])), ha="center", color="white", fontsize=8)
        ax.text(i, real.values[i] + syn.values[i]/2, str(int(syn.values[i])), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=20)
    ax.set_ylabel("Number of emails")
    ax.set_title("Final dataset composition: real + synthetic per class", fontweight="bold")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); p = os.path.join(fig_dir, "fig2_composition.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); return p


# ── Figure 3: text-length distribution (real vs synthetic) ────────────────────
def fig_length(df_combined, fig_dir):
    d = df_combined.copy()
    d["len"] = d["text"].astype(str).str.len().clip(upper=600)
    fig, ax = plt.subplots(figsize=(11, 5))
    for src, color in [("real", "#455A64"), ("synthetic", "#FF9800")]:
        s = d.loc[d["source"] == src, "len"] if "source" in d else d["len"]
        if len(s):
            ax.hist(s, bins=50, alpha=0.6, label=f"{src} (median={s.median():.0f})", color=color)
    ax.set_xlabel("Character length (clipped @600)"); ax.set_ylabel("Frequency")
    ax.set_title("Email length distribution", fontweight="bold")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); p = os.path.join(fig_dir, "fig3_length_distribution.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); return p


# ── Figures 4 & 5: classification on the balanced set ─────────────────────────
def fig_classification(y_true, y_pred, fig_dir):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    labels = ORDER
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Zero-shot confusion matrix (balanced eval)", fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); p4 = os.path.join(fig_dir, "fig4_confusion_matrix.png")
    fig.savefig(p4, dpi=150, bbox_inches="tight"); plt.close(fig)

    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    x = np.arange(len(labels)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w, pr, w, label="precision", color="#42A5F5")
    ax.bar(x,     rc, w, label="recall",    color="#66BB6A")
    ax.bar(x + w, f1, w, label="F1",        color="#FFA726")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Per-class zero-shot metrics (balanced eval)", fontweight="bold")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); p5 = os.path.join(fig_dir, "fig5_perclass_metrics.png")
    fig.savefig(p5, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p4, p5, (pr, rc, f1, sup)


# ── Tables + REPORT.md ────────────────────────────────────────────────────────
def write_tables_and_report(df_real, df_combined, out_dir,
                            class_metrics=None, overall_acc=None, model_id="",
                            wall_clock_min=None):
    fig_dir, tab_dir = _dirs(out_dir)

    p1 = fig_balance(df_real, df_combined, fig_dir)
    p2 = fig_composition(df_combined, fig_dir)
    p3 = fig_length(df_combined, fig_dir)

    # balance table
    comp = (df_combined.groupby(["category", "source"]).size()
            .unstack(fill_value=0).reindex(ORDER))
    bt = pd.DataFrame({
        "category": ORDER,
        "real":      [int(comp.get("real", pd.Series(0, index=ORDER))[c]) for c in ORDER],
        "synthetic": [int(comp.get("synthetic", pd.Series(0, index=ORDER))[c]) for c in ORDER],
    })
    bt["total"] = bt["real"] + bt["synthetic"]
    bt.to_csv(os.path.join(tab_dir, "balance_table.csv"), index=False)

    # sample emails per class
    sample_lines = ["# Sample emails per class\n"]
    for c in ORDER:
        sample_lines.append(f"\n## {c} ({config.CATEGORY_BN.get(c,'')})\n")
        for src in ["real", "synthetic"]:
            sub = df_combined[(df_combined.category == c) & (df_combined.get("source") == src)]
            if len(sub):
                sample_lines.append(f"\n**{src}:**\n")
                for t in sub["text"].head(3):
                    sample_lines.append(f"- {str(t).strip()[:200]}")
    with open(os.path.join(tab_dir, "sample_emails.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(sample_lines))

    # classification figures + table
    cls_md = ""
    if class_metrics is not None:
        y_true, y_pred = class_metrics
        p4, p5, (pr, rc, f1, sup) = fig_classification(y_true, y_pred, fig_dir)
        cr = pd.DataFrame({"category": ORDER, "precision": pr.round(3),
                           "recall": rc.round(3), "f1": f1.round(3), "support": sup})
        cr.to_csv(os.path.join(tab_dir, "classification_report.csv"), index=False)
        macro_f1 = float(np.mean(f1))
        cls_md = (
            f"\n## Zero-shot classification on the balanced set\n\n"
            f"Model: `{model_id}` · overall accuracy: "
            f"**{(overall_acc or 0)*100:.1f}%** · macro-F1: **{macro_f1:.3f}**\n\n"
            + cr.to_markdown(index=False) +
            f"\n\n![confusion](figures/{os.path.basename(p4)})\n"
            f"![per-class](figures/{os.path.basename(p5)})\n"
        )

    # REPORT.md
    raw_counts = df_real["category"].value_counts().reindex(ORDER, fill_value=0)
    imbalance_ratio = raw_counts.max() / max(raw_counts.min(), 1)
    bal_counts = df_combined["category"].value_counts().reindex(ORDER, fill_value=0)
    wc = f"{wall_clock_min:.1f} min" if wall_clock_min is not None else "n/a"
    md = [
        "# Bangla Email Dataset — Data Creation Report\n",
        f"Generation model: `{model_id}` · wall-clock: {wc}\n",
        "## 1. The problem: the raw corpus is imbalanced\n",
        f"The raw corpus has **{len(df_real):,}** emails with a "
        f"**{imbalance_ratio:.2f}×** imbalance between the largest and smallest class "
        "(min {} `{}` … max {} `{}`). Class imbalance depresses per-class precision/recall, "
        "which is exactly what the zero-shot evaluation showed.\n".format(
            int(raw_counts.min()), raw_counts.idxmin(),
            int(raw_counts.max()), raw_counts.idxmax()),
        "## 2. The fix: balance every class to a fixed target\n",
        f"Every class is up-sampled to **{int(bal_counts.max()):,}** "
        f"(real kept, only the deficit synthesised), giving a perfectly balanced "
        f"**{len(df_combined):,}-email** dataset.\n",
        bt.to_markdown(index=False) + "\n",
        f"Balanced? **{'YES — all classes equal' if bal_counts.nunique()==1 else 'NO'}** "
        f"(per-class counts: {sorted(set(bal_counts.values))}).\n",
        "![balance](figures/fig1_balance_before_after.png)\n",
        "## 3. Composition & length\n",
        "![composition](figures/fig2_composition.png)\n",
        "![length](figures/fig3_length_distribution.png)\n",
        cls_md,
        "## Files\n",
        "- `Bangla_Email_Dataset_Augmented.csv` / `.xlsx` — the balanced dataset\n"
        "- `tables/` — CSV tables · `figures/` — PNGs · `tables/sample_emails.md` — samples\n",
    ]
    with open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"📝 Report written: {os.path.join(out_dir, 'REPORT.md')}")
    return os.path.join(out_dir, "REPORT.md")
