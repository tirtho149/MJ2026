#!/usr/bin/env python
"""LoRA fine-tuning of Qwen2.5 as a 6-way Bangla email classifier.

This is the experiment that pushes accuracy past the zero/few-shot ceiling.
Stratified 80/10/10 split of the balanced 12k dataset; LoRA adapters on a
sequence-classification head; report test accuracy / macro-F1 / per-class P/R/F1
/ confusion matrix.  Also extracts hidden-state features (base vs fine-tuned) and
runs t-SNE with a silhouette score — which is what resolves the poor clustering.

  python finetune.py                       # full: 3 epochs on Qwen2.5-3B
  python finetune.py --smoke               # tiny subset, 1 epoch (wiring check)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_recall_fscore_support, silhouette_score)
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from peft import LoraConfig, get_peft_model, TaskType

from bangla_email import config

LABELS = config.CATEGORIES
ID2LABEL = {i: c for i, c in enumerate(LABELS)}      # contiguous 0..5 == config targets
LABEL2ID = {c: i for i, c in enumerate(LABELS)}
COLORS = {"primary": "#4CAF50", "updates": "#2196F3", "spam": "#F44336",
          "promotions": "#FF9800", "social": "#9C27B0", "important": "#00BCD4"}


def compute_metrics(p):
    preds = np.asarray(p.predictions).argmax(-1)
    labels = p.label_ids
    return {"accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro")}


@torch.no_grad()
def extract_features(model, tok, texts, device, max_len=192, batch=32):
    """Mean-pooled last hidden state from the (LoRA-wrapped) backbone."""
    model.eval()
    feats = []
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for i in range(0, len(texts), batch):
        chunk = [str(t)[:600] for t in texts[i:i + batch]]
        enc = tok(chunk, return_tensors="pt", truncation=True, max_length=max_len,
                  padding=True).to(device)
        out = base.model(**enc, output_hidden_states=False)   # backbone forward
        h = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1)
        feats.append(pooled.float().cpu().numpy())
    return np.concatenate(feats, 0)


def tsne_figure(feats, labels, title, path):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import normalize
    X = normalize(feats)                                # L2 -> cosine geometry
    n = len(X)
    perp = max(5, min(30, (n - 1) // 3))
    emb = TSNE(n_components=2, perplexity=perp, init="pca", metric="cosine",
               random_state=config.SEED).fit_transform(X)
    try:
        sil = silhouette_score(emb, labels)
    except Exception:
        sil = float("nan")
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in LABELS:
        m = np.array(labels) == c
        if m.any():
            ax.scatter(emb[m, 0], emb[m, 1], s=22, alpha=0.75, label=c,
                       color=COLORS[c], edgecolors="white", linewidths=0.3)
    ax.set_title(f"{title}\n(silhouette={sil:.3f}, perplexity={perp})", fontweight="bold")
    ax.legend(markerscale=1.5, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return sil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    ap.add_argument("--data", default=os.path.join(config.DATA_DIR, "Bangla_Email_Dataset_Augmented.csv"))
    ap.add_argument("--out-dir", default=os.path.join(config.REPO_DIR, "runs", "ft"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--bf16", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    config.seed_everything(config.SEED)          # python / numpy / torch
    set_seed(config.SEED)                         # transformers (train sampling, init)
    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(config.DATA_DIR, "figures"); os.makedirs(fig_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── data: stratified 80/10/10 ─────────────────────────────────────────────
    df = pd.read_csv(args.data)
    df["label"] = df["category"].map(LABEL2ID)
    if args.smoke:
        df = df.groupby("category", group_keys=False).sample(60, random_state=config.SEED)
    train, tmp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=config.SEED)
    val, test  = train_test_split(tmp, test_size=0.5, stratify=tmp["label"], random_state=config.SEED)
    print(f"split: train={len(train)} val={len(val)} test={len(test)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_ds(d):
        ds = Dataset.from_pandas(d[["text", "label"]].rename(columns={"label": "labels"}),
                                 preserve_index=False)
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                      batched=True, remove_columns=["text"])
    ds_train, ds_val, ds_test = to_ds(train), to_ds(val), to_ds(test)

    # ── model + LoRA ──────────────────────────────────────────────────────────
    # Load in fp32 and let the Trainer's bf16=True do mixed-precision autocast.
    # Loading the whole model (incl. the fresh score head) in bf16 makes the
    # classification loss go NaN; fp32 weights + bf16 autocast is stable.
    # IMPORTANT: load in fp32. transformers 5.8.1 creates the new `score`
    # classification head in the config dtype (bf16) and leaves it NON-FINITE
    # (NaN), which makes logits/loss NaN before any training.  fp32 load +
    # explicit re-init of the head fixes it; bf16=True below then does stable
    # mixed-precision autocast with fp32 master weights.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        problem_type="single_label_classification", attn_implementation="eager",
        dtype=torch.float32)
    torch.nn.init.normal_(model.score.weight, mean=0.0, std=0.02)
    if getattr(model.score, "bias", None) is not None:
        torch.nn.init.zeros_(model.score.bias)
    assert torch.isfinite(model.score.weight).all(), "score head still non-finite"
    model.config.pad_token_id = tok.pad_token_id
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # checkpoints to node-local scratch — NFS checkpoint rotation throws
    # ".nfs ... Device or resource busy" while files are still open.
    local_out = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                             f"ft_{os.environ.get('SLURM_JOB_ID', 'local')}")
    targs = TrainingArguments(
        output_dir=local_out, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=64,
        gradient_accumulation_steps=1, learning_rate=args.lr, num_train_epochs=args.epochs,
        warmup_ratio=0.05, weight_decay=0.01, bf16=args.bf16, logging_steps=10,
        max_grad_norm=1.0, save_total_limit=1, report_to="none", dataloader_num_workers=2,
        seed=config.SEED, data_seed=config.SEED)
    trainer = Trainer(model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
                      processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute_metrics)

    trainer.train()

    # ── test evaluation ───────────────────────────────────────────────────────
    pred = trainer.predict(ds_test)
    y_pred = np.asarray(pred.predictions).argmax(-1)
    y_true = pred.label_ids
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABELS))), zero_division=0)
    print(f"\n★ TEST accuracy={acc*100:.2f}%  macro-F1={macro:.3f}")

    cr = pd.DataFrame({"category": LABELS, "precision": pr.round(3), "recall": rc.round(3),
                       "f1": f1.round(3), "support": sup})
    cr.to_csv(os.path.join(config.DATA_DIR, "tables", "finetune_classification_report.csv"),
              index=False)
    os.makedirs(os.path.join(config.DATA_DIR, "tables"), exist_ok=True)
    metrics = {"model": args.model, "accuracy": float(acc), "macro_f1": float(macro),
               "n_train": len(train), "n_test": len(test), "epochs": args.epochs}
    with open(os.path.join(config.DATA_DIR, "finetune_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # confusion matrix figure
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=40, ha="right")
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    th = cm.max() / 2 if cm.max() else 0
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > th else "black", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Fine-tuned Qwen — confusion (acc={acc*100:.1f}%)", fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig7_finetune_confusion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    trainer.save_model(os.path.join(args.out_dir, "adapter"))

    # ── t-SNE: base vs fine-tuned features (resolves the clustering) ──────────
    samp = pd.concat([test[test.category == c].sample(
        min(100, (test.category == c).sum()), random_state=config.SEED) for c in LABELS])
    texts, labs = samp["text"].tolist(), samp["category"].tolist()
    ft_feats = extract_features(model, tok, texts, device, max_len=args.max_len)
    sil_ft = tsne_figure(ft_feats, labs, "t-SNE of FINE-TUNED Qwen features",
                         os.path.join(fig_dir, "fig8_tsne_finetuned.png"))

    base = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), dtype=torch.bfloat16).to(device)
    base.config.pad_token_id = tok.pad_token_id
    base_feats = extract_features(base, tok, texts, device, max_len=args.max_len)
    sil_base = tsne_figure(base_feats, labs, "t-SNE of BASE Qwen features (no fine-tuning)",
                           os.path.join(fig_dir, "fig8b_tsne_base.png"))

    metrics["silhouette_base"] = float(sil_base)
    metrics["silhouette_finetuned"] = float(sil_ft)
    with open(os.path.join(config.DATA_DIR, "finetune_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"t-SNE silhouette: base={sil_base:.3f} -> fine-tuned={sil_ft:.3f}")
    print(cr.to_string(index=False))
    print("✅ fine-tune complete")


if __name__ == "__main__":
    main()
