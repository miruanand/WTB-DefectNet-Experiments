"""
evaluate.py
===========
Run this EXACTLY ONCE, after training is completely finished, on best.pt:

    python evaluate.py --data_root "D:\\...\\WTBs2025" --checkpoint ./runs/exp1/checkpoints/best.pt --out_dir ./runs/exp1

This is the only script in the project that touches the test split. It
rebuilds the SAME 60:20:20 stratified split used during training (same
seed, same fractions -> same test set every time, no leakage risk), runs
best.pt on it once, and writes:

    <out_dir>/eval/test_metrics.json   -- macro/micro metrics + per-class F1
    <out_dir>/eval/confusion_matrix.png
    <out_dir>/eval/classification_report.txt

Do NOT run this repeatedly while tuning anything -- if you go back and
change the model/hyperparameters after looking at these numbers and
re-run evaluate.py again, the test set has effectively become a second
validation set and these numbers stop being trustworthy for your paper.
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # no display needed, just save PNGs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from wtb.config import Config, CLASS_NAMES, NUM_CLASSES, set_seed, get_device
from wtb.model import build_model
from wtb.dataset import build_loaders
from wtb.utils import compute_metrics, load_checkpoint


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True,
                     help="Path to best.pt (NOT last.pt -- best.pt holds the "
                          "highest-val-F1 weights, which is what you should report).")
    ap.add_argument("--out_dir", type=str, default="./runs/exp1")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42,
                     help="MUST match the seed train.py used, or you will "
                          "reconstruct a different split and leak test data.")
    return ap.parse_args()


@torch.no_grad()
def run_inference(model, loader, device, channels_last: bool):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        logits, _ = model(imgs)
        probs = torch.softmax(logits, dim=1)
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
        y_true.extend(labels.tolist())
    return y_true, y_pred, np.array(y_prob)


def plot_roc_curves(y_true, y_prob: np.ndarray, class_names, out_path: str) -> dict:
    """
    One-vs-rest ROC curve + AUC per class, all on one figure, plus the
    micro-average ROC. Returns a {class_name: auc} dict for test_metrics.json.
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(9, 8))
    per_class_auc = {}
    for i, name in enumerate(class_names):
        # a class can be entirely absent/present in y_true_bin[:, i]; guard it
        if y_true_bin[:, i].sum() == 0 or y_true_bin[:, i].sum() == len(y_true_bin):
            per_class_auc[name] = float("nan")
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        per_class_auc[name] = roc_auc
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{name} (AUC={roc_auc:.3f})")

    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color="black", linestyle="--", linewidth=2,
            label=f"micro-average (AUC={auc_micro:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Test-set ROC curves (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    per_class_auc["micro_average"] = auc_micro
    return per_class_auc


def plot_confusion_matrix(cm: np.ndarray, class_names, out_path: str):
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test-set confusion matrix (row-normalized)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = Config(data_root=args.data_root, out_dir=args.out_dir,
                 batch_size=args.batch_size, num_workers=args.num_workers,
                 seed=args.seed)
    set_seed(cfg.seed)
    device = get_device()

    print(f"[evaluate] Rebuilding the same 60:20:20 split (seed={cfg.seed}) "
          f"used during training -- only the test_loader below will be used.")
    _, _, test_loader, train_counts = build_loaders(
        data_root=cfg.data_root, img_size=cfg.img_size, batch_size=cfg.batch_size,
        val_fraction=cfg.val_fraction, test_fraction=cfg.test_fraction,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, seed=cfg.seed,
    )

    print(f"[evaluate] Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    classes = ckpt.get("classes", CLASS_NAMES)
    print(f"[evaluate] Checkpoint is from epoch {ckpt.get('epoch', '?')}, "
          f"best_f1 recorded at save time = {ckpt.get('best_f1', '?')}")

    # Older checkpoints (exp1, Try_2, Try_4) predate the "backbone" field
    # and default to "dsps" -- exactly what they were trained with.
    cfg.backbone = ckpt.get("backbone") or "dsps"
    print(f"[evaluate] backbone = {cfg.backbone}")
    model = build_model(cfg, NUM_CLASSES).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["state_dict"])
    model.head.set_prior(train_counts)   # harmless at eval time (LTCP only applies it in train mode)

    print(f"[evaluate] Running inference on the TEST set "
          f"({len(test_loader.dataset)} images) -- this is a ONE-TIME touch.")
    y_true, y_pred, y_prob = run_inference(model, test_loader, device, cfg.channels_last)

    metrics = compute_metrics(y_true, y_pred, NUM_CLASSES)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    report_txt = classification_report(
        y_true, y_pred, target_names=classes, digits=4, zero_division=0
    )

    eval_dir = os.path.join(cfg.out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    roc_auc_per_class = plot_roc_curves(
        y_true, y_prob, classes, os.path.join(eval_dir, "roc_curves.png")
    )
    metrics["roc_auc"] = roc_auc_per_class

    print("\n=== TEST SET RESULTS ===")
    print(f"  accuracy        : {metrics['accuracy']:.4f}")
    print(f"  macro precision : {metrics['macro_precision']:.4f}")
    print(f"  macro recall    : {metrics['macro_recall']:.4f}")
    print(f"  macro F1        : {metrics['macro_f1']:.4f}")
    print(f"  balanced acc    : {metrics['balanced_acc']:.4f}")
    print(f"  cohen's kappa   : {metrics['kappa']:.4f}")
    print(f"  MCC             : {metrics['mcc']:.4f}")
    print("\n  per-class F1:")
    for name, f1 in zip(classes, metrics["per_class_f1"]):
        print(f"    {name:24s}: {f1:.4f}")

    with open(os.path.join(eval_dir, "test_metrics.json"), "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": ckpt.get("epoch"),
            "n_test_images": len(test_loader.dataset),
            "class_names": classes,
            **metrics,
        }, f, indent=2)

    with open(os.path.join(eval_dir, "classification_report.txt"), "w") as f:
        f.write(report_txt)

    plot_confusion_matrix(cm, classes, os.path.join(eval_dir, "confusion_matrix.png"))

    print(f"\n[evaluate] Saved:")
    print(f"    {eval_dir}\\test_metrics.json")
    print(f"    {eval_dir}\\classification_report.txt")
    print(f"    {eval_dir}\\confusion_matrix.png")
    print(f"    {eval_dir}\\roc_curves.png")
    print("\n[evaluate] Next: python gradcam.py --data_root <...> --checkpoint "
          f"{args.checkpoint} --out_dir {cfg.out_dir}")


if __name__ == "__main__":
    main()
