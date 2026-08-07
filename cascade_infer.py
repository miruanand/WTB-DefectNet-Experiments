"""
cascade_infer.py
=================
Combines your main 9-class model (e.g. Try_9's best.pt) with the binary
"localized damage" vs "coating detachment" specialist from train_cascade.py.

Runs the main model on the full test set as usual. Any image the main
model predicts as EITHER "localized damage" OR "coating detachment" gets
a second opinion from the specialist, which overrides the main model's
prediction for just that image. Every other prediction is left untouched.

    python cascade_infer.py --data_root "C:\\...\\WTBs2025" ^
        --main_checkpoint runs\\Try_9\\checkpoints\\best.pt --main_img_size 384 ^
        --cascade_checkpoint runs\\Try_9_cascade\\checkpoints\\best.pt --cascade_img_size 384 ^
        --out_dir runs\\Try_9_cascade_eval --tta

Writes the same three artifacts evaluate.py does (test_metrics.json,
classification_report.txt, confusion_matrix.png) into --out_dir/eval, PLUS
a before/after comparison so you can see exactly how much the cascade
step helped.
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from wtb.config import Config, CLASS_NAMES, NUM_CLASSES, set_seed, get_device
from wtb.model import build_model
from wtb.dataset import index_dataset, stratified_split_3way, build_transforms, WTBDataset
from wtb.utils import compute_metrics, load_checkpoint
from torch.utils.data import DataLoader

from train_cascade import CASCADE_TO_MAIN_IDX   # {0: "localized damage" idx, 1: "coating detachment" idx}

MAIN_IDX_TO_CASCADE_LABEL = {v: k for k, v in CASCADE_TO_MAIN_IDX.items()}  # {2: 0, 6: 1}
CASCADE_TRIGGER_IDXS = set(CASCADE_TO_MAIN_IDX.values())  # {2, 6}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--main_checkpoint", type=str, required=True)
    ap.add_argument("--main_img_size", type=int, default=384,
                     help="MUST match the img_size the main checkpoint was trained "
                          "with (384 if you already bumped it for Try_9; 224 for "
                          "earlier runs). Checkpoints don't store this, so it's not "
                          "auto-detected -- double check before trusting the numbers.")
    ap.add_argument("--cascade_checkpoint", type=str, required=True)
    ap.add_argument("--cascade_img_size", type=int, default=384)
    ap.add_argument("--out_dir", type=str, default="./runs/cascade_eval")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42,
                     help="Must match the seed train.py used to build the main "
                          "model's test split, or this reconstructs a different split.")
    ap.add_argument("--tta", action="store_true",
                     help="4-way flip TTA on the MAIN model only (matches how you "
                          "evaluated Try_9). The cascade specialist runs single-view.")
    return ap.parse_args()


@torch.no_grad()
def infer_main(model, loader, device, channels_last, tta):
    model.eval()
    y_pred, y_prob = [], []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        if tta:
            views = [imgs, torch.flip(imgs, dims=[3]), torch.flip(imgs, dims=[2]),
                      torch.flip(imgs, dims=[2, 3])]
            probs = None
            for v in views:
                logits, _ = model(v)
                p = torch.softmax(logits, dim=1)
                probs = p if probs is None else probs + p
            probs = probs / len(views)
        else:
            logits, _ = model(imgs)
            probs = torch.softmax(logits, dim=1)
        y_pred.extend(probs.argmax(dim=1).cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
    return y_pred, np.array(y_prob)


@torch.no_grad()
def infer_cascade(model, loader, device, channels_last):
    model.eval()
    y_pred = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        logits, _ = model(imgs)
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return y_pred


def plot_confusion_matrix(cm, class_names, out_path, title):
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
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
    set_seed(args.seed)
    device = get_device()

    # ---- rebuild the SAME 9-class test split evaluate.py / train.py used ----
    samples = index_dataset(args.data_root)
    _, _, test_samples = stratified_split_3way(samples, 0.2, 0.2, args.seed)
    print(f"[cascade_infer] Test set: {len(test_samples)} images (same split as evaluate.py)")

    y_true = [label for _, label in test_samples]

    # ---- load main model ----
    print(f"[cascade_infer] Loading MAIN checkpoint: {args.main_checkpoint}")
    main_ckpt = load_checkpoint(args.main_checkpoint, map_location=device)
    main_classes = main_ckpt.get("classes", CLASS_NAMES)
    cfg = Config(data_root=args.data_root, batch_size=args.batch_size)
    cfg.backbone = main_ckpt.get("backbone") or "dsps"
    main_model = build_model(cfg, NUM_CLASSES).to(device)
    if cfg.channels_last:
        main_model = main_model.to(memory_format=torch.channels_last)
    main_model.load_state_dict(main_ckpt["state_dict"])

    main_ds = WTBDataset(test_samples, build_transforms(args.main_img_size, train=False))
    main_loader = DataLoader(main_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=cfg.pin_memory)

    print(f"[cascade_infer] Running MAIN model inference (img_size={args.main_img_size}"
          f"{', TTA' if args.tta else ''})...")
    y_pred_main, y_prob_main = infer_main(main_model, main_loader, device, cfg.channels_last, args.tta)

    # ---- baseline metrics (main model alone, for comparison) ----
    metrics_before = compute_metrics(y_true, y_pred_main, NUM_CLASSES)
    print(f"\n[cascade_infer] BEFORE cascade (main model only):")
    print(f"    accuracy  : {metrics_before['accuracy']:.4f}")
    print(f"    macro F1  : {metrics_before['macro_f1']:.4f}")

    # ---- find images that need a second opinion ----
    trigger_indices = [i for i, p in enumerate(y_pred_main) if p in CASCADE_TRIGGER_IDXS]
    print(f"\n[cascade_infer] {len(trigger_indices)}/{len(test_samples)} test images "
          f"predicted as 'localized damage' or 'coating detachment' -- "
          f"routing these through the specialist.")

    y_pred_final = list(y_pred_main)   # copy; only trigger_indices get overwritten

    if trigger_indices:
        print(f"[cascade_infer] Loading CASCADE checkpoint: {args.cascade_checkpoint}")
        cascade_ckpt = load_checkpoint(args.cascade_checkpoint, map_location=device)
        cascade_cfg = Config(data_root=args.data_root, batch_size=args.batch_size)
        cascade_cfg.backbone = cascade_ckpt.get("backbone") or "resnet18"
        cascade_model = build_model(cascade_cfg, num_classes=2).to(device)
        if cascade_cfg.channels_last:
            cascade_model = cascade_model.to(memory_format=torch.channels_last)
        cascade_model.load_state_dict(cascade_ckpt["state_dict"])

        trigger_samples = [test_samples[i] for i in trigger_indices]
        cascade_ds = WTBDataset(trigger_samples, build_transforms(args.cascade_img_size, train=False))
        cascade_loader = DataLoader(cascade_ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=cascade_cfg.pin_memory)

        print(f"[cascade_infer] Running specialist on {len(trigger_indices)} images...")
        cascade_preds = infer_cascade(cascade_model, cascade_loader, device, cascade_cfg.channels_last)

        n_changed = 0
        for local_i, global_i in enumerate(trigger_indices):
            new_main_idx = CASCADE_TO_MAIN_IDX[cascade_preds[local_i]]
            if new_main_idx != y_pred_final[global_i]:
                n_changed += 1
            y_pred_final[global_i] = new_main_idx
        print(f"[cascade_infer] Specialist changed {n_changed}/{len(trigger_indices)} predictions.")

    # ---- final metrics ----
    metrics_after = compute_metrics(y_true, y_pred_final, NUM_CLASSES)
    cm_before = confusion_matrix(y_true, y_pred_main, labels=list(range(NUM_CLASSES)))
    cm_after = confusion_matrix(y_true, y_pred_final, labels=list(range(NUM_CLASSES)))
    report_after = classification_report(
        y_true, y_pred_final, target_names=main_classes, digits=4, zero_division=0
    )

    print(f"\n=== AFTER cascade ===")
    print(f"  accuracy  : {metrics_after['accuracy']:.4f}  "
          f"(was {metrics_before['accuracy']:.4f}, delta {metrics_after['accuracy']-metrics_before['accuracy']:+.4f})")
    print(f"  macro F1  : {metrics_after['macro_f1']:.4f}  "
          f"(was {metrics_before['macro_f1']:.4f}, delta {metrics_after['macro_f1']-metrics_before['macro_f1']:+.4f})")
    print("\n  per-class F1 (after):")
    for name, f1 in zip(main_classes, metrics_after["per_class_f1"]):
        print(f"    {name:24s}: {f1:.4f}")

    eval_dir = os.path.join(args.out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "test_metrics.json"), "w") as f:
        json.dump({
            "main_checkpoint": args.main_checkpoint,
            "cascade_checkpoint": args.cascade_checkpoint,
            "n_test_images": len(test_samples),
            "n_routed_through_cascade": len(trigger_indices),
            "metrics_before_cascade": metrics_before,
            "metrics_after_cascade": metrics_after,
        }, f, indent=2)
    with open(os.path.join(eval_dir, "classification_report.txt"), "w") as f:
        f.write(report_after)
    plot_confusion_matrix(cm_before, main_classes, os.path.join(eval_dir, "confusion_matrix_before.png"),
                           "Test-set confusion matrix -- BEFORE cascade")
    plot_confusion_matrix(cm_after, main_classes, os.path.join(eval_dir, "confusion_matrix_after.png"),
                           "Test-set confusion matrix -- AFTER cascade")

    print(f"\n[cascade_infer] Saved:")
    print(f"    {eval_dir}\\test_metrics.json")
    print(f"    {eval_dir}\\classification_report.txt")
    print(f"    {eval_dir}\\confusion_matrix_before.png")
    print(f"    {eval_dir}\\confusion_matrix_after.png")


if __name__ == "__main__":
    main()
