"""
ensemble_infer.py
==================
Averages softmax probabilities across MULTIPLE main-model checkpoints, then
optionally routes the result through the cascade specialist (same soft-blend
logic as cascade_infer.py). This is the "ensemble Try_9 + Try_9_cascade's
main model + their EMA variants" idea from earlier -- it was flagged as a
free, no-retraining accuracy source but never actually run. This script runs
it.

Basic usage -- ensemble two or more independently-trained checkpoints:

    python ensemble_infer.py --data_root "C:\\...\\WTBs2025" ^
        --checkpoints runs\\Try_8\\checkpoints\\best.pt runs\\Try_9\\checkpoints\\best.pt ^
        --out_dir runs\\ensemble_Try8_Try9 --tta

With the cascade specialist chained on afterward:

    python ensemble_infer.py --data_root "C:\\...\\WTBs2025" ^
        --checkpoints runs\\Try_8\\checkpoints\\best.pt runs\\Try_9\\checkpoints\\best.pt ^
        --cascade_checkpoint runs\\Try_9_cascade\\checkpoints\\best.pt ^
        --out_dir runs\\ensemble_Try8_Try9_cascade --tta --cascade_tta --cascade_weight 0.6

Notes on which checkpoints are actually worth ensembling:

  - Averaging a model with its OWN EMA shadow is NOT guaranteed to help --
    on this project, EMA weights scored LOWER than raw weights on Try_7
    (70.6% vs 75.1%) and Try_7_phase2 (75.4% vs 76.0%). Check both
    individually with evaluate.py --use_ema first before assuming EMA is
    free accuracy for a given checkpoint.
  - Ensembling INDEPENDENTLY-trained checkpoints (different Try_ runs, e.g.
    Try_8 + Try_9) is the more reliable lever: even if one is individually
    a bit weaker, two models that make DIFFERENT mistakes average out some
    of each other's errors. That's the classic ensembling assumption -- it
    can still fail if the models are too similar or one is much worse, so
    always compare against the single best checkpoint's own number.
  - Each checkpoint can have a different backbone/img_size; this script
    handles that automatically per-checkpoint (auto-detected from the
    checkpoint if it was saved with the img_size fix in wtb/utils.py,
    otherwise falls back to Config's default with a warning).
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

from train_cascade import CASCADE_TO_MAIN_IDX


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--checkpoints", type=str, nargs="+", required=True,
                     help="2 or more checkpoint paths to ensemble (e.g. "
                          "runs\\Try_8\\checkpoints\\best.pt runs\\Try_9\\checkpoints\\best.pt). "
                          "Softmax probabilities are averaged EQUALLY across all of them.")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                     help="Optional per-checkpoint weights (same length/order as "
                          "--checkpoints), e.g. --weights 0.4 0.6 to trust the second "
                          "checkpoint more. Defaults to equal weighting. Doesn't need "
                          "to sum to 1 -- it's renormalized automatically.")
    ap.add_argument("--img_sizes", type=int, nargs="+", default=None,
                     help="Optional per-checkpoint img_size override (same length/order "
                          "as --checkpoints). Only needed for checkpoints saved before "
                          "the img_size fix -- otherwise auto-detected.")
    ap.add_argument("--cascade_checkpoint", type=str, default=None,
                     help="Optional: chain the same soft-blend cascade specialist step "
                          "(see cascade_infer.py) on top of the ensembled probabilities.")
    ap.add_argument("--cascade_img_size", type=int, default=None)
    ap.add_argument("--cascade_weight", type=float, default=0.6,
                     help="Same meaning as in cascade_infer.py: blend weight for the "
                          "specialist vs the (already-ensembled) main probabilities.")
    ap.add_argument("--cascade_tta", action="store_true")
    ap.add_argument("--out_dir", type=str, default="./runs/ensemble_eval")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42,
                     help="Must match the seed train.py used, or this reconstructs a "
                          "different test split than the one your checkpoints were "
                          "evaluated on elsewhere.")
    ap.add_argument("--tta", action="store_true",
                     help="4-way flip TTA on EVERY checkpoint in the ensemble.")
    args = ap.parse_args()

    if args.weights is not None and len(args.weights) != len(args.checkpoints):
        raise SystemExit("--weights must have the same length as --checkpoints")
    if args.img_sizes is not None and len(args.img_sizes) != len(args.checkpoints):
        raise SystemExit("--img_sizes must have the same length as --checkpoints")
    return args


def detect_img_size(ckpt: dict, cli_override, default: int, label: str) -> int:
    if cli_override is not None:
        print(f"[ensemble_infer] {label} img_size = {cli_override} (CLI override)")
        return cli_override
    stored = ckpt.get("img_size")
    if stored is not None:
        print(f"[ensemble_infer] {label} img_size = {stored} (auto-detected from checkpoint)")
        return stored
    print(f"[ensemble_infer] WARNING: {label} checkpoint has no stored img_size -- "
          f"falling back to default={default}.")
    return default


@torch.no_grad()
def infer_probs(model, loader, device, channels_last, tta):
    model.eval()
    y_prob = []
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
        y_prob.extend(probs.cpu().tolist())
    return np.array(y_prob)


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

    samples = index_dataset(args.data_root)
    _, _, test_samples = stratified_split_3way(samples, 0.2, 0.2, args.seed)
    print(f"[ensemble_infer] Test set: {len(test_samples)} images (same split as evaluate.py)")
    y_true = [label for _, label in test_samples]

    weights = args.weights or [1.0] * len(args.checkpoints)
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    img_size_overrides = args.img_sizes or [None] * len(args.checkpoints)

    ensembled_probs = np.zeros((len(test_samples), NUM_CLASSES), dtype=np.float64)
    per_model_accuracy = []

    for i, ckpt_path in enumerate(args.checkpoints):
        print(f"\n[ensemble_infer] --- checkpoint {i+1}/{len(args.checkpoints)}: {ckpt_path} "
              f"(weight={weights[i]:.3f}) ---")
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        classes = ckpt.get("classes", CLASS_NAMES)
        if list(classes) != list(CLASS_NAMES):
            raise SystemExit(
                f"[ensemble_infer] {ckpt_path} has classes={classes}, which doesn't match "
                f"the main 9-class CLASS_NAMES. ensemble_infer.py is for combining several "
                f"9-class checkpoints -- pass the cascade specialist via --cascade_checkpoint "
                f"instead of --checkpoints."
            )
        cfg = Config(data_root=args.data_root, batch_size=args.batch_size)
        cfg.backbone = ckpt.get("backbone") or "dsps"
        img_size = detect_img_size(ckpt, img_size_overrides[i], cfg.img_size, f"checkpoint[{i}]")

        model = build_model(cfg, NUM_CLASSES).to(device)
        if cfg.channels_last:
            model = model.to(memory_format=torch.channels_last)
        model.load_state_dict(ckpt["state_dict"])

        ds = WTBDataset(test_samples, build_transforms(img_size, train=False))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=cfg.pin_memory)

        print(f"[ensemble_infer] Running inference (backbone={cfg.backbone}, "
              f"img_size={img_size}{', TTA' if args.tta else ''})...")
        probs = infer_probs(model, loader, device, cfg.channels_last, args.tta)
        ensembled_probs += weights[i] * probs

        this_acc = compute_metrics(y_true, probs.argmax(axis=1).tolist(), NUM_CLASSES)["accuracy"]
        per_model_accuracy.append(this_acc)
        print(f"[ensemble_infer] This checkpoint alone: accuracy = {this_acc:.4f}")

    y_pred_ensemble = ensembled_probs.argmax(axis=1).tolist()
    metrics_ensemble_only = compute_metrics(y_true, y_pred_ensemble, NUM_CLASSES)
    print(f"\n[ensemble_infer] === ENSEMBLE of {len(args.checkpoints)} checkpoints (before cascade) ===")
    print(f"    accuracy  : {metrics_ensemble_only['accuracy']:.4f}")
    print(f"    macro F1  : {metrics_ensemble_only['macro_f1']:.4f}")
    print(f"    (individual checkpoints scored: "
          f"{', '.join(f'{a:.4f}' for a in per_model_accuracy)})")
    if metrics_ensemble_only["accuracy"] <= max(per_model_accuracy):
        print(f"[ensemble_infer] NOTE: the ensemble did NOT beat the best individual "
              f"checkpoint ({max(per_model_accuracy):.4f}). This can happen if the "
              f"checkpoints are too similar (correlated errors) or one is much weaker "
              f"than the others -- try --weights to favor the stronger one(s), or drop "
              f"the weakest checkpoint from --checkpoints.")

    # ---- optional cascade specialist step, chained on top of the ensemble ----
    y_prob_final = ensembled_probs.copy()
    n_corrections, n_regressions = 0, 0
    n_routed = 0

    if args.cascade_checkpoint:
        y_pred_pre_cascade = ensembled_probs.argmax(axis=1).tolist()
        trigger_indices = [i for i, p in enumerate(y_pred_pre_cascade)
                            if p in CASCADE_TO_MAIN_IDX.values()]
        n_routed = len(trigger_indices)
        print(f"\n[ensemble_infer] {n_routed}/{len(test_samples)} images routed through "
              f"the cascade specialist (blend weight = {args.cascade_weight:.2f})...")

        if trigger_indices and args.cascade_weight > 0.0:
            cascade_ckpt = load_checkpoint(args.cascade_checkpoint, map_location=device)
            cascade_cfg = Config(data_root=args.data_root, batch_size=args.batch_size)
            cascade_cfg.backbone = cascade_ckpt.get("backbone") or "resnet18"
            cascade_img_size = detect_img_size(cascade_ckpt, args.cascade_img_size,
                                                cascade_cfg.img_size, "CASCADE")
            cascade_model = build_model(cascade_cfg, num_classes=2).to(device)
            if cascade_cfg.channels_last:
                cascade_model = cascade_model.to(memory_format=torch.channels_last)
            cascade_model.load_state_dict(cascade_ckpt["state_dict"])

            trigger_samples = [test_samples[i] for i in trigger_indices]
            cascade_ds = WTBDataset(trigger_samples, build_transforms(cascade_img_size, train=False))
            cascade_loader = DataLoader(cascade_ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=cascade_cfg.pin_memory)

            cascade_probs = infer_probs(cascade_model, cascade_loader, device,
                                         cascade_cfg.channels_last, args.cascade_tta)

            main_idx_ld = CASCADE_TO_MAIN_IDX[0]
            main_idx_cd = CASCADE_TO_MAIN_IDX[1]
            w = args.cascade_weight

            for local_i, global_i in enumerate(trigger_indices):
                p_main_pair = np.array([ensembled_probs[global_i, main_idx_ld],
                                         ensembled_probs[global_i, main_idx_cd]])
                p_spec_pair = cascade_probs[local_i]

                p_main_pair_norm = p_main_pair / max(p_main_pair.sum(), 1e-8)
                blended_pair = w * p_spec_pair + (1.0 - w) * p_main_pair_norm
                blended_pair = blended_pair / max(blended_pair.sum(), 1e-8)

                original_pair_mass = p_main_pair.sum()
                y_prob_final[global_i, main_idx_ld] = blended_pair[0] * original_pair_mass
                y_prob_final[global_i, main_idx_cd] = blended_pair[1] * original_pair_mass

                old_pred = y_pred_pre_cascade[global_i]
                new_pred = int(y_prob_final[global_i].argmax())
                true_label = y_true[global_i]
                if new_pred != old_pred:
                    if old_pred != true_label and new_pred == true_label:
                        n_corrections += 1
                    elif old_pred == true_label and new_pred != true_label:
                        n_regressions += 1

            print(f"[ensemble_infer]   -> {n_corrections} corrections, {n_regressions} regressions "
                  f"(net {n_corrections - n_regressions:+d})")

    y_pred_final = y_prob_final.argmax(axis=1).tolist()
    metrics_final = compute_metrics(y_true, y_pred_final, NUM_CLASSES)

    print(f"\n=== FINAL (ensemble" + (" + cascade" if args.cascade_checkpoint else "") + ") ===")
    print(f"  accuracy  : {metrics_final['accuracy']:.4f}")
    print(f"  macro F1  : {metrics_final['macro_f1']:.4f}")
    print("\n  per-class F1:")
    for name, f1 in zip(CLASS_NAMES, metrics_final["per_class_f1"]):
        print(f"    {name:24s}: {f1:.4f}")

    cm_final = confusion_matrix(y_true, y_pred_final, labels=list(range(NUM_CLASSES)))
    report_final = classification_report(
        y_true, y_pred_final, target_names=CLASS_NAMES, digits=4, zero_division=0
    )

    eval_dir = os.path.join(args.out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "test_metrics.json"), "w") as f:
        json.dump({
            "checkpoints": args.checkpoints,
            "weights": weights.tolist(),
            "tta": bool(args.tta),
            "cascade_checkpoint": args.cascade_checkpoint,
            "cascade_weight": args.cascade_weight if args.cascade_checkpoint else None,
            "n_test_images": len(test_samples),
            "per_model_accuracy": per_model_accuracy,
            "metrics_ensemble_only": metrics_ensemble_only,
            "n_routed_through_cascade": n_routed,
            "n_corrections": n_corrections,
            "n_regressions": n_regressions,
            "metrics_final": metrics_final,
        }, f, indent=2)
    with open(os.path.join(eval_dir, "classification_report.txt"), "w") as f:
        f.write(report_final)
    plot_confusion_matrix(cm_final, CLASS_NAMES, os.path.join(eval_dir, "confusion_matrix.png"),
                           "Test-set confusion matrix -- ensemble" +
                           (" + cascade" if args.cascade_checkpoint else ""))

    print(f"\n[ensemble_infer] Saved:")
    print(f"    {eval_dir}\\test_metrics.json")
    print(f"    {eval_dir}\\classification_report.txt")
    print(f"    {eval_dir}\\confusion_matrix.png")


if __name__ == "__main__":
    main()
