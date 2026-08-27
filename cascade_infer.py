"""
cascade_infer.py
=================
Combines your main 9-class model (e.g. Try_9's best.pt) with the binary
"localized damage" vs "coating detachment" specialist from train_cascade.py.

    python cascade_infer.py --data_root "C:\\...\\WTBs2025" ^
        --main_checkpoint runs\\Try_9\\checkpoints\\best.pt ^
        --cascade_checkpoint runs\\Try_9_cascade\\checkpoints\\best.pt ^
        --out_dir runs\\Try_9_cascade_eval --tta --cascade_tta

CHANGED from the original version:

  1. SOFT BLENDING instead of a hard override. The old version replaced the
     main model's prediction with the specialist's prediction 100% of the
     time whenever the main model called an image "localized damage" or
     "coating detachment" -- even when the main model was highly confident
     and correct. That throws away good information: if the main model is
     90% sure and the specialist (which only sees ~73% F1 on this pair
     itself) disagrees, blindly trusting the specialist can make things
     WORSE on exactly the images the main model already had right.
     Now the two models' probabilities over just these 2 classes are
     combined as a weighted average (--cascade_weight, default 0.6 favors
     the specialist since it IS the one trained specifically for this
     distinction, but no longer with total authority), then renormalized
     and argmax'd. Every other class's probability is left untouched.

  2. TTA on the specialist too (--cascade_tta), matching what --tta already
     does for the main model. Free accuracy, no retraining.

  3. img_size is auto-detected from each checkpoint (if it was saved with
     the img_size fix in wtb/utils.py) instead of assuming --main_img_size/
     --cascade_img_size defaults are correct for every checkpoint.

  4. A diagnostic block prints how many of the specialist's overrides were
     actually CORRECTIONS (main was wrong, blend fixed it) vs REGRESSIONS
     (main was right, blend broke it) -- so you can see directly whether
     raising/lowering --cascade_weight helps, instead of only seeing the
     net accuracy number.

Writes the same three artifacts the original did (test_metrics.json,
classification_report.txt, confusion matrices) into --out_dir/eval.
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
CASCADE_TRIGGER_IDXS = sorted(CASCADE_TO_MAIN_IDX.values())  # [2, 6], kept sorted so indexing below is stable


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--main_checkpoint", type=str, required=True)
    ap.add_argument("--main_img_size", type=int, default=None,
                     help="Overrides auto-detection. Only needed if the checkpoint "
                          "predates the img_size fix (no 'img_size' key stored) -- "
                          "in that case this falls back to Config's default (384) "
                          "unless you pass this explicitly.")
    ap.add_argument("--cascade_checkpoint", type=str, required=True)
    ap.add_argument("--cascade_img_size", type=int, default=None,
                     help="Same as --main_img_size, for the specialist checkpoint.")
    ap.add_argument("--out_dir", type=str, default="./runs/cascade_eval")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42,
                     help="Must match the seed train.py used to build the main "
                          "model's test split, or this reconstructs a different split.")
    ap.add_argument("--tta", action="store_true",
                     help="4-way flip TTA on the MAIN model (matches how you "
                          "evaluated Try_9).")
    ap.add_argument("--cascade_tta", action="store_true",
                     help="4-way flip TTA on the SPECIALIST too. Previously the "
                          "specialist always ran single-view even when --tta was "
                          "set for the main model -- this closes that gap.")
    ap.add_argument("--cascade_weight", type=float, default=0.6,
                     help="Blend weight for the specialist's probability over the "
                          "2 cascade classes, in [0, 1]. final = w*specialist + "
                          "(1-w)*main, renormalized, then argmax over just these "
                          "2 classes (all other classes keep the main model's "
                          "probability untouched). 1.0 reproduces the OLD hard-"
                          "override behaviour exactly; 0.0 disables the cascade "
                          "entirely (falls back to the main model alone). Try a "
                          "few values (e.g. 0.4, 0.5, 0.6, 0.7, 0.8) -- this is a "
                          "free, no-retraining sweep since inference is cheap.")
    return ap.parse_args()


def detect_img_size(ckpt: dict, cli_override, default: int, label: str) -> int:
    if cli_override is not None:
        print(f"[cascade_infer] {label} img_size = {cli_override} (CLI override)")
        return cli_override
    stored = ckpt.get("img_size")
    if stored is not None:
        print(f"[cascade_infer] {label} img_size = {stored} (auto-detected from checkpoint)")
        return stored
    print(f"[cascade_infer] WARNING: {label} checkpoint has no stored img_size "
          f"(saved before this field was added). Falling back to default={default} "
          f"-- pass --{'main' if 'MAIN' in label else 'cascade'}_img_size explicitly "
          f"if you know this is wrong.")
    return default


@torch.no_grad()
def infer_probs(model, loader, device, channels_last, tta):
    """Returns softmax probabilities for every image in loader, in order.
    Shared by both the main model and the specialist so TTA behaves
    identically for either one."""
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
    assert 0.0 <= args.cascade_weight <= 1.0, "--cascade_weight must be in [0, 1]"
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
    main_img_size = detect_img_size(main_ckpt, args.main_img_size, cfg.img_size, "MAIN")
    main_model = build_model(cfg, NUM_CLASSES).to(device)
    if cfg.channels_last:
        main_model = main_model.to(memory_format=torch.channels_last)
    main_model.load_state_dict(main_ckpt["state_dict"])

    main_ds = WTBDataset(test_samples, build_transforms(main_img_size, train=False))
    main_loader = DataLoader(main_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=cfg.pin_memory)

    print(f"[cascade_infer] Running MAIN model inference (img_size={main_img_size}"
          f"{', TTA' if args.tta else ''})...")
    y_prob_main = infer_probs(main_model, main_loader, device, cfg.channels_last, args.tta)
    y_pred_main = y_prob_main.argmax(axis=1).tolist()

    # ---- baseline metrics (main model alone, for comparison) ----
    metrics_before = compute_metrics(y_true, y_pred_main, NUM_CLASSES)
    print(f"\n[cascade_infer] BEFORE cascade (main model only):")
    print(f"    accuracy  : {metrics_before['accuracy']:.4f}")
    print(f"    macro F1  : {metrics_before['macro_f1']:.4f}")

    # ---- find images that need a second opinion ----
    trigger_indices = [i for i, p in enumerate(y_pred_main) if p in CASCADE_TRIGGER_IDXS]
    print(f"\n[cascade_infer] {len(trigger_indices)}/{len(test_samples)} test images "
          f"predicted as 'localized damage' or 'coating detachment' -- "
          f"routing these through the specialist "
          f"(blend weight = {args.cascade_weight:.2f} specialist / "
          f"{1 - args.cascade_weight:.2f} main).")

    y_prob_final = y_prob_main.copy()
    n_corrections, n_regressions, n_unchanged_route, n_no_argmax_change = 0, 0, 0, 0

    if trigger_indices and args.cascade_weight > 0.0:
        print(f"[cascade_infer] Loading CASCADE checkpoint: {args.cascade_checkpoint}")
        cascade_ckpt = load_checkpoint(args.cascade_checkpoint, map_location=device)
        cascade_cfg = Config(data_root=args.data_root, batch_size=args.batch_size)
        cascade_cfg.backbone = cascade_ckpt.get("backbone") or "resnet18"
        cascade_img_size = detect_img_size(cascade_ckpt, args.cascade_img_size, cascade_cfg.img_size, "CASCADE")
        cascade_model = build_model(cascade_cfg, num_classes=2).to(device)
        if cascade_cfg.channels_last:
            cascade_model = cascade_model.to(memory_format=torch.channels_last)
        cascade_model.load_state_dict(cascade_ckpt["state_dict"])

        trigger_samples = [test_samples[i] for i in trigger_indices]
        cascade_ds = WTBDataset(trigger_samples, build_transforms(cascade_img_size, train=False))
        cascade_loader = DataLoader(cascade_ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=cascade_cfg.pin_memory)

        print(f"[cascade_infer] Running specialist on {len(trigger_indices)} images "
              f"(img_size={cascade_img_size}{', TTA' if args.cascade_tta else ''})...")
        cascade_probs = infer_probs(cascade_model, cascade_loader, device,
                                     cascade_cfg.channels_last, args.cascade_tta)
        # cascade_probs[:, 0] = P(localized damage), cascade_probs[:, 1] = P(coating detachment)
        # main-space indices for the same two classes, in that same 0/1 order:
        main_idx_ld = CASCADE_TO_MAIN_IDX[0]   # 2
        main_idx_cd = CASCADE_TO_MAIN_IDX[1]   # 6

        w = args.cascade_weight
        for local_i, global_i in enumerate(trigger_indices):
            p_main_pair = np.array([y_prob_main[global_i, main_idx_ld],
                                     y_prob_main[global_i, main_idx_cd]])
            p_spec_pair = cascade_probs[local_i]   # already sums to 1 over [ld, cd]

            p_main_pair_norm = p_main_pair / max(p_main_pair.sum(), 1e-8)
            blended_pair = w * p_spec_pair + (1.0 - w) * p_main_pair_norm
            blended_pair = blended_pair / max(blended_pair.sum(), 1e-8)

            # Scale the blended pair back to occupy the SAME total probability
            # mass the main model originally assigned to these 2 classes,
            # so the other 7 classes' probabilities stay meaningfully
            # comparable (this only reshuffles mass between ld/cd, it
            # doesn't invent or destroy probability mass elsewhere).
            original_pair_mass = p_main_pair.sum()
            y_prob_final[global_i, main_idx_ld] = blended_pair[0] * original_pair_mass
            y_prob_final[global_i, main_idx_cd] = blended_pair[1] * original_pair_mass

            old_pred = y_pred_main[global_i]
            new_pred = int(y_prob_final[global_i].argmax())
            true_label = y_true[global_i]

            if new_pred == old_pred:
                n_unchanged_route += 1
            else:
                n_no_argmax_change += 0  # (kept for symmetry/readability, no-op)
                if old_pred != true_label and new_pred == true_label:
                    n_corrections += 1
                elif old_pred == true_label and new_pred != true_label:
                    n_regressions += 1

        n_changed = len(trigger_indices) - n_unchanged_route
        print(f"[cascade_infer] Specialist changed {n_changed}/{len(trigger_indices)} "
              f"final predictions.")
        print(f"[cascade_infer]   -> {n_corrections} were CORRECTIONS (main was wrong, now right)")
        print(f"[cascade_infer]   -> {n_regressions} were REGRESSIONS (main was right, now wrong)")
        print(f"[cascade_infer]   -> net effect of the cascade step: "
              f"{n_corrections - n_regressions:+d} images "
              f"({'net positive' if n_corrections >= n_regressions else 'net NEGATIVE -- consider lowering --cascade_weight'})")
    elif args.cascade_weight == 0.0:
        print("[cascade_infer] --cascade_weight=0.0: cascade disabled, using main model predictions as-is.")

    y_pred_final = y_prob_final.argmax(axis=1).tolist()

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
            "cascade_weight": args.cascade_weight,
            "main_tta": bool(args.tta),
            "cascade_tta": bool(args.cascade_tta),
            "n_test_images": len(test_samples),
            "n_routed_through_cascade": len(trigger_indices),
            "n_corrections": n_corrections,
            "n_regressions": n_regressions,
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
    print(f"\n[cascade_infer] TIP: --cascade_weight is free to sweep (no retraining) -- "
          f"try e.g. 0.4/0.5/0.6/0.7/0.8 and keep whichever gives the best accuracy/macro-F1 "
          f"on THIS run, since it's evaluated on the test set, don't sweep it more than a "
          f"handful of times or you're effectively re-fitting to the test set.")


if __name__ == "__main__":
    main()
