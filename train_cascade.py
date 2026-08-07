"""
train_cascade.py
=================
Trains a SEPARATE binary specialist for the one pair the main 9-class
model keeps confusing: "localized damage" vs "coating detachment"
(81 of ~1509 test images misclassified between just these two in Try_9 --
the single largest error block in the whole confusion matrix).

Does NOT touch wtb/model.py, wtb/dataset.py, wtb/losses.py, wtb/utils.py,
or wtb/config.py -- reuses all of them exactly as-is. Only difference from
train.py: this script's own local 2-class index/split, so it doesn't need
a separate copied dataset folder -- point it at the SAME WTBs2025 root you
already use for the main model.

Usage (same DATA_ROOT you already use for train.py):

    python train_cascade.py --data_root "C:\\...\\WTBs2025" --exp_name Try_9_cascade --backbone resnet18 --epochs 60

Then run cascade_infer.py to combine this with your main Try_9 checkpoint
and see the corrected metrics.
"""

import argparse
import csv
import glob
import os
import time

import numpy as np
import torch

from wtb.config import Config, set_seed, get_device, ensure_out_dir
from wtb.model import build_model
from wtb.losses import CompositeLoss
from wtb.dataset import (
    IMG_EXTS, build_transforms, WTBDataset, stratified_split_3way,
    make_weighted_sampler,
)
from wtb.utils import (
    compute_metrics, EarlyStopping, save_checkpoint, apply_mixup_cutmix, ModelEMA,
)
from torch.utils.data import DataLoader

# The two classes this specialist tells apart, and which index in the
# MAIN 9-class model's CLASS_NAMES each one corresponds to. cascade_infer.py
# uses CASCADE_TO_MAIN_IDX to map this model's 0/1 output back onto the
# main model's label space.
CASCADE_CLASSES = ["localized damage", "coating detachment"]
CASCADE_TO_MAIN_IDX = {0: 2, 1: 6}   # must match wtb/config.py CLASS_NAMES order


def cascade_class_distribution(samples):
    """Like wtb.dataset.class_distribution, but counts over CASCADE_CLASSES (2)
    instead of the global 9-class CLASS_NAMES. Using the wtb.dataset version
    here was the original bug: it silently returns a 9-length list (indices
    0/1 populated, 2-8 zero), which then corrupts LTCP's log_prior buffer
    size in set_prior() -- "size of tensor a (2) must match tensor b (9)".
    """
    from collections import Counter
    counts = Counter(label for _, label in samples)
    return [counts.get(i, 0) for i in range(len(CASCADE_CLASSES))]


def index_cascade_dataset(data_root: str):
    """Same on-disk convention as wtb/dataset.py's index_dataset (<class>/images/*.jpg),
    but only scans the two classes above instead of all 9."""
    samples = []
    missing = []
    for label, cls_name in enumerate(CASCADE_CLASSES):
        cls_dir = os.path.join(data_root, cls_name, "images")
        if not os.path.isdir(cls_dir):
            missing.append(cls_name)
            continue
        for ext in IMG_EXTS:
            for path in glob.glob(os.path.join(cls_dir, f"*{ext}")):
                samples.append((path, label))
    if missing:
        raise FileNotFoundError(
            f"Could not find an 'images' folder for {missing} under '{data_root}'. "
            f"Point --data_root at the same WTBs2025 folder you use for train.py."
        )
    if len(samples) == 0:
        raise RuntimeError(f"Found 0 images under {data_root}. Check the path.")
    return samples


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--exp_name", type=str, default="cascade_LDvsCD")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--base_lr", type=float, default=None)
    ap.add_argument("--img_size", type=int, default=None,
                     help="Defaults to Config.img_size (384 if you already bumped it "
                          "for Try_9). This is a focused 2-class problem so you can "
                          "also try pushing this higher (e.g. 512) with less overfit "
                          "risk than on the full 9-class run.")
    ap.add_argument("--backbone", type=str, default="resnet18", choices=["dsps", "resnet18", "resnet34"])
    ap.add_argument("--unfreeze_stem", action="store_true")
    ap.add_argument("--require_gpu", action="store_true")
    return ap.parse_args()


def build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    import math
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, total_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def run_validation(model, loader, criterion, device, channels_last):
    model.eval()
    y_true, y_pred = [], []
    loss_sum, n_batches = 0.0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        logits, feat = model(imgs)
        loss, _ = criterion(logits, feat, model.head.prototypes, labels)
        loss_sum += loss.item()
        n_batches += 1
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.cpu().tolist())
    metrics = compute_metrics(y_true, y_pred, num_classes=2)
    metrics["loss"] = loss_sum / max(1, n_batches)
    return metrics


def main():
    args = parse_args()
    cfg = Config(data_root=args.data_root, backbone=args.backbone)
    if args.out_dir:
        cfg.out_dir = args.out_dir
    else:
        cfg.out_dir = os.path.join(".", "runs", args.exp_name)
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.base_lr:
        cfg.base_lr = args.base_lr
    if args.img_size:
        cfg.img_size = args.img_size
    if args.unfreeze_stem:
        cfg.freeze_stem = False
    cfg.epochs = args.epochs

    set_seed(cfg.seed)
    device = get_device(require_gpu=args.require_gpu)
    ensure_out_dir(cfg.out_dir)
    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
    best_path = os.path.join(ckpt_dir, "best.pt")
    best_ema_path = os.path.join(ckpt_dir, "best_ema.pt")
    log_path = os.path.join(cfg.out_dir, "log.csv")

    print(f"[cascade] data_root = {cfg.data_root}")
    print(f"[cascade] out_dir   = {cfg.out_dir}")
    print(f"[cascade] classes   = {CASCADE_CLASSES} (binary specialist)")

    samples = index_cascade_dataset(cfg.data_root)
    dist = cascade_class_distribution(samples)
    print(f"[cascade] Found {len(samples)} images:")
    for name, n in zip(CASCADE_CLASSES, dist):
        print(f"    {name:24s}: {n:5d}")

    train_samples, val_samples, test_samples = stratified_split_3way(
        samples, cfg.val_fraction, cfg.test_fraction, cfg.seed
    )
    train_counts = cascade_class_distribution(train_samples)
    print(f"[cascade] Split -> train {len(train_samples)} | val {len(val_samples)} | test {len(test_samples)}")

    train_ds = WTBDataset(train_samples, build_transforms(cfg.img_size, train=True))
    val_ds = WTBDataset(val_samples, build_transforms(cfg.img_size, train=False))

    sampler = make_weighted_sampler([l for _, l in train_samples])
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=cfg.pin_memory,
        drop_last=True, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    # Note: unlike the main 9-class run, the weighted sampler is fine (in
    # fact recommended) here -- there's no logit-adjustment term at play
    # for this standalone binary head, so none of the config.py caveat
    # about stacking correction methods applies.

    print(f"[cascade] backbone = {cfg.backbone}, img_size = {cfg.img_size}")
    model = build_model(cfg, num_classes=2).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.head.set_prior(train_counts)

    criterion = CompositeLoss(
        train_counts, gamma=cfg.focal_gamma, beta_la=cfg.beta_la,
        beta_pc=cfg.beta_pc, cb_beta=cfg.cb_beta, label_smoothing=cfg.label_smoothing,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, cfg.warmup_epochs, cfg.epochs, steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    early_stopping = EarlyStopping(patience=cfg.patience, mode="max")

    ema = ModelEMA(model, decay=cfg.model_ema_decay) if cfg.use_model_ema else None
    ema_best = -float("inf")

    write_header = not os.path.isfile(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow([
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
            "val_macro_f1", "val_balanced_acc", "lr", "epoch_seconds",
        ])
        log_file.flush()

    print(f"[cascade] {steps_per_epoch} steps/epoch, {cfg.epochs} epochs")
    best_epoch = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_start = time.time()
        running_loss, n_seen, running_correct = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            if cfg.channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            imgs, labels_a, labels_b, lam, mixed = apply_mixup_cutmix(
                imgs, labels, cfg.mixup_alpha, cfg.cutmix_alpha,
                cfg.mixup_cutmix_prob, cfg.use_mixup, cfg.use_cutmix,
            )

            with torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
                logits, feat = model(imgs)
                if mixed:
                    loss_a, _ = criterion(logits, feat, model.head.prototypes, labels_a)
                    loss_b, _ = criterion(logits, feat, model.head.prototypes, labels_b)
                    loss = lam * loss_a + (1.0 - lam) * loss_b
                else:
                    loss, _ = criterion(logits, feat, model.head.prototypes, labels_a)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(model)
            if not mixed:
                model.head.update_prototypes(feat.detach(), labels_a)

            running_loss += loss.item() * imgs.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_seen += imgs.size(0)

            if (step + 1) % cfg.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  epoch {epoch+1}/{cfg.epochs}  step {step+1}/{len(train_loader)}  "
                      f"loss {loss.item():.4f}  lr {lr_now:.2e}")

        train_loss = running_loss / max(1, n_seen)
        train_acc = running_correct / max(1, n_seen)
        val_metrics = run_validation(model, val_loader, criterion, device, cfg.channels_last)
        val_metrics_ema = run_validation(ema.module, val_loader, criterion, device, cfg.channels_last) if ema else None
        epoch_seconds = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"[epoch {epoch+1}/{cfg.epochs}] train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
              f"val_loss {val_metrics['loss']:.4f} val_acc {val_metrics['accuracy']:.4f} "
              f"val_macro_f1 {val_metrics['macro_f1']:.4f} ({epoch_seconds:.1f}s/epoch)")
        if val_metrics_ema:
            print(f"           [EMA] val_acc {val_metrics_ema['accuracy']:.4f} "
                  f"val_macro_f1 {val_metrics_ema['macro_f1']:.4f}")

        log_writer.writerow([
            epoch + 1, f"{train_loss:.6f}", f"{train_acc:.6f}",
            f"{val_metrics['loss']:.6f}", f"{val_metrics['accuracy']:.6f}",
            f"{val_metrics['macro_f1']:.6f}", f"{val_metrics['balanced_acc']:.6f}",
            f"{lr_now:.8f}", f"{epoch_seconds:.2f}",
        ])
        log_file.flush()

        should_stop = early_stopping.step(val_metrics["macro_f1"], model)
        if early_stopping.since == 0:
            best_epoch = epoch
            save_checkpoint(best_path, model, CASCADE_CLASSES, epoch, early_stopping.best, backbone=cfg.backbone)
            print(f"           -> new best (val_macro_f1={early_stopping.best:.4f}), saved best.pt")

        if ema and val_metrics_ema["macro_f1"] > ema_best:
            ema_best = val_metrics_ema["macro_f1"]
            save_checkpoint(best_ema_path, ema.module, CASCADE_CLASSES, epoch, ema_best, backbone=cfg.backbone)
            print(f"           -> new best EMA (val_macro_f1={ema_best:.4f}), saved best_ema.pt")

        if should_stop:
            print(f"[cascade] Early stopping at epoch {epoch+1}. Best val macro-F1 = {early_stopping.best:.4f}")
            break

    log_file.close()
    early_stopping.restore_best(model)
    save_checkpoint(best_path, model, CASCADE_CLASSES, best_epoch, early_stopping.best, backbone=cfg.backbone)
    print(f"[cascade] Training finished. Best val macro-F1 = {early_stopping.best:.4f}")
    print(f"[cascade] Best weights saved to {best_path}")
    print(f"[cascade] Next: python cascade_infer.py --data_root <...> "
          f"--main_checkpoint runs\\Try_9\\checkpoints\\best.pt "
          f"--cascade_checkpoint {best_path} --out_dir runs\\Try_9_cascade_eval")


if __name__ == "__main__":
    main()