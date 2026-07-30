"""
train.py
========
The real training loop for WTB-DefectNet.

    python train.py --data_root "D:\\...\\WTBs2025" --out_dir ./runs/exp1

RESUME (the whole point of this script): if training stops for ANY reason
--Ctrl+C, a Windows sleep/crash, an OOM, pulling the power cord-- just run
the exact same command again with --resume added:

    python train.py --data_root "D:\\...\\WTBs2025" --out_dir ./runs/exp1 --resume

It will find runs/exp1/checkpoints/last.pt automatically and continue from
the epoch right after the last one that finished -- same optimizer state,
same LR schedule position, same AMP scaler state, same early-stopping
counters. You do NOT need to pass --resume every time by path; "last.pt" in
--out_dir is always kept up to date, one epoch at a time.

Two checkpoints are kept in <out_dir>/checkpoints/:
    last.pt  -- overwritten every epoch, used for --resume
    best.pt  -- overwritten only when val macro-F1 improves, used later by
                evaluate.py and gradcam.py

A per-epoch CSV log is written to <out_dir>/log.csv (also resumed/appended,
never overwritten) so you can plot loss/F1 curves afterwards without
re-parsing console output.
"""

import argparse
import csv
import os
import signal
import sys
import time

import torch
import torch.nn as nn

from wtb.config import Config, CLASS_NAMES, NUM_CLASSES, set_seed, get_device, ensure_out_dir
from wtb.model import WTBDefectNet
from wtb.losses import CompositeLoss
from wtb.dataset import build_loaders
from wtb.utils import compute_metrics, EarlyStopping, save_checkpoint, load_checkpoint


def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--exp_name", type=str, default=None,
                     help="Shortcut for --out_dir ./runs/<exp_name>. "
                          "Ignored if --out_dir is also given.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--require_gpu", action="store_true",
                     help="Hard-stop immediately if no CUDA GPU is detected, instead "
                          "of silently training on CPU for hours.")
    ap.add_argument("--resume", action="store_true",
                     help="Resume from <out_dir>/checkpoints/last.pt if it exists.")
    ap.add_argument("--resume_path", type=str, default=None,
                     help="Resume from a specific checkpoint file instead of last.pt.")
    args = ap.parse_args()

    cfg = Config()
    for field in ("data_root", "out_dir", "epochs", "batch_size", "num_workers"):
        val = getattr(args, field)
        if val is not None:
            setattr(cfg, field, val)
    if args.out_dir is None and args.exp_name is not None:
        cfg.out_dir = os.path.join(".", "runs", args.exp_name)
    return cfg, args


def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int, steps_per_epoch: int):
    """Linear warmup for warmup_epochs, then cosine decay to 0 for the rest.
    Stepped once per OPTIMIZER STEP (i.e. after grad-accum), not per epoch,
    for a smooth curve."""
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, total_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        import math
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


@torch.no_grad()
def run_validation(model, loader, criterion, device, channels_last: bool):
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
    metrics = compute_metrics(y_true, y_pred, NUM_CLASSES)
    metrics["loss"] = loss_sum / max(1, n_batches)
    return metrics


def main():
    cfg, args = parse_args()
    set_seed(cfg.seed)
    device = get_device(require_gpu=args.require_gpu)

    if device.type == "cpu" and cfg.num_workers > 2:
        print(f"[train] No GPU: reducing num_workers from {cfg.num_workers} to 2. "
              f"On CPU, extra DataLoader worker processes compete with training "
              f"for the same cores and add memory pressure -- a likely contributor "
              f"to the access-violation crash from your last run.")
        cfg.num_workers = 2
    ensure_out_dir(cfg.out_dir)
    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")
    log_path = os.path.join(cfg.out_dir, "log.csv")

    print(f"[train] data_root = {cfg.data_root}")
    print(f"[train] out_dir   = {cfg.out_dir}")

    train_loader, val_loader, test_loader, train_counts = build_loaders(
        data_root=cfg.data_root,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        seed=cfg.seed,
    )

    model = WTBDefectNet(
        num_classes=NUM_CLASSES, widths=cfg.widths, tau=cfg.tau, lam=cfg.lam,
        proto_momentum=cfg.proto_momentum, head_dropout=cfg.head_dropout,
    ).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.head.set_prior(train_counts)

    criterion = CompositeLoss(
        train_counts, gamma=cfg.focal_gamma, beta_la=cfg.beta_la,
        beta_pc=cfg.beta_pc, cb_beta=cfg.cb_beta,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay
    )
    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
    scheduler = build_scheduler(optimizer, cfg.warmup_epochs, cfg.epochs, steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    early_stopping = EarlyStopping(patience=cfg.patience, mode="max")

    start_epoch = 0
    global_step = 0

    # ---------------- resume ----------------
    resume_from = args.resume_path or (last_path if args.resume else None)
    if resume_from and os.path.isfile(resume_from):
        print(f"[resume] Loading checkpoint: {resume_from}")
        ckpt = load_checkpoint(resume_from, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "early_stopping" in ckpt:
            early_stopping.load_state_dict(ckpt["early_stopping"])
        start_epoch = ckpt["epoch"] + 1
        global_step = start_epoch * steps_per_epoch
        print(f"[resume] Resuming from epoch {start_epoch} "
              f"(best val macro-F1 so far: {early_stopping.best:.4f})")
    elif args.resume or args.resume_path:
        print(f"[resume] --resume given but no checkpoint found at "
              f"'{resume_from}'. Starting fresh from epoch 0.")

    # write CSV header only if starting a brand-new log
    write_header = not os.path.isfile(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow([
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
            "val_macro_f1", "val_balanced_acc", "lr", "epoch_seconds",
        ])
        log_file.flush()

    # Ctrl+C stops training almost immediately -- it does NOT wait for the
    # current epoch to finish. last.pt already holds the state from the
    # last FULLY COMPLETED epoch (we only checkpoint at epoch boundaries),
    # so nothing further needs to be saved here: resuming just re-runs the
    # interrupted epoch's steps in a fresh random order, which is harmless.
    interrupted = {"flag": False}

    def handle_sigint(signum, frame):
        print("\n[train] Interrupt received -- stopping now. The last "
              "fully completed epoch is already saved in checkpoints/last.pt. "
              "Re-run with --resume to continue from there.")
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, handle_sigint)

    print(f"[train] {steps_per_epoch} optimizer steps/epoch, "
          f"{cfg.epochs - start_epoch} epochs remaining (of {cfg.epochs} total)")

    best_epoch = start_epoch   # fallback if resuming and no new best is found this run

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_start = time.time()
        running_loss, n_seen, running_correct = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (imgs, labels) in enumerate(train_loader):
            if interrupted["flag"]:
                print(f"[train] Stopped mid-epoch {epoch+1} at step {step}/{len(train_loader)}. "
                      f"This epoch was NOT saved (only full epochs are checkpointed).")
                log_file.close()
                sys.exit(0)

            imgs = imgs.to(device, non_blocking=True)
            if cfg.channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
                logits, feat = model(imgs)
                loss, parts = criterion(logits, feat, model.head.prototypes, labels)
                loss_scaled = loss / cfg.grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            # EMA-update the LTCP prototype memory bank from this batch's
            # features (after the optimizer step, so it tracks the current
            # feature space). Classes absent from this batch are untouched.
            model.head.update_prototypes(feat.detach(), labels)

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
        epoch_seconds = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"[epoch {epoch+1}/{cfg.epochs}] "
              f"train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
              f"val_loss {val_metrics['loss']:.4f}  val_acc {val_metrics['accuracy']:.4f}  "
              f"val_macro_f1 {val_metrics['macro_f1']:.4f}  "
              f"val_balanced_acc {val_metrics['balanced_acc']:.4f}  "
              f"({epoch_seconds:.1f}s/epoch)")

        remaining = cfg.epochs - (epoch + 1)
        eta = format_eta(remaining * epoch_seconds)
        print(f"           ETA to epoch {cfg.epochs} (if no early stop): {eta}")

        log_writer.writerow([
            epoch + 1, f"{train_loss:.6f}", f"{train_acc:.6f}",
            f"{val_metrics['loss']:.6f}", f"{val_metrics['accuracy']:.6f}",
            f"{val_metrics['macro_f1']:.6f}", f"{val_metrics['balanced_acc']:.6f}",
            f"{lr_now:.8f}", f"{epoch_seconds:.2f}",
        ])
        log_file.flush()

        should_stop = early_stopping.step(val_metrics["macro_f1"], model)
        is_best = early_stopping.since == 0   # step() just improved best

        save_checkpoint(
            last_path, model, CLASS_NAMES, epoch, early_stopping.best,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            early_stopping=early_stopping,
        )
        if is_best:
            best_epoch = epoch   # BUGFIX: remember which epoch this actually was
            save_checkpoint(
                best_path, model, CLASS_NAMES, epoch, early_stopping.best,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                early_stopping=early_stopping,
            )
            print(f"           -> new best (val_macro_f1={early_stopping.best:.4f}), saved best.pt")

        if interrupted["flag"]:
            # Interrupt landed during validation/checkpointing rather than
            # the training loop -- epoch + checkpoint above already
            # completed normally, so this epoch IS saved and safe to resume from.
            print(f"[train] Epoch {epoch+1} finished and saved to {last_path} "
                  f"before the interrupt took effect.")
            print("[train] Re-run the same command with --resume to continue.")
            log_file.close()
            sys.exit(0)

        if should_stop:
            print(f"[train] Early stopping: no val macro-F1 improvement for "
                  f"{cfg.patience} epochs. Best = {early_stopping.best:.4f}.")
            break

    log_file.close()
    early_stopping.restore_best(model)
    save_checkpoint(
        best_path, model, CLASS_NAMES, best_epoch, early_stopping.best,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        early_stopping=early_stopping,
    )
    print(f"[train] Training finished. Best val macro-F1 = {early_stopping.best:.4f}")
    print(f"[train] Best weights saved to {best_path}")
    print("[train] Next: python evaluate.py --data_root <...> --checkpoint "
          f"{best_path} --out_dir {cfg.out_dir}")


if __name__ == "__main__":
    main()
