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
from wtb.model import build_model
from wtb.losses import CompositeLoss
from wtb.dataset import build_loaders
from wtb.utils import (
    compute_metrics, EarlyStopping, save_checkpoint, load_checkpoint,
    apply_mixup_cutmix, ModelEMA,
)


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
    ap.add_argument("--img_size", type=int, default=None,
                     help="Override Config.img_size (default 384). Higher values "
                          "(e.g. 448) preserve more detail for small defects "
                          "(pinholes, hairline paint cracks) at the cost of more "
                          "VRAM/epoch time. Must match at evaluate.py/cascade_infer.py "
                          "time -- but those already auto-detect it from the saved "
                          "checkpoint, so no extra flag is needed there.")
    ap.add_argument("--base_lr", type=float, default=None,
                     help="Override Config.base_lr. Mainly for phase-2 fine-tuning: "
                          "use a lower LR (e.g. 3e-5) than the phase-1 default (3e-4) "
                          "when unfreezing a previously-frozen pretrained stem.")
    ap.add_argument("--init_from", type=str, default=None,
                     help="Load ONLY model weights (state_dict) from this checkpoint, "
                          "then start a FRESH run (new optimizer/scheduler/early-stopping, "
                          "epoch 0) -- mutually exclusive with --resume. This is for "
                          "phase-2 stem fine-tuning: train once with the pretrained stem "
                          "frozen (the default), then run again with "
                          "`--backbone resnet18 --unfreeze_stem --init_from "
                          "runs/<phase1>/checkpoints/best.pt --base_lr 3e-5 "
                          "--exp_name <phase1>_phase2`. Does not change wtb/model.py or "
                          "any block in it -- only which checkpoint the weights start from "
                          "and which stem layers are frozen.")
    ap.add_argument("--no_mixup_cutmix", action="store_true",
                     help="Disable MixUp/CutMix for this run (falls back to the plain "
                          "augmentation pipeline in wtb/dataset.py).")
    ap.add_argument("--no_model_ema", action="store_true",
                     help="Disable model-weight EMA (checkpoints/best_ema.pt) for this run.")
    ap.add_argument("--label_smoothing", type=float, default=None,
                     help="Override Config.label_smoothing.")
    ap.add_argument("--backbone", type=str, default=None,
                     choices=["dsps", "resnet18", "resnet34"],
                     help="'dsps' = fully custom WTBDefectNet from scratch. "
                          "'resnet18'/'resnet34' = ImageNet-pretrained stem, custom "
                          "TSDB/ASA/DRFB/WGFR/MSCA/LTCP on top. See config.py.")
    ap.add_argument("--no_pretrained_stem", action="store_true",
                     help="With --backbone resnet18/34, randomly init the stem "
                          "instead of loading ImageNet weights (ablation only).")
    ap.add_argument("--unfreeze_stem", action="store_true",
                     help="With --backbone resnet18/34, fine-tune the pretrained "
                          "stem instead of freezing it.")
    ap.add_argument("--use_weighted_sampler", action="store_true",
                     help="Restore the old always-on WeightedRandomSampler "
                          "(stacks with CompositeLoss's own imbalance correction "
                          "-- see config.py before using this).")
    ap.add_argument("--require_gpu", action="store_true",
                     help="Hard-stop immediately if no CUDA GPU is detected, instead "
                          "of silently training on CPU for hours.")
    ap.add_argument("--resume", action="store_true",
                     help="Resume from <out_dir>/checkpoints/last.pt if it exists.")
    ap.add_argument("--resume_path", type=str, default=None,
                     help="Resume from a specific checkpoint file instead of last.pt.")
    args = ap.parse_args()

    if args.resume and args.init_from:
        raise SystemExit(
            "[train] --resume and --init_from are mutually exclusive: --resume "
            "continues an interrupted run with its OLD optimizer/scheduler/early-"
            "stopping state; --init_from starts a brand-new run from just the "
            "weights of a (usually different) checkpoint. Pick one."
        )

    cfg = Config()
    for field in ("data_root", "out_dir", "epochs", "batch_size", "num_workers",
                  "backbone", "base_lr", "img_size"):
        val = getattr(args, field)
        if val is not None:
            setattr(cfg, field, val)
    if args.out_dir is None and args.exp_name is not None:
        cfg.out_dir = os.path.join(".", "runs", args.exp_name)
    if args.no_pretrained_stem:
        cfg.pretrained_stem = False
    if args.unfreeze_stem:
        cfg.freeze_stem = False
    if args.use_weighted_sampler:
        cfg.use_weighted_sampler = True
    if args.no_mixup_cutmix:
        cfg.use_mixup = False
        cfg.use_cutmix = False
    if args.no_model_ema:
        cfg.use_model_ema = False
    if args.label_smoothing is not None:
        cfg.label_smoothing = args.label_smoothing
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
        use_weighted_sampler=cfg.use_weighted_sampler,
    )

    print(f"[train] backbone = {cfg.backbone}"
          + (f" (pretrained={cfg.pretrained_stem}, freeze_stem={cfg.freeze_stem})"
             if cfg.backbone != "dsps" else ""))
    model = build_model(cfg, NUM_CLASSES).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.head.set_prior(train_counts)

    if args.init_from:
        print(f"[train] --init_from: loading model weights ONLY from "
              f"{args.init_from} (no optimizer/scheduler/early-stopping state -- "
              f"this is a fresh run starting at epoch 0).")
        init_ckpt = load_checkpoint(args.init_from, map_location=device)
        missing, unexpected = model.load_state_dict(init_ckpt["state_dict"], strict=False)
        if missing or unexpected:
            print(f"[train] --init_from: {len(missing)} missing / "
                  f"{len(unexpected)} unexpected key(s) vs the current model "
                  f"(expected if --backbone/--unfreeze_stem differ from the "
                  f"source checkpoint's config).")

    criterion = CompositeLoss(
        train_counts, gamma=cfg.focal_gamma, beta_la=cfg.beta_la,
        beta_pc=cfg.beta_pc, cb_beta=cfg.cb_beta, label_smoothing=cfg.label_smoothing,
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
    ckpt = None
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

    # ---------------- model-weight EMA (see wtb/utils.py ModelEMA) ----------------
    # Built AFTER --init_from/--resume have already set the model's starting
    # weights, so the shadow copy starts from the right place. On resume, if
    # the checkpoint carries its own ema_state_dict, that's restored instead
    # of re-deriving the shadow from the (possibly already-drifted) raw weights.
    ema = None
    best_ema_path = os.path.join(ckpt_dir, "best_ema.pt")
    ema_best = -float("inf")
    if cfg.use_model_ema:
        ema = ModelEMA(model, decay=cfg.model_ema_decay)
        if ckpt is not None and ckpt.get("ema_state_dict") is not None:
            ema.load_state_dict(ckpt["ema_state_dict"])
            print("[resume] Restored model-EMA shadow weights from checkpoint.")
        else:
            print(f"[train] Model EMA enabled (decay={cfg.model_ema_decay}); "
                  f"shadow weights initialized from the current model state. "
                  f"Tracked separately in {best_ema_path}.")

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

            # MixUp/CutMix (image-level only; wtb/model.py is untouched).
            # `mixed=False` on most batches (mixup_cutmix_prob<1) makes
            # labels_a=labels_b=labels, lam=1.0 -- identical to no mixing.
            imgs, labels_a, labels_b, lam, mixed = apply_mixup_cutmix(
                imgs, labels, cfg.mixup_alpha, cfg.cutmix_alpha,
                cfg.mixup_cutmix_prob, cfg.use_mixup, cfg.use_cutmix,
            )

            with torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
                logits, feat = model(imgs)
                if mixed:
                    loss_a, parts = criterion(logits, feat, model.head.prototypes, labels_a)
                    loss_b, _ = criterion(logits, feat, model.head.prototypes, labels_b)
                    loss = lam * loss_a + (1.0 - lam) * loss_b
                else:
                    loss, parts = criterion(logits, feat, model.head.prototypes, labels_a)
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
                if ema is not None:
                    ema.update(model)

            # EMA-update the LTCP prototype memory bank from this batch's
            # features (after the optimizer step, so it tracks the current
            # feature space). Classes absent from this batch are untouched.
            # Skipped on mixed (MixUp/CutMix) steps: those features don't
            # cleanly belong to one class, and feeding blended features into
            # the prototype bank would inject noise into exactly the
            # rare-class stability mechanism LTCP exists for.
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
        val_metrics_ema = None
        if ema is not None:
            val_metrics_ema = run_validation(ema.module, val_loader, criterion, device, cfg.channels_last)
        epoch_seconds = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"[epoch {epoch+1}/{cfg.epochs}] "
              f"train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
              f"val_loss {val_metrics['loss']:.4f}  val_acc {val_metrics['accuracy']:.4f}  "
              f"val_macro_f1 {val_metrics['macro_f1']:.4f}  "
              f"val_balanced_acc {val_metrics['balanced_acc']:.4f}  "
              f"({epoch_seconds:.1f}s/epoch)")
        if val_metrics_ema is not None:
            print(f"           [EMA]  val_loss {val_metrics_ema['loss']:.4f}  "
                  f"val_acc {val_metrics_ema['accuracy']:.4f}  "
                  f"val_macro_f1 {val_metrics_ema['macro_f1']:.4f}")

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

        ema_state = ema.state_dict() if ema is not None else None

        save_checkpoint(
            last_path, model, CLASS_NAMES, epoch, early_stopping.best,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            early_stopping=early_stopping, backbone=cfg.backbone,
            ema_state_dict=ema_state, img_size=cfg.img_size,
        )
        if is_best:
            best_epoch = epoch   # BUGFIX: remember which epoch this actually was
            save_checkpoint(
                best_path, model, CLASS_NAMES, epoch, early_stopping.best,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                early_stopping=early_stopping, backbone=cfg.backbone,
                ema_state_dict=ema_state, img_size=cfg.img_size,
            )
            print(f"           -> new best (val_macro_f1={early_stopping.best:.4f}), saved best.pt")

        # best_ema.pt is selected independently, by the EMA model's OWN val
        # macro-F1 -- this never affects early stopping or which epoch
        # best.pt/last.pt point to; it's purely an extra artifact so you can
        # compare "raw best" vs "EMA best" at evaluate.py time and keep
        # whichever scores higher on the one-time test run.
        if val_metrics_ema is not None and val_metrics_ema["macro_f1"] > ema_best:
            ema_best = val_metrics_ema["macro_f1"]
            save_checkpoint(
                best_ema_path, ema.module, CLASS_NAMES, epoch, ema_best,
                backbone=cfg.backbone, img_size=cfg.img_size,
            )
            print(f"           -> new best EMA (val_macro_f1={ema_best:.4f}), saved best_ema.pt")

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
        early_stopping=early_stopping, backbone=cfg.backbone,
        ema_state_dict=(ema.state_dict() if ema is not None else None),
        img_size=cfg.img_size,
    )
    print(f"[train] Training finished. Best val macro-F1 = {early_stopping.best:.4f}")
    print(f"[train] Best weights saved to {best_path}")
    if ema is not None:
        print(f"[train] Best EMA weights (val_macro_f1={ema_best:.4f}) saved to {best_ema_path}")
        print("[train] Evaluate BOTH on the test set (evaluate.py --use_ema for the "
              "EMA one) and keep whichever scores higher -- neither is assumed better.")
    print("[train] Next: python evaluate.py --data_root <...> --checkpoint "
          f"{best_path} --out_dir {cfg.out_dir}")


if __name__ == "__main__":
    main()
