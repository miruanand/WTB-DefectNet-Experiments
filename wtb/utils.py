"""
utils.py
========
Metrics (macro-P/R/F1, balanced accuracy, kappa, MCC, per-class F1),
checkpoint save/load, and an EarlyStopping helper -- shared by whichever
training script (single-split or 5-fold CV) you run next.

Checkpoints now carry everything needed for a TRUE resume: model weights,
optimizer state, scheduler state, AMP GradScaler state, and EarlyStopping's
internal counters/best-state. Interrupting train.py (Ctrl+C, crash, power
loss) and re-running with --resume continues from the exact epoch it left
off on, not from epoch 0.
"""

import copy
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef, accuracy_score,
)


def compute_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "per_class_f1": f1_score(
            y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
        ).tolist(),
    }


class EarlyStopping:
    """Stops when val macro-F1 hasn't improved for `patience` epochs; keeps best weights."""

    def __init__(self, patience: int = 20, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.best_state = None
        self.since = 0

    def step(self, value: float, model: torch.nn.Module) -> bool:
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.since = 0
        else:
            self.since += 1
        return self.since >= self.patience     # True -> caller should stop

    def restore_best(self, model: torch.nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

    # ---- resume support ----
    def state_dict(self) -> Dict:
        """Everything needed to resume early-stopping exactly where it left off.
        best_state (the best model weights so far) is included so a resumed
        run can still restore_best() at the end even if training is
        interrupted before val improves again."""
        return {
            "patience": self.patience,
            "mode": self.mode,
            "best": self.best,
            "since": self.since,
            "best_state": self.best_state,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.patience = state["patience"]
        self.mode = state["mode"]
        self.best = state["best"]
        self.since = state["since"]
        self.best_state = state["best_state"]


# ======================================================================
# MixUp / CutMix -- image-level augmentation, applied in train.py before
# the forward pass. Neither touches wtb/model.py; they operate purely on
# the input tensor and the label bookkeeping around it.
# ======================================================================
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Beta(alpha,alpha)-weighted linear blend of two shuffled copies of
    the batch. Returns (mixed_x, y_a, y_b, lam) -- the loss should be
    computed as lam*criterion(out, y_a) + (1-lam)*criterion(out, y_b)."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """Pastes a random rectangular patch from a shuffled copy of the batch
    into each image; lam is corrected to the ACTUAL pasted-area fraction
    (not the sampled one) since the patch is clipped to image bounds."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[index]

    H, W = x.size(2), x.size(3)
    cut_rat = float(np.sqrt(max(1.0 - lam, 0.0)))
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = int(np.clip(cy - cut_h // 2, 0, H)), int(np.clip(cy + cut_h // 2, 0, H))
    x1, x2 = int(np.clip(cx - cut_w // 2, 0, W)), int(np.clip(cx + cut_w // 2, 0, W))

    x[:, :, y1:y2, x1:x2] = x[index][:, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(H * W))
    return x, y_a, y_b, lam


def apply_mixup_cutmix(
    x: torch.Tensor, y: torch.Tensor,
    mixup_alpha: float, cutmix_alpha: float, prob: float,
    use_mixup: bool, use_cutmix: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
    """
    With probability `prob`, mixes this batch (picking MixUp or CutMix
    50/50 when both are enabled); otherwise returns it unchanged.
    Returns (x, y_a, y_b, lam, mixed) -- when mixed=False, y_a is y and
    y_b is y and lam=1.0, so `lam*crit(out,y_a) + (1-lam)*crit(out,y_b)`
    is always the correct loss whether or not mixing happened.
    """
    if (not use_mixup and not use_cutmix) or np.random.rand() > prob:
        return x, y, y, 1.0, False
    do_cutmix = use_cutmix if not (use_mixup and use_cutmix) else (np.random.rand() < 0.5)
    if do_cutmix:
        x, y_a, y_b, lam = cutmix_data(x, y, cutmix_alpha)
    else:
        x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
    return x, y_a, y_b, lam, True


# ======================================================================
# Model weight EMA -- a shadow copy of the ENTIRE model (all params +
# buffers), decayed toward the live training weights every optimizer
# step. This is separate from LTCP's own prototype EMA (wtb/model.py) --
# that one smooths the classifier's class prototypes; this one smooths
# every weight in the network, DSPS/TSDB/ASA/DRFB/WGFR/MSCA/LTCP included,
# without changing what any of those blocks compute.
# ======================================================================
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            new_val = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(new_val, alpha=1.0 - self.decay)
            else:
                # non-float buffers (e.g. LTCP's `initialized` bool mask) --
                # EMA is meaningless for these, just track the live value.
                v.copy_(new_val)

    def state_dict(self) -> Dict:
        return self.module.state_dict()

    def load_state_dict(self, state: Dict) -> None:
        self.module.load_state_dict(state)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    classes: List[str],
    epoch: int,
    best_f1: float,
    optimizer=None,
    scheduler=None,
    scaler=None,
    early_stopping: Optional[EarlyStopping] = None,
    backbone: Optional[str] = None,
    ema_state_dict: Optional[Dict] = None,
) -> None:
    """
    Saves a FULL resume-capable checkpoint. `epoch` should be the index of
    the epoch that just finished (0-based) -- on resume, training continues
    from epoch+1.

    `backbone` (e.g. "dsps", "resnet18") is stored so evaluate.py and
    gradcam.py can call wtb.model.build_model() with the SAME architecture
    the checkpoint was trained with, instead of assuming WTBDefectNet.

    `ema_state_dict`, if given, stores the model-weight-EMA shadow weights
    (see ModelEMA above) alongside the raw weights, so a resumed run can
    restore the EMA shadow exactly and evaluate.py can optionally load the
    EMA weights instead of the raw ones (--use_ema). Older checkpoints
    (saved before this was added) simply won't have this key -- everything
    that reads it uses .get(...) with a fallback, so old checkpoints still
    load fine.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "state_dict": model.state_dict(),
        "classes": classes,
        "epoch": epoch,
        "best_f1": best_f1,
        "backbone": backbone,
    }
    if ema_state_dict is not None:
        state["ema_state_dict"] = ema_state_dict
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if early_stopping is not None:
        state["early_stopping"] = early_stopping.state_dict()

    # Write to a temp file first, then atomically replace -- if the process
    # is killed mid-write (power loss, OOM kill) the previous good
    # checkpoint on disk is never left half-written/corrupted.
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str, map_location=None) -> Dict:
    return torch.load(path, map_location=map_location)
