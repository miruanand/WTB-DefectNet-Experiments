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

import os
from typing import Dict, List, Optional

import torch
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
) -> None:
    """
    Saves a FULL resume-capable checkpoint. `epoch` should be the index of
    the epoch that just finished (0-based) -- on resume, training continues
    from epoch+1.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "state_dict": model.state_dict(),
        "classes": classes,
        "epoch": epoch,
        "best_f1": best_f1,
    }
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
