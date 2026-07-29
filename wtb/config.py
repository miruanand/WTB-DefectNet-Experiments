"""
config.py
=========
Single source of truth for paths, class names, and hyperparameters.

Nothing in here is Colab-specific — everything reads from local disk / CLI args,
so this runs unchanged on any Linux/Windows machine with a CUDA GPU.
"""

import os
import random
import dataclasses
from typing import List, Optional

import numpy as np
import torch


# ----------------------------------------------------------------------
# Class names — taken directly from your WTBs2025 data.yaml so label
# indices match the dataset's own convention (not alphabetical order).
# ----------------------------------------------------------------------
CLASS_NAMES: List[str] = [
    "oil leakage",           # 0
    "paint cracks",          # 1
    "localized damage",      # 2
    "lightning strikes",     # 3  <- rare / safety-critical class
    "surface stains",        # 4
    "erosion",                # 5
    "coating detachment",    # 6
    "protective film damage",  # 7
    "pinholes",               # 8
]
NUM_CLASSES = len(CLASS_NAMES)

# Folder-name -> label-index map (must match CLASS_NAMES order above).
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclasses.dataclass
class Config:
    # ---- paths ----
    data_root: str = "./WTBs2025"      # folder produced by unzipping WTBs2025.zip
    out_dir: str = "./runs/exp1"       # checkpoints, logs, plots go here

    # ---- data ----
    img_size: int = 224
    val_fraction: float = 0.2          # 60:20:20 train:val:test split
    test_fraction: float = 0.2         # touched exactly once, after training is done
    num_workers: int = 8               # set to os.cpu_count() // 2 as a sane default
    pin_memory: bool = True

    # ---- model ----
    widths: tuple = (64, 128, 256, 512)
    tau: float = 16.0                  # LTCP cosine-logit temperature
    lam: float = 1.0                   # LTCP logit-adjustment strength

    # ---- optimization (from your plan doc, section 5) ----
    batch_size: int = 32
    epochs: int = 120
    base_lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    patience: int = 20                 # early stopping, on val macro-F1
    grad_accum_steps: int = 1          # bump this if you must lower batch_size for VRAM
    max_grad_norm: float = 5.0         # gradient clipping, stabilizes AMP training

    # ---- loss weights (from your plan doc, section 4) ----
    focal_gamma: float = 2.0
    beta_la: float = 1.0
    beta_pc: float = 0.1
    cb_beta: float = 0.999             # class-balanced effective-number beta

    # ---- misc ----
    seed: int = 42
    amp: bool = True                   # mixed precision — needs a CUDA GPU to matter
    channels_last: bool = True         # small free speedup on modern NVIDIA GPUs
    log_every: int = 50                # steps between console loss prints


def set_seed(seed: int = 42) -> None:
    """Full reproducibility across python/numpy/torch/cuda RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(verbose: bool = True) -> torch.device:
    """
    Picks CUDA if present and prints a clear diagnostic. Designed for a
    dedicated GPU box, not Colab — no drive mounting, no !pip magics.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            vram = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
            print(f"[config] Using GPU: {name} ({vram:.1f} GB VRAM)")
            torch.backends.cudnn.benchmark = True   # fixed input size -> faster convs
    else:
        device = torch.device("cpu")
        if verbose:
            print("[config] WARNING: no CUDA GPU detected, falling back to CPU. "
                  "Training WTB-DefectNet on CPU will be extremely slow — "
                  "check your CUDA / driver install if you expected a GPU here.")
    return device


def ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    return path
