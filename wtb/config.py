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

# Must be set BEFORE torch/numpy get imported anywhere, and this is the
# first module every entrypoint (train.py/evaluate.py/gradcam.py) imports,
# so this is the one place to put it. Fixes:
#   "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
#    already initialized" -- a duplicate-OpenMP-runtime conflict that's
#    extremely common with Anaconda + PyTorch + numpy/MKL on Windows, and
#    which killed train.py immediately (exit code 3) on your first two
#    attempts before any epoch even started.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
    proto_momentum: float = 0.9        # LTCP EMA prototype-update momentum
    head_dropout: float = 0.15         # dropout on pooled features before LTCP

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


def get_device(verbose: bool = True, require_gpu: bool = False) -> torch.device:
    """
    Picks CUDA if present and prints a LOUD, impossible-to-miss diagnostic
    (previous version's single print line was easy to lose in scrollback /
    got silently dropped from pipeline_log.txt due to Windows stdout
    buffering when the child process wasn't run with -u).

    require_gpu=True hard-stops immediately if no CUDA GPU is found, instead
    of quietly training on CPU for hours. Use --require_gpu on train.py.
    """
    banner = "=" * 60
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            vram = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
            print(banner)
            print(f"[config] DEVICE = GPU  ({name}, {vram:.1f} GB VRAM)")
            print(banner)
            torch.backends.cudnn.benchmark = True   # fixed input size -> faster convs
    else:
        device = torch.device("cpu")
        if verbose:
            print(banner)
            print("[config] DEVICE = CPU  --  NO CUDA GPU DETECTED.")
            print("[config] Training on CPU will be extremely slow (likely 10-30x "
                  "slower per epoch than GPU). Most common causes on a Windows/"
                  "Anaconda box:")
            print("[config]   1. PyTorch was installed as the CPU-only build. Check "
                  "with: python -c \"import torch; print(torch.__version__, "
                  "torch.version.cuda)\" -- if torch.version.cuda prints None, "
                  "reinstall from https://pytorch.org/get-started/locally/ picking "
                  "your CUDA version.")
            print("[config]   2. NVIDIA driver / CUDA toolkit not installed or not on "
                  "PATH -- check with: nvidia-smi")
            print(banner)
        if require_gpu:
            raise RuntimeError(
                "[config] --require_gpu was set and no CUDA GPU was detected. "
                "Stopping now instead of silently training on CPU for hours. "
                "See the diagnostics printed above."
            )
    return device


def ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    return path
