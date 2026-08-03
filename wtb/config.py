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

    # `backbone` picks which class build_model() (wtb/model.py) constructs:
    #   "dsps"      -> WTBDefectNet, fully custom, trained from scratch
    #                  (DSPS stem -> Stage1 -> ... -> LTCP, exactly as in
    #                  Proposed_Architecture.docx). This is what exp1,
    #                  Try_2, and Try_4 all used -- best result so far is
    #                  exp1's 65.37% macro-F1, and every doc-alignment
    #                  edit since then has scored LOWER, not higher.
    #   "resnet18"/"resnet34" -> WTBDefectNetHybrid: an ImageNet-pretrained
    #                  ResNet stem+layer1+layer2 (56x56x64 -> 28x28x128)
    #                  replaces ONLY the generic-feature part of the
    #                  network (DSPS + Stage1's TSDB/ASA). TSDB, ASA, DRFB,
    #                  WGFR, MSCA, and LTCP are all still there, at the
    #                  same doc-specified stage placements, doing the same
    #                  job on the same 28x28x128 -> 14x14x256 -> 7x7x512
    #                  pyramid -- nothing about the novel-block story
    #                  changes. Only the part every from-scratch CNN has to
    #                  relearn from ~7.5k images (generic edges/textures/
    #                  colour blobs) is borrowed instead of trained from
    #                  nothing. Recommended first thing to try: a 9-class
    #                  classifier trained fully from scratch on ~7.5k
    #                  images is a genuinely hard data regime, and it's a
    #                  more likely explanation for exp1/Try_2/Try_4 all
    #                  landing in the 54-65% macro-F1 band than any single
    #                  block being "wrong".
    backbone: str = "dsps"
    pretrained_stem: bool = True       # only used when backbone != "dsps"
    freeze_stem: bool = True           # freeze the pretrained conv1/layer1/layer2 for
                                        # the first run (fewer params to overfit with on
                                        # ~7.5k images); set False for a second, longer
                                        # fine-tuning pass once the new head has adapted

    # WTBDefectNet's imbalance handling currently stacks FOUR separate
    # corrections on top of each other: (1) this WeightedRandomSampler,
    # (2) class-balanced focal loss weights in CompositeLoss, (3) train-
    # time logit adjustment baked into LTCP's logits, (4) focal loss's own
    # easy-example down-weighting. (2)-(4) are exactly what the doc's Gap
    # G4 / "Composite Loss" column call for, so they stay. But (1) and (3)
    # work against each other: logit adjustment (Menon et al., 2021) is
    # derived assuming the model is trained on the NATURAL imbalanced
    # distribution -- its whole point is to correct for a bias that only
    # exists if training saw the real class frequencies. Rebalancing the
    # batches with a sampler removes that bias at the data level, so the
    # logit-adjustment term is then correcting for an imbalance the model
    # was never actually trained on. Set to True to restore the old
    # (stacked) behaviour for comparison.
    use_weighted_sampler: bool = False

    # ---- accuracy-push additions (training procedure only -- NONE of these
    # touch DSPS/TSDB/ASA/DRFB/WGFR/MSCA/LTCP or the doc's block placements
    # in wtb/model.py; they only change how the same architecture is
    # trained/evaluated) ----

    # MixUp / CutMix: applied on a random subset of batches (mixup_cutmix_prob)
    # to the IMAGES only. Helps the texture-heavy, low-support classes
    # (surface stains n=263, paint cracks n=447, localized damage n=1027)
    # generalize better instead of memorizing the ~150-300 train images
    # each one actually has after the 60:20:20 split. When both are True,
    # each mixed batch randomly picks one (50/50) rather than applying both.
    use_mixup: bool = True
    mixup_alpha: float = 0.2            # Beta(alpha,alpha) mixing coefficient
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    mixup_cutmix_prob: float = 0.5      # fraction of batches that get mixed at all

    # Label smoothing on the logit-adjusted CE term inside CompositeLoss
    # (wtb/losses.py). 0.0 = disabled = identical to before this change.
    label_smoothing: float = 0.05

    # Model weight EMA (NOT the same as LTCP's own prototype EMA -- this
    # tracks a shadow copy of the WHOLE model's weights, updated every
    # optimizer step). Cheap, near-guaranteed small val-F1 bump, standard
    # practice (timm/EfficientNet-style). Tracked as a SEPARATE checkpoint
    # (checkpoints/best_ema.pt) alongside the normal best.pt so you can
    # compare both and keep whichever scores higher on your one-time test
    # evaluation -- it does not replace or change how best.pt is selected.
    use_model_ema: bool = True
    model_ema_decay: float = 0.999

    # ---- optimization (from your plan doc, section 5) ----
    batch_size: int = 32
    epochs: int = 120
    base_lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    patience: int = 35                 # early stopping, on val macro-F1 (was 20 -- too
                                        # short: exp1's own best epoch was 120/120, and
                                        # Try_3 got cut off mid-recovery at patience=20)
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
