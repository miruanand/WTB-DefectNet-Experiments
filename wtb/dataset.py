"""
dataset.py
==========
Handles the REAL on-disk layout of WTBs2025 after you unzip WTBs2025.zip:

    WTBs2025/
        oil leakage/
            images/*.jpg
            labels/*.txt        <- YOLO bbox labels, unused here (classification only)
        paint cracks/
            images/*.jpg
            labels/*.txt
        ...

This is a Roboflow object-detection export, NOT a plain torchvision
ImageFolder tree (class/*.jpg). Using torchvision.datasets.ImageFolder
directly on this would silently find 0 images. WTBDataset below walks the
correct nested "<class>/images/" path and ignores the bbox labels.
"""

import os
import glob
from collections import Counter
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T

from .config import CLASS_NAMES, CLASS_TO_IDX

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def index_dataset(data_root: str) -> List[Tuple[str, int]]:
    """
    Scans data_root/<class name>/images/* for every class in CLASS_NAMES
    and returns a list of (filepath, label_idx) — the label comes from the
    folder name via CLASS_TO_IDX, so it always matches data.yaml's indexing.
    """
    samples = []
    missing = []
    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(data_root, cls_name, "images")
        if not os.path.isdir(cls_dir):
            missing.append(cls_name)
            continue
        label = CLASS_TO_IDX[cls_name]
        for ext in IMG_EXTS:
            for path in glob.glob(os.path.join(cls_dir, f"*{ext}")):
                samples.append((path, label))

    if missing:
        raise FileNotFoundError(
            f"Could not find an 'images' folder for these classes under "
            f"'{data_root}': {missing}. Check --data_root points at the "
            f"unzipped WTBs2025 folder (the one containing the 9 class "
            f"subfolders), not the zip file itself."
        )
    if len(samples) == 0:
        raise RuntimeError(f"Found 0 images under {data_root}. Check the path.")

    return samples


def class_distribution(samples: List[Tuple[str, int]]) -> List[int]:
    counts = Counter(label for _, label in samples)
    return [counts.get(i, 0) for i in range(len(CLASS_NAMES))]


def build_transforms(img_size: int = 224, train: bool = True) -> T.Compose:
    """RandAugment + defect-preserving crops on train; deterministic resize on eval."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    if train:
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),          # UAV shots have no fixed "up" — safe to use
            T.RandomRotation(15),
            T.RandAugment(num_ops=2, magnitude=7),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(imagenet_mean, imagenet_std),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(imagenet_mean, imagenet_std),
    ])


class WTBDataset(Dataset):
    """Thin (path, label) -> (tensor, label) dataset. Loads lazily, one image at a time."""

    def __init__(self, samples: List[Tuple[str, int]], transform: T.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {path}: {e}")
        return self.transform(img), label


def make_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Inverse-frequency sampler so rare classes (lightning strikes: 285 images)
    are seen roughly as often as common ones (erosion: 2799 images) per epoch.
    This is on TOP of the class-balanced loss — sampling balances what the
    model sees, the loss balances how hard it's penalized either way.
    """
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def stratified_split_3way(
    samples: List[Tuple[str, int]],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Per-class 60:20:20 train:val:test split (fractions default to 0.2/0.2,
    train gets the remainder). Splitting PER CLASS, not globally, matters
    here because lightning-strikes has only 285 images — a global shuffle
    can easily starve its val or test slice; this guarantees every one of
    the 9 classes is represented proportionally in all three splits.

    val_fraction / test_fraction apply to the FULL per-class pool, so with
    the defaults (0.2, 0.2) you get exactly the requested 60:20:20.
    """
    assert 0 < val_fraction < 1 and 0 < test_fraction < 1
    assert val_fraction + test_fraction < 1, "val + test fractions must leave room for train"

    rng = np.random.default_rng(seed)
    by_class = {}
    for path, label in samples:
        by_class.setdefault(label, []).append((path, label))

    train, val, test = [], [], []
    for label, items in by_class.items():
        items = items.copy()
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(round(n * test_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_loaders(
    data_root: str,
    img_size: int = 224,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    num_workers: int = 8,
    pin_memory: bool = True,
    seed: int = 42,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[int]]:
    """
    One-call convenience: index -> 60:20:20 stratified split -> transforms -> loaders.
    Returns (train_loader, val_loader, test_loader, train_class_counts).
    train_class_counts is needed by CompositeLoss / LTCP's prior.

    Use val_loader during training (early stopping, model selection, the
    checkpoint you keep). Touch test_loader exactly ONCE, after training is
    completely finished, for the number you report in the paper — using it
    more than that turns it into a second validation set and the reported
    numbers stop being trustworthy.

    use_weighted_sampler=False (new default) trains on the natural class
    distribution and leaves imbalance correction entirely to CompositeLoss
    (class-balanced focal weights + logit adjustment) and LTCP's EMA
    prototype bank. See the long comment on Config.use_weighted_sampler in
    config.py for why stacking this sampler on top of logit-adjusted
    training is a bug, not an extra safety net. Pass True to restore the
    old always-on behaviour for a side-by-side comparison run.
    """
    samples = index_dataset(data_root)
    print(f"[dataset] Found {len(samples)} images across {len(CLASS_NAMES)} classes.")
    dist = class_distribution(samples)
    for name, n in zip(CLASS_NAMES, dist):
        print(f"    {name:24s}: {n:5d}")

    train_samples, val_samples, test_samples = stratified_split_3way(
        samples, val_fraction, test_fraction, seed
    )
    train_counts = class_distribution(train_samples)
    print(f"[dataset] Split -> train {len(train_samples)} | "
          f"val {len(val_samples)} | test {len(test_samples)} "
          f"({100*len(train_samples)/len(samples):.0f}:"
          f"{100*len(val_samples)/len(samples):.0f}:"
          f"{100*len(test_samples)/len(samples):.0f})")

    train_ds = WTBDataset(train_samples, build_transforms(img_size, train=True))
    val_ds = WTBDataset(val_samples, build_transforms(img_size, train=False))
    test_ds = WTBDataset(test_samples, build_transforms(img_size, train=False))

    if use_weighted_sampler:
        print("[dataset] use_weighted_sampler=True -- rebalancing batches with "
              "WeightedRandomSampler ON TOP of CompositeLoss's own imbalance "
              "correction (class-balanced weights + logit adjustment). See "
              "Config.use_weighted_sampler in config.py before trusting this "
              "combination's numbers.")
        sampler = make_weighted_sampler([l for _, l in train_samples])
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=True, persistent_workers=(num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=True, persistent_workers=(num_workers > 0),
        )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader, train_counts
