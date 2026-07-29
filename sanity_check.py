"""
sanity_check.py
================
Run this FIRST on your GPU machine, before touching train.py:

    python sanity_check.py --data_root /path/to/WTBs2025

It does three things, each printed clearly so you can see exactly where a
problem is if one exists:
  1. Builds WTBDefectNet and pushes a random 224x224 batch through it,
     printing the output shape after every stage (stem/stage1..4/head).
  2. If --data_root is given, indexes your real dataset, prints the
     per-class counts, and pulls one real batch through the model.
  3. Runs one backward pass with the composite loss to confirm gradients
     flow (catches shape mismatches that only appear with grad enabled,
     e.g. inplace-op errors from GELU/BN inside residual branches).

No CUDA is required for this script (it works on CPU too, just slower) --
but it WILL use your GPU automatically if one is available.
"""

import argparse
import torch

from wtb.config import CLASS_NAMES, NUM_CLASSES, set_seed, get_device
from wtb.model import WTBDefectNet
from wtb.losses import CompositeLoss
from wtb.dataset import index_dataset, class_distribution, build_transforms, WTBDataset
from torch.utils.data import DataLoader


def check_forward_shapes(model, device):
    print("\n=== 1) Random-tensor forward pass (shape trace) ===")
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        h = model.stem(x);   print(f"  stem   -> {tuple(h.shape)}  (expect 2,64,56,56)")
        h = model.stage1(h); print(f"  stage1 -> {tuple(h.shape)}  (expect 2,64,56,56)")
        h = model.stage2(h); print(f"  stage2 -> {tuple(h.shape)}  (expect 2,128,28,28)")
        h = model.stage3(h); print(f"  stage3 -> {tuple(h.shape)}  (expect 2,256,14,14)")
        h = model.stage4(h); print(f"  stage4 -> {tuple(h.shape)}  (expect 2,512,7,7)")
        feat = model.gap(h).flatten(1)
        logits, f = model.head(feat)
        print(f"  logits -> {tuple(logits.shape)}  (expect 2,{NUM_CLASSES})")
    print(f"  Total trainable params: {model.count_params()/1e6:.2f} M "
          f"(plan doc target: 8-12 M)")


def check_backward_pass(model, device):
    print("\n=== 2) One backward pass with CompositeLoss (gradient sanity) ===")
    fake_counts = [100] * NUM_CLASSES
    criterion = CompositeLoss(fake_counts).to(device)
    x = torch.randn(4, 3, 224, 224, device=device)
    y = torch.randint(0, NUM_CLASSES, (4,), device=device)
    model.train()
    logits, feat = model(x)
    loss, parts = criterion(logits, feat, model.head.prototypes, y)
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  loss = {loss.item():.4f}  ({parts})")
    print(f"  parameters with gradients: {n_grad}/{n_total} "
          f"{'OK' if n_grad == n_total else '<-- SOME PARAMS GOT NO GRADIENT, investigate!'}")


def check_real_dataset(data_root, model, device, img_size=224, batch_size=8):
    print(f"\n=== 3) Real dataset check: {data_root} ===")
    samples = index_dataset(data_root)
    print(f"  Found {len(samples)} images across {NUM_CLASSES} classes:")
    for name, n in zip(CLASS_NAMES, class_distribution(samples)):
        print(f"    {name:24s}: {n:5d}")

    ds = WTBDataset(samples[:batch_size], build_transforms(img_size, train=True))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    imgs, labels = next(iter(loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        logits, _ = model(imgs)
    print(f"  Real batch -> images {tuple(imgs.shape)}, logits {tuple(logits.shape)}")
    print("  Dataset + model plumbing OK.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="",
                     help="Path to the unzipped WTBs2025 folder (optional for this check)")
    args = ap.parse_args()

    set_seed(42)
    device = get_device()

    model = WTBDefectNet(num_classes=NUM_CLASSES).to(device)

    check_forward_shapes(model, device)
    check_backward_pass(model, device)

    if args.data_root:
        check_real_dataset(args.data_root, model, device)
    else:
        print("\n(--data_root not given, skipping real-dataset check)")

    print("\nAll sanity checks passed.")
