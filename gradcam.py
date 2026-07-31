"""
gradcam.py
==========
Generates Grad-CAM attention heatmaps -- the "where is the model looking"
figure for your report/defense.

    python gradcam.py --data_root "D:\\...\\WTBs2025" --checkpoint ./runs/exp1/checkpoints/best.pt --out_dir ./runs/exp1

For each of the 9 classes, it picks a few TEST-set images (same split as
evaluate.py -- read-only, no leakage), runs the model, and overlays a
heatmap showing which pixels drove the prediction. Useful for spotting
whether the model is actually looking at the defect (good) or at some
unrelated background cue like sky, mounting hardware, or watermark text
(bad -- and worth mentioning in your report if you see it).

Grad-CAM is computed on the LAST feature map before global-average-pooling
(model.stage4's output, 7x7x512) -- the standard placement, since that's
the last spot where spatial location information still exists before the
model collapses to a single feature vector.

Outputs to <out_dir>/gradcam/:
    <class>_<n>.png     -- one figure per example: original | heatmap overlay
    grid.png             -- all examples combined into one grid, handy for
                             a single figure in your report
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from wtb.config import Config, CLASS_NAMES, NUM_CLASSES, set_seed, get_device
from wtb.model import build_model
from wtb.dataset import index_dataset, stratified_split_3way, build_transforms
from wtb.utils import load_checkpoint


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./runs/exp1")
    ap.add_argument("--num_per_class", type=int, default=2,
                     help="How many test-set examples to visualize per class.")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42,
                     help="Must match train.py's seed to pick from the TEST split only.")
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--test_fraction", type=float, default=0.2)
    return ap.parse_args()


def compute_gradcam(model, x: torch.Tensor, target_class: int):
    """
    Returns (cam as HxW numpy in [0,1], predicted_class, confidence) for one
    image tensor x of shape (1,3,H,W). Uses the model's stored
    _last_feat_map (set in WTBDefectNet.forward) as the CAM source layer.
    """
    model.zero_grad(set_to_none=True)
    logits, _ = model(x)
    probs = F.softmax(logits, dim=1)[0]
    pred_class = int(logits.argmax(dim=1).item())

    feat_map = model._last_feat_map          # (1, C, h, w), still in the graph
    feat_map.retain_grad()
    score = logits[0, target_class]
    score.backward()

    grads = feat_map.grad[0]                  # (C, h, w)
    activations = feat_map.detach()[0]        # (C, h, w)
    weights = grads.mean(dim=(1, 2))          # (C,)  -- global-average-pooled gradient per channel

    cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))  # (h, w)
    cam = cam - cam.min()
    denom = cam.max().clamp(min=1e-8)
    cam = (cam / denom).cpu().numpy()

    return cam, pred_class, float(probs[pred_class].item()), float(probs[target_class].item())


def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, img_size: int, alpha: float = 0.45) -> np.ndarray:
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((img_size, img_size), Image.BILINEAR)
    cam_arr = np.asarray(cam_img).astype(np.float64) / 255.0
    heat = cm.jet(cam_arr)[:, :, :3]                         # (H,W,3) in [0,1]
    base = np.asarray(pil_img.resize((img_size, img_size))).astype(np.float64) / 255.0
    overlay = (1 - alpha) * base + alpha * heat
    return np.clip(overlay, 0, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(f"[gradcam] Indexing dataset and rebuilding the same split (seed={args.seed})...")
    samples = index_dataset(args.data_root)
    _, _, test_samples = stratified_split_3way(
        samples, args.val_fraction, args.test_fraction, args.seed
    )
    by_class = {i: [] for i in range(NUM_CLASSES)}
    for path, label in test_samples:
        by_class[label].append(path)

    rng = random.Random(args.seed)
    selected = []   # list of (path, label)
    for label in range(NUM_CLASSES):
        paths = by_class[label]
        if not paths:
            print(f"[gradcam] WARNING: no test images for class "
                  f"'{CLASS_NAMES[label]}', skipping.")
            continue
        rng.shuffle(paths)
        selected.extend((p, label) for p in paths[:args.num_per_class])

    print(f"[gradcam] Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    cfg = Config(backbone=ckpt.get("backbone") or "dsps")
    print(f"[gradcam] backbone = {cfg.backbone}")
    model = build_model(cfg, NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()   # eval mode -- BN uses running stats; gradients still flow fine for Grad-CAM

    eval_transform = build_transforms(args.img_size, train=False)
    out_dir = os.path.join(args.out_dir, "gradcam")
    os.makedirs(out_dir, exist_ok=True)

    grid_cells = []   # (overlay_array, title) in class order, for the combined grid
    for path, true_label in selected:
        pil_img = Image.open(path).convert("RGB")
        x = eval_transform(pil_img).unsqueeze(0).to(device)

        cam, pred_class, pred_conf, true_conf = compute_gradcam(model, x, target_class=true_label)
        overlay = overlay_heatmap(pil_img, cam, args.img_size)

        correct = (pred_class == true_label)
        title = (f"true: {CLASS_NAMES[true_label]}\n"
                 f"pred: {CLASS_NAMES[pred_class]} ({pred_conf:.2f}) "
                 f"{'OK' if correct else 'WRONG'}")
        grid_cells.append((overlay, title, correct))

        # individual figure: original | heatmap overlay
        fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
        axes[0].imshow(pil_img.resize((args.img_size, args.img_size)))
        axes[0].set_title("original")
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title(title, fontsize=9,
                           color=("green" if correct else "red"))
        axes[1].axis("off")
        fig.tight_layout()
        safe_name = CLASS_NAMES[true_label].replace(" ", "_")
        idx = sum(1 for c in os.listdir(out_dir) if c.startswith(safe_name))
        fig.savefig(os.path.join(out_dir, f"{safe_name}_{idx}.png"), dpi=150)
        plt.close(fig)

    # combined grid, num_per_class columns x NUM_CLASSES rows
    ncols = args.num_per_class
    nrows = NUM_CLASSES
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    if nrows == 1:
        axes = axes[None, :]
    if ncols == 1:
        axes = axes[:, None]
    cell_iter = iter(grid_cells)
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            try:
                overlay, title, correct = next(cell_iter)
            except StopIteration:
                ax.axis("off")
                continue
            ax.imshow(overlay)
            ax.set_title(title, fontsize=7, color=("green" if correct else "red"))
            ax.axis("off")
    fig.tight_layout()
    grid_path = os.path.join(out_dir, "grid.png")
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)

    print(f"[gradcam] Saved {len(selected)} individual figures + grid.png to {out_dir}")


if __name__ == "__main__":
    main()
