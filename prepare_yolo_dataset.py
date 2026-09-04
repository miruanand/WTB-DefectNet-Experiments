"""
prepare_yolo_dataset.py
========================
WTBs2025.zip (once extracted) is organized BY CLASS:

    WTBs2025/
      coating detachment/images/*.jpg   coating detachment/labels/*.txt
      erosion/images/*.jpg              erosion/labels/*.txt
      ... (9 folders total)

Ultralytics YOLO expects it organized BY SPLIT instead:

    WTBs2025_yolo/
      train/images/*.jpg   train/labels/*.txt
      val/images/*.jpg     val/labels/*.txt
      test/images/*.jpg    test/labels/*.txt
      data.yaml

This script does that conversion with a per-class stratified 7:2:1
split (matching the ratio the WTBs2025 dataset paper itself used), so
every split gets a proportional share of each of the 9 defect types --
important given how skewed the classes are (erosion: 2799 images vs.
lightning strikes: 57).

NEW: --oversample_max_ratio (default 4.0) duplicates minority-class
TRAIN images (never val/test -- duplicating into your evaluation sets
would inflate your reported accuracy dishonestly) to reduce class
imbalance, capped at a max multiplier per class so tiny classes don't
get duplicated to the point of just memorizing the same handful of
images repeatedly. Set --oversample_max_ratio 1.0 to disable (no
duplication, original behavior).

Usage
-----
    python prepare_yolo_dataset.py --src "C:\\path\\to\\WTBs2025" --dst "C:\\path\\to\\WTBs2025_yolo"

By default files are COPIED (safe, doesn't touch your original
extracted WTBs2025 folder). Pass --move if you want to save disk space
instead (only do this once you're sure you don't need the by-class
layout anymore).
"""

import argparse
import random
import shutil
from pathlib import Path

CLASS_NAMES = [
    "oil leakage",
    "paint cracks",
    "localized damage",
    "lightning strikes",
    "surface stains",
    "erosion",
    "coating detachment",
    "protective film damage",
    "pinholes",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to extracted WTBs2025 folder (by-class layout)")
    ap.add_argument("--dst", required=True, help="output path for the restructured (by-split) dataset")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.2)
    # test = 1 - train - val
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move", action="store_true", help="move instead of copy (saves disk space)")
    ap.add_argument(
        "--oversample_max_ratio",
        type=float,
        default=4.0,
        help="Cap on how many times a minority class's TRAIN images get duplicated "
             "to reduce class imbalance. E.g. with the default 4.0, a class whose "
             "'natural' target (to match the largest class) would need 10x "
             "duplication instead gets capped at 4x -- enough to meaningfully "
             "increase its training exposure without just re-showing the model "
             "the same handful of images over and over. Set to 1.0 to disable "
             "oversampling entirely (original, un-duplicated behavior). Never "
             "applied to val/test -- only train images are duplicated.",
    )
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    random.seed(args.seed)

    for split in ("train", "val", "test"):
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)

    file_op = shutil.move if args.move else shutil.copy2

    # ---- Phase 1: collect per-class splits (no writing yet) ----
    per_class_splits = {}  # class_name -> {"train": [...], "val": [...], "test": [...]}
    for class_folder in sorted(src.iterdir()):
        if not class_folder.is_dir():
            continue
        img_dir = class_folder / "images"
        lbl_dir = class_folder / "labels"
        if not img_dir.is_dir():
            continue

        images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg")) + sorted(img_dir.glob("*.png"))
        pairs = []
        for img_path in images:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))
            else:
                print(f"  WARNING: no label for {img_path.name}, skipping")

        random.shuffle(pairs)
        n = len(pairs)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        per_class_splits[class_folder.name] = {
            "train": pairs[:n_train],
            "val": pairs[n_train:n_train + n_val],
            "test": pairs[n_train + n_val:],
        }

    # ---- Phase 2: compute oversampling multiplier per class (train only) ----
    max_train_count = max(len(s["train"]) for s in per_class_splits.values())
    multipliers = {}
    for class_name, s in per_class_splits.items():
        n_train = len(s["train"])
        if n_train == 0:
            multipliers[class_name] = 1
            continue
        natural_ratio = max_train_count / n_train
        capped_ratio = min(natural_ratio, args.oversample_max_ratio)
        multipliers[class_name] = max(1, round(capped_ratio))

    # ---- Phase 3: write everything out ----
    totals = {"train": 0, "val": 0, "test": 0}
    for class_name, s in per_class_splits.items():
        safe_prefix = class_name.replace(" ", "_")
        mult = multipliers[class_name]

        print(f"{class_name:30s} total={sum(len(v) for v in s.values()):5d}  "
              f"train={len(s['train']):5d} (x{mult} oversample)  "
              f"val={len(s['val']):5d}  test={len(s['test']):5d}")

        for split, pairs in s.items():
            n_copies = mult if split == "train" else 1  # NEVER oversample val/test
            for img_path, lbl_path in pairs:
                for copy_idx in range(n_copies):
                    suffix = f"__dup{copy_idx}" if copy_idx > 0 else ""
                    new_img_name = f"{safe_prefix}__{img_path.stem}{suffix}{img_path.suffix}"
                    new_lbl_name = f"{safe_prefix}__{lbl_path.stem}{suffix}{lbl_path.suffix}"
                    # copy2 (not move) for any duplicate beyond the first, even
                    # in --move mode, since a "moved" file can't be moved twice.
                    op = file_op if copy_idx == 0 else shutil.copy2
                    op(img_path, dst / split / "images" / new_img_name)
                    op(lbl_path, dst / split / "labels" / new_lbl_name)
                    totals[split] += 1

    yaml_lines = [
        f"path: {dst.as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "names:",
    ] + [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    (dst / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    print(f"\nDone. train={totals['train']}  val={totals['val']}  test={totals['test']}")
    print(f"data.yaml written to: {dst / 'data.yaml'}")
    if args.oversample_max_ratio > 1.0:
        print(f"\nNote: train counts above include oversampling duplicates (capped at "
              f"{args.oversample_max_ratio}x per class). val/test are never duplicated, "
              f"so your reported test-set accuracy stays honest.")
    print("\nIMPORTANT: the 9 class-name -> id mapping above must match the "
          "original data.yaml shipped inside WTBs2025.zip. It was verified "
          "to be: 0 oil leakage, 1 paint cracks, 2 localized damage, "
          "3 lightning strikes, 4 surface stains, 5 erosion, "
          "6 coating detachment, 7 protective film damage, 8 pinholes -- "
          "if your download differs, edit CLASS_NAMES above to match your "
          "own data.yaml before training.")


if __name__ == "__main__":
    main()
