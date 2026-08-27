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

Usage
-----
    python prepare_yolo_dataset.py --src "C:\\Users\\Student\\Desktop\\23BLC1224_FYP-1_Karthik_Sir\\WTBs2025" --dst "C:\\Users\\Student\\Desktop\\23BLC1224_FYP-1_Karthik_Sir\\WTBs2025_yolo"

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
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    random.seed(args.seed)

    for split in ("train", "val", "test"):
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)

    file_op = shutil.move if args.move else shutil.copy2
    totals = {"train": 0, "val": 0, "test": 0}

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
        splits = {
            "train": pairs[:n_train],
            "val": pairs[n_train:n_train + n_val],
            "test": pairs[n_train + n_val:],
        }

        print(f"{class_folder.name:30s} total={n:5d}  train={len(splits['train']):5d}  "
              f"val={len(splits['val']):5d}  test={len(splits['test']):5d}")

        for split, split_pairs in splits.items():
            for img_path, lbl_path in split_pairs:
                # Prefix filenames with the class folder name to avoid any
                # cross-class filename collisions once everything lands in
                # one flat train/images folder.
                safe_prefix = class_folder.name.replace(" ", "_")
                new_img_name = f"{safe_prefix}__{img_path.name}"
                new_lbl_name = f"{safe_prefix}__{lbl_path.name}"
                file_op(img_path, dst / split / "images" / new_img_name)
                file_op(lbl_path, dst / split / "labels" / new_lbl_name)
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
    print("\nIMPORTANT: the 9 class-name -> id mapping above must match the "
          "original data.yaml shipped inside WTBs2025.zip. It was verified "
          "to be: 0 oil leakage, 1 paint cracks, 2 localized damage, "
          "3 lightning strikes, 4 surface stains, 5 erosion, "
          "6 coating detachment, 7 protective film damage, 8 pinholes -- "
          "if your download differs, edit CLASS_NAMES above to match your "
          "own data.yaml before training.")


if __name__ == "__main__":
    main()
