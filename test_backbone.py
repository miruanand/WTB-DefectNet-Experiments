"""
test_backbone.py
=================
Quick sanity check -- run this FIRST, before any real training, to
confirm the backbone swap builds correctly on your machine (this exact
script is what was used to verify the integration end-to-end before
handing it over: model build -> forward pass -> one real training step
on synthetic data, all passing).

    python test_backbone.py

Expected output ends with "ALL CHECKS PASSED".
"""

import torch

from wtb_yolo_modules import register_wtb_modules

register_wtb_modules()

from ultralytics import YOLO


def main():
    print("[1/3] Building model from yolo11-wtbdefectnet.yaml ...")
    model = YOLO("yolo11-wtbdefectnet.yaml")
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"      OK -- {n_params:,} params, stride {model.model.stride.tolist()}")

    print("[2/3] Forward pass on a dummy 640x640 image ...")
    model.model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model.model(x)
    # In eval mode Detect returns (predictions, raw_feature_maps); we only
    # care about the predictions tensor here.
    preds = out[0] if isinstance(out, (list, tuple)) else out
    print(f"      OK -- output shape {tuple(preds.shape)} "
          f"(expect [1, 4+num_classes, num_anchors])")

    print("[3/3] One real training step on synthetic data (checks the loss "
          "computes and backprops without shape errors) ...")
    import tempfile, os
    from PIL import Image
    import random

    with tempfile.TemporaryDirectory() as tmp:
        for split in ("train", "val"):
            os.makedirs(f"{tmp}/{split}/images", exist_ok=True)
            os.makedirs(f"{tmp}/{split}/labels", exist_ok=True)
            for i in range(4):
                Image.new("RGB", (640, 640), (random.randint(0, 255),) * 3).save(
                    f"{tmp}/{split}/images/img{i}.jpg"
                )
                with open(f"{tmp}/{split}/labels/img{i}.txt", "w") as f:
                    f.write(f"{random.randint(0, 8)} 0.5 0.5 0.2 0.2\n")

        yaml_path = f"{tmp}/data.yaml"
        with open(yaml_path, "w") as f:
            f.write(
                f"path: {tmp}\ntrain: train/images\nval: val/images\n"
                "names:\n  0: oil leakage\n  1: paint cracks\n  2: localized damage\n"
                "  3: lightning strikes\n  4: surface stains\n  5: erosion\n"
                "  6: coating detachment\n  7: protective film damage\n  8: pinholes\n"
            )

        model = YOLO("yolo11-wtbdefectnet.yaml")
        model.train(
            data=yaml_path, epochs=1, imgsz=640, batch=2,
            device="cpu", workers=0, plots=False, val=False, verbose=False,
        )
    print("      OK -- training step completed without errors")

    print("\nALL CHECKS PASSED. Safe to move on to real training with train_yolo.py.")


if __name__ == "__main__":
    main()
