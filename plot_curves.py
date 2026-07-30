r"""
plot_curves.py
===============
Reads <run_dir>/log.csv and produces epoch-wise curves into
<run_dir>/Curves/. Adapts to whichever columns exist, so it still works on
older log.csv files that predate accuracy logging.

    python plot_curves.py --run_dir ./runs/Try_2

(Previously this hardcoded a single D:\... path -- that's gone so the same
script works for every experiment folder without editing the file.)
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True,
                     help="Experiment folder containing log.csv, e.g. ./runs/Try_2")
    return ap.parse_args()


def main():
    args = parse_args()
    csv_path = os.path.join(args.run_dir, "log.csv")
    save_path = os.path.join(args.run_dir, "Curves")
    os.makedirs(save_path, exist_ok=True)

    df = pd.read_csv(csv_path)
    has_acc = "train_acc" in df.columns and "val_acc" in df.columns

    plt.style.use("ggplot")

    def single(col, color, ylabel, title, fname):
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df[col], color=color, linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, fname), dpi=300)
        plt.close()

    # ---- Loss ----
    single("train_loss", "blue", "Loss", "Training Loss", "train_loss.png")
    single("val_loss", "red", "Loss", "Validation Loss", "val_loss.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Training Loss", linewidth=2)
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(save_path, "combined_loss_curve.png"), dpi=300)
    plt.close()

    # ---- Accuracy (new) ----
    if has_acc:
        single("train_acc", "blue", "Accuracy", "Training Accuracy", "train_accuracy.png")
        single("val_acc", "red", "Accuracy", "Validation Accuracy", "val_accuracy.png")

        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["train_acc"], label="Training Accuracy", linewidth=2)
        plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy", linewidth=2)
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training vs Validation Accuracy")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(save_path, "combined_accuracy_curve.png"), dpi=300)
        plt.close()
    else:
        print("[plot_curves] WARNING: no train_acc/val_acc columns found in log.csv "
              "(this run predates accuracy logging) -- skipping accuracy curves.")

    # ---- Val Macro F1 ----
    single("val_macro_f1", "green", "Macro F1", "Validation Macro F1", "val_macro_f1.png")

    # ---- Val Balanced Accuracy ----
    single("val_balanced_acc", "purple", "Balanced Accuracy",
           "Validation Balanced Accuracy", "val_balanced_accuracy.png")

    # ---- Combined validation metrics ----
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["val_macro_f1"], label="Macro F1", linewidth=2)
    plt.plot(df["epoch"], df["val_balanced_acc"], label="Balanced Accuracy", linewidth=2)
    if has_acc:
        plt.plot(df["epoch"], df["val_acc"], label="Accuracy", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("Validation Performance")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(save_path, "combined_validation_metrics.png"), dpi=300)
    plt.close()

    print("=" * 60)
    print("Plots generated successfully!")
    print("Saved in:", save_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
