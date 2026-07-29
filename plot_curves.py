import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"D:\23BLC1224_Mirunalini_FYP1_Karthik_Sir\wtb_defectnet\runs\exp1"

csv_path = os.path.join(BASE_DIR, "log.csv")
save_path = os.path.join(BASE_DIR, "Curves")

os.makedirs(save_path, exist_ok=True)

df = pd.read_csv(csv_path)

plt.style.use("ggplot")

# -----------------------------
# Training Loss
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["train_loss"], color="blue", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "train_loss.png"), dpi=300)
plt.close()

# -----------------------------
# Validation Loss
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["val_loss"], color="red", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "val_loss.png"), dpi=300)
plt.close()

# -----------------------------
# Combined Loss
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["train_loss"], label="Training Loss", linewidth=2)
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "combined_loss_curve.png"), dpi=300)
plt.close()

# -----------------------------
# Validation Macro F1
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["val_macro_f1"], color="green", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("Validation Macro F1")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "val_macro_f1.png"), dpi=300)
plt.close()

# -----------------------------
# Validation Balanced Accuracy
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["val_balanced_acc"], color="purple", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Balanced Accuracy")
plt.title("Validation Balanced Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "val_balanced_accuracy.png"), dpi=300)
plt.close()

# -----------------------------
# Combined Validation Metrics
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["val_macro_f1"], label="Macro F1", linewidth=2)
plt.plot(df["epoch"], df["val_balanced_acc"], label="Balanced Accuracy", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "combined_validation_metrics.png"), dpi=300)
plt.close()

print("=" * 60)
print("Plots generated successfully!")
print("Saved in:", save_path)
print("=" * 60)