$ErrorActionPreference = "Stop"
$DATA_ROOT = "C:\Users\Student\Desktop\23BLC1224_FYP-1_Karthik_Sir\WTBs2025"

# ============================================================
# Try_9_phase2: unfreeze the ResNet18 stem and fine-tune the
# WHOLE model end-to-end, starting from Try_9's best weights.
#
# Try_9 (your current best single model, 77.67% test acc) trained
# for 120 epochs with the pretrained stem completely FROZEN --
# only the custom TSDB/ASA/DRFB/WGFR/MSCA/LTCP blocks learned.
# This phase never ran on your best checkpoint before (you only
# ever did it on the weaker Try_7 -> Try_7_phase2, which alone
# added +0.9% accuracy). Unfreezing lets the stem's low/mid-level
# filters specialize to blade textures instead of staying generic
# ImageNet features.
#
# base_lr=3e-5 is 10x lower than phase-1's 3e-4 -- necessary so
# fine-tuning doesn't wreck the already-good pretrained weights in
# the first few epochs. epochs=40 with patience=35 (config.py
# default) is enough runway without wasting a full 120-epoch budget
# on a fine-tuning pass that should converge much faster than
# training from scratch did.
# ============================================================

Write-Host "=== Training Try_9_phase2 (unfreeze + fine-tune) ===" -ForegroundColor Cyan
python train.py --data_root "$DATA_ROOT" --exp_name Try_9_phase2 --backbone resnet18 `
    --unfreeze_stem --init_from runs\Try_9\checkpoints\best.pt `
    --base_lr 3e-5 --epochs 40
if ($LASTEXITCODE -ne 0) { Write-Host "Training failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Evaluating Try_9_phase2 ===" -ForegroundColor Cyan
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs\Try_9_phase2\checkpoints\best.pt --out_dir runs\Try_9_phase2 --tta
if ($LASTEXITCODE -ne 0) { Write-Host "Evaluate failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Grad-CAM Try_9_phase2 ===" -ForegroundColor Cyan
python gradcam.py --data_root "$DATA_ROOT" --checkpoint runs\Try_9_phase2\checkpoints\best.pt --out_dir runs\Try_9_phase2
if ($LASTEXITCODE -ne 0) { Write-Host "Grad-CAM failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Pushing to GitHub ===" -ForegroundColor Cyan
git add -A
git commit -m "Try_9_phase2: unfreeze resnet18 stem, fine-tune from Try_9 best.pt, base_lr=3e-5"
git push origin

Write-Host "=== ALL DONE: Try_9_phase2 trained, evaluated, and pushed ===" -ForegroundColor Green
Write-Host "Compare runs\Try_9\eval\test_metrics.json vs runs\Try_9_phase2\eval\test_metrics.json before deciding which to keep." -ForegroundColor Yellow
