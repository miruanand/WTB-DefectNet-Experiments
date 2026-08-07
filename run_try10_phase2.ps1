$ErrorActionPreference = "Stop"
$DATA_ROOT = "C:\Users\Student\Desktop\23BLC1224_FYP-1_Karthik_Sir\WTBs2025"

# ============================================================
# Try_10_phase2: unfreeze the ResNet34 stem, fine-tune from
# Try_10's best weights.
#
# ONLY run this after checking that Try_10 (frozen stem, phase 1)
# actually beat Try_9 -- if resnet34 @ 448px didn't help in phase 1,
# fine-tuning it further is unlikely to close that gap and just
# costs you GPU time. If Try_10 < Try_9, skip straight to fine-
# tuning Try_9 instead (run_try9_phase2.ps1) and treat Try_10 as a
# dead end worth reporting as a negative result (still useful for
# your report/ablation table).
# ============================================================

Write-Host "=== Training Try_10_phase2 (unfreeze + fine-tune resnet34) ===" -ForegroundColor Cyan
python train.py --data_root "$DATA_ROOT" --exp_name Try_10_phase2 --backbone resnet34 --img_size 448 `
    --unfreeze_stem --init_from runs\Try_10\checkpoints\best.pt `
    --base_lr 3e-5 --epochs 40
if ($LASTEXITCODE -ne 0) { Write-Host "Training failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Evaluating Try_10_phase2 ===" -ForegroundColor Cyan
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs\Try_10_phase2\checkpoints\best.pt --out_dir runs\Try_10_phase2 --tta
if ($LASTEXITCODE -ne 0) { Write-Host "Evaluate failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Grad-CAM Try_10_phase2 ===" -ForegroundColor Cyan
python gradcam.py --data_root "$DATA_ROOT" --checkpoint runs\Try_10_phase2\checkpoints\best.pt --out_dir runs\Try_10_phase2
if ($LASTEXITCODE -ne 0) { Write-Host "Grad-CAM failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Pushing to GitHub ===" -ForegroundColor Cyan
git add -A
git commit -m "Try_10_phase2: unfreeze resnet34 stem, fine-tune from Try_10 best.pt, base_lr=3e-5"
git push origin

Write-Host "=== ALL DONE: Try_10_phase2 trained, evaluated, and pushed ===" -ForegroundColor Green
