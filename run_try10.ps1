$ErrorActionPreference = "Stop"
$DATA_ROOT = "C:\Users\Student\Desktop\23BLC1224_FYP-1_Karthik_Sir\WTBs2025"

# ============================================================
# Try_10: ResNet34 backbone (more capacity than Try_9's ResNet18)
# at img_size=448 (vs 384 for every previous run).
#
# Rationale for each change:
#  - resnet34: roughly 2x the layer1/layer2 depth of resnet18, still
#    small enough to fine-tune later without overfitting ~7.5k images
#    when frozen for phase 1. Never tried in this project before --
#    every prior run (Try_7/8/9) used resnet18.
#  - img_size=448: your weakest/most confusable classes (localized
#    damage, coating detachment, pinholes, hairline paint cracks) are
#    small or textural. Downsampling to 384 already discards detail;
#    448 keeps more of it. Costs ~35% more VRAM/epoch time -- your
#    20GB RTX 4000 Ada has headroom for this at batch_size=16.
#
# Phase 1 here keeps the stem FROZEN (Config default), same as Try_9,
# so this is an apples-to-apples test of "does resnet34 @ 448px beat
# resnet18 @ 384px" before you also try unfreezing it in phase 2.
# ============================================================

Write-Host "=== Training Try_10 (resnet34, img_size=448) ===" -ForegroundColor Cyan
python train.py --data_root "$DATA_ROOT" --exp_name Try_10 --backbone resnet34 --img_size 448 --epochs 120
if ($LASTEXITCODE -ne 0) { Write-Host "Training failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Evaluating Try_10 ===" -ForegroundColor Cyan
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs\Try_10\checkpoints\best.pt --out_dir runs\Try_10 --tta
if ($LASTEXITCODE -ne 0) { Write-Host "Evaluate failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Grad-CAM Try_10 ===" -ForegroundColor Cyan
python gradcam.py --data_root "$DATA_ROOT" --checkpoint runs\Try_10\checkpoints\best.pt --out_dir runs\Try_10
if ($LASTEXITCODE -ne 0) { Write-Host "Grad-CAM failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Pushing to GitHub ===" -ForegroundColor Cyan
git add -A
git commit -m "Try_10: img_size=448, resnet34, 120 epochs"
git push origin

Write-Host "=== ALL DONE: Try_10 trained, evaluated, and pushed ===" -ForegroundColor Green
Write-Host "Compare against Try_9 (77.67%) before deciding whether to phase-2 fine-tune this one too." -ForegroundColor Yellow
