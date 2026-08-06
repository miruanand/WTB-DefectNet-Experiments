$ErrorActionPreference = "Stop"
$DATA_ROOT = "C:\Users\Student\Desktop\23BLC1224_FYP-1_Karthik_Sir\WTBs2025"

Write-Host "=== Training Try_9 ===" -ForegroundColor Cyan
python train.py --data_root "$DATA_ROOT" --exp_name Try_9 --backbone resnet18 --epochs 120
if ($LASTEXITCODE -ne 0) { Write-Host "Training failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Evaluating Try_9 ===" -ForegroundColor Cyan
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs\Try_9\checkpoints\best.pt --out_dir runs\Try_9 --tta
if ($LASTEXITCODE -ne 0) { Write-Host "Evaluate failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Grad-CAM Try_9 ===" -ForegroundColor Cyan
python gradcam.py --data_root "$DATA_ROOT" --checkpoint runs\Try_9\checkpoints\best.pt --out_dir runs\Try_9
if ($LASTEXITCODE -ne 0) { Write-Host "Grad-CAM failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Pushing to GitHub ===" -ForegroundColor Cyan
git add -A
git commit -m "Try_9: img_size=384, resnet18, 120 epochs"
git push origin

Write-Host "=== ALL DONE: Try_9 trained, evaluated, and pushed ===" -ForegroundColor Green
