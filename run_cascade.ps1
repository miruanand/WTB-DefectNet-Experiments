$ErrorActionPreference = "Stop"
$DATA_ROOT = "C:\Users\Student\Desktop\23BLC1224_FYP-1_Karthik_Sir\WTBs2025"

Write-Host "=== Training cascade specialist (Try_9_cascade) ===" -ForegroundColor Cyan
python train_cascade.py --data_root "$DATA_ROOT" --exp_name Try_9_cascade --backbone resnet18 --epochs 60
if ($LASTEXITCODE -ne 0) { Write-Host "Cascade training failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Running combined cascade inference ===" -ForegroundColor Cyan
python cascade_infer.py --data_root "$DATA_ROOT" --main_checkpoint runs\Try_9\checkpoints\best.pt --main_img_size 384 --cascade_checkpoint runs\Try_9_cascade\checkpoints\best.pt --cascade_img_size 384 --out_dir runs\Try_9_cascade_eval --tta
if ($LASTEXITCODE -ne 0) { Write-Host "Cascade inference failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Pushing to GitHub ===" -ForegroundColor Cyan
git add -A
git commit -m "Try_9_cascade: LD/CD binary specialist + combined cascade eval"
git push origin

Write-Host "=== ALL DONE: cascade trained, evaluated, and pushed ===" -ForegroundColor Green