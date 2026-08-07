$ErrorActionPreference = "Stop"
$DATA_ROOT = "C:\Users\Student\Desktop\23BLC1224_FYP-1_Karthik_Sir\WTBs2025"

Write-Host "=== [1/3] Cascade weight sweep: Try_9 + Try_9_cascade (cascade_weight=0.5) ===" -ForegroundColor Cyan
python cascade_infer.py --data_root "$DATA_ROOT" --main_checkpoint runs\Try_9\checkpoints\best.pt --cascade_checkpoint runs\Try_9_cascade\checkpoints\best.pt --out_dir runs\Try_9_cascade_v2 --tta --cascade_tta --cascade_weight 0.5
if ($LASTEXITCODE -ne 0) { Write-Host "Step 1 (cascade_infer) failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== [2/3] Ensembling Try_8 + Try_9 ===" -ForegroundColor Cyan
python ensemble_infer.py --data_root "$DATA_ROOT" --checkpoints runs\Try_8\checkpoints\best.pt runs\Try_9\checkpoints\best.pt --out_dir runs\ensemble_8_9 --tta
if ($LASTEXITCODE -ne 0) { Write-Host "Step 2 (ensemble_infer) failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== [3/3] Ensembling Try_8 + Try_9, then chaining the cascade specialist ===" -ForegroundColor Cyan
python ensemble_infer.py --data_root "$DATA_ROOT" --checkpoints runs\Try_8\checkpoints\best.pt runs\Try_9\checkpoints\best.pt --cascade_checkpoint runs\Try_9_cascade\checkpoints\best.pt --out_dir runs\ensemble_8_9_cascade --tta --cascade_tta --cascade_weight 0.5
if ($LASTEXITCODE -ne 0) { Write-Host "Step 3 (ensemble_infer + cascade) failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Comparing results ===" -ForegroundColor Cyan
Write-Host "Try_9 + cascade (v2, blend):" -ForegroundColor Yellow
python -c "import json; d=json.load(open('runs/Try_9_cascade_v2/eval/test_metrics.json')); print(f\"  accuracy={d['metrics_after_cascade']['accuracy']:.4f}  macro_f1={d['metrics_after_cascade']['macro_f1']:.4f}\")"
Write-Host "Ensemble Try_8 + Try_9 (no cascade):" -ForegroundColor Yellow
python -c "import json; d=json.load(open('runs/ensemble_8_9/eval/test_metrics.json')); print(f\"  accuracy={d['metrics_ensemble_only']['accuracy']:.4f}  macro_f1={d['metrics_ensemble_only']['macro_f1']:.4f}\")"
Write-Host "Ensemble Try_8 + Try_9 + cascade:" -ForegroundColor Yellow
python -c "import json; d=json.load(open('runs/ensemble_8_9_cascade/eval/test_metrics.json')); print(f\"  accuracy={d['metrics_final']['accuracy']:.4f}  macro_f1={d['metrics_final']['macro_f1']:.4f}\")"

Write-Host "=== Pushing to GitHub (https://github.com/miruanand/WTB-DefectNet-Experiments) ===" -ForegroundColor Cyan
git add -A
git commit -m "Cascade weight-blend sweep + Try_8/Try_9 ensemble runs (with and without cascade)"
git push origin

Write-Host "=== ALL DONE: 3 runs finished, compared above, and pushed to GitHub ===" -ForegroundColor Green
Write-Host "Check runs\Try_9_cascade_v2, runs\ensemble_8_9, and runs\ensemble_8_9_cascade for full eval artifacts." -ForegroundColor Green
