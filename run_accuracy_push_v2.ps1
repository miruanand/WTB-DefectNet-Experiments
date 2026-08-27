$ErrorActionPreference = "Stop"
$DATA_ROOT = "C:\Users\Student\Desktop\23BLC1224_FYP-1_Karthik_Sir\WTBs2025"

# ============================================================
# EDIT THIS ONE LINE before running: point $WINNER at whichever
# checkpoint scored highest accuracy in evaluate.py across
# Try_9 (77.67%), Try_9_phase2, Try_10, Try_10_phase2.
# Check with:
#   python compare_results.py runs\Try_9\eval\test_metrics.json runs\Try_9_phase2\eval\test_metrics.json runs\Try_10\eval\test_metrics.json runs\Try_10_phase2\eval\test_metrics.json
# ============================================================
$WINNER_EXP = "Try_9_phase2"          # <-- change to whichever run actually won
$WINNER_BACKBONE = "resnet18"         # <-- "resnet18" for Try_9*, "resnet34" for Try_10*
$WINNER_IMG_SIZE = 384                # <-- 384 for Try_9*, 448 for Try_10*

$WINNER_CKPT = "runs\$WINNER_EXP\checkpoints\best.pt"

Write-Host "=== [1/2] Training a NEW cascade specialist on top of $WINNER_EXP ===" -ForegroundColor Cyan
python train_cascade.py --data_root "$DATA_ROOT" --exp_name "${WINNER_EXP}_cascade" `
    --backbone $WINNER_BACKBONE --img_size $WINNER_IMG_SIZE --epochs 60
if ($LASTEXITCODE -ne 0) { Write-Host "Cascade training failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== [2/2] Cascade weight sweep (0.4 / 0.5 / 0.6) ===" -ForegroundColor Cyan
# Sweeping only 3 values (not more) -- see cascade_infer.py's own TIP: sweeping
# the test set more than a handful of times starts re-fitting to it.
foreach ($w in @(0.4, 0.5, 0.6)) {
    Write-Host "  -- cascade_weight=$w --" -ForegroundColor DarkCyan
    python cascade_infer.py --data_root "$DATA_ROOT" `
        --main_checkpoint $WINNER_CKPT `
        --cascade_checkpoint "runs\${WINNER_EXP}_cascade\checkpoints\best.pt" `
        --out_dir "runs\${WINNER_EXP}_cascade_w$($w -replace '\.','')" `
        --tta --cascade_tta --cascade_weight $w
    if ($LASTEXITCODE -ne 0) { Write-Host "cascade_infer failed at weight=$w, stopping." -ForegroundColor Red; exit 1 }
}

Write-Host "=== Comparing everything ===" -ForegroundColor Cyan
python compare_results.py `
    "runs\Try_9\eval\test_metrics.json" `
    "runs\Try_9_phase2\eval\test_metrics.json" `
    "runs\Try_10\eval\test_metrics.json" `
    "runs\Try_10_phase2\eval\test_metrics.json" `
    "runs\${WINNER_EXP}_cascade_w04\eval\test_metrics.json" `
    "runs\${WINNER_EXP}_cascade_w05\eval\test_metrics.json" `
    "runs\${WINNER_EXP}_cascade_w06\eval\test_metrics.json"

Write-Host "=== Pushing to GitHub ===" -ForegroundColor Cyan
git add -A
git commit -m "Accuracy push v2: cascade specialist + weight sweep on top of $WINNER_EXP"
git push origin

Write-Host "=== ALL DONE ===" -ForegroundColor Green
Write-Host "Report whichever row compare_results.py marked BEST." -ForegroundColor Yellow
