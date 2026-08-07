$ErrorActionPreference = "Stop"
$DATA_ROOT = "C:\Users\Student\Desktop\23BLC1224_FYP-1_Karthik_Sir\WTBs2025"

function Push-Stage($message) {
    Write-Host "=== Pushing to GitHub: $message ===" -ForegroundColor Cyan
    git add -A
    git commit -m "$message" --allow-empty-message -q
    git push origin
}

# ============================================================
# STAGE 1: Try_9_phase2 -- unfreeze Try_9's stem, fine-tune end-to-end
# ============================================================
Write-Host "`n########## STAGE 1: Try_9_phase2 ##########" -ForegroundColor Magenta

Write-Host "=== Training Try_9_phase2 ===" -ForegroundColor Cyan
python train.py --data_root "$DATA_ROOT" --exp_name Try_9_phase2 --backbone resnet18 `
    --unfreeze_stem --init_from runs\Try_9\checkpoints\best.pt --base_lr 3e-5 --epochs 40
if ($LASTEXITCODE -ne 0) { Write-Host "Try_9_phase2 training failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Evaluating Try_9_phase2 ===" -ForegroundColor Cyan
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs\Try_9_phase2\checkpoints\best.pt --out_dir runs\Try_9_phase2 --tta
if ($LASTEXITCODE -ne 0) { Write-Host "Try_9_phase2 evaluate failed, stopping." -ForegroundColor Red; exit 1 }

python gradcam.py --data_root "$DATA_ROOT" --checkpoint runs\Try_9_phase2\checkpoints\best.pt --out_dir runs\Try_9_phase2
if ($LASTEXITCODE -ne 0) { Write-Host "Try_9_phase2 gradcam failed, stopping." -ForegroundColor Red; exit 1 }

Push-Stage "Try_9_phase2: unfreeze resnet18 stem, fine-tune from Try_9 best.pt"

# ============================================================
# STAGE 2: Try_10 -- resnet34 backbone, img_size=448, frozen stem
# ============================================================
Write-Host "`n########## STAGE 2: Try_10 ##########" -ForegroundColor Magenta

Write-Host "=== Training Try_10 ===" -ForegroundColor Cyan
python train.py --data_root "$DATA_ROOT" --exp_name Try_10 --backbone resnet34 --img_size 448 --epochs 120
if ($LASTEXITCODE -ne 0) { Write-Host "Try_10 training failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Evaluating Try_10 ===" -ForegroundColor Cyan
python evaluate.py --data_root "$DATA_ROOT" --checkpoint runs\Try_10\checkpoints\best.pt --out_dir runs\Try_10 --tta
if ($LASTEXITCODE -ne 0) { Write-Host "Try_10 evaluate failed, stopping." -ForegroundColor Red; exit 1 }

python gradcam.py --data_root "$DATA_ROOT" --checkpoint runs\Try_10\checkpoints\best.pt --out_dir runs\Try_10
if ($LASTEXITCODE -ne 0) { Write-Host "Try_10 gradcam failed, stopping." -ForegroundColor Red; exit 1 }

Push-Stage "Try_10: resnet34, img_size=448, 120 epochs"

# ============================================================
# STAGE 3: auto-pick the winner (Try_9 vs Try_9_phase2 vs Try_10)
# ============================================================
Write-Host "`n########## STAGE 3: Picking winner ##########" -ForegroundColor Magenta
python pick_winner.py
if ($LASTEXITCODE -ne 0) { Write-Host "pick_winner.py failed, stopping." -ForegroundColor Red; exit 1 }

$winner = Get-Content runs\winner.json | ConvertFrom-Json
$WINNER_EXP = $winner.exp
$WINNER_BACKBONE = $winner.backbone
$WINNER_IMG_SIZE = $winner.img_size
Write-Host "Winner: $WINNER_EXP (backbone=$WINNER_BACKBONE, img_size=$WINNER_IMG_SIZE, accuracy=$($winner.accuracy))" -ForegroundColor Yellow

$WINNER_CKPT = "runs\$WINNER_EXP\checkpoints\best.pt"

# ============================================================
# STAGE 4: train a new cascade specialist on top of the winner
# ============================================================
Write-Host "`n########## STAGE 4: Cascade specialist on $WINNER_EXP ##########" -ForegroundColor Magenta

python train_cascade.py --data_root "$DATA_ROOT" --exp_name "${WINNER_EXP}_cascade" `
    --backbone $WINNER_BACKBONE --img_size $WINNER_IMG_SIZE --epochs 60
if ($LASTEXITCODE -ne 0) { Write-Host "Cascade training failed, stopping." -ForegroundColor Red; exit 1 }

Write-Host "=== Cascade weight sweep (0.4 / 0.5 / 0.6) ===" -ForegroundColor Cyan
$sweepOutputs = @()
foreach ($w in @(0.4, 0.5, 0.6)) {
    $wtag = ($w -replace '\.', '')
    $outDir = "runs\${WINNER_EXP}_cascade_w$wtag"
    Write-Host "  -- cascade_weight=$w --" -ForegroundColor DarkCyan
    python cascade_infer.py --data_root "$DATA_ROOT" `
        --main_checkpoint $WINNER_CKPT `
        --cascade_checkpoint "runs\${WINNER_EXP}_cascade\checkpoints\best.pt" `
        --out_dir $outDir `
        --tta --cascade_tta --cascade_weight $w
    if ($LASTEXITCODE -ne 0) { Write-Host "cascade_infer failed at weight=$w, stopping." -ForegroundColor Red; exit 1 }
    $sweepOutputs += "$outDir\eval\test_metrics.json"
}

Push-Stage "Cascade specialist + weight sweep on top of $WINNER_EXP"

# ============================================================
# STAGE 5: final comparison across everything
# ============================================================
Write-Host "`n########## STAGE 5: Final comparison ##########" -ForegroundColor Magenta
$allResults = @(
    "runs\Try_9\eval\test_metrics.json",
    "runs\Try_9_phase2\eval\test_metrics.json",
    "runs\Try_10\eval\test_metrics.json"
) + $sweepOutputs

python compare_results.py $allResults

Write-Host "`n########## PIPELINE COMPLETE ##########" -ForegroundColor Green
Write-Host "Winner backbone was: $WINNER_EXP -- final cascade results are in runs\${WINNER_EXP}_cascade_w04/05/06\eval\" -ForegroundColor Green
Write-Host "Report whichever row compare_results.py marked BEST." -ForegroundColor Yellow
