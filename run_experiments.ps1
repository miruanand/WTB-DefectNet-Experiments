# run_experiments.ps1
# ====================
# Runs both training commands back-to-back, generates the comparison
# report, and pushes everything (code + COMPARISON.md, NOT the large
# .pt checkpoint files) to your GitHub repo.
#
# Usage (from the repo root, with your venv already activated -- same
# prompt you've been running `python train_yolo.py ...` from):
#
#     .\run_experiments.ps1
#
# If PowerShell blocks the script from running (common default on
# Windows), run this ONCE first in the same terminal:
#
#     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#
# That only affects the current terminal session, not your whole system.

$ErrorActionPreference = "Stop"   # stop the whole script if any command below fails
$DataYaml = "D:\23BLC1224_Mirunalini_FYP1_Karthik_Sir\WTBs2025_yolo\data.yaml"
$Epochs = 300
$Batch = 16
$Device = 0

function Run-Step($Name, $ScriptBlock) {
    Write-Host "`n========== $Name ==========" -ForegroundColor Cyan
    $start = Get-Date
    & $ScriptBlock
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit code $LASTEXITCODE) -- stopping." -ForegroundColor Red
        exit 1
    }
    $elapsed = (Get-Date) - $start
    Write-Host "Done: $Name  (took $($elapsed.ToString('hh\:mm\:ss')))" -ForegroundColor Green
}

# 1. Train WTB-DefectNet-backbone YOLO
Run-Step "Training WTB-DefectNet + YOLO" {
    python train_yolo.py --data $DataYaml --epochs $Epochs --batch $Batch --device $Device
}

# 2. Train stock YOLOv11n baseline
Run-Step "Training YOLOv11n baseline" {
    python train_yolo.py --data $DataYaml --epochs $Epochs --batch $Batch --device $Device --baseline
}

# 3. Build the comparison report from both runs' results_summary.json
Run-Step "Generating comparison report" {
    python generate_comparison.py
}

# 4. Commit and push -- code + results + comparison report only.
#    Checkpoint (.pt) files are deliberately excluded (see .gitignore
#    note below) since GitHub isn't a good place for multi-hundred-MB
#    binaries and they're not needed to reproduce/verify the numbers.
Run-Step "Pushing to GitHub" {
    # Make sure large checkpoint files never get staged, even if no
    # .gitignore exists yet in this repo.
    if (-not (Test-Path ".gitignore") -or -not (Select-String -Path ".gitignore" -Pattern "\.pt$" -Quiet -ErrorAction SilentlyContinue)) {
        Add-Content -Path ".gitignore" -Value "`n# large checkpoint files -- not tracked`nruns/**/*.pt`nruns/**/weights/`n"
    }

    git add wtb_yolo_modules.py yolo11-wtbdefectnet.yaml train_yolo.py test_backbone.py `
        prepare_yolo_dataset.py generate_comparison.py run_experiments.ps1 .gitignore COMPARISON.md `
        runs/detect/wtbdefectnet_yolo/results_summary.json `
        runs/detect/yolo11n_baseline/results_summary.json `
        runs/detect/wtbdefectnet_yolo/results.png `
        runs/detect/yolo11n_baseline/results.png `
        2>$null

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
    git commit -m "Add YOLO11 + WTB-DefectNet backbone: training results ($timestamp)"
    git push
}

Write-Host "`nAll steps completed successfully." -ForegroundColor Green
Write-Host "See COMPARISON.md in the repo root for the results table."
