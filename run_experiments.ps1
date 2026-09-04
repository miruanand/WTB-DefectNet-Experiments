# run_experiments.ps1
# ====================
# Runs all three training commands back-to-back (P2 architecture, no-P2
# architecture, stock YOLOv11n baseline), generates the 3-way comparison
# report, and pushes everything to git automatically -- no manual steps
# in between.
#
# Usage (from the repo root, with your venv already activated):
#
#     .\run_experiments.ps1
#
# If PowerShell blocks the script from running, run this ONCE first in
# the same terminal:
#
#     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = "Stop"   # stop the whole script if any command below fails
$DataYaml = "D:\23BLC1224_Mirunalini_FYP1_Karthik_Sir\WTBs2025_yolo\data.yaml"
$Epochs = 120
$Batch = 8
$Device = 0
$GitBranch = "main"   # <-- CHECK THIS matches your actual default branch name (case-sensitive:
                      #     GitHub's default is lowercase "main" -- if your repo genuinely uses
                      #     "MAIN" in caps, change this to match exactly, or `git push` will fail
                      #     with "branch not found" at the very last step, after all the training.

function Run-Step($Name, $ScriptBlock) {
    Write-Host "`n========== $Name ==========" -ForegroundColor Cyan
    $start = Get-Date
    & $ScriptBlock
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit code $LASTEXITCODE) -- stopping. Nothing after this step ran, including git push." -ForegroundColor Red
        exit 1
    }
    $elapsed = (Get-Date) - $start
    Write-Host "Done: $Name  (took $($elapsed.ToString('hh\:mm\:ss')))" -ForegroundColor Green
}

# 1. Train WTB-DefectNet with the P2 head (default architecture)
Run-Step "Training WTB-DefectNet (P2)" {
    python train_yolo.py --data $DataYaml --epochs $Epochs --batch $Batch --device $Device --arch p2
}

# 2. Train WTB-DefectNet WITHOUT the P2 head (ablation)
Run-Step "Training WTB-DefectNet (no P2)" {
    python train_yolo.py --data $DataYaml --epochs $Epochs --batch $Batch --device $Device --arch no_p2
}

# 3. Train stock YOLOv11n baseline
Run-Step "Training YOLOv11n baseline" {
    python train_yolo.py --data $DataYaml --epochs $Epochs --batch $Batch --device $Device --baseline
}

# 4. Build the 3-way comparison report
Run-Step "Generating comparison report" {
    python generate_comparison.py
}

# 5. Commit and push -- code + results + comparison report only.
#    Checkpoint (.pt) files are deliberately excluded: GitHub isn't a
#    good place for multi-hundred-MB binaries, and they're not needed
#    to verify the reported numbers.
Run-Step "Pushing to git" {
    if (-not (Test-Path ".gitignore") -or -not (Select-String -Path ".gitignore" -Pattern "\.pt$" -Quiet -ErrorAction SilentlyContinue)) {
        Add-Content -Path ".gitignore" -Value "`n# large checkpoint files -- not tracked`nruns/**/*.pt`nruns/**/weights/`n"
    }

    git add .
    git commit -m "P2 vs No-P2 vs YOLO11n baseline comparison"
    git push origin $GitBranch
}

Write-Host "`nAll steps completed successfully." -ForegroundColor Green
Write-Host "See COMPARISON.md in the repo root for the 3-way results table."
