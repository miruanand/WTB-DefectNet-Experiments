@echo off
setlocal

set DATA_ROOT=C:\Users\Student\Desktop\23BLC1224_FYP-1_Karthik_Sir\WTBs2025

REM ============================================================
REM PHASE 1: Try_7 -- resnet18 hybrid, frozen stem, mixup/cutmix + EMA on
REM ============================================================
python train.py --data_root "%DATA_ROOT%" --exp_name Try_7 --backbone resnet18 --num_workers 8
if errorlevel 1 goto :error

python evaluate.py --data_root "%DATA_ROOT%" --checkpoint runs\Try_7\checkpoints\best.pt --out_dir runs\Try_7 --tta
if errorlevel 1 goto :error

python gradcam.py --data_root "%DATA_ROOT%" --checkpoint runs\Try_7\checkpoints\best.pt --out_dir runs\Try_7
if errorlevel 1 goto :error

python evaluate.py --data_root "%DATA_ROOT%" --checkpoint runs\Try_7\checkpoints\best_ema.pt --out_dir runs\Try_7_ema --tta --use_ema
if errorlevel 1 goto :error

python gradcam.py --data_root "%DATA_ROOT%" --checkpoint runs\Try_7\checkpoints\best_ema.pt --out_dir runs\Try_7_ema
if errorlevel 1 goto :error

REM ============================================================
REM PHASE 2: Try_7_phase2 -- unfreeze the pretrained stem, fine-tune at a lower LR
REM ============================================================
python train.py --data_root "%DATA_ROOT%" --exp_name Try_7_phase2 --backbone resnet18 --unfreeze_stem --init_from runs\Try_7\checkpoints\best.pt --base_lr 3e-5 --epochs 60 --num_workers 8
if errorlevel 1 goto :error

python evaluate.py --data_root "%DATA_ROOT%" --checkpoint runs\Try_7_phase2\checkpoints\best.pt --out_dir runs\Try_7_phase2 --tta
if errorlevel 1 goto :error

python gradcam.py --data_root "%DATA_ROOT%" --checkpoint runs\Try_7_phase2\checkpoints\best.pt --out_dir runs\Try_7_phase2
if errorlevel 1 goto :error

python evaluate.py --data_root "%DATA_ROOT%" --checkpoint runs\Try_7_phase2\checkpoints\best_ema.pt --out_dir runs\Try_7_phase2_ema --tta --use_ema
if errorlevel 1 goto :error

python gradcam.py --data_root "%DATA_ROOT%" --checkpoint runs\Try_7_phase2\checkpoints\best_ema.pt --out_dir runs\Try_7_phase2_ema
if errorlevel 1 goto :error

REM ============================================================
REM Push everything
REM ============================================================
git add -A
git commit -m "Try_7 (resnet18 + mixup/cutmix/EMA) + Try_7_phase2 (unfrozen stem fine-tune): eval + Grad-CAM for raw and EMA checkpoints"
git push origin
goto :end

:error
echo.
echo [run_all] A step failed -- stopping here so later steps don't run on a broken/missing checkpoint.
echo [run_all] Fix the issue, then re-run individual remaining lines by hand ^(no need to redo finished steps^).

:end
endlocal
