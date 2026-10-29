@echo off
REM ============================================================
REM  Hemorrhage anomaly map - one-click runner (Windows)
REM  1. Edit the paths below
REM  2. Double-click this file
REM  3. When done, push the shareable results (see message at the end)
REM ============================================================
setlocal

REM ----- Edit these -------------------------------------------
REM Folder of hemorrhage images (test only, NOT used in training)
set "HEMO_DIR=C:\path\to\hemorrhage_images"
REM Folder of normal images that were NOT in the training set
set "NORMAL_DIR=C:\path\to\normal_test_images"
REM Optional: folder of hemorrhage masks (same file name as image, white = bleeding). Leave empty if none.
set "MASK_DIR="
REM LoRA weights (default: the file in this repo)
set "LORA=%~dp0pytorch_lora_weights.safetensors"
set "STRENGTHS=0.2 0.3 0.4 0.5 0.6"
set "N_SEEDS=5"
REM ------------------------------------------------------------

cd /d "%~dp0"

if not exist "%HEMO_DIR%\" goto :no_hemo
if not exist "%NORMAL_DIR%\" goto :no_normal
if not exist "%LORA%" goto :no_lora

set "MASK_ARG="
if defined MASK_DIR set MASK_ARG=--mask_dir "%MASK_DIR%"

for %%S in (%STRENGTHS%) do (
    echo.
    echo ===== strength %%S : hemorrhage =====
    python anomaly_map.py --input_dir "%HEMO_DIR%" --output_dir "results\s%%S\hemorrhage" --strength %%S --n_seeds %N_SEEDS% --lora "%LORA%" %MASK_ARG% || goto :fail
    echo ===== strength %%S : normal =====
    python anomaly_map.py --input_dir "%NORMAL_DIR%" --output_dir "results\s%%S\normal" --strength %%S --n_seeds %N_SEEDS% --lora "%LORA%" || goto :fail
)

echo.
echo ===== Summary =====
python evaluate_anomaly.py --results_dir results || goto :fail

echo.
echo ============================================================
echo  Done. Shareable results (heatmaps, scores, summary) can be pushed with:
echo.
echo    git add results
echo    git status            ^<-- check: NO *_panel.png / *.npy / id_map.csv listed
echo    git commit -m "Add anomaly map results"
echo    git push
echo.
echo  *_panel.png contain the original images and stay on this computer only.
echo ============================================================
pause
exit /b 0

:no_hemo
echo [ERROR] HEMO_DIR not found: "%HEMO_DIR%"
goto :fail
:no_normal
echo [ERROR] NORMAL_DIR not found: "%NORMAL_DIR%"
goto :fail
:no_lora
echo [ERROR] LoRA file not found: "%LORA%"
goto :fail
:fail
echo.
echo Failed. See the error message above.
pause
exit /b 1
