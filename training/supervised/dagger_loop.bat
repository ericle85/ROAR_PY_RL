@echo off
setlocal enabledelayedexpansion

REM === Configuration ===
set NUM_ITERATIONS=5
set EVAL_ENV=roar_competition
set TRAIN_ENV=learner
set CHECKPOINT_PATH=training\supervised\checkpoints\best_model.pt
set DATA_DIR=training\expert_data

REM === Main Loop ===
for /L %%i in (1,1,%NUM_ITERATIONS%) do (
    echo.
    echo ========================================
    echo DAgger Iteration %%i of %NUM_ITERATIONS%
    echo ========================================

    REM Find next available dagger episode number
    call :get_next_ep
    echo Next episode number: !NEXT_EP!

    REM Run evaluation with DAgger collection
    echo Running evaluation in %EVAL_ENV%...
    call conda run -n %EVAL_ENV% python training/eval_pretrained.py ^
        --checkpoint %CHECKPOINT_PATH% ^
        --episodes 1 ^
        --dagger ^
        --dagger-output-dir %DATA_DIR%

    REM Rename dagger_ep0.npz to next episode number
    if exist "%DATA_DIR%\dagger_ep0.npz" (
        echo Renaming dagger_ep0.npz to dagger_ep!NEXT_EP!.npz
        move "%DATA_DIR%\dagger_ep0.npz" "%DATA_DIR%\dagger_ep!NEXT_EP!.npz"
    ) else (
        echo WARNING: dagger_ep0.npz not found!
    )

    REM Run training on all accumulated data
    echo Running training in %TRAIN_ENV%...
    call conda run -n %TRAIN_ENV% python -m training.supervised.train ^
        --data-dir %DATA_DIR% ^
        --output-dir training/supervised/checkpoints
)

echo.
echo ========================================
echo DAgger loop complete!
echo ========================================
goto :eof

REM === Helper: Find next episode number ===
:get_next_ep
set NEXT_EP=0
for %%f in ("%DATA_DIR%\dagger_ep*.npz") do (
    set "fname=%%~nf"
    set "num=!fname:dagger_ep=!"
    if !num! GEQ !NEXT_EP! set /a NEXT_EP=!num!+1
)
goto :eof
