@echo off
REM Launch script for distributed PPO learner (Python 3.11 + CUDA)
REM
REM Usage: launch_learner.bat [num_workers] [total_timesteps]
REM Example: launch_learner.bat 2 10000000

setlocal enabledelayedexpansion

REM ============================================
REM CONFIGURATION - Edit these paths
REM ============================================

REM Path to Python 3.11 environment with CUDA
REM Option 1: Conda environment
set CONDA_ENV_NAME=py311_cuda

REM Option 2: Virtual environment (uncomment if using venv instead)
REM set VENV_PATH=C:\path\to\py311_venv

REM Redis host (change if running on different machine)
REM For WSL: use localhost if Redis is bound to 0.0.0.0
set REDIS_HOST=localhost
set REDIS_PORT=6379
set REDIS_PASSWORD=your_redis_password_here

REM Checkpoint directory
set CHECKPOINT_DIR=checkpoints

REM ============================================
REM Parse arguments
REM ============================================

set NUM_WORKERS=%1
if "%NUM_WORKERS%"=="" set NUM_WORKERS=1

set TOTAL_TIMESTEPS=%2
if "%TOTAL_TIMESTEPS%"=="" set TOTAL_TIMESTEPS=10000000

REM ============================================
REM Activate environment and run
REM ============================================

echo ============================================
echo Distributed PPO Learner
echo ============================================
echo Number of workers: %NUM_WORKERS%
echo Total timesteps: %TOTAL_TIMESTEPS%
echo Redis: %REDIS_HOST%:%REDIS_PORT%
echo ============================================

REM Change to project root directory
cd /d %~dp0..

REM Activate conda environment
call conda activate %CONDA_ENV_NAME%
if errorlevel 1 (
    echo Failed to activate conda environment: %CONDA_ENV_NAME%
    echo Please edit this script and set the correct environment name.
    pause
    exit /b 1
)

REM Uncomment below if using venv instead of conda
REM call %VENV_PATH%\Scripts\activate.bat

REM Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')"

REM Run learner
python distributed/learner.py ^
    --num-workers %NUM_WORKERS% ^
    --total-timesteps %TOTAL_TIMESTEPS% ^
    --redis-host %REDIS_HOST% ^
    --redis-port %REDIS_PORT% ^
    --redis-password %REDIS_PASSWORD% ^
    --checkpoint-dir %CHECKPOINT_DIR% ^
    --log-level INFO

pause
