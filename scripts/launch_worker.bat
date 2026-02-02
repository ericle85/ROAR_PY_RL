@echo off
REM Launch script for distributed PPO worker (Python 3.8)
REM
REM Usage: launch_worker.bat <worker_id>
REM Example: launch_worker.bat 0
REM          launch_worker.bat 1

setlocal enabledelayedexpansion

REM ============================================
REM CONFIGURATION - Edit these paths
REM ============================================

REM Path to Python 3.8 environment (for CARLA 0.9.12)
REM Option 1: Conda environment
set CONDA_ENV_NAME=roar_competition

REM Option 2: Virtual environment (uncomment if using venv instead)
REM set VENV_PATH=C:\path\to\py38_venv

REM Path to CARLA executable
set CARLA_EXE=C:\Users\shrek\Downloads\Monza_V1.1\Monza\CarlaUE4.exe

REM Path to racing line file
set RACING_LINE=C:\Users\shrek\ROAR_PY_RL\racingline\main.npz

REM Redis host (change if running on different machine)
REM For WSL: use localhost if Redis is bound to 0.0.0.0
set REDIS_HOST=localhost
set REDIS_PORT=6379
set REDIS_PASSWORD=your_redis_password_here

REM Steps per rollout
set N_STEPS=2048

REM ============================================
REM Parse arguments
REM ============================================

set WORKER_ID=%1
if "%WORKER_ID%"=="" (
    echo Error: Worker ID required
    echo Usage: launch_worker.bat ^<worker_id^>
    echo Example: launch_worker.bat 0
    pause
    exit /b 1
)

REM ============================================
REM Validate paths
REM ============================================

if not exist "%CARLA_EXE%" (
    echo Error: CARLA executable not found: %CARLA_EXE%
    echo Please edit this script and set CARLA_EXE to the correct path.
    pause
    exit /b 1
)

if not exist "%RACING_LINE%" (
    echo Error: Racing line file not found: %RACING_LINE%
    echo Please edit this script and set RACING_LINE to the correct path.
    pause
    exit /b 1
)

REM ============================================
REM Activate environment and run
REM ============================================

echo ============================================
echo Distributed PPO Worker %WORKER_ID%
echo ============================================
echo CARLA: %CARLA_EXE%
echo Racing line: %RACING_LINE%
echo Redis: %REDIS_HOST%:%REDIS_PORT%
echo Steps per rollout: %N_STEPS%
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

REM Run worker
python distributed/worker.py ^
    --worker-id %WORKER_ID% ^
    --carla-exe "%CARLA_EXE%" ^
    --racing-line "%RACING_LINE%" ^
    --redis-host %REDIS_HOST% ^
    --redis-port %REDIS_PORT% ^
    --redis-password %REDIS_PASSWORD% ^
    --n-steps %N_STEPS% ^
    --log-level INFO

pause
