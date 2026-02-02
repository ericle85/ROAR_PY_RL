@echo off
REM Launch all components for distributed training
REM
REM This script starts:
REM   1. Redis server
REM   2. Learner (Python 3.11 + CUDA)
REM   3. Multiple workers (Python 3.8)
REM
REM Usage: launch_all.bat [num_workers]
REM Example: launch_all.bat 2

setlocal enabledelayedexpansion

REM ============================================
REM CONFIGURATION
REM ============================================

set NUM_WORKERS=%1
if "%NUM_WORKERS%"=="" set NUM_WORKERS=2

set WORKER_START_DELAY=30

REM ============================================
REM Launch Components
REM ============================================

echo ============================================
echo Distributed PPO Training Launcher
echo ============================================
echo Number of workers: %NUM_WORKERS%
echo ============================================
echo.

REM Change to scripts directory
cd /d %~dp0

echo Step 1: Checking Redis connection (running in WSL)...
REM Redis is expected to be running in WSL
REM To start Redis in WSL: wsl redis-server --requirepass your_password
echo   Note: Make sure Redis is running in WSL before continuing.
echo   Start with: wsl redis-server --requirepass your_password
timeout /t 3 /nobreak > nul

echo Step 2: Starting learner...
start "Learner" cmd /k launch_learner.bat %NUM_WORKERS%
timeout /t 5 /nobreak > nul

echo Step 3: Starting workers...
for /L %%i in (0,1,%NUM_WORKERS%) do (
    if %%i lss %NUM_WORKERS% (
        echo   Starting worker %%i...
        start "Worker %%i" cmd /k launch_worker.bat %%i

        REM Wait between workers to stagger CARLA startup
        if %%i lss %NUM_WORKERS% (
            echo   Waiting %WORKER_START_DELAY%s before next worker...
            timeout /t %WORKER_START_DELAY% /nobreak > nul
        )
    )
)

echo.
echo ============================================
echo All components launched!
echo ============================================
echo.
echo Windows opened:
echo   - Learner
for /L %%i in (0,1,%NUM_WORKERS%) do (
    if %%i lss %NUM_WORKERS% (
        echo   - Worker %%i
    )
)
echo.
echo To stop: Close each window or press Ctrl+C in each
echo.
pause
