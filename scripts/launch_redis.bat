@echo off
REM Launch Redis server for distributed training
REM
REM Redis for Windows: https://github.com/tporadowski/redis/releases
REM Download, extract, and set REDIS_PATH below

setlocal

REM ============================================
REM CONFIGURATION
REM ============================================

REM Path to Redis installation
set REDIS_PATH=C:\Redis

REM ============================================
REM Launch Redis
REM ============================================

if not exist "%REDIS_PATH%\redis-server.exe" (
    echo Error: Redis not found at %REDIS_PATH%
    echo.
    echo Please download Redis for Windows from:
    echo https://github.com/tporadowski/redis/releases
    echo.
    echo Extract and set REDIS_PATH in this script.
    pause
    exit /b 1
)

echo ============================================
echo Starting Redis Server
echo ============================================
echo Press Ctrl+C to stop
echo ============================================

"%REDIS_PATH%\redis-server.exe"

pause
