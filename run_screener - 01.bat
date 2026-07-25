@echo off
title SCREENER ML SERVER

echo ======================================
echo Killing processes using port 8001...
echo ======================================


for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001') do (
    echo Killing PID %%a
    taskkill /F /PID %%a >nul 2>&1
)

timeout /t 2 /nobreak >nul

echo.
echo ======================================
echo Starting ML SCREENER...
echo ======================================


D:
cd /d "E:\application\Office-Scripts\SCRIPTS\SCREENER-ML-V1-TEST-01"

python ml_service/ml_service.py

pause