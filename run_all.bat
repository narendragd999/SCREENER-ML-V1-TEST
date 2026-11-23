@echo off
title SCREENER + ML SERVICE LAUNCHER
echo ===================================================
echo Starting Complete SCREENER Suite
echo ===================================================

cd /d "%~dp0"

echo [1/2] Launching ML Service on http://localhost:8001...
start "ML Service (Port 8001)" cmd /k "cd /d "%~dp0" && python ml_service/ml_service.py"

timeout /t 3 /nobreak >nul

echo [2/2] Launching Main Screener on http://localhost:8000...
start "Main Screener (Port 8000)" cmd /k "cd /d "%~dp0" && python main.py"

echo.
echo ===================================================
echo Both services launched!
echo Main Screener: http://localhost:8000
echo ML Service:    http://localhost:8001
echo ===================================================
pause
