@echo off
title EQUITY 9-EMA & FAIR VALUE 3% SCREENER

echo ========================================================
echo Starting Equity 9-EMA & Fair Value 3% Screener Platform
echo ========================================================

cd /d "%~dp0"

echo Dashboard URL: http://localhost:8000
echo API Docs:      http://localhost:8000/docs
echo.

echo Keep this window open while using the dashboard.
python -B -m uvicorn main:app --host 127.0.0.1 --port 8000
if errorlevel 1 echo If port 8000 is already in use, open the running dashboard instead of starting a second copy.

pause
