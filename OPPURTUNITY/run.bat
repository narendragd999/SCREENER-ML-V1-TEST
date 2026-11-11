@echo off
title Opportunity Engine — No-Stop Equity Momentum and Fair Value
cd /d "%~dp0"
echo =====================================================================
echo  OPPORTUNITY ENGINE: No-Stop Equity Momentum and Fair Value Platform
echo =====================================================================
echo  Universe: 307 Vetted Strict-4 Healthy Stocks (tickers.csv)
echo  Features: Horizontal Support Floors + 9-EMA + Fair Value + No-Stop Lab
echo =====================================================================
echo  Starting local server on http://localhost:8500 ...
python -m uvicorn main:app --host 0.0.0.0 --port 8500 --reload
pause
