@echo off
title StockMarketAI Standalone Launcher
echo ========================================================
echo        StockMarketAI: Standalone Launcher
echo ========================================================
echo Starting local AI Inference & Broker Execution Engine...

start /B python run_api.py --host 127.0.0.1 --port 8000

echo Waiting for backend engine to initialize...
timeout /t 3 /nobreak >nul

echo Launching StockMarketAI Dashboard...
start http://localhost:8000/dashboard

echo [OK] StockMarketAI is running standalone at http://localhost:8000/dashboard
echo You can minimize this window. Close this window to stop the server.
pause
