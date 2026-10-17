#!/usr/bin/env python3
"""
StockMarketAI: Phase 5 API & Interactive Dashboard Runner
=========================================================
Starts the FastAPI ML inference server and serves the Interactive Web Dashboard.

Usage:
    python run_api.py
    python run_api.py --port 8080 --reload
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.main import main

if __name__ == "__main__":
    main()
