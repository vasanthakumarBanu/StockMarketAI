"""
StockMarketAI: API Server Runner
================================
Run with:
    python -m src.api.main
or:
    python run_api.py
"""

import uvicorn
import argparse
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="StockMarketAI REST API & Dashboard Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    args = parser.parse_args()

    print(f"Starting StockMarketAI Server on http://localhost:{args.port}")
    print(f"Interactive Dashboard: http://localhost:{args.port}/dashboard")
    print(f"Swagger API Docs:      http://localhost:{args.port}/docs")

    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
