"""
StockMarketAI: FastAPI Application Factory
==========================================
Configures middleware, routes, static frontend dashboard, and lifecycle events.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

from src.api.routes import health_router, predict_router, backtest_router, stocks_router, brokers_router
from src.stockmarketai.service import get_service

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "src", "stockmarketai", "frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm ML service and load models on server startup."""
    print("🚀 Starting StockMarketAI REST API Server...")
    # Initialize singleton service and load models
    service = get_service()
    print(f"✅ Models loaded: {len(service.models)} regression targets, classification ready={service.classification_model is not None}")
    yield
    print("🛑 Shutting down StockMarketAI REST API Server...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="StockMarketAI Production ML API",
        description=(
            "Production REST API for StockMarketAI — Multi-Level Stock Market Prediction & Backtesting System.\n\n"
            "- **Level 1**: Direction Classification (Up vs Down Probability)\n"
            "- **Level 2**: OHLC Price Regression (Tomorrow's Open, High, Low, Close)\n"
            "- **Level 3**: AI Trading Recommendation (BUY/HOLD/SELL) with Dynamic ATR Stops & Risk/Reward\n"
            "- **Phase 4**: Event-Driven Backtesting Simulator\n"
            "- **Phase 5**: Reusable Microservice & Live Interactive Dashboard"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routers
    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(backtest_router)
    app.include_router(stocks_router)
    app.include_router(brokers_router)

    # Mount static frontend dashboard if available
    if os.path.exists(FRONTEND_DIR):
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

        @app.get("/", include_in_schema=False)
        @app.get("/dashboard", include_in_schema=False)
        def serve_dashboard():
            index_path = os.path.join(FRONTEND_DIR, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return RedirectResponse(url="/docs")

        @app.get("/about", include_in_schema=False)
        def serve_about():
            about_path = os.path.join(PROJECT_ROOT, "ABOUT_STOCKMARKETAI.md")
            if os.path.exists(about_path):
                from fastapi.responses import PlainTextResponse
                with open(about_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return PlainTextResponse(content, media_type="text/markdown; charset=utf-8")
            return RedirectResponse(url="/docs")
    else:
        @app.get("/", include_in_schema=False)
        def root_redirect():
            return RedirectResponse(url="/docs")

    return app


app = create_app()
