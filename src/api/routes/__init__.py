"""
API Route Handlers
"""
from src.api.routes.health import router as health_router
from src.api.routes.predict import router as predict_router
from src.api.routes.backtest import router as backtest_router
from src.api.routes.stocks import router as stocks_router
from src.api.routes.brokers import router as brokers_router

__all__ = ["health_router", "predict_router", "backtest_router", "stocks_router", "brokers_router"]
