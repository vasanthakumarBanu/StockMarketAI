"""
StockMarketAI Backtesting Package
"""

from src.backtesting.metrics import calculate_metrics, format_metrics_table
from src.backtesting.strategy import (
    BaseStrategy,
    DirectionStrategy,
    RegressionExpectedReturnStrategy,
    EnhancedHybridStrategy
)
from src.backtesting.backtest import Backtester

__all__ = [
    "calculate_metrics",
    "format_metrics_table",
    "BaseStrategy",
    "DirectionStrategy",
    "RegressionExpectedReturnStrategy",
    "EnhancedHybridStrategy",
    "Backtester"
]
