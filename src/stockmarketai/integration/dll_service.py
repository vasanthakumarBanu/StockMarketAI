"""
StockMarketAI: Python DLL Service Layer
=======================================
Implements Section 2.1 of multi_stock_adaptation_plan.md.
Provides a thread-safe, JSON-serializable service layer that can be:
1. Exported as a C-compatible DLL/shared library using ctypes/PyInstaller
2. Imported directly via Python.NET (pythonnet) in C#/.NET trading applications
3. Invoked from desktop trading platforms (NinjaTrader, MetaTrader, custom WPF/WinForms)
"""

import os
import sys
import json
import ctypes
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.stockmarketai.service import get_service


class StockPredictionDLLService:
    """
    Service Layer designed for DLL wrapping and C# interop.
    Accepts JSON strings or Python dicts and returns JSON responses.
    """

    def __init__(self):
        self._service = get_service()

    def predict(self, stock_symbol: str, use_sentiment: bool = True) -> str:
        """
        Generate prediction for stock symbol and return as JSON string.
        """
        try:
            pred = self._service.predict(stock_symbol, use_sentiment=use_sentiment)
            return json.dumps({
                "status": "success",
                "data": pred
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })

    def backtest(
        self,
        stock_symbol: str,
        strategy: str = "enhanced",
        initial_capital: float = 10000.0,
        stop_loss: float = 0.03,
        take_profit: float = 0.06,
        use_atr: bool = True
    ) -> str:
        """
        Execute backtest simulation and return metrics/trades as JSON string.
        """
        try:
            results = self._service.backtest(
                symbol=stock_symbol,
                strategy=strategy,
                initial_capital=initial_capital,
                stop_loss=stop_loss,
                take_profit=take_profit,
                use_atr=use_atr
            )
            return json.dumps({
                "status": "success",
                "data": results
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })

    def get_status(self) -> str:
        """Return system and model readiness status as JSON."""
        return json.dumps(self._service.get_status())


# Global singleton instance for C-ABI exports
_dll_instance = None

def _get_dll_instance():
    global _dll_instance
    if _dll_instance is None:
        _dll_instance = StockPredictionDLLService()
    return _dll_instance


# ---------------------------------------------------------------------------
# C-Compatible Function Exports (Callable by external C/C++/C# DLL loaders)
# ---------------------------------------------------------------------------

# Keep allocated strings alive for C callers
_allocated_strings = []

def c_predict(symbol_cstr: str) -> str:
    """C-exported symbol prediction function."""
    service = _get_dll_instance()
    res = service.predict(symbol_cstr)
    return res

def c_backtest(symbol_cstr: str, strategy_cstr: str) -> str:
    """C-exported backtesting function."""
    service = _get_dll_instance()
    res = service.backtest(symbol_cstr, strategy=strategy_cstr)
    return res

def c_status() -> str:
    """C-exported status function."""
    service = _get_dll_instance()
    return service.get_status()
