"""
StockMarketAI: Python Developer SDK Client
==========================================
Provides an intuitive, idiomatic Python client interface for StockMarketAI.
Supports both in-process service mode and remote REST API mode.
"""

from typing import Dict, List, Optional, Any, Union
import json
import logging
import requests

logger = logging.getLogger("stockmarketai.client")


class StockMarketAIClient:
    """
    Client for StockMarketAI.
    
    Usage:
        # In-process mode (direct Python service execution)
        client = StockMarketAIClient()
        prediction = client.predict("MSFT")
        print(prediction['level3_trading_signal']['signal'])

        # Remote API mode (connects to running FastAPI server)
        client = StockMarketAIClient(base_url="http://localhost:8000")
        prediction = client.predict("AAPL")
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = timeout
        self._local_service = None

        if not self.base_url:
            from src.stockmarketai.service import get_service
            self._local_service = get_service()

    def predict(self, symbol: str, use_sentiment: bool = True) -> Dict[str, Any]:
        """
        Get 3-Level Prediction and AI trade signal for a stock symbol.
        """
        symbol = symbol.upper().strip()
        if self.base_url:
            url = f"{self.base_url}/api/v1/predict/{symbol}?use_sentiment={str(use_sentiment).lower()}"
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            return self._local_service.predict(symbol, use_sentiment=use_sentiment)

    def backtest(
        self,
        symbol: str,
        strategy: str = "enhanced",
        initial_capital: float = 10000.0,
        stop_loss: float = 0.03,
        take_profit: float = 0.06,
        use_atr: bool = True
    ) -> Dict[str, Any]:
        """
        Run backtest simulation for a stock symbol.
        """
        symbol = symbol.upper().strip()
        if self.base_url:
            url = f"{self.base_url}/api/v1/backtest"
            payload = {
                "symbol": symbol,
                "strategy": strategy,
                "initial_capital": initial_capital,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "use_atr": use_atr
            }
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            return self._local_service.backtest(
                symbol=symbol,
                strategy=strategy,
                initial_capital=initial_capital,
                stop_loss=stop_loss,
                take_profit=take_profit,
                use_atr=use_atr
            )

    def scan(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Scan a list of symbols and rank trading signals.
        """
        if self.base_url:
            url = f"{self.base_url}/api/v1/stocks/scan"
            resp = requests.post(url, json={"symbols": symbols}, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            return self._local_service.scan_watchlist(symbols)

    def status(self) -> Dict[str, Any]:
        """Get system health and model status."""
        if self.base_url:
            url = f"{self.base_url}/api/v1/status"
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            return self._local_service.get_status()
