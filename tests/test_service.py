"""
Unit Tests for StockMarketAIService (Built-in unittest)
"""

import unittest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.stockmarketai.service import StockMarketAIService, get_service


class TestStockMarketAIService(unittest.TestCase):

    def test_service_singleton(self):
        """Verify service follows singleton pattern."""
        s1 = get_service()
        s2 = get_service()
        self.assertIs(s1, s2)

    def test_service_status(self):
        """Verify service health and metadata."""
        service = get_service()
        status = service.get_status()
        self.assertEqual(status["status"], "healthy")
        self.assertEqual(status["version"], "1.0.0")
        self.assertIn("models_loaded", status)
        self.assertGreaterEqual(status["models_loaded"]["feature_count"], 17)

    def test_service_prediction(self):
        """Verify complete 3-Level prediction generation."""
        service = get_service()
        pred = service.predict("MSFT", retrain_if_needed=False, use_sentiment=False)

        self.assertEqual(pred["symbol"], "MSFT")
        self.assertIn("level1_direction", pred)
        self.assertIn(pred["level1_direction"]["prediction"], ["UP", "DOWN"])
        self.assertTrue(0.0 <= pred["level1_direction"]["probability_up"] <= 1.0)

        self.assertIn("level2_price_targets", pred)
        l2 = pred["level2_price_targets"]
        self.assertGreaterEqual(l2["tomorrow_high"], l2["tomorrow_open"])
        self.assertGreaterEqual(l2["tomorrow_high"], l2["tomorrow_close"])
        self.assertLessEqual(l2["tomorrow_low"], l2["tomorrow_open"])
        self.assertLessEqual(l2["tomorrow_low"], l2["tomorrow_close"])

        self.assertIn("level3_trading_signal", pred)
        l3 = pred["level3_trading_signal"]
        self.assertIn(l3["signal"], ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"])
        self.assertTrue(0.0 <= l3["confidence"] <= 1.0)

    def test_service_backtest(self):
        """Verify backtesting execution through service."""
        service = get_service()
        result = service.backtest("MSFT", strategy="enhanced", initial_capital=10000.0)

        self.assertEqual(result["symbol"], "MSFT")
        self.assertEqual(result["initial_capital"], 10000.0)
        self.assertGreater(result["final_equity"], 0)
        self.assertIn("sharpe_ratio", result)
        self.assertIn("max_drawdown_pct", result)
        self.assertIn("equity_curve", result)
        self.assertGreater(len(result["equity_curve"]), 0)

    def test_service_scan(self):
        """Verify multi-stock scanning."""
        service = get_service()
        results = service.scan_watchlist(["MSFT"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["symbol"], "MSFT")
        self.assertIn("signal", results[0])


if __name__ == "__main__":
    unittest.main()
