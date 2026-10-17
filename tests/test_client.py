"""
Unit Tests for StockMarketAI Developer SDK Client (Built-in unittest)
"""

import unittest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.stockmarketai.client import StockMarketAIClient


class TestStockMarketAIClient(unittest.TestCase):

    def setUp(self):
        self.client = StockMarketAIClient()

    def test_client_status(self):
        """Test status through client."""
        status = self.client.status()
        self.assertEqual(status["status"], "healthy")

    def test_client_prediction(self):
        """Test prediction through client."""
        pred = self.client.predict("MSFT", use_sentiment=False)
        self.assertEqual(pred["symbol"], "MSFT")
        self.assertIn("level3_trading_signal", pred)
        self.assertIn(pred["level3_trading_signal"]["signal"], ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"])

    def test_client_scan(self):
        """Test scanning through client."""
        scan_res = self.client.scan(["MSFT"])
        self.assertEqual(len(scan_res), 1)
        self.assertEqual(scan_res[0]["symbol"], "MSFT")


if __name__ == "__main__":
    unittest.main()
