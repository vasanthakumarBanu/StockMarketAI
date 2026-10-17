"""
Unit & Integration Tests for Phase 6 Platform Integration, DLL Service & Brokers
"""

import unittest
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.stockmarketai.integration.dll_service import (
    StockPredictionDLLService,
    c_predict,
    c_backtest,
    c_status
)
from src.stockmarketai.brokers import (
    AngelOneConnector,
    INDmoneyConnector,
    AutomatedTradeExecutionEngine,
    OrderSide,
    OrderType,
    OrderStatus
)


class TestPhase6Integration(unittest.TestCase):

    def setUp(self):
        self.dll_service = StockPredictionDLLService()

    def test_dll_service_json_predict(self):
        """Verify DLL service returns valid JSON prediction."""
        raw_json = self.dll_service.predict("MSFT", use_sentiment=False)
        data = json.loads(raw_json)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["data"]["symbol"], "MSFT")
        self.assertIn("level3_trading_signal", data["data"])

    def test_dll_service_json_backtest(self):
        """Verify DLL service returns valid JSON backtest."""
        raw_json = self.dll_service.backtest("MSFT", strategy="enhanced", initial_capital=10000.0)
        data = json.loads(raw_json)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["data"]["symbol"], "MSFT")
        self.assertIn("final_equity", data["data"])

    def test_c_abi_exports(self):
        """Verify C-compatible function exports."""
        status_json = c_status()
        status = json.loads(status_json)
        self.assertEqual(status["status"], "healthy")

        pred_json = c_predict("MSFT")
        pred = json.loads(pred_json)
        self.assertEqual(pred["status"], "success")

    def test_angel_one_connector(self):
        """Test Angel One SmartAPI connector."""
        connector = AngelOneConnector(is_paper_trading=True)
        self.assertTrue(connector.connect())

        # Test Quote
        quote = connector.get_quote("MSFT")
        self.assertGreater(quote["ltp"], 0)

        # Test Order
        order = connector.place_order(
            symbol="MSFT",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            stop_loss=quote["ltp"] * 0.95,
            take_profit=quote["ltp"] * 1.05
        )
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertEqual(order.quantity, 10)

        # Check Position
        positions = connector.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["symbol"], "MSFT")

    def test_indmoney_connector(self):
        """Test INDmoney US fractional equities connector."""
        connector = INDmoneyConnector(is_paper_trading=True)
        self.assertTrue(connector.connect())

        order = connector.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=2.5,
            order_type=OrderType.MARKET
        )
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertEqual(order.quantity, 2.5)

    def test_execution_engine(self):
        """Test automated trade execution engine with circuit breaker."""
        connector = AngelOneConnector(is_paper_trading=True)
        connector.connect()

        engine = AutomatedTradeExecutionEngine(
            broker=connector,
            allocated_capital=10000.0,
            risk_per_trade=0.02,
            dry_run=False
        )

        # Force circuit breaker validation
        self.assertFalse(connector.circuit_breaker.is_tripped)
        connector.circuit_breaker.record_pnl(-5.0)  # -5% loss trips circuit breaker
        self.assertTrue(connector.circuit_breaker.is_tripped)

        # Execution should be blocked by circuit breaker
        res = engine.evaluate_and_execute("MSFT")
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
