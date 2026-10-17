"""
Integration Tests for FastAPI REST API Endpoints (Built-in unittest)
"""

import unittest
import os
import sys
from starlette.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.app import create_app


class TestStockMarketAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def test_health_endpoints(self):
        """Test health and status endpoints."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "healthy")

        res_v1 = self.client.get("/api/v1/health")
        self.assertEqual(res_v1.status_code, 200)
        self.assertEqual(res_v1.json()["status"], "healthy")

        res_status = self.client.get("/api/v1/status")
        self.assertEqual(res_status.status_code, 200)
        self.assertIn("models_loaded", res_status.json())

    def test_models_endpoint(self):
        """Test models specs endpoint."""
        res = self.client.get("/api/v1/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("feature_columns", data)
        self.assertEqual(data["status"], "ready")

    def test_stocks_endpoint(self):
        """Test stocks list endpoint."""
        res = self.client.get("/api/v1/stocks")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreater(data["total"], 0)
        self.assertTrue(any(s["symbol"] == "MSFT" for s in data["supported_stocks"]))

    def test_predict_get(self):
        """Test GET /api/v1/predict/{symbol}."""
        res = self.client.get("/api/v1/predict/MSFT?use_sentiment=false")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["symbol"], "MSFT")
        self.assertIn("level1_direction", data)
        self.assertIn("level2_price_targets", data)
        self.assertIn("level3_trading_signal", data)
        self.assertIn(data["level3_trading_signal"]["signal"], ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"])

    def test_predict_post(self):
        """Test POST /api/v1/predict."""
        res = self.client.post("/api/v1/predict", json={"symbol": "MSFT", "use_sentiment": False})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["symbol"], "MSFT")

    def test_backtest_post(self):
        """Test POST /api/v1/backtest."""
        res = self.client.post("/api/v1/backtest", json={
            "symbol": "MSFT",
            "strategy": "enhanced",
            "initial_capital": 10000.0,
            "use_atr": True
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["symbol"], "MSFT")
        self.assertIn("final_equity", data)
        self.assertIn("sharpe_ratio", data)
        self.assertIn("equity_curve", data)
        self.assertGreater(len(data["equity_curve"]), 0)

    def test_broker_execution_status(self):
        """Test GET /api/v1/brokers/execution-status."""
        res = self.client.get("/api/v1/brokers/execution-status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("auto_execution_enabled", data)
        self.assertIn("broker_name", data)

    def test_broker_toggle_and_execute(self):
        """Test toggling the master auto-execution boolean and verifying block/permit."""
        # 1. Ensure disabled initially
        res = self.client.post("/api/v1/brokers/toggle-execution", json={"enabled": False})
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["auto_execution_enabled"])

        # 2. When False, execute-now must be blocked
        res_exec_blocked = self.client.post("/api/v1/brokers/execute-now", json={"symbol": "MSFT"})
        self.assertEqual(res_exec_blocked.status_code, 200)
        self.assertEqual(res_exec_blocked.json()["status"], "blocked")
        self.assertFalse(res_exec_blocked.json()["auto_execution_enabled"])

        # 3. Toggle to True
        res_toggle_on = self.client.post("/api/v1/brokers/toggle-execution", json={"enabled": True, "paper_trading": True})
        self.assertEqual(res_toggle_on.status_code, 200)
        self.assertTrue(res_toggle_on.json()["auto_execution_enabled"])

        # 4. Verify status endpoint reflects True
        res_status = self.client.get("/api/v1/brokers/execution-status")
        self.assertTrue(res_status.json()["auto_execution_enabled"])

        # 5. Clean up - toggle back to False for safety
        res_cleanup = self.client.post("/api/v1/brokers/toggle-execution", json={"enabled": False})
        self.assertFalse(res_cleanup.json()["auto_execution_enabled"])

    def test_stock_search_endpoint(self):
        """Test GET /api/v1/stocks/search for Indian and US equities."""
        res = self.client.get("/api/v1/stocks/search?query=SRM%20Energy&limit=5")
        self.assertEqual(res.status_code, 200)
        items = res.json()
        self.assertGreater(len(items), 0)
        self.assertTrue(any(it["symbol"] == "SRMENERGY.BO" for it in items))

    def test_predict_company_name_resolution(self):
        """Test GET /api/v1/predict/ with company name 'SRM Energy Ltd' auto-resolves to SRMENERGY.BO."""
        res = self.client.get("/api/v1/predict/SRM%20Energy%20Ltd?use_sentiment=false")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["symbol"], "SRMENERGY.BO")
        self.assertEqual(data["query_entered"], "SRM Energy Ltd")
        self.assertIn("SRM ENERGY", data.get("company_name", "").upper())
        self.assertIn("level1_direction", data)
        self.assertIn("level2_price_targets", data)
        self.assertIn("level3_trading_signal", data)


if __name__ == "__main__":
    unittest.main()
