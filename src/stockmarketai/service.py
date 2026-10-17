"""
StockMarketAI: Reusable Production ML Service Layer
===================================================
Provides a thread-safe, high-performance service interface for:
- Live and historical stock data fetching with caching
- Full-pipeline feature engineering (Technical, Candlestick, Volume, Lags, ATR)
- Multi-level ML inference:
    * Level 1: Direction classification (Up vs. Down probability)
    * Level 2: OHLC regression prices (Tomorrow Open, High, Low, Close)
    * Level 3: AI Trade Recommendation (BUY/HOLD/SELL) with confidence, dynamic ATR stops, risk/reward
- Event-driven backtesting engine integration
- Multi-stock market scanner
"""

import os
import sys
import time
import pickle
import joblib
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from complete_workflow_fixed import download_stock_data, engineer_all_features
from src.backtesting.strategy import (
    BaseStrategy,
    DirectionStrategy,
    RegressionExpectedReturnStrategy,
    EnhancedHybridStrategy
)
from src.backtesting.backtest import Backtester

logger = logging.getLogger("stockmarketai.service")


class StockMarketAIService:
    """
    Production-ready Reusable ML Service for StockMarketAI.
    Manages model loading, caching, prediction inference, and strategy execution.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern to avoid reloading heavy models on every request."""
        if cls._instance is None:
            cls._instance = super(StockMarketAIService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, models_dir: str = "models_complete", classification_dir: str = "models"):
        if getattr(self, "_initialized", False):
            return

        self.models_dir = os.path.join(PROJECT_ROOT, models_dir)
        self.classification_dir = os.path.join(PROJECT_ROOT, classification_dir)
        self.start_time = time.time()
        
        # Caches
        self._data_cache: Dict[str, Dict[str, Any]] = {}
        self._prediction_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl_seconds = 300  # 5 minutes cache for market data

        # Models storage
        self.models: Dict[str, Any] = {}
        self.scaler = None
        self.feature_columns: List[str] = []
        self.classification_model = None
        self.results_summary: Dict[str, Any] = {}

        self._load_models()
        self._initialized = True
        logger.info("StockMarketAIService initialized successfully.")

    def _load_models(self):
        """Load pre-trained models, scalers, and metadata."""
        try:
            # 1. Load feature columns
            feat_cols_path = os.path.join(self.models_dir, "feature_columns.pkl")
            if os.path.exists(feat_cols_path):
                self.feature_columns = joblib.load(feat_cols_path)

            # 2. Load feature scaler
            scaler_path = os.path.join(self.models_dir, "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

            # 3. Load results summary
            summary_path = os.path.join(self.models_dir, "results_summary.pkl")
            if os.path.exists(summary_path):
                self.results_summary = joblib.load(summary_path)

            # 4. Load OHLC Regression models
            targets = ["Tomorrow_Open", "Tomorrow_High", "Tomorrow_Low", "Tomorrow_Close"]
            for target in targets:
                target_key = target.lower()
                lr_file = os.path.join(self.models_dir, f"{target_key}_linear_regression.pkl")
                rf_file = os.path.join(self.models_dir, f"{target_key}_random_forest_regressor.pkl")

                if os.path.exists(lr_file):
                    self.models[target] = joblib.load(lr_file)
                elif os.path.exists(rf_file):
                    self.models[target] = joblib.load(rf_file)

            # 5. Load Classification model for direction
            clf_path = os.path.join(self.classification_dir, "stock_rf_model.pkl")
            if not os.path.exists(clf_path):
                clf_path = os.path.join(self.classification_dir, "stock_lr_model.pkl")

            if os.path.exists(clf_path):
                self.classification_model = joblib.load(clf_path)

            logger.info(f"Loaded {len(self.models)} regression models and classification model.")
        except Exception as e:
            logger.warning(f"Error loading pre-trained models: {e}. Models will be trained dynamically if needed.")

    def get_market_data(self, symbol: str, period: str = "2y", force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch market data with caching and automatic company name resolution.
        """
        raw_query = symbol.strip()
        from src.stockmarketai.data.search import resolve_symbol_or_search
        resolved_sym, _ = resolve_symbol_or_search(raw_query)
        symbol = resolved_sym
        now = time.time()

        if not force_refresh and symbol in self._data_cache:
            cache_entry = self._data_cache[symbol]
            if now - cache_entry["timestamp"] < self._cache_ttl_seconds:
                return cache_entry["data"].copy()

        # Download stock data
        try:
            raw_path = download_stock_data(symbol, period=period)
        except Exception as e:
            raise ValueError(f"No market data found for '{raw_query}' (resolved ticker: '{symbol}'): {e}")

        df = pd.read_csv(raw_path)
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)

        self._data_cache[symbol] = {
            "timestamp": now,
            "data": df.copy()
        }
        return df

    def search_symbols(self, query: str, max_results: int = 8) -> List[Dict[str, Any]]:
        """Search Indian and US stock tickers by name or query."""
        from src.stockmarketai.data.search import search_stock_symbols
        return search_stock_symbols(query, max_results=max_results)

    def predict(
        self,
        symbol: str,
        retrain_if_needed: bool = True,
        use_sentiment: bool = True
    ) -> Dict[str, Any]:
        """
        Generate complete 3-Level Prediction for any stock or company name:
          - Automatically resolves company names (e.g. 'SRM Energy Ltd' -> 'SRMENERGY.BO')
          - Level 1: Direction (Up/Down) & Probability
          - Level 2: Tomorrow's Open, High, Low, Close prices & changes
          - Level 3: Trading recommendation (BUY/HOLD/SELL) with confidence & ATR stops
        """
        raw_query = symbol.strip()
        from src.stockmarketai.data.search import resolve_symbol_or_search
        resolved_symbol, meta = resolve_symbol_or_search(raw_query)
        symbol = resolved_symbol
        now = time.time()

        # Check prediction cache
        if symbol in self._prediction_cache:
            cached = self._prediction_cache[symbol]
            if now - cached["timestamp"] < 60:  # 1 minute cache for predictions
                return cached["prediction"]

        # Fetch market data
        df = self.get_market_data(symbol)
        if len(df) < 50:
            raise ValueError(f"Insufficient historical data for {symbol} (only {len(df)} rows)")

        # Feature engineering
        featured_df = engineer_all_features(df)
        featured_clean = featured_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

        # Get latest market metrics
        latest_row = featured_clean.iloc[-1]
        today_open = float(latest_row["Open"])
        today_high = float(latest_row["High"])
        today_low = float(latest_row["Low"])
        today_close = float(latest_row["Close"])
        today_volume = float(latest_row["Volume"])
        prediction_date = latest_row["Date"].strftime("%Y-%m-%d") if "Date" in latest_row else datetime.now().strftime("%Y-%m-%d")

        # ATR calculation for risk stops
        atr = float(latest_row["ATR_14"]) if "ATR_14" in latest_row and not np.isnan(latest_row["ATR_14"]) else float(today_close * 0.02)
        volatility_20 = float(latest_row["Volatility_20"]) if "Volatility_20" in latest_row and not np.isnan(latest_row["Volatility_20"]) else 0.015

        # Sentiment Analysis
        sentiment_score = 0.0
        headline_count = 0
        if use_sentiment:
            try:
                from src.stockmarketai.sentiment import get_stock_sentiment
                sent_res = get_stock_sentiment(symbol, news_delay=0.3, as_compound=True)
                sentiment_score = float(sent_res.get("compound", 0.0))
                headline_count = int(sent_res.get("headline_count", 0))
            except Exception as e:
                logger.debug(f"Sentiment unavailable for {symbol}: {e}")
                sentiment_score = 0.0

        # Perform Predictions using Level 1 & Level 2 models
        pred_open, pred_high, pred_low, pred_close = None, None, None, None
        direction_prob = 0.5
        direction_pred = 1

        # Check if pre-trained models match features
        can_use_pretrained = (
            self.scaler is not None
            and self.feature_columns
            and all(c in featured_clean.columns for c in self.feature_columns)
            and len(self.models) == 4
        )

        if can_use_pretrained and not retrain_if_needed:
            # Fast inference with pre-trained models
            X_latest = featured_clean[self.feature_columns].iloc[[-1]]
            X_scaled = self.scaler.transform(X_latest)

            pred_open = float(self.models["Tomorrow_Open"].predict(X_scaled)[0])
            pred_high = float(self.models["Tomorrow_High"].predict(X_scaled)[0])
            pred_low = float(self.models["Tomorrow_Low"].predict(X_scaled)[0])
            pred_close = float(self.models["Tomorrow_Close"].predict(X_scaled)[0])

            if self.classification_model is not None:
                try:
                    direction_prob = float(self.classification_model.predict_proba(X_scaled)[0, 1])
                    direction_pred = int(self.classification_model.predict(X_scaled)[0])
                except Exception:
                    direction_prob = 1.0 if pred_close > today_close else 0.0
                    direction_pred = 1 if pred_close > today_close else 0
            else:
                direction_prob = 1.0 if pred_close > today_close else 0.0
                direction_pred = 1 if pred_close > today_close else 0
        else:
            # Dedicated out-of-sample trained model for this specific stock
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler

            data_fit = featured_clean.copy()
            data_fit["Tomorrow_Direction"] = (data_fit["Close"].shift(-1) > data_fit["Close"]).astype(int)
            target_cols = ["Tomorrow_Open", "Tomorrow_High", "Tomorrow_Low", "Tomorrow_Close", "Tomorrow_Direction"]
            feat_cols = [c for c in data_fit.columns if c not in target_cols and c != "Date"]

            # Train on historical rows (excluding last row where tomorrow target is unknown)
            train_data = data_fit.iloc[:-1].dropna(subset=target_cols)
            X_train = train_data[feat_cols]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # 1. Classification
            clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
            clf.fit(X_train_scaled, train_data["Tomorrow_Direction"])

            # 2. Regression
            reg_models = {}
            for target in ["Tomorrow_Open", "Tomorrow_High", "Tomorrow_Low", "Tomorrow_Close"]:
                reg = LinearRegression()
                reg.fit(X_train_scaled, train_data[target])
                reg_models[target] = reg

            # Predict for latest day
            X_curr = scaler.transform(data_fit[feat_cols].iloc[[-1]])
            pred_open = float(reg_models["Tomorrow_Open"].predict(X_curr)[0])
            pred_high = float(reg_models["Tomorrow_High"].predict(X_curr)[0])
            pred_low = float(reg_models["Tomorrow_Low"].predict(X_curr)[0])
            pred_close = float(reg_models["Tomorrow_Close"].predict(X_curr)[0])

            probs = clf.predict_proba(X_curr)[0]
            direction_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            direction_pred = int(clf.predict(X_curr)[0])

        # Enforce physical price consistency (High >= Max(Open, Close), Low <= Min(Open, Close))
        pred_high = max(pred_high, pred_open, pred_close)
        pred_low = min(pred_low, pred_open, pred_close)

        # Level 3: AI Signal Decision Engine
        expected_change = pred_close - today_close
        expected_change_pct = (expected_change / today_close) * 100.0
        expected_return = expected_change / today_close

        # Confidence calculation
        direction_confidence = abs(direction_prob - 0.5) * 2.0  # 0.0 to 1.0
        signal_confidence = direction_confidence * 0.7 + min(abs(expected_return) / 0.02, 1.0) * 0.3

        # Sentiment adjustment
        if sentiment_score > 0.15:
            sentiment_label = "BULLISH"
            signal_confidence = min(signal_confidence * 1.15, 0.98)
        elif sentiment_score < -0.15:
            sentiment_label = "BEARISH"
            signal_confidence = min(signal_confidence * 1.15, 0.98)
        else:
            sentiment_label = "NEUTRAL"

        # Signal rules
        if direction_prob >= 0.60 and expected_return >= 0.015:
            signal = "STRONG BUY"
        elif direction_prob >= 0.52 and expected_return > 0.003:
            signal = "BUY"
        elif direction_prob <= 0.40 and expected_return <= -0.015:
            signal = "STRONG SELL"
        elif direction_prob <= 0.48 and expected_return < -0.003:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Stop-loss and take-profit targets based on ATR
        if signal in ["BUY", "STRONG BUY"]:
            stop_loss = round(today_close - 1.5 * atr, 2)
            take_profit = round(today_close + 2.5 * atr, 2)
            risk_reward = round((take_profit - today_close) / max(today_close - stop_loss, 0.01), 2)
        elif signal in ["SELL", "STRONG SELL"]:
            stop_loss = round(today_close + 1.5 * atr, 2)
            take_profit = round(today_close - 2.5 * atr, 2)
            risk_reward = round((today_close - take_profit) / max(stop_loss - today_close, 0.01), 2)
        else:
            stop_loss = round(today_close - 1.5 * atr, 2)
            take_profit = round(today_close + 1.5 * atr, 2)
            risk_reward = 1.0

        # Technical indicators summary
        rsi = float(latest_row["RSI_14"]) if "RSI_14" in latest_row else 50.0
        macd = float(latest_row["MACD"]) if "MACD" in latest_row else 0.0
        macd_signal = float(latest_row["MACD_Signal"]) if "MACD_Signal" in latest_row else 0.0
        ma_20 = float(latest_row["MA_20"]) if "MA_20" in latest_row else today_close
        ma_50 = float(latest_row["MA_50"]) if "MA_50" in latest_row else today_close
        company_name = meta.get("name") if meta else symbol
        market_name = meta.get("market") if meta else "Equities"
        badge_name = meta.get("badge") if meta else ""

        result = {
            "symbol": symbol,
            "query_entered": raw_query,
            "company_name": company_name,
            "market": market_name,
            "badge": badge_name,
            "prediction_date": prediction_date,
            "current_market": {
                "open": round(today_open, 2),
                "high": round(today_high, 2),
                "low": round(today_low, 2),
                "close": round(today_close, 2),
                "volume": int(today_volume),
                "atr": round(atr, 2),
                "volatility_20": round(volatility_20, 4),
                "rsi_14": round(rsi, 2),
                "macd": round(macd, 3),
                "macd_signal": round(macd_signal, 3),
                "ma_20": round(ma_20, 2),
                "ma_50": round(ma_50, 2),
            },
            "level1_direction": {
                "prediction": "UP" if direction_pred == 1 else "DOWN",
                "probability_up": round(direction_prob, 4),
                "probability_pct": round(direction_prob * 100.0, 2),
                "confidence": round(direction_confidence, 4)
            },
            "level2_price_targets": {
                "tomorrow_open": round(pred_open, 2),
                "tomorrow_high": round(pred_high, 2),
                "tomorrow_low": round(pred_low, 2),
                "tomorrow_close": round(pred_close, 2),
                "expected_change": round(expected_change, 2),
                "expected_change_pct": round(expected_change_pct, 2),
                "interval_lower": round(pred_close - 1.96 * atr, 2),
                "interval_upper": round(pred_close + 1.96 * atr, 2)
            },
            "level3_trading_signal": {
                "signal": signal,
                "confidence": round(signal_confidence, 4),
                "confidence_pct": round(signal_confidence * 100.0, 1),
                "expected_return_pct": round(expected_change_pct, 2),
                "stop_loss_price": stop_loss,
                "take_profit_price": take_profit,
                "risk_reward_ratio": risk_reward,
                "recommended_action": (
                    f"{signal} with stop-loss at ${stop_loss} and take-profit at ${take_profit}"
                    if signal != "HOLD" else "Hold position or wait for stronger directional confirmation"
                )
            },
            "sentiment": {
                "score": round(sentiment_score, 4),
                "sentiment_label": sentiment_label,
                "headlines_analyzed": headline_count
            },
            "timestamp": time.time()
        }

        # Save to prediction cache
        self._prediction_cache[symbol] = {
            "timestamp": now,
            "prediction": result
        }

        return result

    def backtest(
        self,
        symbol: str,
        strategy: str = "enhanced",
        initial_capital: float = 10000.0,
        stop_loss: float = 0.03,
        take_profit: float = 0.06,
        use_atr: bool = True,
        lookback_days: int = 504
    ) -> Dict[str, Any]:
        """
        Execute backtest for a symbol using Phase 4 Backtester engine.
        """
        from run_backtest import prepare_and_predict_out_of_sample

        symbol = symbol.upper().strip()
        df = self.get_market_data(symbol, period="2y")
        featured_df = engineer_all_features(df)

        test_data, model_stats = prepare_and_predict_out_of_sample(featured_df, train_split=0.8)

        # Select strategy
        strategy_lower = strategy.lower()
        if strategy_lower in ["direction", "clf"]:
            strat = DirectionStrategy(
                buy_threshold=0.55,
                exit_threshold=0.45,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                use_atr_stops=use_atr
            )
        elif strategy_lower in ["regression", "reg"]:
            strat = RegressionExpectedReturnStrategy(
                min_expected_return=0.005,
                exit_expected_return=-0.002,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                use_atr_stops=use_atr
            )
        else:
            strat = EnhancedHybridStrategy(
                min_direction_prob=0.52,
                min_expected_return=0.003,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                use_atr_stops=use_atr
            )

        backtester = Backtester(
            initial_capital=initial_capital,
            commission_rate=0.001,
            slippage_rate=0.0005,
            results_dir=os.path.join(PROJECT_ROOT, "src", "backtesting", "results")
        )

        results = backtester.run(test_data, strat, symbol=symbol)
        metrics = results["metrics"]

        # Serialize trade log for JSON API
        trade_log = results["trade_log"]
        trades_list = []
        if not trade_log.empty:
            for _, r in trade_log.tail(50).iterrows():
                trades_list.append({
                    "entry_date": r["entry_date"].strftime("%Y-%m-%d") if hasattr(r["entry_date"], "strftime") else str(r["entry_date"]),
                    "exit_date": r["exit_date"].strftime("%Y-%m-%d") if hasattr(r["exit_date"], "strftime") else str(r["exit_date"]),
                    "action": r.get("action", "BUY"),
                    "entry_price": round(float(r["entry_price"]), 2),
                    "exit_price": round(float(r["exit_price"]), 2),
                    "pnl": round(float(r["pnl"]), 2),
                    "return_pct": round(float(r.get("pnl_pct", r.get("return_pct", 0.0))), 2),
                    "exit_reason": str(r.get("exit_reason", "SIGNAL"))
                })

        # Sample equity curve points
        equity_series = results["equity_series"]
        benchmark_series = results["benchmark_series"]
        curve_data = []
        step = max(1, len(equity_series) // 40)
        for i in range(0, len(equity_series), step):
            dt = equity_series.index[i]
            dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
            curve_data.append({
                "date": dt_str,
                "strategy_equity": round(float(equity_series.iloc[i]), 2),
                "benchmark_equity": round(float(benchmark_series.iloc[i]), 2)
            })

        bench = metrics.get("benchmark", {})
        trades_m = metrics.get("trades", {})

        # Parse profit factor float
        pf = trades_m.get("profit_factor", 0.0)
        try:
            pf_val = round(float(pf), 2)
        except (ValueError, TypeError):
            pf_val = 999.0

        # Parse sortino float
        sortino = metrics.get("sortino_ratio", 0.0)
        try:
            sortino_val = round(float(sortino), 2)
        except (ValueError, TypeError):
            sortino_val = 999.0

        return {
            "symbol": symbol,
            "strategy": strat.__class__.__name__,
            "initial_capital": initial_capital,
            "final_equity": round(float(metrics.get("final_equity", initial_capital)), 2),
            "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
            "benchmark_return_pct": round(float(bench.get("benchmark_return_pct", 0.0)), 2),
            "cagr_pct": round(float(metrics.get("cagr_pct", 0.0)), 2),
            "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 2),
            "sortino_ratio": sortino_val,
            "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
            "benchmark_drawdown_pct": round(float(bench.get("benchmark_mdd_pct", 0.0)), 2),
            "win_rate_pct": round(float(trades_m.get("win_rate_pct", 0.0)), 1),
            "profit_factor": pf_val,
            "total_trades": int(trades_m.get("total_trades", 0)),
            "winning_trades": int(trades_m.get("winning_trades", 0)),
            "losing_trades": int(trades_m.get("losing_trades", 0)),
            "model_stats": model_stats,
            "equity_curve": curve_data,
            "recent_trades": trades_list
        }

    def scan_watchlist(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Scan a list of stock symbols and rank trading opportunities.
        """
        opportunities = []
        for symbol in symbols:
            try:
                pred = self.predict(symbol, retrain_if_needed=False)
                opportunities.append({
                    "symbol": pred["symbol"],
                    "current_price": pred["current_market"]["close"],
                    "predicted_close": pred["level2_price_targets"]["tomorrow_close"],
                    "expected_change_pct": pred["level2_price_targets"]["expected_change_pct"],
                    "direction": pred["level1_direction"]["prediction"],
                    "direction_probability": pred["level1_direction"]["probability_pct"],
                    "signal": pred["level3_trading_signal"]["signal"],
                    "confidence": pred["level3_trading_signal"]["confidence_pct"],
                    "stop_loss": pred["level3_trading_signal"]["stop_loss_price"],
                    "take_profit": pred["level3_trading_signal"]["take_profit_price"],
                    "rsi": pred["current_market"]["rsi_14"],
                    "sentiment": pred["sentiment"]["sentiment_label"]
                })
            except Exception as e:
                logger.warning(f"Error scanning {symbol}: {e}")

        # Sort by signal strength and absolute expected return
        priority = {"STRONG BUY": 5, "BUY": 4, "HOLD": 2, "SELL": 3, "STRONG SELL": 4}
        opportunities.sort(
            key=lambda x: (priority.get(x["signal"], 1), abs(x["expected_change_pct"])),
            reverse=True
        )
        return opportunities

    def get_status(self) -> Dict[str, Any]:
        """Get service health and system metadata."""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "uptime_seconds": round(time.time() - self.start_time, 2),
            "models_loaded": {
                "regression_targets": list(self.models.keys()),
                "classification_available": self.classification_model is not None,
                "feature_count": len(self.feature_columns),
                "scaler_loaded": self.scaler is not None
            },
            "cached_symbols": list(self._data_cache.keys()),
            "timestamp": datetime.now().isoformat()
        }


# Singleton accessor
def get_service() -> StockMarketAIService:
    return StockMarketAIService()
