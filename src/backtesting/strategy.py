"""
StockMarketAI: Strategy Module
Implements trading strategies leveraging ML model predictions (Direction & Price Targets).
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """Abstract Base Class for all ML-driven trading strategies."""

    def __init__(
        self,
        stop_loss_pct: float = 0.03,        # 3% stop loss default
        take_profit_pct: float = 0.06,      # 6% take profit default
        position_size_pct: float = 1.0,     # Allocate 100% of available capital
        max_holding_days: int = 15,         # Maximum holding period before forced review/exit
        use_atr_stops: bool = False,        # If true, stop-loss and take-profit scale with ATR
        atr_multiplier_stop: float = 2.0,   # ATR stop-loss multiplier
        atr_multiplier_profit: float = 3.0  # ATR take-profit multiplier
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_size_pct = min(max(position_size_pct, 0.05), 1.0)
        self.max_holding_days = max_holding_days
        self.use_atr_stops = use_atr_stops
        self.atr_multiplier_stop = atr_multiplier_stop
        self.atr_multiplier_profit = atr_multiplier_profit

    @abstractmethod
    def generate_signal(self, row: pd.Series, prev_row: pd.Series = None) -> dict:
        """
        Generate a trading signal for the current time step.

        Args:
            row: Feature & prediction row for current date.
            prev_row: Feature & prediction row for previous date.

        Returns:
            dict with:
                - action: 'BUY', 'SELL', 'HOLD'
                - confidence: float (0.0 to 1.0)
                - reason: str explanation
                - target_stop_loss_pct: optional dynamic stop loss
                - target_take_profit_pct: optional dynamic take profit
        """
        pass


class DirectionStrategy(BaseStrategy):
    """
    Strategy based on Level 1 Classification Model (Up vs. Down Probability).
    Buys when upward probability exceeds buy_threshold.
    Exits when upward probability falls below exit_threshold.
    """

    def __init__(
        self,
        buy_threshold: float = 0.53,
        exit_threshold: float = 0.47,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.buy_threshold = buy_threshold
        self.exit_threshold = exit_threshold

    def generate_signal(self, row: pd.Series, prev_row: pd.Series = None) -> dict:
        # Check available probability or direction column
        prob_up = row.get("direction_prob", row.get("pred_prob_up", None))
        if prob_up is None and "predicted_direction" in row:
            prob_up = 0.60 if row["predicted_direction"] == 1 else 0.40
        elif prob_up is None:
            # Fallback if only binary direction column is present
            prob_up = 0.50

        # Calculate dynamic ATR stops if requested
        dynamic_sl = self.stop_loss_pct
        dynamic_tp = self.take_profit_pct
        if self.use_atr_stops and "ATR_14" in row and "Close" in row and row["Close"] > 0:
            atr_pct = float(row["ATR_14"]) / float(row["Close"])
            dynamic_sl = atr_pct * self.atr_multiplier_stop
            dynamic_tp = atr_pct * self.atr_multiplier_profit

        if prob_up >= self.buy_threshold:
            confidence = (prob_up - 0.5) * 2.0  # Scales 0.5..1.0 to 0.0..1.0
            return {
                "action": "BUY",
                "confidence": min(max(confidence, 0.1), 1.0),
                "reason": f"P(Up) = {prob_up:.3f} >= threshold {self.buy_threshold:.2f}",
                "stop_loss_pct": dynamic_sl,
                "take_profit_pct": dynamic_tp
            }
        elif prob_up <= self.exit_threshold:
            confidence = (0.5 - prob_up) * 2.0
            return {
                "action": "SELL",
                "confidence": min(max(confidence, 0.1), 1.0),
                "reason": f"P(Up) = {prob_up:.3f} <= exit threshold {self.exit_threshold:.2f}",
                "stop_loss_pct": dynamic_sl,
                "take_profit_pct": dynamic_tp
            }
        else:
            return {
                "action": "HOLD",
                "confidence": 0.5,
                "reason": f"P(Up) = {prob_up:.3f} within neutral zone",
                "stop_loss_pct": dynamic_sl,
                "take_profit_pct": dynamic_tp
            }


class RegressionExpectedReturnStrategy(BaseStrategy):
    """
    Strategy based on Level 2 Regression Models (Tomorrow Close Prediction).
    Calculates expected return: (predicted_close - current_close) / current_close.
    Buys when expected return exceeds min_expected_return.
    """

    def __init__(
        self,
        min_expected_return: float = 0.005,  # 0.5% expected gain
        exit_expected_return: float = -0.002, # -0.2% expected loss
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_expected_return = min_expected_return
        self.exit_expected_return = exit_expected_return

    def generate_signal(self, row: pd.Series, prev_row: pd.Series = None) -> dict:
        current_close = float(row.get("Close", 0))
        pred_close = float(row.get("pred_Tomorrow_Close", row.get("predicted_close", current_close)))

        if current_close <= 0:
            return {"action": "HOLD", "confidence": 0.0, "reason": "Invalid close price"}

        expected_return = (pred_close - current_close) / current_close

        # Dynamic ATR stops
        dynamic_sl = self.stop_loss_pct
        dynamic_tp = self.take_profit_pct
        if self.use_atr_stops and "ATR_14" in row and current_close > 0:
            atr_pct = float(row["ATR_14"]) / current_close
            dynamic_sl = atr_pct * self.atr_multiplier_stop
            dynamic_tp = atr_pct * self.atr_multiplier_profit

        if expected_return >= self.min_expected_return:
            confidence = min(expected_return / (self.min_expected_return * 3.0), 1.0)
            return {
                "action": "BUY",
                "confidence": max(confidence, 0.2),
                "reason": f"Expected return {expected_return*100:+.2f}% >= {self.min_expected_return*100:+.2f}%",
                "stop_loss_pct": dynamic_sl,
                "take_profit_pct": dynamic_tp
            }
        elif expected_return <= self.exit_expected_return:
            return {
                "action": "SELL",
                "confidence": min(abs(expected_return) / 0.02, 1.0),
                "reason": f"Expected return {expected_return*100:+.2f}% <= {self.exit_expected_return*100:+.2f}%",
                "stop_loss_pct": dynamic_sl,
                "take_profit_pct": dynamic_tp
            }
        else:
            return {
                "action": "HOLD",
                "confidence": 0.5,
                "reason": f"Expected return {expected_return*100:+.2f}% in neutral zone",
                "stop_loss_pct": dynamic_sl,
                "take_profit_pct": dynamic_tp
            }


class EnhancedHybridStrategy(BaseStrategy):
    """
    Combines Direction Probability (Level 1) + Price Prediction (Level 2)
    with Volatility & ATR Filtering.
    """

    def __init__(
        self,
        min_direction_prob: float = 0.52,
        min_expected_return: float = 0.003, # 0.3% expected return
        exit_prob_threshold: float = 0.47,
        volatility_filter_std: float = 0.04, # Avoid trading when 20d volatility is abnormally high (> 4% daily)
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_direction_prob = min_direction_prob
        self.min_expected_return = min_expected_return
        self.exit_prob_threshold = exit_prob_threshold
        self.volatility_filter_std = volatility_filter_std

    def generate_signal(self, row: pd.Series, prev_row: pd.Series = None) -> dict:
        current_close = float(row.get("Close", 0))
        pred_close = float(row.get("pred_Tomorrow_Close", row.get("predicted_close", current_close)))
        prob_up = float(row.get("direction_prob", row.get("pred_prob_up", 0.50)))
        volatility_20 = float(row.get("Volatility_20", 0.0))

        if current_close <= 0:
            return {"action": "HOLD", "confidence": 0.0, "reason": "Invalid close price"}

        expected_return = (pred_close - current_close) / current_close

        # Check volatility filter (extreme volatility = danger / reduce trade risk)
        if volatility_20 > self.volatility_filter_std:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": f"High volatility regime ({volatility_20*100:.2f}% > {self.volatility_filter_std*100:.1f}%), staying in cash"
            }

        # Dynamic ATR stops
        dynamic_sl = self.stop_loss_pct
        dynamic_tp = self.take_profit_pct
        if self.use_atr_stops and "ATR_14" in row and current_close > 0:
            atr_pct = float(row["ATR_14"]) / current_close
            dynamic_sl = atr_pct * self.atr_multiplier_stop
            dynamic_tp = atr_pct * self.atr_multiplier_profit

        # Combined BUY condition:
        # Both direction probability AND expected return are positive
        if prob_up >= self.min_direction_prob and expected_return >= self.min_expected_return:
            confidence = (prob_up * 0.5) + (min(expected_return / 0.02, 1.0) * 0.5)
            return {
                "action": "BUY",
                "confidence": min(max(confidence, 0.2), 0.95),
                "reason": f"Consensus BUY: P(Up)={prob_up:.2f}, ExpRet={expected_return*100:+.2f}%",
                "stop_loss_pct": dynamic_sl,
                "take_profit_pct": dynamic_tp
            }
        # Combined EXIT/SELL condition:
        elif prob_up <= self.exit_prob_threshold or expected_return <= -self.min_expected_return:
            return {
                "action": "SELL",
                "confidence": max(1.0 - prob_up, 0.4),
                "reason": f"Consensus SELL: P(Up)={prob_up:.2f}, ExpRet={expected_return*100:+.2f}%",
                "stop_loss_pct": dynamic_sl,
                "take_profit_pct": dynamic_tp
            }
        else:
            return {
                "action": "HOLD",
                "confidence": 0.5,
                "reason": f"Neutral: P(Up)={prob_up:.2f}, ExpRet={expected_return*100:+.2f}%",
                "stop_loss_pct": dynamic_sl,
                "take_profit_pct": dynamic_tp
            }
