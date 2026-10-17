"""
StockMarketAI: Automated Trade Execution Engine
===============================================
Binds AI signals directly to broker gateways with automated risk management.
Automatic execution is controlled by a master boolean flag (auto_execution_enabled),
which can be toggled on/off via UI button or REST API.
"""

import logging
from typing import Dict, Any, Optional
from src.stockmarketai.service import get_service
from src.stockmarketai.brokers.base import (
    BaseBrokerConnector,
    OrderSide,
    OrderType,
    OrderResult
)
from src.stockmarketai.brokers.angel_one import AngelOneConnector

logger = logging.getLogger("stockmarketai.execution")


class AutomatedTradeExecutionEngine:
    """
    Automated execution engine that monitors AI trading recommendations
    and executes orders through a designated broker connector with risk filters.
    
    Controlled by a boolean toggle: auto_execution_enabled.
    When False (default/safe): All automated orders are blocked.
    When True: AI signals meeting risk criteria are sent to the broker.
    """

    def __init__(
        self,
        broker: Optional[BaseBrokerConnector] = None,
        allocated_capital: float = 10000.0,
        risk_per_trade: float = 0.02, # 2% risk per trade default
        auto_execution_enabled: bool = False, # Master boolean switch
        dry_run: bool = False
    ):
        self.broker = broker if broker is not None else AngelOneConnector(is_paper_trading=True)
        self.allocated_capital = allocated_capital
        self.risk_per_trade = risk_per_trade
        self.auto_execution_enabled = auto_execution_enabled
        self.dry_run = dry_run
        self.service = get_service()

        if hasattr(self.broker, "connect") and not getattr(self.broker, "_connected", False):
            self.broker.connect()

    def set_auto_execution(self, enabled: bool) -> bool:
        """
        Master toggle controlled by the button on the UI or API.
        Enables or disables automated broker order execution.
        """
        self.auto_execution_enabled = bool(enabled)
        logger.info(f"Master Auto-Execution switch changed: auto_execution_enabled={self.auto_execution_enabled}")
        return self.auto_execution_enabled

    def set_broker(self, broker: BaseBrokerConnector):
        """Switch active broker connector (e.g. Angel One, INDmoney)."""
        self.broker = broker
        if not getattr(self.broker, "_connected", False):
            self.broker.connect()
        logger.info(f"Switched active broker to: {self.broker.__class__.__name__}")

    def get_status(self) -> Dict[str, Any]:
        """Return live execution status, boolean switch state, and broker info."""
        return {
            "auto_execution_enabled": self.auto_execution_enabled,
            "broker_name": self.broker.__class__.__name__,
            "is_paper_trading": getattr(self.broker, "is_paper_trading", True),
            "allocated_capital": self.allocated_capital,
            "risk_per_trade_pct": self.risk_per_trade * 100.0,
            "circuit_breaker_tripped": self.broker.circuit_breaker.is_tripped,
            "circuit_breaker_reason": self.broker.circuit_breaker.trip_reason,
            "open_positions": self.broker.get_positions(),
            "dry_run": self.dry_run
        }

    def evaluate_and_execute(self, symbol: str) -> Optional[OrderResult]:
        """
        1. Check master boolean switch (auto_execution_enabled).
        2. Predict stock movement (Level 1, Level 2, Level 3).
        3. Verify Circuit Breaker & Risk validation.
        4. Size position according to ATR stop distance.
        5. Route order to broker.
        """
        # --- Master Boolean Switch Check ---
        if not self.auto_execution_enabled:
            logger.warning(
                f"Auto-Execution is currently DISABLED (auto_execution_enabled=False). "
                f"Click the button on the dashboard to enable live automated execution."
            )
            return None

        pred = self.service.predict(symbol)
        l3 = pred["level3_trading_signal"]
        signal = l3["signal"]
        current_price = pred["current_market"]["close"]
        atr = pred["current_market"]["atr"]

        logger.info(f"Signal for {symbol}: {signal} (Confidence: {l3['confidence_pct']}%)")

        # Check circuit breaker
        open_pos = self.broker.get_positions()
        if not self.broker.circuit_breaker.validate_trade(l3, len(open_pos)):
            logger.info(f"Trade for {symbol} did not pass risk rules.")
            return None

        # Position Sizing
        # Risk amount = allocated_capital * risk_per_trade (e.g. $10,000 * 2% = $200)
        # Risk per share = 1.5 * ATR
        risk_amount = self.allocated_capital * self.risk_per_trade
        stop_dist = max(1.5 * atr, current_price * 0.01)
        quantity = max(1, int(risk_amount / stop_dist))

        if signal in ["STRONG BUY", "BUY"]:
            side = OrderSide.BUY
            stop_loss = l3["stop_loss_price"]
            take_profit = l3["take_profit_price"]

            if self.dry_run:
                logger.info(f"[DRY RUN] Would execute BUY {quantity} shares of {symbol} at ${current_price:.2f}")
                return None

            return self.broker.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

        elif signal in ["STRONG SELL", "SELL"]:
            # If we hold a position, close it
            side = OrderSide.SELL
            if any(p["symbol"] == symbol.upper() for p in open_pos):
                pos = next(p for p in open_pos if p["symbol"] == symbol.upper())
                qty_to_sell = pos["quantity"]

                if self.dry_run:
                    logger.info(f"[DRY RUN] Would execute SELL {qty_to_sell} shares of {symbol}")
                    return None

                return self.broker.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=qty_to_sell,
                    order_type=OrderType.MARKET,
                    price=current_price
                )

        logger.info(f"Signal is {signal}. No execution needed.")
        return None


# Global execution engine singleton
_global_engine: Optional[AutomatedTradeExecutionEngine] = None

def get_execution_engine() -> AutomatedTradeExecutionEngine:
    global _global_engine
    if _global_engine is None:
        _global_engine = AutomatedTradeExecutionEngine(
            auto_execution_enabled=False  # Safe default: starts in OFF state
        )
    return _global_engine
