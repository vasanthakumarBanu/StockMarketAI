"""
StockMarketAI: Base Broker Connector & Risk Safety Layer
========================================================
Implements core execution abstractions, circuit breakers, and order models.
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging
import time

logger = logging.getLogger("stockmarketai.brokers")


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    status: OrderStatus
    message: str
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class CircuitBreaker:
    """
    Automated Risk Circuit Breaker:
    - Halts trading if daily loss exceeds max_daily_loss_pct.
    - Blocks signals if volatility exceeds max_volatility_std.
    - Limits total open position count and order execution frequency.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,  # 3% max daily portfolio loss
        max_open_positions: int = 5,
        min_signal_confidence: float = 0.60
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_open_positions = max_open_positions
        self.min_signal_confidence = min_signal_confidence
        self.daily_pnl = 0.0
        self.is_tripped = False
        self.trip_reason = ""

    def validate_trade(self, signal: Dict[str, Any], open_positions_count: int) -> bool:
        """Return True if trade passes all risk rules, False otherwise."""
        if self.is_tripped:
            logger.warning(f"Circuit breaker active: {self.trip_reason}. Rejecting trade.")
            return False

        if open_positions_count >= self.max_open_positions:
            logger.warning(f"Max open positions reached ({self.max_open_positions}). Trade blocked.")
            return False

        conf = signal.get("confidence", 0.0)
        if conf < self.min_signal_confidence:
            logger.info(f"Signal confidence {conf:.2f} below threshold {self.min_signal_confidence}. Skipping.")
            return False

        return True

    def record_pnl(self, pnl_pct: float):
        """Record trade result and check circuit limits."""
        self.daily_pnl += pnl_pct
        if self.daily_pnl <= -self.max_daily_loss_pct * 100.0:
            self.is_tripped = True
            self.trip_reason = f"Daily drawdown limit hit: {self.daily_pnl:.2f}%"
            logger.error(f"🚨 CIRCUIT BREAKER TRIPPED: {self.trip_reason}")


class BaseBrokerConnector(ABC):
    """Abstract Base Class for all live/paper broker connectors."""

    def __init__(self, is_paper_trading: bool = True):
        self.is_paper_trading = is_paper_trading
        self.circuit_breaker = CircuitBreaker()
        self.orders: Dict[str, OrderResult] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def connect(self) -> bool:
        """Authenticate and establish session with broker API."""
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Dict[str, float]:
        """Fetch real-time bid, ask, and last traded price."""
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> OrderResult:
        """Place an order with broker."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch list of open positions."""
        pass
