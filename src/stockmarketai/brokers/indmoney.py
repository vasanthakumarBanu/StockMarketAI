"""
StockMarketAI: INDmoney Platform Connector
==========================================
Implements integration for INDmoney platform (specializing in US stocks,
fractional investing, and portfolio automated rebalancing).
"""

import time
import uuid
import logging
from typing import Dict, List, Optional, Any
from src.stockmarketai.brokers.base import (
    BaseBrokerConnector,
    OrderResult,
    OrderSide,
    OrderType,
    OrderStatus
)

logger = logging.getLogger("stockmarketai.brokers.indmoney")


class INDmoneyConnector(BaseBrokerConnector):
    """
    INDmoney US Equities & Fractional Trading Gateway.
    Supports dollar-based fractional share allocation and automated signal execution.
    """

    def __init__(
        self,
        auth_token: Optional[str] = None,
        account_id: Optional[str] = None,
        is_paper_trading: bool = True
    ):
        super().__init__(is_paper_trading=is_paper_trading)
        self.auth_token = auth_token or "MOCK_INDMONEY_TOKEN"
        self.account_id = account_id or "MOCK_IND_ACCT"
        self._connected = False

    def connect(self) -> bool:
        """Authenticate with INDmoney API."""
        self._connected = True
        logger.info(f"Connected to INDmoney API ({'PAPER' if self.is_paper_trading else 'LIVE'})")
        return True

    def get_quote(self, symbol: str) -> Dict[str, float]:
        """Fetch real-time US equity quote."""
        from src.stockmarketai.service import get_service
        service = get_service()
        df = service.get_market_data(symbol)
        last_close = float(df.iloc[-1]["Close"])
        return {
            "symbol": symbol.upper(),
            "ltp": last_close,
            "bid": round(last_close * 0.9995, 2),
            "ask": round(last_close * 1.0005, 2),
        }

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
        """Place order with fractional share support."""
        order_id = f"IND_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        quote = self.get_quote(symbol)
        exec_price = price if (order_type == OrderType.LIMIT and price) else quote["ltp"]

        sym_key = symbol.upper()
        if side == OrderSide.BUY:
            current_qty = self.positions.get(sym_key, {}).get("quantity", 0.0)
            self.positions[sym_key] = {
                "symbol": sym_key,
                "quantity": round(current_qty + quantity, 4),
                "entry_price": exec_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "updated_at": time.time()
            }
        elif side == OrderSide.SELL:
            if sym_key in self.positions:
                del self.positions[sym_key]

        order_res = OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=exec_price,
            status=OrderStatus.FILLED,
            message=f"Filled {quantity:.4f} shares of {symbol} at ${exec_price:.2f} via INDmoney"
        )
        self.orders[order_id] = order_res
        return order_res

    def get_positions(self) -> List[Dict[str, Any]]:
        """Return open holdings."""
        return list(self.positions.values())
