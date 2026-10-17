"""
StockMarketAI: Broker Connectors & Automated Execution Package
"""
from src.stockmarketai.brokers.base import BaseBrokerConnector, OrderResult, OrderType, OrderSide, OrderStatus
from src.stockmarketai.brokers.angel_one import AngelOneConnector
from src.stockmarketai.brokers.indmoney import INDmoneyConnector
from src.stockmarketai.brokers.execution_engine import AutomatedTradeExecutionEngine

__all__ = [
    "BaseBrokerConnector",
    "OrderResult",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "AngelOneConnector",
    "INDmoneyConnector",
    "AutomatedTradeExecutionEngine"
]
