"""
StockMarketAI: API Schemas (Pydantic V2)
========================================
Defines strict type models, field validations, and OpenAPI documentation schemas.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    version: str = Field(..., example="1.0.0")
    uptime_seconds: float = Field(..., example=120.45)
    models_loaded: Dict[str, Any]
    cached_symbols: List[str]
    timestamp: str


class CurrentMarketInfo(BaseModel):
    open: float = Field(..., example=519.09)
    high: float = Field(..., example=522.50)
    low: float = Field(..., example=513.66)
    close: float = Field(..., example=514.02)
    volume: int = Field(..., example=18400000)
    atr: float = Field(..., example=7.25)
    volatility_20: float = Field(..., example=0.014)
    rsi_14: float = Field(..., example=52.8)
    macd: float = Field(..., example=1.24)
    macd_signal: float = Field(..., example=0.98)
    ma_20: float = Field(..., example=510.4)
    ma_50: float = Field(..., example=498.2)


class Level1DirectionInfo(BaseModel):
    prediction: str = Field(..., example="UP")
    probability_up: float = Field(..., example=0.68)
    probability_pct: float = Field(..., example=68.0)
    confidence: float = Field(..., example=0.36)


class Level2PriceTargetsInfo(BaseModel):
    tomorrow_open: float = Field(..., example=514.76)
    tomorrow_high: float = Field(..., example=518.20)
    tomorrow_low: float = Field(..., example=512.10)
    tomorrow_close: float = Field(..., example=517.16)
    expected_change: float = Field(..., example=3.14)
    expected_change_pct: float = Field(..., example=0.61)
    interval_lower: float = Field(..., example=503.20)
    interval_upper: float = Field(..., example=531.12)


class Level3TradingSignalInfo(BaseModel):
    signal: str = Field(..., example="BUY")
    confidence: float = Field(..., example=0.74)
    confidence_pct: float = Field(..., example=74.0)
    expected_return_pct: float = Field(..., example=0.61)
    stop_loss_price: Optional[float] = Field(None, example=503.14)
    take_profit_price: Optional[float] = Field(None, example=532.14)
    risk_reward_ratio: float = Field(..., example=1.75)
    recommended_action: str = Field(..., example="BUY with stop-loss at $503.14 and take-profit at $532.14")


class SentimentInfo(BaseModel):
    score: float = Field(..., example=0.18)
    sentiment_label: str = Field(..., example="BULLISH")
    headlines_analyzed: int = Field(..., example=10)


class PredictionRequest(BaseModel):
    symbol: str = Field(..., example="MSFT", description="Stock ticker symbol (e.g. MSFT, AAPL, AGYS, TSLA)")
    use_sentiment: bool = Field(True, description="Whether to include news sentiment analysis")


class PredictionResponse(BaseModel):
    symbol: str = Field(..., example="MSFT")
    query_entered: Optional[str] = Field(None, example="SRM Energy Ltd", description="Original raw company name or ticker query entered")
    company_name: Optional[str] = Field(None, example="SRM Energy Limited", description="Official company name")
    market: Optional[str] = Field(None, example="India (BSE)", description="Exchange / market region")
    badge: Optional[str] = Field(None, example="[IN-BSE]", description="Market badge identifier")
    prediction_date: str = Field(..., example="2026-10-17")
    current_market: CurrentMarketInfo
    level1_direction: Level1DirectionInfo
    level2_price_targets: Level2PriceTargetsInfo
    level3_trading_signal: Level3TradingSignalInfo
    sentiment: SentimentInfo
    timestamp: float


class BacktestRequest(BaseModel):
    symbol: str = Field(..., example="MSFT", description="Stock ticker symbol to backtest")
    strategy: str = Field("enhanced", description="Strategy type: enhanced | direction | regression")
    initial_capital: float = Field(10000.0, description="Starting capital in USD", ge=100.0)
    stop_loss: float = Field(0.03, description="Stop loss percentage (e.g. 0.03 = 3%)", ge=0.005, le=0.20)
    take_profit: float = Field(0.06, description="Take profit percentage (e.g. 0.06 = 6%)", ge=0.01, le=0.50)
    use_atr: bool = Field(True, description="Enable dynamic ATR-based stops")


class TradeLogItem(BaseModel):
    entry_date: str
    exit_date: str
    action: str
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    exit_reason: str


class EquityCurvePoint(BaseModel):
    date: str
    strategy_equity: float
    benchmark_equity: float


class BacktestResponse(BaseModel):
    symbol: str
    strategy: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    benchmark_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    benchmark_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    model_stats: Dict[str, Any]
    equity_curve: List[EquityCurvePoint]
    recent_trades: List[TradeLogItem]


class ScanRequest(BaseModel):
    symbols: List[str] = Field(
        default=["AAPL", "MSFT", "TSLA", "NVDA", "AGYS"],
        description="List of ticker symbols to scan"
    )


class StockScanItem(BaseModel):
    symbol: str
    current_price: float
    predicted_close: float
    expected_change_pct: float
    direction: str
    direction_probability: float
    signal: str
    confidence: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    rsi: float
    sentiment: str


class StockListResponse(BaseModel):
    supported_stocks: List[Dict[str, str]]
    total: int
