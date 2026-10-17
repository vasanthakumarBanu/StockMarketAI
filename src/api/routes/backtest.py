"""
Backtesting Routes
"""

from fastapi import APIRouter, HTTPException, Query
from src.stockmarketai.service import get_service
from src.api.schemas import BacktestRequest, BacktestResponse

router = APIRouter(tags=["Backtesting"])


@router.get(
    "/api/v1/backtest/{symbol}",
    response_model=BacktestResponse,
    summary="Run Backtest for Stock Symbol"
)
def backtest_stock_get(
    symbol: str,
    strategy: str = Query("enhanced", description="Strategy: enhanced | direction | regression"),
    initial_capital: float = Query(10000.0, description="Starting capital in USD"),
    stop_loss: float = Query(0.03, description="Stop loss percentage"),
    take_profit: float = Query(0.06, description="Take profit percentage"),
    use_atr: bool = Query(True, description="Enable dynamic ATR stops")
):
    """
    Run an event-driven backtest simulation for any stock symbol:
    - Calculates Sharpe Ratio, Sortino Ratio, CAGR, Max Drawdown, Win Rate, Profit Factor.
    - Compares strategy equity curve directly with Buy & Hold benchmark.
    - Returns trade log history and execution points.
    """
    try:
        service = get_service()
        result = service.backtest(
            symbol=symbol,
            strategy=strategy,
            initial_capital=initial_capital,
            stop_loss=stop_loss,
            take_profit=take_profit,
            use_atr=use_atr
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtesting error for {symbol}: {str(e)}")


@router.post(
    "/api/v1/backtest",
    response_model=BacktestResponse,
    summary="Run Backtest with Custom Parameters (POST)"
)
def backtest_stock_post(req: BacktestRequest):
    """Execute backtest via POST body."""
    try:
        service = get_service()
        result = service.backtest(
            symbol=req.symbol,
            strategy=req.strategy,
            initial_capital=req.initial_capital,
            stop_loss=req.stop_loss,
            take_profit=req.take_profit,
            use_atr=req.use_atr
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtesting error for {req.symbol}: {str(e)}")
