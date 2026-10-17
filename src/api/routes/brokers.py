"""
Broker Execution & Master Auto-Trading Control Routes
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.stockmarketai.brokers.execution_engine import get_execution_engine
from src.stockmarketai.brokers.angel_one import AngelOneConnector
from src.stockmarketai.brokers.indmoney import INDmoneyConnector

router = APIRouter(tags=["Broker Execution Controls"])


class ToggleExecutionRequest(BaseModel):
    enabled: bool = Field(..., description="Master boolean flag controlling automated trade execution")
    broker: Optional[str] = Field("angel_one", description="Broker: angel_one | indmoney")
    paper_trading: Optional[bool] = Field(True, description="Whether to run in paper trading mode")


class ExecuteTradeRequest(BaseModel):
    symbol: str = Field(..., example="MSFT", description="Stock ticker symbol to evaluate & execute")


@router.get("/api/v1/brokers/execution-status", summary="Get Auto-Execution Status & Boolean Switch State")
def get_execution_status():
    """Returns the live state of the automated execution engine and master boolean switch."""
    engine = get_execution_engine()
    return engine.get_status()


@router.post("/api/v1/brokers/toggle-execution", summary="Toggle Auto-Execution Boolean Switch (Button Control)")
def toggle_auto_execution(req: ToggleExecutionRequest):
    """
    Controlled by the button on the UI:
    Toggles the master boolean value `auto_execution_enabled`.
    When True: Live/Paper automated orders can be routed to broker.
    When False: All orders are strictly blocked.
    """
    engine = get_execution_engine()

    # Switch broker connector if specified
    if req.broker == "indmoney":
        engine.set_broker(INDmoneyConnector(is_paper_trading=req.paper_trading))
    else:
        engine.set_broker(AngelOneConnector(is_paper_trading=req.paper_trading))

    # Set master boolean switch
    state = engine.set_auto_execution(req.enabled)

    return {
        "status": "success",
        "auto_execution_enabled": state,
        "message": f"Auto-Execution master switch is now {'ENABLED (ACTIVE)' if state else 'DISABLED (SAFE)'}",
        "broker": engine.broker.__class__.__name__,
        "is_paper_trading": req.paper_trading
    }


@router.post("/api/v1/brokers/execute-now", summary="Evaluate & Execute Order for Symbol")
def execute_trade_now(req: ExecuteTradeRequest):
    """
    Evaluates AI signal and routes order to broker ONLY if auto_execution_enabled == True.
    """
    engine = get_execution_engine()

    if not engine.auto_execution_enabled:
        return {
            "status": "blocked",
            "auto_execution_enabled": False,
            "message": "Auto-Execution is currently DISABLED. Click the Auto-Execution button to turn it ON before executing live trades."
        }

    try:
        order = engine.evaluate_and_execute(req.symbol)
        if order is None:
            return {
                "status": "no_action",
                "message": f"AI signal for {req.symbol} did not warrant a trade or was blocked by risk rules."
            }
        return {
            "status": "executed",
            "order": {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "price": order.price,
                "status": order.status,
                "message": order.message
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error for {req.symbol}: {str(e)}")
