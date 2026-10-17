"""
Prediction Routes
"""

from fastapi import APIRouter, HTTPException, Query
from src.stockmarketai.service import get_service
from src.api.schemas import PredictionRequest, PredictionResponse

router = APIRouter(tags=["Predictions"])


@router.get(
    "/api/v1/predict/{symbol}",
    response_model=PredictionResponse,
    summary="Get 3-Level Stock Prediction"
)
def predict_stock_get(
    symbol: str,
    use_sentiment: bool = Query(True, description="Enable FinViz/VADER news sentiment analysis")
):
    """
    Get 3-Level AI Stock Predictions:
    - **Level 1**: Direction (Up vs. Down) probability & confidence
    - **Level 2**: Tomorrow's predicted Open, High, Low, Close prices & changes
    - **Level 3**: AI Trading Recommendation (BUY/HOLD/SELL) with ATR stops & risk/reward
    """
    try:
        service = get_service()
        result = service.predict(symbol, use_sentiment=use_sentiment)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error for {symbol}: {str(e)}")


@router.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    summary="Post Prediction Request"
)
def predict_stock_post(req: PredictionRequest):
    """Generate prediction via POST body."""
    try:
        service = get_service()
        result = service.predict(req.symbol, use_sentiment=req.use_sentiment)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error for {req.symbol}: {str(e)}")
