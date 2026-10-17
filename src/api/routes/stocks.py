"""
Stocks and Watchlist Routes
"""

from typing import List
from fastapi import APIRouter, HTTPException
from src.stockmarketai.service import get_service
from src.api.schemas import StockListResponse, ScanRequest, StockScanItem

router = APIRouter(tags=["Stocks & Scanner"])

DEFAULT_WATCHLIST = [
    {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Semiconductors"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Automotive / EV"},
    {"symbol": "AGYS", "name": "Agilysys Inc.", "sector": "Hospitality Software"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries (NSE)", "sector": "Conglomerate"},
]


@router.get("/api/v1/stocks", response_model=StockListResponse, summary="Get Supported / Configured Stocks")
def get_stock_list():
    """Returns the list of recommended and configured tickers for quick scanning."""
    return {
        "supported_stocks": DEFAULT_WATCHLIST,
        "total": len(DEFAULT_WATCHLIST)
    }


@router.get("/api/v1/stocks/search", summary="Search Indian & US Stocks by Name or Ticker")
def search_stocks(query: str, limit: int = 8):
    """
    Search for similar stocks across Indian (NSE/BSE) and US (NASDAQ/NYSE) markets.
    Accepts company names (e.g. 'SRM Energy', 'Tata Motors', 'Apple') or partial tickers.
    """
    try:
        service = get_service()
        return service.search_symbols(query, max_results=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stock search error: {str(e)}")


@router.post("/api/v1/stocks/scan", response_model=List[StockScanItem], summary="Batch Scan Stocks")
def scan_stocks(req: ScanRequest):
    """
    Scan multiple stock symbols simultaneously.
    Returns ranked trading signals and expected returns.
    """
    try:
        service = get_service()
        results = service.scan_watchlist(req.symbols)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scanning error: {str(e)}")
