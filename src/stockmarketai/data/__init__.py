"""
StockMarketAI Data Module
========================

Handles data loading, preprocessing, and management.
"""

from .load_stock_data import fetch_live_stock_data
from .preprocessing import preprocess_stock_data

__all__ = [
    'fetch_live_stock_data',
    'preprocess_stock_data'
]
