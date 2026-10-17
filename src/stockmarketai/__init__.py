"""
StockMarketAI Package
====================

A comprehensive stock market prediction and trading system.

Modules:
--------
data       : Data loading and preprocessing
features   : Feature engineering
models     : Machine learning models
pipelines  : Complete workflow implementations
trading    : Trading systems and strategies
cli        : Command-line interfaces
"""

from . import data
from . import features
from . import models
from . import pipelines
from . import trading
from . import cli

__all__ = [
    'data',
    'features', 
    'models',
    'pipelines',
    'trading',
    'cli'
]

__version__ = '1.0.0'
__author__ = 'StockMarketAI Team'
