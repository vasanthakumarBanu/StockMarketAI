"""Sentiment Analysis Module for StockMarketAI.

Provides VADER-based sentiment scoring from FinViz news headlines.
Designed to integrate with the enhanced trading signal system for
sentiment-adjusted confidence and position sizing.

Features:
- FinViz news scraping (ticker-specific, with User-Agent rotation)
- VADER sentiment compound/pos/neu/neg scoring
- Sentiment score normalization (-1 to +1 range)
- Headline count tracking for confidence weighting
- Graceful error handling (placeholder scores on failure)
- Compatible with yfinance for additional stock data
"""

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)


# Ensure VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


# Module-level VADER analyzer (avoid re-downloading on every call)
_vader_analyzer = None


def _get_vader_analyzer():
    """ Lazily get or create the VADER sentiment analyzer. """
    global _vader_analyzer
    if _vader_analyzer is None:
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


# FinViz URL template
_FINVIZ_URL = 'https://finviz.com/quote.ashx?t='

# User-Agent rotation to reduce blocking risk
_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0',
]


def _get_random_user_agent():
    """Return a random User-Agent string from the rotation pool."""
    import random
    return random.choice(_USER_AGENTS)


def scrape_finviz_news(ticker, delay=1.0):
    """
    Scrape stock news headlines from FinViz for a given ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT'). Case-insensitive.
    delay : float, optional
        Seconds to sleep between requests (default 1.0). Increase to reduce
        FinViz blocking risk.

    Returns
    -------
    list of dict
        Each dict contains 'title', 'time', 'date' keys. Empty list if
        no news found or on error (caller should handle gracefully).

    Example
    -------
    >>> headlines = scrape_finviz_news('MSFT')
    >>> len(headline)
    5
    """
    try:
        url = _FINVIZ_URL + ticker.upper()
        user_agent = _get_random_user_agent()

        req = Request(url=url, headers={'User-Agent': user_agent})
        response = urlopen(req, timeout=15)

        html = BeautifulSoup(response, 'html.parser')
        news_table = html.find(id='news-table')

        if news_table is None:
            logger.warning(f"No news-table found for {ticker}; page structure may have changed.")
            return []

        headlines = []
        for row in news_table.find_all('tr'):
            anchor = row.find('a')
            if anchor is None or anchor.text.strip() == '':
                continue

            title = anchor.get_text(strip=True)

            # FinViz news table: first td is date/time, second td may be empty
            tds = row.find_all('td')
            if len(tds) < 1:
                continue

            date_time_text = tds[0].get_text(strip=True)
            date_time_parts = date_time_text.split()

            if len(date_time_parts) == 1:
                date = date_time_parts[0]
                time_str = ''
            else:
                date = date_time_parts[0]
                time_str = ' '.join(date_time_parts[1:])

            headlines.append({
                'title': title,
                'date': date,
                'time': time_str
            })

        # Respectful delay between requests
        time.sleep(delay)

        return headlines

    except Exception as e:
        logger.error(f"Error scraping FinViz news for {ticker}: {e}")
        return []


def compute_vader_sentiment(headlines, as_compound=True):
    """
    Compute VADER sentiment scores for a list of news headlines.

    Parameters
    ----------
    headlines : list of dict
        List of headline dicts from scrape_finviz_news(). Must contain
        'title' key. If list is empty/None, returns neutral scores.
    as_compound : bool, optional
        If True (default), return compound score in [-1, +1] range.
        If False, return dict with 'pos', 'neu', 'neg' keys in [0, 1] range.

    Returns
    -------
    dict
        Sentiment scores. Example:
        {'compound': 0.4532, 'pos': 0.654, 'neu': 0.302, 'neg': 0.044}
        Or if no headlines: {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}

    Example
    -------
    >>> headlines = [{'title': 'Apple beats earnings estimates'}, {'title': 'New product launch'}]
    >>> sentiment = compute_vader_sentiment(headlines)
    >>> sentiment['compound']
    0.6211
    """
    if not headlines:
        default = {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        if not as_compound:
            return {'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        return default

    analyzer = _get_vader_analyzer()

    # Combine all headline titles into one text for overall sentiment
    # (alternative: average individual headline scores)
    combined_title = ' '.join(h['title'] for h in headlines)

    # Get VADER scores
    vs = analyzer.polarity_scores(combined_title)

    # Normalize: ensure all keys present
    result = {
        'compound': round(float(vs.get('compound', 0.0)), 4),
        'pos': round(float(vs.get('pos', 0.0)), 4),
        'neu': round(float(vs.get('neu', 0.0)), 4),
        'neg': round(float(vs.get('neg', 0.0)), 4),
    }

    # Add headline count for confidence weighting
    result['headline_count'] = len(headlines)

    # Ensure valid ranges
    result['compound'] = max(-1.0, min(1.0, result['compound']))
    result['pos'] = max(0.0, min(1.0, result['pos']))
    result['neu'] = max(0.0, min(1.0, result['neu']))
    result['neg'] = max(0.0, min(1.0, result['neg']))

    if not as_compound:
        # Return only the traditional VADER keys (no compound)
        return {'pos': result['pos'], 'neu': result['neu'], 'neg': result['neg']}

    return result


def get_stock_sentiment(ticker, news_delay=1.0, as_compound=True):
    """
    Convenience function: scrape FinViz news and compute VADER sentiment.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    news_delay : float, optional
        Delay in seconds between FinViz requests (default 1.0).
    as_compound : bool, optional
        If True, return compound score in [-1, +1].
        If False, return dict with pos/neu/neg in [0, 1].

    Returns
    -------
    dict
        Sentiment scores with headline_count.

    Example
    -------
    >>> sentiment = get_stock_sentiment('MSFT')
    >>> sentiment['compound']
    0.1243
    >>> sentiment['headline_count']
    7
    """
    headlines = scrape_finviz_news(ticker, delay=news_delay)
    return compute_vader_sentiment(headlines, as_compound=as_compound)


def get_stock_info(ticker):
    """
    Get yfinance stock info (price, sector, industry) for a ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    dict
        Keys: 'price', 'sector', 'industry', 'market_cap', 'currency'.
        Values may be None if data unavailable.

    Example
    -------
    >>> info = get_stock_info('MSFT')
    >>> info['price']
    519.09
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        price = info.get('regularMarketPreviousClose') or info.get('regularMarketPrice')
        sector = info.get('sector')
        industry = info.get('industry')
        market_cap = info.get('marketCap')
        currency = info.get('currency', 'USD')

        return {
            'price': price,
            'sector': sector,
            'industry': industry,
            'market_cap': market_cap,
            'currency': currency,
        }
    except Exception as e:
        logger.error(f"Error getting yfinance info for {ticker}: {e}")
        return {
            'price': None,
            'sector': None,
            'industry': None,
            'market_cap': None,
            'currency': 'USD',
        }


def sentiment_to_factor(compound_score, weight=0.5):
    """
    Convert VADER compound score [-1, +1] to an adjustment factor.

    Parameters
    ----------
    compound_score : float
        VADER compound sentiment in [-1 (bearish), +1 (bullish)] range.
    weight : float, optional
        How strongly sentiment influences the adjustment [0, 1].
        - weight=0: sentiment has no effect (factor always 1.0)
        - weight=1: full sentiment influence
        - Default 0.5: moderate influence (recommended)

    Returns
    -------
    float
        Adjustment factor in [0.5, 1.5] range.
        - factor > 1.0: bullish (positive sentiment)
        - factor < 1.0: bearish (negative sentiment)
        - factor = 1.0: neutral sentiment

    Example
    -------
    >>> sentiment_to_factor(0.5, weight=0.5)
    1.25   # Moderately bullish adjusts by +25%
    >>> sentiment_to_factor(-0.5, weight=0.5)
    0.75   # Moderately bearish adjusts by -25%
    >>> sentiment_to_factor(0.0, weight=0.5)
    1.0    # Neutral: no adjustment
    """
    # Map [-1, +1] -> [0.5, 1.5] with weight control
    # compound=+1 → factor=1.5 (max bullish)
    # compound=-1 → factor=0.5 (max bearish)
    # compound=0  → factor=1.0 (neutral)
    # Use sigmoid-like mapping for smooth transitions

    # Clamp compound to valid range
    compound = max(-1.0, min(1.0, compound_score))

    # Linear mapping: -1 → 0.5, 0 → 1.0, +1 → 1.5
    unmapped = 0.5 * compound + 1.0  # maps -1→0.0, 0→1.0, +1→2.0

    # Apply weight: if weight=0, always 1.0; if weight=1, use unmapped
    # Interpolate: factor = 1.0 + weight * (unmapped - 1.0)
    factor = 1.0 + weight * (unmapped - 1.0)

    # Clamp to sensible bounds
    factor = max(0.5, min(1.5, factor))

    return round(float(factor), 4)


# ---------------------------------------------------------------------------
# Batch Processing (for portfolio sentiment)
# ---------------------------------------------------------------------------

def batch_sentiment(tickers, news_delay=1.0, max_workers=1, as_compound=True):
    """
    Compute sentiment for multiple tickers concurrently.

    Parameters
    ----------
    tickers : list of str
        Stock ticker symbols.
    news_delay : float, optional
        Delay between FinViz requests per ticker (default 1.0).
    max_workers : int, optional
        Max concurrent requests (default 1 = sequential for politeness).
    as_compound : bool, optional
        If True, include compound score.

    Returns
    -------
    dict
        Mapping: ticker → sentiment dict (as from get_stock_sentiment).

    Example
    -------
    >>> results = batch_sentiment(['AAPL', 'MSFT', 'TSLA'], news_delay=0.5)
    >>> results['MSFT']['compound']
    0.312
    """
    results = {}

    for ticker in tickers:
        try:
            sentiment = get_stock_sentiment(ticker, news_delay=news_delay,
                                            as_compound=as_compound)
            results[ticker.upper()] = sentiment
        except Exception as e:
            logger.error(f"Batch sentiment failed for {ticker}: {e}")
            # Insert neutral placeholder so callers don't crash
            results[ticker.upper()] = {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0,
                'headline_count': 0,
            }

        # Stagger requests to be kind to FinViz
        if news_delay > 0:
            time.sleep(news_delay)

    return results


# ---------------------------------------------------------------------------
# Module-Level Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python sentiment.py <TICKER> [--compound|--traditional]")
        sys.exit(1)

    ticker = sys.argv[1].upper()

    # Check for flags
    as_compound = True
    if '--traditional' in sys.argv:
        as_compound = False

    # Fetch sentiment
    sentiment = get_stock_sentiment(ticker, as_compound=as_compound)

    # Print result
    indent = 2
    print(f"Sentiment for {ticker}:")
    for key, value in sentiment.items():
        print(f"  {key}: {value}")

    # Also get stock info
    info = get_stock_info(ticker)
    if info['price'] is not None:
        print(f"\nPrice: ${info['price']:.2f} ({info['currency']})")
    if info['sector']:
        print(f"Sector: {info['sector']}")
    if info['industry']:
        print(f"Industry: {info['industry']}")