"""
StockMarketAI: Universal Stock Search & Symbol Resolver Engine
=============================================================
Enables searching any company name or ticker across Indian (NSE, BSE)
and US (NASDAQ, NYSE, AMEX) markets, automatically resolving human-typed
company names (e.g. "SRM Energy Ltd" -> "SRMENERGY.BO") to valid tickers.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import yfinance as yf

logger = logging.getLogger("stockmarketai.search")

# Exchange to friendly market name mapping
EXCHANGE_MARKET_MAP = {
    "BSE": "India (BSE)",
    "NSI": "India (NSE)",
    "NSE": "India (NSE)",
    "NMS": "US (NASDAQ)",
    "NYQ": "US (NYSE)",
    "NGM": "US (NASDAQ)",
    "PCX": "US (NYSE Arca)",
    "BTS": "US (BATS)",
    "PNK": "US (OTC Markets)",
    "LSE": "UK (London)",
    "TOR": "Canada (TSX)",
}

# Country & Badge mapping
MARKET_BADGE_MAP = {
    "India (BSE)": "[IN-BSE]",
    "India (NSE)": "[IN-NSE]",
    "US (NASDAQ)": "[US-NASDAQ]",
    "US (NYSE)": "[US-NYSE]",
    "US (NYSE Arca)": "[US-NYSE]",
    "US (OTC Markets)": "[US-OTC]",
}

COUNTRY_MAP = {
    "India (BSE)": "India",
    "India (NSE)": "India",
    "US (NASDAQ)": "United States",
    "US (NYSE)": "United States",
    "US (NYSE Arca)": "United States",
    "US (OTC Markets)": "United States",
}


def search_stock_symbols(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """
    Search for stocks across Indian and US exchanges by company name or ticker query.
    
    Examples:
        search_stock_symbols("SRM Energy") -> [{'symbol': 'SRMENERGY.BO', 'name': 'SRM Energy Limited', ...}]
        search_stock_symbols("Tata Motors") -> [{'symbol': 'TMCV.NS', ...}, {'symbol': 'TATAMOTORS.NS', ...}]
        search_stock_symbols("Apple") -> [{'symbol': 'AAPL', ...}]
    """
    query = (query or "").strip()
    if not query:
        return []

    results: List[Dict[str, Any]] = []

    try:
        search_engine = yf.Search(query, max_results=max_results)
        quotes = getattr(search_engine, "quotes", []) or []

        for q in quotes:
            # We focus on equity, ETF, and index quotes
            q_type = q.get("quoteType", "").upper()
            if q_type not in ["EQUITY", "ETF", "INDEX", "MUTUALFUND", ""]:
                continue

            symbol = q.get("symbol", "").strip()
            if not symbol:
                continue

            short_name = q.get("shortname") or q.get("longname") or symbol
            long_name = q.get("longname") or short_name
            exch = q.get("exchange", "").upper()
            market = EXCHANGE_MARKET_MAP.get(exch, exch or "Global")
            badge = MARKET_BADGE_MAP.get(market, f"[{exch}]")
            country = COUNTRY_MAP.get(market, "Global")

            # Relevance score
            score = 0
            if exch in ["NSI", "NSE", "BSE"]:
                score += 80
            elif exch in ["NMS", "NYQ", "NGM", "PCX"]:
                score += 70
            elif exch in ["PNK", "OID"]:
                score += 10
            else:
                score += 30

            q_clean = query.upper()
            if symbol.upper() == q_clean:
                score += 100
            elif symbol.upper().startswith(q_clean):
                score += 40

            if short_name.upper().startswith(q_clean):
                score += 50
            elif q_clean in short_name.upper() or q_clean in long_name.upper():
                score += 30

            results.append({
                "symbol": symbol,
                "name": short_name,
                "full_name": long_name,
                "exchange": exch,
                "market": market,
                "badge": badge,
                "country": country,
                "type": q_type or "EQUITY",
                "_score": score
            })

        # Sort results by relevance score descending
        results.sort(key=lambda x: x.get("_score", 0), reverse=True)

    except Exception as e:
        logger.warning(f"Error during symbol search for '{query}': {e}")

    return results


def resolve_symbol_or_search(query: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Intelligently resolves a user query to a valid trading ticker.
    
    1. If query is already an exact ticker with no spaces (e.g. 'MSFT', 'AAPL', 'SRMENERGY.BO'):
       Returns the ticker directly.
    2. If query contains company names (e.g. 'SRM Energy Ltd', 'Tata Motors', 'State Bank of India'):
       Searches Yahoo Finance and selects the most relevant equity ticker.
    
    Returns:
        (resolved_ticker, metadata_dict)
    """
    raw_query = (query or "").strip()
    if not raw_query:
        return "MSFT", None

    clean_symbol = raw_query.upper()

    # If it's a short alphanumeric ticker without spaces or ends with .NS / .BO,
    # try testing if it directly downloads or has an obvious format
    is_direct_ticker = (
        (" " not in clean_symbol) and
        (len(clean_symbol) <= 12) and
        (clean_symbol.replace(".", "").isalnum())
    )

    if is_direct_ticker:
        # Check if direct ticker works
        try:
            t = yf.Ticker(clean_symbol)
            fast_info = getattr(t, "fast_info", None)
            if fast_info and hasattr(fast_info, "last_price") and fast_info.last_price is not None:
                return clean_symbol, {
                    "symbol": clean_symbol,
                    "name": clean_symbol,
                    "market": "Direct Ticker"
                }
        except Exception:
            pass

    # If it has spaces (e.g. "SRM Energy Ltd", "Tata Motors") or direct lookup wasn't obvious, search:
    matches = search_stock_symbols(raw_query, max_results=5)
    if matches:
        top_match = matches[0]
        resolved = top_match["symbol"]
        logger.info(f"Resolved company query '{raw_query}' to ticker '{resolved}' ({top_match.get('name')})")
        return resolved, top_match

    # Fallback to sanitized uppercase
    return clean_symbol, None
