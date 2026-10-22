#!/usr/bin/env python3
"""
StockMarketAI: Consolidated Automatic Stock Prediction
Downloads live historical data dynamically from Yahoo Finance and performs:
  1. Level 1 Directional Forecast (Up/Down classification & confidence)
  2. Level 2 Price Target Forecast (Tomorrow's Open, High, Low, Close regression)
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import yfinance as yf

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def resolve_ticker(query: str) -> str:
    """Resolve natural company names or raw symbols to Yahoo Finance tickers."""
    try:
        from src.stockmarketai.data.search import resolve_symbol_or_search
        symbol, name = resolve_symbol_or_search(query)
        if name and name != symbol:
            print(f"Resolved '{query}' -> {symbol} ({name})")
        return symbol
    except Exception:
        return query.strip().upper()


def download_and_save_stock_data(symbol, period="2y", save_dir="src/stockmarketai/data/raw/"):
    """
    Download stock data dynamically from Yahoo Finance and save in the required format.

    Args:
        symbol (str): Stock ticker or resolved name (e.g., 'MSFT', 'AAPL', 'RELIANCE.NS')
        period (str): Data duration ('1mo', '3mo', '6mo', '1y', '2y', '5y')
        save_dir (str): Directory where CSV will be saved

    Returns:
        str: Path to saved CSV file, or None if failed
    """
    try:
        clean_symbol = symbol.strip().upper()
        print(f"Fetching live market data for {clean_symbol} from Yahoo Finance (period: {period})...")

        ticker = yf.Ticker(clean_symbol)
        hist = ticker.history(period=period)

        if hist.empty:
            print(f"[!] No market data found for symbol '{clean_symbol}'")
            return None

        print(f"Retrieved {len(hist)} historical bars for {clean_symbol}")

        hist.reset_index(inplace=True)

        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [c for c in required_cols if c not in hist.columns]
        if missing_cols:
            print(f"[!] Missing required columns from Yahoo Finance: {missing_cols}")
            return None

        df = hist[required_cols].copy()

        # Handle timezone-aware timestamps
        if hasattr(df['Date'].dt, 'tz') and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
        else:
            df['Date'] = pd.to_datetime(df['Date'])

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{clean_symbol}.csv")
        df.to_csv(save_path, index=False)

        print(f"Live data synchronized: {save_path}")
        print(f"   Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        return save_path

    except Exception as e:
        print(f"[!] Error downloading {symbol}: {e}")
        return None


def engineer_features(df):
    """
    Engineer the exact 17 quantitative indicators expected by the models.

    Args:
        df (pd.DataFrame): Raw OHLCV DataFrame with Date index or Date column.

    Returns:
        tuple[pd.DataFrame, list[str]]: (Featured DataFrame, list of feature column names)
    """
    data = df.copy()

    # Ensure Date is index
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], utc=True)
        data['Date'] = data['Date'].dt.tz_localize(None)
        data.set_index('Date', inplace=True)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # 1. Return and Volatility
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility_20'] = data['Daily_Return'].rolling(20).std()

    # 2. Moving Averages
    data['MA_20'] = data['Close'].rolling(20).mean()
    data['MA_50'] = data['Close'].rolling(50).mean()
    data['MA_200'] = data['Close'].rolling(200).mean()

    # 3. RSI (14-period)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data['RSI_14'] = 100 - (100 / (1 + rs))

    # 4. MACD (12, 26, 9)
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # 5. Bollinger Bands (20-period)
    data['BB_Middle'] = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)

    # 6. VWAP (Volume-Weighted Average Price - 20-period rolling)
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3.0
    cum_vol_price = (typical_price * data['Volume']).rolling(20).sum()
    cum_vol = data['Volume'].rolling(20).sum()
    data['VWAP'] = cum_vol_price / (cum_vol + 1e-9)
    data['VWAP_Distance'] = (data['Close'] - data['VWAP']) / (data['VWAP'] + 1e-9)

    # 19 standard feature columns (including VWAP & VWAP_Distance)
    feature_columns = [
        'Close', 'High', 'Low', 'Open', 'Volume',
        'Daily_Return', 'Volatility_20',
        'MA_20', 'MA_50', 'MA_200',
        'RSI_14',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Middle', 'BB_Upper', 'BB_Lower',
        'VWAP', 'VWAP_Distance'
    ]

    featured_data = data[feature_columns].copy()
    featured_data = featured_data.ffill().bfill()

    return featured_data, feature_columns


def predict_direction(symbol, featured_data=None, model_dir="models/"):
    """
    Predict directional price movement (Level 1 Classification: UP vs DOWN).

    Args:
        symbol (str): Stock symbol
        featured_data (pd.DataFrame, optional): Pre-engineered features
        model_dir (str): Directory containing classification models

    Returns:
        dict: Direction classification forecast results
    """
    clean_symbol = symbol.strip().upper()
    try:
        model_path = os.path.join(model_dir, "stock_lr_model.pkl")
        scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
        features_path = os.path.join(model_dir, "feature_columns.pkl")

        if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
            return {
                'error': f"Classification model files missing in '{model_dir}'. Run classification.py first.",
                'symbol': clean_symbol
            }

        if featured_data is None:
            raw_path = f"src/stockmarketai/data/raw/{clean_symbol}.csv"
            if not os.path.exists(raw_path):
                raw_path = download_and_save_stock_data(clean_symbol)
                if not raw_path or not os.path.exists(raw_path):
                    return {'error': f"Failed to retrieve dynamic data for {clean_symbol}.", 'symbol': clean_symbol}
            df = pd.read_csv(raw_path)
            featured_data, _ = engineer_features(df)

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_cols = joblib.load(features_path)

        X = featured_data[feature_cols]
        X_scaled = scaler.transform(X)

        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]

        latest_pred = int(predictions[-1])
        latest_prob = float(probabilities[-1])
        latest_date = featured_data.index[-1]
        up_probability = float(probabilities.mean())
        distribution = np.bincount(predictions)

        # Market sentiment categorization
        if latest_prob >= 0.70:
            sentiment = "Strong Bullish"
        elif latest_prob >= 0.55:
            sentiment = "Moderately Bullish"
        elif latest_prob >= 0.45:
            sentiment = "Neutral / Uncertain"
        elif latest_prob >= 0.30:
            sentiment = "Moderately Bearish"
        else:
            sentiment = "Strong Bearish"

        return {
            'symbol': clean_symbol,
            'latest_date': latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date),
            'latest_prediction': latest_pred,
            'latest_probability': latest_prob,
            'prediction_label': 'UP' if latest_pred == 1 else 'DOWN / NO MOVE',
            'confidence': f"{latest_prob * 100:.1f}%",
            'market_sentiment': sentiment,
            'overall_up_probability': f"{up_probability * 100:.1f}%",
            'prediction_distribution': {
                'down': int(distribution[0]) if len(distribution) > 0 else 0,
                'up': int(distribution[1]) if len(distribution) > 1 else 0
            },
            'total_samples': len(predictions),
            'data_points_used': len(featured_data),
            'features_engineered': len(feature_cols)
        }

    except Exception as e:
        return {'error': str(e), 'symbol': clean_symbol}


def predict_ohlc(symbol, featured_data=None, raw_df=None, model_dir="models_regression/"):
    """
    Predict tomorrow's Open, High, Low, Close prices (Level 2 Multi-Target Regression).

    Args:
        symbol (str): Stock symbol
        featured_data (pd.DataFrame, optional): Pre-engineered features
        raw_df (pd.DataFrame, optional): Raw OHLCV DataFrame
        model_dir (str): Directory containing regression models

    Returns:
        dict: OHLC regression price forecast results
    """
    clean_symbol = symbol.strip().upper()
    try:
        scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
        features_path = os.path.join(model_dir, "feature_columns.pkl")

        if not all(os.path.exists(p) for p in [scaler_path, features_path]):
            return {
                'error': f"Regression model files missing in '{model_dir}'. Run regression.py first.",
                'symbol': clean_symbol
            }

        if featured_data is None or raw_df is None:
            raw_path = f"src/stockmarketai/data/raw/{clean_symbol}.csv"
            if not os.path.exists(raw_path):
                raw_path = download_and_save_stock_data(clean_symbol)
                if not raw_path or not os.path.exists(raw_path):
                    return {'error': f"Failed to retrieve dynamic data for {clean_symbol}.", 'symbol': clean_symbol}
            raw_df = pd.read_csv(raw_path)
            featured_data, _ = engineer_features(raw_df)

        scaler = joblib.load(scaler_path)
        feature_cols = joblib.load(features_path)

        X = featured_data[feature_cols]
        X_scaled = scaler.transform(X)

        model_files = {
            'Tomorrow_Open': 'tomorrow_open_linear_regression.pkl',
            'Tomorrow_High': 'tomorrow_high_linear_regression.pkl',
            'Tomorrow_Low': 'tomorrow_low_linear_regression.pkl',
            'Tomorrow_Close': 'tomorrow_close_linear_regression.pkl'
        }

        predictions = {}
        for target_name, filename in model_files.items():
            path = os.path.join(model_dir, filename)
            if not os.path.exists(path):
                return {'error': f"Missing model file: {path}", 'symbol': clean_symbol}
            model = joblib.load(path)
            pred = model.predict(X_scaled)[-1]
            predictions[target_name] = float(pred)

        latest_date = featured_data.index[-1]
        today_close = float(raw_df['Close'].iloc[-1])
        predicted_close = predictions['Tomorrow_Close']
        change = predicted_close - today_close
        pct_change = (change / (today_close + 1e-9)) * 100
        vwap_val = float(featured_data['VWAP'].iloc[-1]) if 'VWAP' in featured_data.columns else today_close
        vwap_dist = float(featured_data['VWAP_Distance'].iloc[-1]) if 'VWAP_Distance' in featured_data.columns else 0.0

        return {
            'symbol': clean_symbol,
            'prediction_date': latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date),
            'today_close': today_close,
            'vwap': vwap_val,
            'vwap_distance': vwap_dist,
            'tomorrow_open': predictions['Tomorrow_Open'],
            'tomorrow_high': predictions['Tomorrow_High'],
            'tomorrow_low': predictions['Tomorrow_Low'],
            'tomorrow_close': predicted_close,
            'expected_change': change,
            'expected_pct_change': pct_change,
            'features_engineered': len(feature_cols),
            'data_points_used': len(featured_data)
        }

    except Exception as e:
        return {'error': str(e), 'symbol': clean_symbol}


def auto_predict_stock(symbol, mode="both", period="2y", use_cache=False):
    """
    Unified end-to-end stock prediction pipeline.
    Dynamically resolves symbol, downloads live data, computes features,
    and runs Direction Classification and/or OHLC Regression.

    Args:
        symbol (str): Stock symbol or company name (e.g. 'MSFT', 'Tata Motors')
        mode (str): 'both', 'direction', or 'ohlc'
        period (str): Historical period to download ('2y', '1y', etc.)
        use_cache (bool): If True, re-uses local CSV if present

    Returns:
        dict: Consolidated results dictionary
    """
    resolved_sym = resolve_ticker(symbol)
    raw_path = f"src/stockmarketai/data/raw/{resolved_sym}.csv"

    if not use_cache or not os.path.exists(raw_path):
        csv_path = download_and_save_stock_data(resolved_sym, period=period)
        if csv_path is None:
            return {'error': f"Failed to download dynamic market data for {resolved_sym}", 'symbol': resolved_sym}
    else:
        print(f"Using local cached data: {raw_path}")

    # Load raw data and engineer features
    raw_df = pd.read_csv(raw_path)
    featured_df, _ = engineer_features(raw_df)

    results = {'symbol': resolved_sym}

    if mode in ['both', 'direction']:
        results['direction'] = predict_direction(resolved_sym, featured_data=featured_df)

    if mode in ['both', 'ohlc']:
        results['ohlc'] = predict_ohlc(resolved_sym, featured_data=featured_df, raw_df=raw_df)

    return results


# Backward compatibility alias
def predict_stock_from_file(symbol):
    """Backward-compatible alias for directional prediction."""
    return predict_direction(symbol)


def print_prediction_report(results, mode="both"):
    """Print a clean, formatted report of the prediction results."""
    symbol = results.get('symbol', 'UNKNOWN')

    if 'error' in results:
        print(f"[!] Error predicting {symbol}: {results['error']}")
        return

    print("=" * 70)
    print(f"StockMarketAI: Consolidated Forecast Report for {symbol}")
    print("=" * 70)

    # 1. Directional Prediction
    if 'direction' in results:
        d = results['direction']
        if 'error' in d:
            print(f"[!] Direction Forecast Failed: {d['error']}")
        else:
            print("\n[LEVEL 1] DIRECTIONAL FORECAST (Classification)")
            print(f"   Date:                   {d['latest_date']}")
            print(f"   Forecast Direction:     {d['prediction_label']}")
            print(f"   Model Confidence:       {d['confidence']}")
            print(f"   Market Sentiment:       {d['market_sentiment']}")
            print(f"   Historical Up Ratio:    {d['overall_up_probability']}")
            print(f"   Samples Evaluated:      {d['total_samples']}")

    # 2. OHLC Regression
    if 'ohlc' in results:
        o = results['ohlc']
        if 'error' in o:
            print(f"[!] OHLC Forecast Failed: {o['error']}")
        else:
            change_sign = "+" if o['expected_change'] >= 0 else ""
            print("\n[LEVEL 2] PRICE TARGET FORECAST (OHLC Regression)")
            print(f"   Forecast Date:          {o['prediction_date']}")
            print(f"   Today's Close:          ${o['today_close']:.2f}")
            if 'vwap' in o:
                vwap_val = o['vwap']
                vwap_dist = o.get('vwap_distance', 0) * 100
                vwap_bias = "Above VWAP (Institutional Support)" if o['today_close'] >= vwap_val else "Below VWAP (Institutional Resistance)"
                print(f"   20-Day VWAP:            ${vwap_val:.2f} ({vwap_dist:+.2f}% | {vwap_bias})")
            print(f"   Tomorrow's Open:        ${o['tomorrow_open']:.2f}")
            print(f"   Tomorrow's High:        ${o['tomorrow_high']:.2f}")
            print(f"   Tomorrow's Low:         ${o['tomorrow_low']:.2f}")
            print(f"   Tomorrow's Close:       ${o['tomorrow_close']:.2f}")
            print(f"   Expected Price Move:    {change_sign}${o['expected_change']:.2f} ({change_sign}{o['expected_pct_change']:.2f}%)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="StockMarketAI: Dynamic Directional & OHLC Stock Prediction"
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Stock ticker or company name (e.g., MSFT, AAPL, 'Tata Motors', RELIANCE.NS)"
    )
    parser.add_argument(
        "--mode",
        choices=["both", "direction", "ohlc"],
        default="both",
        help="Prediction mode: 'both' (default), 'direction', or 'ohlc'"
    )
    parser.add_argument(
        "--period",
        default="2y",
        help="Historical data duration (default: '2y')"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Use local cached data if available instead of fresh dynamic download"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive prompt mode"
    )

    args = parser.parse_args()

    if args.interactive:
        print("=" * 70)
        print("StockMarketAI: Interactive Dynamic Prediction Terminal")
        print("=" * 70)
        print("Enter 'quit' or 'exit' to terminate.")
        print()

        while True:
            try:
                user_sym = input("Enter stock symbol or company name: ").strip()
                if not user_sym:
                    continue
                if user_sym.lower() in ['quit', 'exit', 'q']:
                    print("Exiting.")
                    break
                res = auto_predict_stock(user_sym, mode=args.mode, period=args.period, use_cache=args.cache)
                print_prediction_report(res, mode=args.mode)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
        return

    # Non-interactive CLI invocation
    target_symbol = args.symbol
    if not target_symbol:
        if sys.stdin.isatty():
            try:
                target_symbol = input("Enter stock symbol or company name (e.g., MSFT, AAPL, RELIANCE.NS): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                return

        if not target_symbol:
            print("Usage: python auto_predict.py <SYMBOL> [--mode {both,direction,ohlc}] [--period 2y]")
            print("Example: python auto_predict.py NVDA")
            return

    res = auto_predict_stock(
        target_symbol,
        mode=args.mode,
        period=args.period,
        use_cache=args.cache
    )
    print_prediction_report(res, mode=args.mode)


if __name__ == "__main__":
    main()