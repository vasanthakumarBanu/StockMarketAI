#!/usr/bin/env python3
"""
StockMarketAI: Automatic OHLC Prediction from Yahoo Finance
Demonstrates how to enter a stock name, auto-download data, engineer features,
and predict tomorrow's Open, High, Low, Close prices using regression models.
"""

import pandas as pd
import numpy as np
import joblib
import os
import yfinance as yf
from datetime import datetime, timedelta
import sys

def download_and_save_stock_data(symbol, period="2y"):
    """
    Download stock data from Yahoo Finance and save in our required format

    Args:
        symbol (str): Stock symbol (e.g., 'AGYS', 'MSFT', 'AAPL')
        period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')

    Returns:
        str: Path to saved CSV file, or None if failed
    """
    try:
        print(f"Downloading {symbol} data from Yahoo Finance (period: {period})...")

        # Download data
        ticker = yf.Ticker(symbol)
        # Get historical market data
        hist = ticker.history(period=period)

        if hist.empty:
            print(f"No data found for {symbol}")
            return None

        print(f"Downloaded {len(hist)} rows of data for {symbol}")

        # Reset index to have Date as column
        hist.reset_index(inplace=True)

        # Yahoo Finance returns columns: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
        # We need: Date, Open, High, Low, Close, Volume (in this order)
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

        # Check if we have all required columns
        missing_cols = [col for col in required_cols if col not in hist.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return None

        # Select and order columns correctly
        df = hist[required_cols].copy()

        # Ensure proper data types
        # Handle timezone-aware datetime if present
        if hasattr(df['Date'].dt, 'tz') and df['Date'].dt.tz is not None:
            # Convert to UTC then remove timezone for consistency
            df['Date'] = df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
        else:
            df['Date'] = pd.to_datetime(df['Date'])

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by date (oldest first)
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Save to our raw data directory
        raw_dir = "src/stockmarketai/data/raw/"
        os.makedirs(raw_dir, exist_ok=True)
        save_path = os.path.join(raw_dir, f"{symbol.upper()}.csv")

        # Save without index (we want Date as a column)
        df.to_csv(save_path, index=False)

        print(f"Saved data to: {save_path}")
        print(f"   Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

        return save_path

    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def engineer_features(df):
    """Engineer the EXACT 17 features the models expect"""
    data = df.copy()

    # Required features:
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility_20'] = data['Daily_Return'].rolling(20).std()

    data['MA_20'] = data['Close'].rolling(20).mean()
    data['MA_50'] = data['Close'].rolling(50).mean()
    data['MA_200'] = data['Close'].rolling(200).mean()

    # RSI (14-period)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # Bollinger Bands (20-period)
    data['BB_Middle'] = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)

    # Define the exact 17 features (in order)
    feature_columns = [
        'Close', 'High', 'Low', 'Open', 'Volume',
        'Daily_Return', 'Volatility_20',
        'MA_20', 'MA_50', 'MA_200',
        'RSI_14',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Middle', 'BB_Upper', 'BB_Lower'
    ]

    # Select and order features
    featured_data = data[feature_columns].copy()

    # Handle missing values (from rolling windows) - SAME AS IN TRAINING
    # Use forward fill then backward fill to handle NaNs from rolling windows
    featured_data = featured_data.ffill().bfill()

    return featured_data, feature_columns

def predict_ohlc(symbol):
    """Predict tomorrow's OHLC for a given stock symbol"""

    # Check if raw data exists
    raw_path = f"src/stockmarketai/data/raw/{symbol}.csv"
    if not os.path.exists(raw_path):
        return {
            'error': f"Raw data for {symbol} not found. Run download first.",
            'symbol': symbol
        }

    try:
        # Load model components (verify they exist)
        model_dir = "models_regression/"
        scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
        features_path = os.path.join(model_dir, "feature_columns.pkl")

        if not all(os.path.exists(p) for p in [scaler_path, features_path]):
            return {
                'error': "Regression model components not found. Please run phase3_regression.py first.",
                'symbol': symbol
            }

        scaler = joblib.load(scaler_path)
        feature_cols = joblib.load(features_path)

        # Load and preprocess raw data - now in correct Yahoo Finance format
        # Format: Date,Open,High,Low,Close,Volume
        df = pd.read_csv(raw_path)

        # Convert Date to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df['Date'] = df['Date'].dt.tz_localize(None)  # Remove timezone for consistency
        df.set_index('Date', inplace=True)

        # Convert numeric columns (should already be numeric from yfinance, but ensure)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Engineer the EXACT 17 features the model expects
        featured_data, feature_columns = engineer_features(df)

        # Prepare features for prediction
        X = featured_data[feature_cols]
        X_scaled = scaler.transform(X)

        # Load the four regression models
        model_files = {
            'Tomorrow_Open': 'tomorrow_open_linear_regression.pkl',
            'Tomorrow_High': 'tomorrow_high_linear_regression.pkl',
            'Tomorrow_Low': 'tomorrow_low_linear_regression.pkl',
            'Tomorrow_Close': 'tomorrow_close_linear_regression.pkl'
        }

        predictions = {}
        for target_name, model_file in model_files.items():
            model_path = os.path.join(model_dir, model_file)
            model = joblib.load(model_path)
            pred = model.predict(X_scaled)[-1]  # Get latest prediction
            predictions[target_name] = pred

        # Get latest date
        latest_date = featured_data.index[-1]

        result = {
            'symbol': symbol.upper(),
            'prediction_date': latest_date.strftime('%Y-%m-%d'),
            'tomorrow_open': float(predictions['Tomorrow_Open']),
            'tomorrow_high': float(predictions['Tomorrow_High']),
            'tomorrow_low': float(predictions['Tomorrow_Low']),
            'tomorrow_close': float(predictions['Tomorrow_Close']),
            'features_engineered': len(feature_columns),
            'data_points_used': len(featured_data)
        }

        return result

    except Exception as e:
        return {
            'error': str(e),
            'symbol': symbol.upper()
        }

def main():
    print("=" * 70)
    print("StockMarketAI: Auto-Download & Predict Tomorrow's OHLC from Yahoo Finance")
    print("=" * 70)
    print()

    # Show what's currently available
    raw_data_path = "src/stockmarketai/data/raw/"
    if os.path.exists(raw_data_path):
        existing_stocks = [f.replace('.csv', '') for f in os.listdir(raw_data_path) if f.endswith('.csv')]
        print(f"Currently available stocks: {', '.join(existing_stocks) if existing_stocks else 'None'}")
    else:
        print("No stocks currently available in raw data directory")

    print()

    # For demo purposes in automated environment, use MSFT
    # In real usage, this would come from user input or command line argument
    if len(sys.argv) > 1:
        user_input = sys.argv[1].strip().upper()
        print(f"Using stock symbol from command line: {user_input}")
    else:
        user_input = "MSFT"  # Default demo
        print(f"No symbol provided, using default: {user_input}")
        print("(In interactive use, you would be prompted to enter a stock symbol)")

    print()
    print(f"Processing {user_input}...")
    print("-" * 50)

    # Step 1: Download and save data
    csv_path = download_and_save_stock_data(user_input, period="2y")

    if csv_path is None:
        print("Failed to download data. Exiting.")
        return

    print()

    # Step 2: Run OHLC prediction
    print("Running OHLC prediction on downloaded data...")
    result = predict_ohlc(user_input)

    if 'error' not in result:
        print(f"OHLC Prediction successful for {result['symbol']}:")
        print(f"   Date: {result['prediction_date']}")
        print(f"   Tomorrow's Open:  ${result['tomorrow_open']:.2f}")
        print(f"   Tomorrow's High:  ${result['tomorrow_high']:.2f}")
        print(f"   Tomorrow's Low:   ${result['tomorrow_low']:.2f}")
        print(f"   Tomorrow's Close: ${result['tomorrow_close']:.2f}")
        print()
        print(f"   Features engineered: {result['features_engineered']}")
        print(f"   Data points used: {result['data_points_used']}")

        # Calculate expected change from today's close (if we have today's close)
        # We don't have today's close easily without re-reading, but we can approximate
        # by getting the last close from the raw data
        try:
            df_raw = pd.read_csv(csv_path)
            today_close = df_raw['Close'].iloc[-1]
            predicted_close = result['tomorrow_close']
            change = predicted_close - today_close
            pct_change = (change / today_close) * 100
            print(f"   Today's Close:    ${today_close:.2f}")
            print(f"   Expected Change:  ${change:.2f} ({pct_change:+.2f}%)")
        except:
            pass

    else:
        print(f"Prediction failed: {result['error']}")

    print()
    print("=" * 70)
    print("Process Complete!")
    print("To predict another stock's OHLC, run: python auto_predict_ohlc.py SYMBOL")
    print("Example: python auto_predict_ohlc.py AGYS")
    print("Data saved to: src/stockmarketai/data/raw/{SYMBOL}.csv")
    print("=" * 70)

if __name__ == "__main__":
    main()