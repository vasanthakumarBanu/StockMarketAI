import pandas as pd
import numpy as np
import os
import sys
try:
    from stockmarketai.data.load_stock_data import fetch_live_stock_data
except ImportError:
    from src.stockmarketai.data.load_stock_data import fetch_live_stock_data

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26):
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    return ema_fast - ema_slow

def preprocess_stock_data(stock_name, days=365):
    stock_data = fetch_live_stock_data(stock_name, days)
    stock_data = stock_data.apply(pd.to_numeric, errors='coerce')

    # Flatten multi-index columns if present (e.g., from yfinance download)
    if isinstance(stock_data.columns, pd.MultiIndex):
        # Take the first level of the multi-index (assuming we only have one stock)
        stock_data.columns = stock_data.columns.get_level_values(0)

    # Calculate features
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['Volatility_20'] = stock_data['Daily_Return'].rolling(window=20).std()
    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA_200'] = stock_data['Close'].rolling(window=200).mean()  # Added
    stock_data['RSI_14'] = calculate_rsi(stock_data['Close'], window=14)
    macd_line = calculate_macd(stock_data['Close'], fast=12, slow=26)
    stock_data['MACD'] = macd_line
    stock_data['MACD_Signal'] = macd_line.ewm(span=9, adjust=False).mean()  # Added
    stock_data['MACD_Hist'] = stock_data['MACD'] - stock_data['MACD_Signal']  # Added

    ma = stock_data['Close'].rolling(window=20).mean()
    std = stock_data['Close'].rolling(window=20).std()
    stock_data['BB_Middle'] = ma
    stock_data['BB_Upper'] = ma + (std * 2)
    stock_data['BB_Lower'] = ma - (std * 2)
    # Volume_Change is calculated but not used in the model, so we skip adding it to the dataframe
    # stock_data['Volume_Change'] = stock_data['Volume'].pct_change()

    stock_data.dropna(inplace=True)

    # Define the exact order of columns as expected by the model
    required_columns = [
        'Close', 'High', 'Low', 'Open', 'Volume',
        'Daily_Return', 'Volatility_20', 'MA_20', 'MA_50', 'MA_200',
        'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Middle', 'BB_Upper', 'BB_Lower'
    ]
    stock_data = stock_data[required_columns]

    return stock_data

if __name__ == "__main__":
    stock_name = "AAPL"
    try:
        processed_data = preprocess_stock_data(stock_name)
        print("Columns:", processed_data.columns.tolist())
        print(f"Number of columns: {len(processed_data.columns)}")
        print(f"Successfully processed data for {stock_name}")
        print(processed_data.head())
        save_path = os.path.join('src', 'stockmarketai', 'data', 'processed', f'{stock_name}_processed_live.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        processed_data.to_csv(save_path, index=False)
        print(f"Processed data saved to {save_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()