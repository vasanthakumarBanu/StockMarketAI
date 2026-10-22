import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_rsi(series, window=14):
    """
    Calculate Relative Strength Index (RSI).

    Args:
        series (pd.Series): Price series.
        window (int): Lookback window for RSI calculation.

    Returns:
        pd.Series: RSI values.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_macd(series, slow=26, fast=12):
    """
    Calculate Moving Average Convergence Divergence (MACD).

    Args:
        series (pd.Series): Price series.
        slow (int): Slow period.
        fast (int): Fast period.

    Returns:
        pd.Series: MACD values.
    """
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    macd = ema_fast - ema_slow
    return macd

def engineer_all_features(df):
    """
    Implement ALL feature engineering steps from the workflow:
    Technical Indicators ? Candlestick Features ? Volume Features ? Lag Features
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data (Date, Open, High, Low, Close, Volume)
        
    Returns:
        pd.DataFrame: DataFrame with all engineered features plus tomorrow's targets
    """
    print("Engineering ALL features as per workflow...")
    data = df.copy()

    # ==================== TECHNICAL INDICATORS ====================
    print("  Calculating technical indicators...")

    # Basic price and volume (already in df)
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility_20'] = data['Daily_Return'].rolling(20).std()

    # Moving Averages (SMA and EMA)
    data['MA_20'] = data['Close'].rolling(20).mean()
    data['MA_50'] = data['Close'].rolling(50).mean()
    data['MA_200'] = data['Close'].rolling(200).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # RSI (14-period)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))

    # EMA (Exponential Moving Average) - Trend Component
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # MACD with EMA
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # Bollinger Bands (20-period)
    data['BB_Middle'] = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)

    # ATR (Average True Range) - Volatility Component
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift(1))
    low_close = abs(data['Low'] - data['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR_14'] = tr.rolling(14).mean()

    # ==================== CANDLESTICK FEATURES ====================
    print("  Calculating candlestick features...")

    # Body and wicks
    data['Body_Size'] = abs(data['Close'] - data['Open'])
    data['Upper_Wick'] = data['High'] - np.maximum(data['Open'], data['Close'])
    data['Lower_Wick'] = np.minimum(data['Open'], data['Close']) - data['Low']

    # Body to total range ratio
    data['Total_Range'] = data['High'] - data['Low']
    data['Body_to_Total_Ratio'] = np.where(data['Total_Range'] > 0,
                                          data['Body_Size'] / data['Total_Range'], 0)

    # Upper and lower wick ratios
    data['Upper_Wick_Ratio'] = np.where(data['Total_Range'] > 0,
                                       data['Upper_Wick'] / data['Total_Range'], 0)
    data['Lower_Wick_Ratio'] = np.where(data['Total_Range'] > 0,
                                       data['Lower_Wick'] / data['Total_Range'], 0)

    # Candlestick direction (1 for bullish, 0 for bearish)
    data['Is_Bullish'] = (data['Close'] > data['Open']).astype(int)

    # Body size relative to previous day
    data['Body_Size_Pct_Change'] = data['Body_Size'].pct_change()

    # ==================== VOLUME FEATURES ====================
    print("  Calculating volume features...")

    # Volume moving averages
    data['Volume_MA_5'] = data['Volume'].rolling(5).mean()
    data['Volume_MA_20'] = data['Volume'].rolling(20).mean()
    data['Volume_Ratio_5_20'] = np.where(data['Volume_MA_20'] > 0,
                                        data['Volume_MA_5'] / data['Volume_MA_20'], 1)

    # Price-Volume relationship
    data['PV_Trend'] = data['Daily_Return'] * np.log1p(data['Volume'])
    data['Volume_Price_Correlation'] = data['Close'].rolling(20).corr(data['Volume'])

    # Volume volatility
    data['Volume_Volatility'] = data['Volume'].rolling(20).std()

    # VWAP (Volume-Weighted Average Price - 20-period rolling)
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3.0
    cum_vol_price = (typical_price * data['Volume']).rolling(20).sum()
    cum_vol = data['Volume'].rolling(20).sum()
    data['VWAP'] = cum_vol_price / (cum_vol + 1e-9)
    data['VWAP_Distance'] = (data['Close'] - data['VWAP']) / (data['VWAP'] + 1e-9)

    # ==================== LAG FEATURES ====================
    print("  Calculating lag features...")

    # Lag OHLCV (1, 2, 3 periods)
    for lag in [1, 2, 3]:
        data[f'Open_Lag{lag}'] = data['Open'].shift(lag)
        data[f'High_Lag{lag}'] = data['High'].shift(lag)
        data[f'Low_Lag{lag}'] = data['Low'].shift(lag)
        data[f'Close_Lag{lag}'] = data['Close'].shift(lag)
        data[f'Volume_Lag{lag}'] = data['Volume'].shift(lag)

    # Lag returns
    for lag in [1, 2, 3, 5, 10]:
        data[f'Return_Lag{lag}'] = data['Daily_Return'].shift(lag)

    # Lag technical indicators (select key ones)
    for lag in [1, 2, 3]:
        data[f'RSI_14_Lag{lag}'] = data['RSI_14'].shift(lag)
        data[f'MACD_Lag{lag}'] = data['MACD'].shift(lag)
        data[f'Volatility_20_Lag{lag}'] = data['Volatility_20'].shift(lag)

    # Create tomorrow's OHLC targets (shift by -1) - this goes at the end
    data['Tomorrow_Open'] = data['Open'].shift(-1)
    data['Tomorrow_High'] = data['High'].shift(-1)
    data['Tomorrow_Low'] = data['Low'].shift(-1)
    data['Tomorrow_Close'] = data['Close'].shift(-1)

    print(f"  Total features engineered: {len(data.columns)}")

    return data

def generate_features_basic(stock_data):
    """
    Generate basic features from stock data (backward compatibility).
    
    Args:
        stock_data (pd.DataFrame): Stock data with columns like 'Close', 'Volume', etc.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # Ensure the data is numeric before applying rolling calculations
    stock_data = stock_data.apply(pd.to_numeric, errors='coerce')

    # Example features
    stock_data['SMA_5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['RSI'] = calculate_rsi(stock_data['Close'], window=14)
    stock_data['MACD'] = calculate_macd(stock_data['Close'])
    stock_data['Volume_Change'] = stock_data['Volume'].pct_change()

    # Drop rows with NaN values
    stock_data.dropna(inplace=True)

    return stock_data

# Example usage
if __name__ == "__main__":
    stock_name = "AAPL"
    try:
        # Try to load processed data first
        processed_path = os.path.join('src', 'stockmarketai', 'data', 'processed', f'{stock_name}_processed_live.csv')
        if os.path.exists(processed_path):
            stock_data = pd.read_csv(processed_path)
            features_data = engineer_all_features(stock_data)
        else:
            # Fallback to raw data
            raw_path = os.path.join('src', 'stockmarketai', 'data', 'raw', f'{stock_name}.csv')
            if os.path.exists(raw_path):
                stock_data = pd.read_csv(raw_path)
                features_data = engineer_all_features(stock_data)
            else:
                raise FileNotFoundError(f"No data found for {stock_name}")
                
        print(f"Successfully generated features for {stock_name}")
        print(f"Features shape: {features_data.shape}")
        print(f"Feature columns: {list(features_data.columns)}")
        
        # Save features
        features_path = os.path.join('src', 'stockmarketai', 'data', 'processed', f'{stock_name}_features.csv')
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        features_data.to_csv(features_path, index=False)
        print(f"Features saved to {features_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
