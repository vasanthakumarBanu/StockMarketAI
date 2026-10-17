import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_live_stock_data(stock_name, days=365):
    """
    Fetch live stock data for a given stock name.

    Args:
        stock_name (str): Name of the stock (e.g., "AAPL").
        days (int): Number of days of historical data to fetch.

    Returns:
        pd.DataFrame: DataFrame containing historical stock data.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Fetch data from Yahoo Finance
    data = yf.download(stock_name, start=start_date, end=end_date)

    if data.empty:
        raise ValueError(f"No data found for {stock_name}")

    return data

def download_stock_data(symbol: str, period: str = "2y", save_dir: str = None) -> str:
    """
    Download stock data from Yahoo Finance, clean schema, and save to raw CSV.

    Args:
        symbol (str): Stock ticker symbol (e.g., "MSFT", "RELIANCE.NS").
        period (str): Historical period to download (default: "2y").
        save_dir (str, optional): Target directory to save CSV. Defaults to 'src/stockmarketai/data/raw'.

    Returns:
        str: Absolute or relative path to saved CSV file.
    """
    print(f"Downloading {symbol} data from Yahoo Finance (period: {period})...")
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)

    if hist.empty:
        raise ValueError(f"No data found for {symbol}")

    print(f"Downloaded {len(hist)} rows of data for {symbol}")

    # Reset index to have Date as column
    hist.reset_index(inplace=True)

    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in hist.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df = hist[required_cols].copy()

    # Handle timezone-aware datetime if present
    if hasattr(df['Date'].dt, 'tz') and df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        df['Date'] = pd.to_datetime(df['Date'])

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    if save_dir is None:
        # Detect project root if running from anywhere
        save_dir = os.path.join("src", "stockmarketai", "data", "raw")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{symbol.upper()}.csv")
    df.to_csv(save_path, index=False)

    print(f"Saved data to: {save_path}")
    if not df.empty and 'Date' in df:
        print(f"   Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

    return save_path


# Example usage
if __name__ == "__main__":
    stock_name = "AAPL"  # Default stock for testing
    try:
        stock_data = fetch_live_stock_data(stock_name)
        print(f"Successfully fetched data for {stock_name}")
        print(stock_data.head())

        # Use path relative to project root for saving the file
        save_path = os.path.join('src', 'stockmarketai', 'data', 'raw', f'{stock_name}_live.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        stock_data.to_csv(save_path)
        print(f"Data saved to {save_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()