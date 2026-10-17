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