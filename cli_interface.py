import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from stockmarketai.data.load_stock_data import fetch_live_stock_data
from stockmarketai.data.preprocessing import preprocess_stock_data
from stockmarketai.models.prediction import predict_stock_trend

def main():
    print("Welcome to StockMarketAI!")
    stock_name = input("Enter the stock name (e.g., AAPL, GOOGL): ")

    try:
        # Fetch live data
        stock_data = fetch_live_stock_data(stock_name)
        print(f"\nLive Data for {stock_name}:")
        print(stock_data.tail())

        # Preprocess data
        processed_data = preprocess_stock_data(stock_name)
        print(f"\nProcessed Data with Features:")
        print(processed_data.tail())

        # Predict trend
        prediction = predict_stock_trend(stock_name)
        print(f"\nPredicted Trend: {prediction}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()