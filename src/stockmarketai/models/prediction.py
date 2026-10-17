# Removing pickle import since we'll use joblib instead
import pandas as pd
import os
import sys
import joblib

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import the preprocessing function
from stockmarketai.data.preprocessing import preprocess_stock_data

def predict_stock_trend(stock_name):
    """
    Predict stock trend using a saved model.

    Args:
        stock_name (str): Name of the stock (e.g., "AAPL").

    Returns:
        str: Prediction ("Increase" or "Decrease").
    """
    # Fetch and preprocess live data
    processed_data = preprocess_stock_data(stock_name)

    # Verify the number of features in the processed data
    print("DataFrame columns:", processed_data.columns.tolist())

    # Define the correct path to the model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'stockmarketai', 'models', 'stock_rf_model.pkl')

    # Load the model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Ensure the latest data point is used for prediction
    latest_data = processed_data.iloc[-1:]

    # Make prediction
    try:
        prediction = model.predict(latest_data)
        return "Increase" if prediction[0] == 1 else "Decrease"
    except Exception as e:
        print(f"Error making prediction: {e}")
        raise

# Example usage
if __name__ == "__main__":
    stock_name = "AAPL"
    try:
        prediction = predict_stock_trend(stock_name)
        print(f"Predicted Trend for {stock_name}: {prediction}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()