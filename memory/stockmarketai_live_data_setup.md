# StockMarketAI Live Data Setup Progress

## Overview
This document summarizes the progress made in setting up the StockMarketAI project to fetch live stock data and make predictions.

## Steps Completed

### 1. **Data Collection and Preprocessing Scripts**
- **`load_stock_data.py`**: 
  - Created a script to fetch live stock data using the `yfinance` library.
  - Successfully fetches historical data for stocks like AAPL.

- **`preprocessing.py`**:
  - Updated to generate the required technical indicators and features (e.g., moving averages, RSI, Bollinger Bands, MACD).
  - Ensures the processed data contains exactly the features expected by the model.
  - Features include:
    - Daily returns
    - Volatility (20-day)
    - Moving Averages (MA_20, MA_50)
    - Relative Strength Index (RSI_14)
    - MACD
    - Bollinger Bands (BB_Middle, BB_Upper, BB_Lower)
    - Volume Change

### 2. **Model Loading and Prediction Scripts**
- **`prediction.py`**:
  - Updated to load the saved model using `joblib`.
  - Ensures the correct path for the model file (`stock_rf_model.pkl`).
  - Designed to predict stock trends based on processed data.

### 3. **Feature Matching**
- Ensured the preprocessing script generates exactly the 17 features expected by the RandomForestClassifier model.
- Adjusted the feature engineering to match the model's expected input format.

## Current Status
- **Permissions**: 
  - Permission issues were encountered while executing scripts. Steps to resolve permissions were outlined but not yet executed.

## Next Steps
- **Permissions Setup**: 
  - Set permissions for the directory and files using `icacls` or equivalent methods.
  - Ensure the scripts can be executed without permission errors.

- **Execution**: 
  - Run the preprocessing script to generate processed data with the correct features.
  - Execute the prediction script to make predictions using the model.

- **Verification**: 
  - Verify the processed data file contains the expected 17 features.
  - Ensure the prediction script outputs the predicted trend for the stock.

## Files Created and Updated
- **`load_stock_data.py`**: Updated to fetch live stock data.
- **`preprocessing.py`**: Updated to generate the correct features.
- **`prediction.py`**: Updated to load the model and make predictions.
- **`scripts/feature_engineering.py`**: Contains helper functions for feature calculations.

## Summary
The project is set up to fetch live stock data, preprocess it into the required features, and make predictions using a trained model. The next step involves resolving permission issues and executing the scripts to complete the setup.

---