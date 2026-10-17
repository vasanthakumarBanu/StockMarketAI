# StockMarketAI Project Continuation Guide

## Overview
This document serves as a continuation guide for the StockMarketAI project. It outlines the current status, steps taken, and instructions for resuming development.

--- 

## Current Status

### **Completed Tasks**
1. **Data Collection and Preprocessing**:
   - Created scripts to fetch live stock data using `yfinance`.
   - Updated preprocessing scripts to generate required technical indicators and features.
   - Successfully generated processed data with the correct features for model input.

2. **Model and Prediction Setup**:
   - Updated `prediction.py` to load the saved model using `joblib`.
   - Ensured the model path is correctly set for `stock_rf_model.pkl`.

3. **Feature Engineering**:
   - Adjusted feature engineering to match the model's expected input format (17 features).

### **Pending Tasks**
1. **Permissions Setup**:
   - Resolve permission issues for executing scripts in the `F:\IT\AI and ML\Data Science\Projects\StockMarketAI` directory.

2. **Execution**:
   - Run preprocessing script to generate processed data.
   - Execute prediction script to make predictions.

--- 

## Instructions for Resuming Development

### **Step 1: Set Permissions**
To execute scripts, ensure the following permissions are set:

1. **Open PowerShell as Administrator**:
   ```powershell
   Start-Process powershell -Verb RunAs
   ```

2. **Grant Full Control Permissions**:
   ```powershell
   $env:USERNAME
   icacls "F:\IT\AI and ML\Data Science\Projects\StockMarketAI" /grant "$env:USERNAME:(F)"
   icacls "F:\IT\AI and ML\Data Science\Projects\StockMarketAI" /T /grant "$env:USERNAME:(F)"
   ```

### **Step 2: Run Preprocessing Script**
Once permissions are set, run the preprocessing script to generate the processed data:

```powershell
python "F:\IT\AI and ML\Data Science\Projects\StockMarketAI\scripts\preprocessing.py"
```

### **Step 3: Run Prediction Script**
After preprocessing, run the prediction script to make predictions:

```powershell
python "F:\IT\AI and ML\Data Science\Projects\StockMarketAI\prediction.py"
```

### **Step 4: Verify Output**
- Check that the processed data file contains the expected 17 features.
- Ensure the prediction script outputs the predicted trend for the stock.

--- 

## Files and Directories

### **Key Files**
- **`load_stock_data.py`**: Fetches live stock data.
- **`preprocessing.py`**: Preprocesses data and generates features.
- **`prediction.py`**: Loads the model and makes predictions.
- **`scripts/feature_engineering.py`**: Helper functions for feature calculations.

### **Data Directories**
- **`data/raw/`**: Contains raw stock data files.
- **`data/processed/`**: Contains processed data files with engineered features.
- **`models/`**: Contains saved model files.

--- 

## Troubleshooting

### **Permission Issues**
- If permission issues persist, ensure you are running PowerShell as Administrator.
- Double-check that the path and command syntax are correct.

### **Feature Mismatch**
- Ensure the preprocessing script generates exactly 17 features as expected by the model.
- Verify the feature columns match those used during model training.

--- 

## Next Steps
1. **Permissions**: Set permissions as outlined above.
2. **Execution**: Run the preprocessing and prediction scripts.
3. **Verification**: Check the output and ensure everything works as expected.

--- 

## Contact
If you encounter any issues or need further assistance, feel free to reach out for help.

---