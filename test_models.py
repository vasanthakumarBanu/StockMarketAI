import joblib
import numpy as np

# Load scaler and feature columns to get shape
scaler = joblib.load("models_regression/feature_scaler.pkl")
feature_cols = joblib.load("models_regression/feature_columns.pkl")
print(f"Number of features: {len(feature_cols)}")
print(f"Feature columns: {feature_cols}")

# Create a dummy input (random values) to test model differences
dummy = np.random.randn(1, len(feature_cols))
dummy_scaled = scaler.transform(dummy)

print("\nTesting model predictions on random input:")
for target in ['tomorrow_open', 'tomorrow_high', 'tomorrow_low', 'tomorrow_close']:
    model = joblib.load(f"models_regression/{target}_linear_regression.pkl")
    pred = model.predict(dummy_scaled)[0]
    print(f"{target}: {pred:.6f}")

# Also try using the first row of actual data from reliance_ml_ready.csv to see if they differ
import pandas as pd
df = pd.read_csv("src/stockmarketai/data/processed/reliance_ml_ready.csv", index_col="Date", parse_dates=True)
# Use last row
X_last = df[feature_cols].iloc[-1:].fillna(method='ffill').fillna(method='bfill')
X_last_scaled = scaler.transform(X_last)
print("\nPredictions using last row of reliance data:")
for target in ['tomorrow_open', 'tomorrow_high', 'tomorrow_low', 'tomorrow_close']:
    model = joblib.load(f"models_regression/{target}_linear_regression.pkl")
    pred = model.predict(X_last_scaled)[0]
    print(f"{target}: {pred:.6f}")