import joblib
import pandas as pd
import numpy as np

# Load the scaler and feature columns
scaler = joblib.load("models_regression/feature_scaler.pkl")
feature_cols = joblib.load("models_regression/feature_columns.pkl")

# Load the processed data and get the last row
df = pd.read_csv("src/stockmarketai/data/processed/reliance_ml_ready.csv", index_col="Date", parse_dates=True)
X_last = df[feature_cols].iloc[-1:].copy()

print("Last row feature values:")
for col in feature_cols:
    print(f"  {col}: {X_last[col].iloc[0]:.6f}")

# Handle any missing values (though there shouldn't be any in the last row)
X_last = X_last.ffill().bfill()

# Scale the features
X_last_scaled = scaler.transform(X_last)

print(f"\nShape of scaled features: {X_last_scaled.shape}")

# Load and test each model
targets = ['Tomorrow_Open', 'Tomorrow_High', 'Tomorrow_Low', 'Tomorrow_Close']
print("\nPredictions from each model:")

for target in targets:
    model = joblib.load(f"models_regression/{target.lower().replace(' ', '_')}_linear_regression.pkl")
    pred = model.predict(X_last_scaled)[0]
    print(f"  {target}: {pred:.6f}")

# Also let's check what the actual tomorrow values would be (from the Target column or by looking ahead)
# Since we shifted by -1 to create tomorrow's values, the last row's tomorrow values would be NaN
# Let's check the second to last row to see what today's values predict for tomorrow
print("\n" + "="*50)
print("Checking second to last row (which has tomorrow's actual values):")

X_second_last = df[feature_cols].iloc[-2:-1].copy()
X_second_last = X_second_last.ffill().bfill()
X_second_last_scaled = scaler.transform(X_second_last)

print("Second to last row feature values (today's features):")
for col in ['Open', 'High', 'Low', 'Close']:  # Just show OHLC for brevity
    print(f"  {col}: {X_second_last[col].iloc[0]:.6f}")

print("\nPredictions for second to last row:")
for target in targets:
    model = joblib.load(f"models_regression/{target.lower().replace(' ', '_')}_linear_regression.pkl")
    pred = model.predict(X_second_last_scaled)[0]
    print(f"  {target}: {pred:.6f}")

# What were the actual tomorrow values? (these would be in the original data shifted)
# For the second to last row, tomorrow's values are the last row's values
actual_tomorrow_open = df.iloc[-1]['Open']
actual_tomorrow_high = df.iloc[-1]['High']
actual_tomorrow_low = df.iloc[-1]['Low']
actual_tomorrow_close = df.iloc[-1]['Close']

print(f"\nActual tomorrow's values (from last row):")
print(f"  Tomorrow_Open:  {actual_tomorrow_open:.6f}")
print(f"  Tomorrow_High:  {actual_tomorrow_high:.6f}")
print(f"  Tomorrow_Low:   {actual_tomorrow_low:.6f}")
print(f"  Tomorrow_Close: {actual_tomorrow_close:.6f}")