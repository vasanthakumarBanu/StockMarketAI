#!/usr/bin/env python3
"""
Script to backtest the best-performing model for Tomorrow_Open prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load the best model and feature scaler
model_path = 'models_complete/tomorrow_open_linear_regression.pkl'
feature_scaler_path = 'models_complete/feature_scaler.pkl'
feature_columns_path = 'models_complete/feature_columns.pkl'

model = pickle.load(open(model_path, 'rb'))
feature_scaler = pickle.load(open(feature_scaler_path, 'rb'))
feature_columns = pickle.load(open(feature_columns_path, 'rb'))

# Load your dataset
# Assuming your dataset is in a CSV file named 'data.csv' in the data directory
# Replace 'data.csv' with your actual dataset filename
try:
    data = pd.read_csv('data/data.csv')
except FileNotFoundError:
    print("Error: Data file not found. Please ensure 'data.csv' exists in the 'data' directory.")
    exit()

# Check if the required columns are present in the dataset
missing_columns = [col for col in feature_columns if col not in data.columns]
if missing_columns:
    print(f"Error: Missing columns in the dataset: {missing_columns}")
    exit()

# Preprocess the data
X = data[feature_columns]
X_scaled = feature_scaler.transform(X)
y = data['Tomorrow_Open']

# Split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Actual Tomorrow Open Price', color='blue')
plt.plot(y_pred, label='Predicted Tomorrow Open Price', color='red', linestyle='--')
plt.title('Actual vs Predicted Tomorrow Open Price')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Save the plot
output_plot_path = 'visualizations/backtest_results.png'
plt.savefig(output_plot_path)
print(f"Plot saved to {output_plot_path}")

# Show the plot
plt.show()