#!/usr/bin/env python3
"""
StockMarketAI: Phase 3 Regression Model for OHLC Prediction
Predicts tomorrow's Open, High, Low, Close values using technical indicators
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import joblib
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up directories
os.makedirs("models_regression", exist_ok=True)

def load_and_prepare_data(data_path):
    """Load and prepare data for regression modeling"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

    # Create tomorrow's OHLC targets (shift by -1)
    df['Tomorrow_Open'] = df['Open'].shift(-1)
    df['Tomorrow_High'] = df['High'].shift(-1)
    df['Tomorrow_Low'] = df['Low'].shift(-1)
    df['Tomorrow_Close'] = df['Close'].shift(-1)

    # Remove last row (will have NaN targets)
    df = df[:-1]

    print(f"After creating tomorrow targets: {df.shape[0]} rows")

    # Define features (same 17 technical indicators as classification model)
    feature_columns = [
        'Close', 'High', 'Low', 'Open', 'Volume',
        'Daily_Return', 'Volatility_20',
        'MA_20', 'MA_50', 'MA_200',
        'RSI_14',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Middle', 'BB_Upper', 'BB_Lower'
    ]

    # Verify all features exist
    missing_features = [f for f in feature_columns if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Prepare features and targets
    X = df[feature_columns].copy()

    # Handle any missing values (from rolling windows)
    print(f"Handling missing values...")
    missing_before = X.isnull().sum().sum()
    X = X.ffill().bfill()  # Forward fill then backward fill
    missing_after = X.isnull().sum().sum()
    print(f"Missing values before: {missing_before}, after: {missing_after}")

    # Define targets
    targets = ['Tomorrow_Open', 'Tomorrow_High', 'Tomorrow_Low', 'Tomorrow_Close']
    y = df[targets].copy()

    print(f"Features: {len(feature_columns)}")
    print(f"Targets: {list(targets)}")
    print(f"Feature names: {feature_columns}")

    return X, y, feature_columns, targets

def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, target_name):
    """Train and evaluate a regression model"""
    print(f"\nTraining {model_name} for {target_name}...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"{model_name} Results for {target_name}:")
    print(f"   Train MAE: {train_mae:.4f}")
    print(f"   Test MAE:  {test_mae:.4f}")
    print(f"   Train RMSE: {train_rmse:.4f}")
    print(f"   Test RMSE:  {test_rmse:.4f}")
    print(f"   Train R2:   {train_r2:.4f}")
    print(f"   Test R2:    {test_r2:.4f}")

    return {
        'model': model,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_pred_test': y_pred_test
    }

def main():
    print("=" * 70)
    print("StockMarketAI: Phase 3 Regression Model for OHLC Prediction")
    print("=" * 70)

    # Load and prepare data
    data_path = "src/stockmarketai/data/processed/reliance_ml_ready.csv"
    X, y, feature_columns, targets = load_and_prepare_data(data_path)

    # Split data (temporal split - important for time series)
    split_idx = int(len(X) * 0.8)  # 80% for training, 20% for testing
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\nData split:")
    print(f"   Training set:  {X_train.shape[0]} samples")
    print(f"   Test set:      {X_test.shape[0]} samples")

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("OK: Features standardized (mean=0, std=1)")

    # Define models to train
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }

    # Store results
    all_results = {}

    # Train models for each target
    for target_name in targets:
        print(f"\n{'='*50}")
        print(f"TARGET: {target_name}")
        print(f"{'='*50}")

        y_train_target = y_train[target_name]
        y_test_target = y_test[target_name]

        target_results = {}

        for model_name, model in models.items():
            result = train_evaluate_model(
                model, X_train_scaled, X_test_scaled,
                y_train_target, y_test_target,
                model_name, target_name
            )
            target_results[model_name] = result

        all_results[target_name] = target_results

    # Find best model for each target based on test R2
    print(f"\n{'='*70}")
    print("BEST MODEL SUMMARY (based on Test R2)")
    print(f"{'='*70}")

    best_models = {}
    for target_name in targets:
        best_model_name = None
        best_r2 = -np.inf

        for model_name in models.keys():
            r2 = all_results[target_name][model_name]['test_r2']
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name

        best_models[target_name] = {
            'model_name': best_model_name,
            'model': all_results[target_name][best_model_name]['model'],
            'test_r2': all_results[target_name][best_model_name]['test_r2'],
            'test_mae': all_results[target_name][best_model_name]['test_mae'],
            'test_rmse': all_results[target_name][best_model_name]['test_rmse']
        }

        print(f"{target_name}:")
        print(f"   Best Model: {best_model_name}")
        print(f"   Test R2:    {all_results[target_name][best_model_name]['test_r2']:.4f}")
        print(f"   Test MAE:   {all_results[target_name][best_model_name]['test_mae']:.4f}")
        print(f"   Test RMSE:  {all_results[target_name][best_model_name]['test_rmse']:.4f}")

    # Save the best models and scaler
    print(f"\nSaving models and scaler...")
    for target_name in targets:
        model_info = best_models[target_name]
        model_filename = f"models_regression/{target_name.lower()}_{model_info['model_name'].lower().replace(' ', '_')}.pkl"
        joblib.dump(model_info['model'], model_filename)
        print(f"OK: Saved {target_name} model: {model_filename}")

    # Save the scaler and feature columns
    joblib.dump(scaler, "models_regression/feature_scaler.pkl")
    joblib.dump(feature_columns, "models_regression/feature_columns.pkl")
    print("OK: Saved scaler: models_regression/feature_scaler.pkl")
    print("OK: Saved feature list: models_regression/feature_columns.pkl")

    # Save all results for reference
    results_summary = {
        'targets': targets,
        'feature_columns': feature_columns,
        'best_models': {k: {
            'model_name': v['model_name'],
            'test_r2': v['test_r2'],
            'test_mae': v['test_mae'],
            'test_rmse': v['test_rmse']
        } for k, v in best_models.items()},
        'all_results': all_results
    }

    # Save results as CSV for easy viewing
    summary_rows = []
    for target_name in targets:
        for model_name in models.keys():
            result = all_results[target_name][model_name]
            summary_rows.append({
                'Target': target_name,
                'Model': model_name,
                'Test_MAE': result['test_mae'],
                'Test_RMSE': result['test_rmse'],
                'Test_R2': result['test_r2']
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("models_regression/model_comparison_summary.csv", index=False)
    print("OK: Saved comparison summary: models_regression/model_comparison_summary.csv")

    print(f"\n{'='*70}")
    print("REGRESSION MODELING COMPLETE!")
    print(f"{'='*70}")
    print("OK: Models saved in models_regression/ directory")
    print("OK: Ready to predict tomorrow's OHLC values")
    print("OK: Use the same feature engineering process as classification model")
    print("=" * 70)

if __name__ == "__main__":
    main()