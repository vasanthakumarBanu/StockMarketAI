#!/usr/bin/env python3
"""
StockMarketAI: Multi-Target OHLC Regression Engine
Predicts tomorrow's Open, High, Low, and Close values using quantitative technical indicators.
Consolidated, production-grade regression training and evaluation pipeline with dynamic data ingestion.
"""

import argparse
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suppress non-critical warnings
warnings.filterwarnings("ignore")


def engineer_17_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the 17 quantitative indicators from raw OHLCV market data."""
    df = raw_df.copy()

    if "Date" in df.columns:
        if hasattr(df["Date"].dt, "tz") and df["Date"].dt.tz is not None:
            df["Date"] = df["Date"].dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. Return and Volatility
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std()

    # 2. Moving Averages
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["MA_200"] = df["Close"].rolling(200).mean()

    # 3. RSI (14-period)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # 4. MACD (12, 26, 9)
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # 5. Bollinger Bands (20-period)
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    # 6. VWAP (Volume-Weighted Average Price - 20-period rolling)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vol_price = (typical_price * df["Volume"]).rolling(20).sum()
    cum_vol = df["Volume"].rolling(20).sum()
    df["VWAP"] = cum_vol_price / (cum_vol + 1e-9)
    df["VWAP_Distance"] = (df["Close"] - df["VWAP"]) / (df["VWAP"] + 1e-9)

    return df


def download_dynamic_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Dynamically download stock data from Yahoo Finance."""
    clean_symbol = symbol.strip().upper()
    print(f"Dynamically downloading {clean_symbol} data from Yahoo Finance (period: {period})...")

    ticker = yf.Ticker(clean_symbol)
    hist = ticker.history(period=period)

    if hist.empty:
        raise ValueError(f"No market data found for symbol '{clean_symbol}'")

    print(f"Downloaded {len(hist)} rows for {clean_symbol}")
    hist.reset_index(inplace=True)

    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in hist.columns]
    if missing:
        raise ValueError(f"Missing columns from Yahoo Finance: {missing}")

    return hist[required_cols].copy()


def load_and_prepare_data(data_path: str = None, symbol: str = None, period: str = "2y"):
    """Load or dynamically download and prepare market data for OHLC multi-target regression."""
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    else:
        target_sym = symbol or "RELIANCE.NS"
        print(f"No static dataset file provided; dynamically fetching live data for {target_sym}...")
        raw_df = download_dynamic_data(target_sym, period=period)
        df = engineer_17_features(raw_df)

    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    if hasattr(df.index, "min") and hasattr(df.index.min(), "strftime"):
        print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

    # Create tomorrow's OHLC targets (shift by -1)
    df["Tomorrow_Open"] = df["Open"].shift(-1)
    df["Tomorrow_High"] = df["High"].shift(-1)
    df["Tomorrow_Low"] = df["Low"].shift(-1)
    df["Tomorrow_Close"] = df["Close"].shift(-1)

    # Drop the final row (has NaN targets due to shift)
    df = df[:-1]
    print(f"After creating tomorrow targets: {df.shape[0]} rows")

    # Define features (19 technical indicators including VWAP)
    feature_columns = [
        "Close", "High", "Low", "Open", "Volume",
        "Daily_Return", "Volatility_20",
        "MA_20", "MA_50", "MA_200",
        "RSI_14",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Middle", "BB_Upper", "BB_Lower",
        "VWAP", "VWAP_Distance"
    ]

    # Verify all feature columns are present
    missing_features = [f for f in feature_columns if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features in dataset: {missing_features}")

    # Prepare features matrix X
    X = df[feature_columns].copy()

    # Handle missing values (from rolling indicator lookbacks)
    missing_before = X.isnull().sum().sum()
    X = X.ffill().bfill()
    missing_after = X.isnull().sum().sum()
    print(f"Missing values handled (before: {missing_before}, after: {missing_after})")

    # Define targets
    targets = ["Tomorrow_Open", "Tomorrow_High", "Tomorrow_Low", "Tomorrow_Close"]
    y = df[targets].copy()

    print(f"Features: {len(feature_columns)}")
    print(f"Targets: {list(targets)}")

    return X, y, feature_columns, targets


def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str, target_name: str):
    """Train and evaluate an individual regression model on scaled features."""
    print(f"\nTraining {model_name} for {target_name}...")
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Compute evaluation metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    train_r2 = float(r2_score(y_train, y_pred_train))
    test_r2 = float(r2_score(y_test, y_pred_test))

    print(f"{model_name} Results for {target_name}:")
    print(f"   Train MAE: {train_mae:.4f}")
    print(f"   Test MAE:  {test_mae:.4f}")
    print(f"   Train RMSE: {train_rmse:.4f}")
    print(f"   Test RMSE:  {test_rmse:.4f}")
    print(f"   Train R2:   {train_r2:.4f}")
    print(f"   Test R2:    {test_r2:.4f}")

    return {
        "model": model,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "y_pred_test": y_pred_test
    }


def main():
    parser = argparse.ArgumentParser(description="StockMarketAI: Dynamic Regression Engine for OHLC Prediction")
    parser.add_argument(
        "--symbol",
        type=str,
        default="RELIANCE.NS",
        help="Stock ticker to dynamically download and train on (default: RELIANCE.NS)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional path to custom feature-engineered dataset CSV (bypasses dynamic download)"
    )
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        help="Data duration for dynamic download (default: 2y)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models_regression",
        help="Directory to save trained regression models and scalers (default: models_regression)"
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Temporal train/test split ratio (default: 0.8 for 80%% train / 20%% test)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("StockMarketAI: Dynamic OHLC Multi-Target Regression Engine")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load or dynamically download and prepare data
    X, y, feature_columns, targets = load_and_prepare_data(
        data_path=args.data,
        symbol=args.symbol,
        period=args.period
    )

    # 2. Temporal train/test split (no lookahead bias)
    split_idx = int(len(X) * args.split)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\nData split:")
    print(f"   Training set:  {X_train.shape[0]} samples")
    print(f"   Test set:      {X_test.shape[0]} samples")

    # 3. Standardize features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("OK: Features standardized (mean=0, std=1)")

    # 4. Model constructors
    model_types = {
        "Linear Regression": LinearRegression,
        "Random Forest Regressor": lambda: RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }

    all_results = {}

    # 5. Train each model candidate for each target
    for target_name in targets:
        print(f"\n{'='*50}")
        print(f"TARGET: {target_name}")
        print(f"{'='*50}")

        y_train_target = y_train[target_name]
        y_test_target = y_test[target_name]

        target_results = {}

        for model_name, model_constructor in model_types.items():
            model_instance = model_constructor()
            result = train_evaluate_model(
                model_instance,
                X_train_scaled,
                X_test_scaled,
                y_train_target,
                y_test_target,
                model_name,
                target_name
            )
            target_results[model_name] = result

        all_results[target_name] = target_results

    # 6. Select best model per OHLC target based on test R2
    print(f"\n{'='*70}")
    print("BEST MODEL SUMMARY (based on Test R2)")
    print(f"{'='*70}")

    best_models = {}
    for target_name in targets:
        best_model_name = None
        best_r2 = -np.inf

        for model_name in model_types.keys():
            r2 = all_results[target_name][model_name]["test_r2"]
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name

        best_models[target_name] = {
            "model_name": best_model_name,
            "model": all_results[target_name][best_model_name]["model"],
            "test_r2": all_results[target_name][best_model_name]["test_r2"],
            "test_mae": all_results[target_name][best_model_name]["test_mae"],
            "test_rmse": all_results[target_name][best_model_name]["test_rmse"]
        }

        print(f"{target_name}:")
        print(f"   Best Model: {best_model_name}")
        print(f"   Test R2:    {all_results[target_name][best_model_name]['test_r2']:.4f}")
        print(f"   Test MAE:   {all_results[target_name][best_model_name]['test_mae']:.4f}")
        print(f"   Test RMSE:  {all_results[target_name][best_model_name]['test_rmse']:.4f}")

    # 7. Persist best models, scaler, and metadata
    print(f"\nSaving best models and scaler to '{args.output_dir}'...")
    for target_name in targets:
        model_info = best_models[target_name]
        slug = model_info["model_name"].lower().replace(" ", "_")
        model_filename = os.path.join(args.output_dir, f"{target_name.lower()}_{slug}.pkl")
        joblib.dump(model_info["model"], model_filename)
        print(f"OK: Saved {target_name} model: {model_filename}")

    scaler_path = os.path.join(args.output_dir, "feature_scaler.pkl")
    features_path = os.path.join(args.output_dir, "feature_columns.pkl")
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_columns, features_path)
    print(f"OK: Saved scaler: {scaler_path}")
    print(f"OK: Saved feature list: {features_path}")

    # 8. Save comparison summary CSV
    summary_rows = []
    for target_name in targets:
        for model_name in model_types.keys():
            res = all_results[target_name][model_name]
            summary_rows.append({
                "Target": target_name,
                "Model": model_name,
                "Test_MAE": res["test_mae"],
                "Test_RMSE": res["test_rmse"],
                "Test_R2": res["test_r2"]
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(args.output_dir, "model_comparison_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"OK: Saved comparison summary: {summary_csv_path}")

    print(f"\n{'='*70}")
    print("REGRESSION MODELING COMPLETE!")
    print(f"{'='*70}")
    print(f"OK: All models and scalers saved in '{args.output_dir}/'")
    print("OK: Ready to predict tomorrow's OHLC values")
    print("=" * 70)


if __name__ == "__main__":
    main()
