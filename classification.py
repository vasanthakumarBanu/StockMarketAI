#!/usr/bin/env python3
"""
StockMarketAI: Directional Trend Classification Engine
Trains and evaluates machine learning classifiers (Logistic Regression & Random Forest)
to predict directional market movement (UP vs. DOWN) from quantitative features.
Production-grade classification pipeline with dynamic live data ingestion.
"""

import argparse
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Check for imblearn availability
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

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
    """Load or dynamically download and prepare features and classification target."""
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

    # Check for target column
    if "Target" not in df.columns:
        if "Tomorrow_Close" in df.columns and "Close" in df.columns:
            df["Target"] = (df["Tomorrow_Close"] > df["Close"]).astype(int)
        elif "Close" in df.columns:
            df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
            df = df[:-1]
        else:
            raise ValueError("Target column ('Target') could not be derived from dataset.")

    # Analyze target distribution
    target_mean = float(df["Target"].mean())
    print("\nTarget Distribution:")
    print(f"   - Class 0 (Down / Neutral): {(1.0 - target_mean) * 100:.1f}%")
    print(f"   - Class 1 (Up):             {target_mean * 100:.1f}%")

    # Standard 17 feature columns matching the quantitative system
    expected_17 = [
        "Close", "High", "Low", "Open", "Volume",
        "Daily_Return", "Volatility_20",
        "MA_20", "MA_50", "MA_200",
        "RSI_14",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Middle", "BB_Upper", "BB_Lower"
    ]

    if all(col in df.columns for col in expected_17):
        feature_columns = expected_17
    else:
        exclude_cols = {"Target", "Tomorrow_Open", "Tomorrow_High", "Tomorrow_Low", "Tomorrow_Close"}
        feature_columns = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_columns].copy()
    y = df["Target"].copy()

    # Handle missing values robustly
    missing_before = X.isnull().sum().sum()
    X = X.ffill().bfill()
    missing_after = X.isnull().sum().sum()
    print(f"Missing values handled (before: {missing_before}, after: {missing_after})")

    print(f"Features: {len(feature_columns)}")
    print(f"Samples:  {len(X)}")

    return X, y, feature_columns


def train_evaluate_classifier(model, X_train, X_test, y_train, y_test, model_name: str):
    """Train and evaluate an individual classification model."""
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    # Compute classification metrics
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_proba))

    print(f"{model_name} Results:")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   ROC AUC:   {auc:.4f}")

    return {
        "model": model,
        "accuracy": acc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": auc,
        "y_pred": y_pred,
        "y_proba": y_proba
    }


def main():
    parser = argparse.ArgumentParser(description="StockMarketAI: Dynamic Directional Trend Classification Engine")
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
        default="models",
        help="Directory to save trained classification models and scaler (default: models)"
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Temporal train/test split ratio (default: 0.8 for 80%% train / 20%% test)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("StockMarketAI: Dynamic Directional Trend Classification Engine")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load or dynamically download and prepare dataset
    X, y, feature_columns = load_and_prepare_data(
        data_path=args.data,
        symbol=args.symbol,
        period=args.period
    )

    # 2. Chronological temporal train/test split (no lookahead bias)
    split_idx = int(len(X) * args.split)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\nData split:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set:     {X_test.shape[0]} samples")

    # 3. Standardize features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("OK: Features standardized (mean=0, std=1)")

    # 4. Check for class imbalance & apply SMOTE if needed
    train_balance = float(y_train.mean())
    print(f"\nTraining set balance: {train_balance:.3f} ({train_balance * 100:.1f}% Class 1)")

    if abs(train_balance - 0.5) > 0.1 and HAS_SMOTE:
        print("Applying SMOTE oversampling to balance training classes...")
        smote = SMOTE(random_state=42)
        X_train_fit, y_train_fit = smote.fit_resample(X_train_scaled, y_train)
        print(f"Balanced training set: {X_train_fit.shape[0]} samples (added {X_train_fit.shape[0] - X_train_scaled.shape[0]} synthetic)")
    else:
        X_train_fit, y_train_fit = X_train_scaled, y_train
        print("No severe imbalance detected or SMOTE omitted - using original training data")

    # 5. Model Candidates
    use_balanced = abs(train_balance - 0.5) > 0.1 and not HAS_SMOTE
    classifiers = {
        "Logistic Regression": LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced" if use_balanced else None,
            n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced" if use_balanced else None,
            n_jobs=-1
        )
    }

    results = {}
    for name, clf in classifiers.items():
        res = train_evaluate_classifier(
            clf,
            X_train_fit,
            X_test_scaled,
            y_train_fit,
            y_test,
            name
        )
        results[name] = res

    # 6. Model Comparison Table
    print(f"\n{'='*70}")
    print("CLASSIFICATION MODEL COMPARISON")
    print(f"{'='*70}")
    summary_rows = []
    for name, res in results.items():
        summary_rows.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "F1-Score": res["f1_score"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "ROC_AUC": res["roc_auc"]
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Best model selection based on F1-Score
    best_row = summary_df.loc[summary_df["F1-Score"].idxmax()]
    best_model_name = best_row["Model"]
    print(f"\nBest Model based on F1-Score: {best_model_name} (F1: {best_row['F1-Score']:.4f}, ROC AUC: {best_row['ROC_AUC']:.4f})")

    # 7. Persist Models and Artifacts
    print(f"\nSaving models and artifacts to '{args.output_dir}'...")

    # Save both models for flexibility
    rf_path = os.path.join(args.output_dir, "stock_rf_model.pkl")
    lr_path = os.path.join(args.output_dir, "stock_lr_model.pkl")
    scaler_path = os.path.join(args.output_dir, "feature_scaler.pkl")
    features_path = os.path.join(args.output_dir, "feature_columns.pkl")
    summary_csv_path = os.path.join(args.output_dir, "classification_summary.csv")

    joblib.dump(results["Random Forest"]["model"], rf_path)
    joblib.dump(results["Logistic Regression"]["model"], lr_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_columns, features_path)
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"OK: Saved Random Forest model:     {rf_path}")
    print(f"OK: Saved Logistic Regression model: {lr_path}")
    print(f"OK: Saved feature scaler:           {scaler_path}")
    print(f"OK: Saved feature columns:          {features_path}")
    print(f"OK: Saved comparison summary:       {summary_csv_path}")

    print(f"\n{'='*70}")
    print("CLASSIFICATION MODELING COMPLETE!")
    print(f"{'='*70}")
    print(f"OK: Models and artifacts ready in '{args.output_dir}/'")
    print("=" * 70)


if __name__ == "__main__":
    main()
