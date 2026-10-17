#!/usr/bin/env python3
"""
StockMarketAI: Phase 4 Backtesting & Strategy Execution CLI
Runs out-of-sample and walk-forward backtests of ML models on any stock symbol,
evaluating trading performance against a Buy & Hold benchmark.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from complete_workflow_fixed import download_stock_data, engineer_all_features
from src.backtesting.strategy import (
    DirectionStrategy,
    RegressionExpectedReturnStrategy,
    EnhancedHybridStrategy
)
from src.backtesting.backtest import Backtester
from src.backtesting.metrics import format_metrics_table


def prepare_and_predict_out_of_sample(df: pd.DataFrame, train_split: float = 0.8):
    """
    Train models on first 80% and generate out-of-sample predictions on remaining 20%.
    Strictly avoids data leakage.
    """
    data = df.copy()

    # Direction target
    data['Tomorrow_Direction'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    # Clean missing / infinite values
    target_cols = ['Tomorrow_Open', 'Tomorrow_High', 'Tomorrow_Low', 'Tomorrow_Close', 'Tomorrow_Direction']
    feature_cols = [c for c in data.columns if c not in target_cols and c != 'Date']

    # Impute rolling NaNs
    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # Drop the final row where tomorrow targets are NaN
    data = data.iloc[:-1].dropna(subset=target_cols)

    split_idx = int(len(data) * train_split)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:].copy()

    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Train Classification Model (Direction)
    y_class_train = train_data['Tomorrow_Direction']
    y_class_test = test_data['Tomorrow_Direction']

    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_class_train)

    test_preds_class = clf.predict(X_test_scaled)
    test_probs = clf.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_class_test, test_preds_class)

    # 2. Train Regression Model (Tomorrow Close)
    y_reg_train = train_data['Tomorrow_Close']
    y_reg_test = test_data['Tomorrow_Close']

    reg = LinearRegression()
    reg.fit(X_train_scaled, y_reg_train)

    test_preds_reg = reg.predict(X_test_scaled)
    r2 = r2_score(y_reg_test, test_preds_reg)
    mae = mean_absolute_error(y_reg_test, test_preds_reg)

    # Attach predictions to test dataframe
    test_data['direction_prob'] = test_probs
    test_data['predicted_direction'] = test_preds_class
    test_data['pred_Tomorrow_Close'] = test_preds_reg

    model_stats = {
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "direction_accuracy": round(acc * 100.0, 2),
        "price_r2": round(r2, 4),
        "price_mae": round(mae, 2)
    }

    return test_data, model_stats


def prepare_and_predict_walk_forward(df: pd.DataFrame, window_size: int = 252, step_size: int = 20):
    """
    Walk-forward rolling training and out-of-sample prediction.
    Trains on window_size bars, predicts next step_size bars, rolls forward.
    """
    data = df.copy()
    data['Tomorrow_Direction'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    target_cols = ['Tomorrow_Open', 'Tomorrow_High', 'Tomorrow_Low', 'Tomorrow_Close', 'Tomorrow_Direction']
    feature_cols = [c for c in data.columns if c not in target_cols and c != 'Date']

    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    data = data.iloc[:-1].dropna(subset=target_cols)

    n_samples = len(data)
    if n_samples <= window_size + step_size:
        # Fall back to out-of-sample split if dataset is too short
        return prepare_and_predict_out_of_sample(df, train_split=0.7)

    all_test_slices = []

    for start_idx in range(0, n_samples - window_size, step_size):
        train_end = start_idx + window_size
        test_end = min(train_end + step_size, n_samples)

        train_slice = data.iloc[start_idx:train_end]
        test_slice = data.iloc[train_end:test_end].copy()

        if len(test_slice) == 0:
            break

        X_tr = train_slice[feature_cols]
        X_te = test_slice[feature_cols]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Classification
        clf = RandomForestClassifier(n_estimators=60, max_depth=5, random_state=42, n_jobs=-1)
        clf.fit(X_tr_s, train_slice['Tomorrow_Direction'])
        test_slice['direction_prob'] = clf.predict_proba(X_te_s)[:, 1]
        test_slice['predicted_direction'] = clf.predict(X_te_s)

        # Regression
        reg = LinearRegression()
        reg.fit(X_tr_s, train_slice['Tomorrow_Close'])
        test_slice['pred_Tomorrow_Close'] = reg.predict(X_te_s)

        all_test_slices.append(test_slice)

    combined_test = pd.concat(all_test_slices)

    acc = accuracy_score(combined_test['Tomorrow_Direction'], combined_test['predicted_direction'])
    r2 = r2_score(combined_test['Tomorrow_Close'], combined_test['pred_Tomorrow_Close'])
    mae = mean_absolute_error(combined_test['Tomorrow_Close'], combined_test['pred_Tomorrow_Close'])

    model_stats = {
        "train_samples": window_size,
        "test_samples": len(combined_test),
        "direction_accuracy": round(acc * 100.0, 2),
        "price_r2": round(r2, 4),
        "price_mae": round(mae, 2)
    }

    return combined_test, model_stats


def run_backtest_pipeline(
    symbol: str = "MSFT",
    period: str = "2y",
    mode: str = "test_split",
    strategy_type: str = "enhanced",
    initial_capital: float = 10000.0,
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.06,
    use_atr_stops: bool = False,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    save_plot: bool = True
) -> dict:
    """Execute complete backtesting pipeline for a stock symbol."""
    print("=" * 70)
    print(f"StockMarketAI: PHASE 4 BACKTESTING & STRATEGY FRAMEWORK")
    print(f"Target Symbol: {symbol.upper()} | Period: {period} | Mode: {mode}")
    print("=" * 70)

    # 1. Download & prepare data
    print(f"\n[1/4] Fetching market data for {symbol.upper()}...")
    raw_path = download_stock_data(symbol, period=period)
    df = pd.read_csv(raw_path)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    # 2. Feature Engineering
    print("\n[2/4] Engineering 65+ Technical, Candlestick, Volume & Lag Features...")
    df_features = engineer_all_features(df)

    # 3. Model Training & Out-of-sample Prediction
    print(f"\n[3/4] Generating Out-Of-Sample Predictions (Mode: {mode})...")
    if mode == "walk_forward":
        test_df, model_stats = prepare_and_predict_walk_forward(df_features)
    else:
        test_df, model_stats = prepare_and_predict_out_of_sample(df_features, train_split=0.8)

    print(f"  Out-Of-Sample Bars: {model_stats['test_samples']}")
    print(f"  Direction Model Accuracy: {model_stats['direction_accuracy']}%")
    print(f"  Price Model Test R2: {model_stats['price_r2']} | MAE: ${model_stats['price_mae']}")

    # 4. Strategy Initialization
    if strategy_type == "direction":
        strategy = DirectionStrategy(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            use_atr_stops=use_atr_stops
        )
    elif strategy_type == "regression":
        strategy = RegressionExpectedReturnStrategy(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            use_atr_stops=use_atr_stops
        )
    else:
        strategy = EnhancedHybridStrategy(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            use_atr_stops=use_atr_stops
        )

    print(f"\n[4/4] Executing Event-Driven Backtest ({strategy.__class__.__name__})...")
    backtester = Backtester(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate
    )

    result = backtester.run(test_df, strategy=strategy, symbol=symbol.upper())

    # Generate Visualization Plot
    plot_path = None
    if save_plot:
        plot_path = backtester.plot_results(result)
        print(f"  Saved Performance Visualization: {plot_path}")

    # Export Trade Log CSV
    trade_log_path = os.path.join(backtester.results_dir, f"trades_{symbol.upper()}_{strategy_type}.csv")
    result["trade_log"].to_csv(trade_log_path, index=False)
    print(f"  Saved Trade Log: {trade_log_path}")

    # Print Formatted Markdown Table
    print("\n" + "=" * 70)
    print(format_metrics_table(result["metrics"], title=f"Backtest Report: {symbol.upper()} ({strategy.__class__.__name__})"))
    print("=" * 70)

    # Save summary report to results directory
    report_path = os.path.join(backtester.results_dir, f"report_{symbol.upper()}_{strategy_type}.md")
    with open(report_path, "w") as f:
        f.write(format_metrics_table(result["metrics"], title=f"Backtest Report: {symbol.upper()}"))
        f.write(f"\n\n- **Model Evaluation**: Accuracy {model_stats['direction_accuracy']}%, Price R2 {model_stats['price_r2']}, MAE ${model_stats['price_mae']}")
        f.write(f"\n- **Execution Parameters**: Initial Capital: ${initial_capital:,.2f}, Commission: {commission_rate*100:.2f}%, Slippage: {slippage_rate*100:.2f}%\n")

    result["plot_path"] = plot_path
    result["model_stats"] = model_stats
    return result


def main():
    parser = argparse.ArgumentParser(description="StockMarketAI Backtesting CLI")
    parser.add_argument("symbol", nargs="?", default="MSFT", help="Stock symbol to backtest (e.g. MSFT, AAPL, AGYS, TSLA)")
    parser.add_argument("--period", default="2y", help="Historical data period (e.g. 1y, 2y, 5y)")
    parser.add_argument("--mode", choices=["test_split", "walk_forward"], default="test_split", help="Backtesting validation mode")
    parser.add_argument("--strategy", choices=["enhanced", "direction", "regression"], default="enhanced", help="Strategy to evaluate")
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Initial portfolio capital in USD")
    parser.add_argument("--stop-loss", type=float, default=0.03, help="Stop loss fraction (e.g. 0.03 for 3 pct)")
    parser.add_argument("--take-profit", type=float, default=0.06, help="Take profit fraction (e.g. 0.06 for 6 pct)")
    parser.add_argument("--use-atr", action="store_true", help="Use dynamic ATR-based stops")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate per trade (e.g. 0.001 for 0.1 pct)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage rate per trade (e.g. 0.0005 for 0.05 pct)")

    args = parser.parse_args()

    run_backtest_pipeline(
        symbol=args.symbol,
        period=args.period,
        mode=args.mode,
        strategy_type=args.strategy,
        initial_capital=args.initial_capital,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        use_atr_stops=args.use_atr,
        commission_rate=args.commission,
        slippage_rate=args.slippage
    )


if __name__ == "__main__":
    main()
