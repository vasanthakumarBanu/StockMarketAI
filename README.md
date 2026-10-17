# StockMarketAI
# 📈 StockMarketAI: Production Quantitative ML & Trading Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Flutter](https://img.shields.io/badge/Flutter-iOS%20|%20Android%20|%20Desktop-02569B.svg)](https://flutter.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-24%20Passing-brightgreen.svg)]()

> **An End-to-End Quantitative Finance, Machine Learning Modeling, Strategy Backtesting & Automated Broker Execution Platform for Global & Indian Equities.**

---

## 🌟 Key Highlights

- **3-Level Hierarchical AI Architecture**:
  - **Level 1**: Directional Trend Classifier (Random Forest / Logistic Regression) with upward probability %.
  - **Level 2**: Multi-target OHLC Regressors predicting tomorrow's exact Open, High, Low, and Close targets.
  - **Level 3**: AI Trading Recommendation Engine (`STRONG BUY`, `BUY`, `HOLD`, `SELL`, `STRONG SELL`) with dynamic ATR-based Stop-Loss and Take-Profit limits.
- **Universal Multi-Market Coverage**:
  - Works with any US stock (`MSFT`, `AAPL`, `NVDA`, `TSLA`) and Indian stock (`RELIANCE.NS`, `TMCV.NS`, `SRMENERGY.BO`, `SBIN.NS`).
  - **Smart Company Resolver & Search**: Type partial or full company names (e.g. `"SRM Energy Ltd"` or `"Tata Motors"`), and it automatically resolves to live exchange tickers!
- **65+ Quantitative Features Pipeline**:
  - Trend (MAs, EMAs), Momentum (RSI 14, MACD), Volatility (Bollinger Bands, ATR 14), Candlestick Geometry, Volume Dynamics, and Time Lags.
- **Event-Driven Strategy Backtesting**:
  - Real-world simulation with slippage, broker commissions, and dynamic trailing ATR stops.
  - Generates Sharpe ratios, Sortino ratios, Max Drawdown, Win Rates, and equity curves.
- **Broker Gateways with Master Boolean Switch**:
  - **Angel One SmartAPI**: Automated Indian equity execution (NSE/BSE).
  - **INDmoney**: US equities & fractional share allocation.
  - **Master Safety Toggle**: Controlled by a boolean switch (`auto_execution_enabled: bool`) via button click to safely lock/permit live trades.
- **Multi-Platform Ready**:
  - **FastAPI REST API & Interactive Web Dashboard**: Dark-mode financial terminal.
  - **Cross-Platform Flutter App (`flutter_app/`)**: Native iOS, Android, macOS, and Windows desktop apps.
  - **C# .NET Standard 2.0 Wrapper & C DLL**: Embeddable in desktop trading terminals.

---

## 🏗️ System Architecture

```
[ Stock Search / Autocomplete ] ──► [ Universal Data Engine (US / India) ]
                                                   │
                                                   ▼
                                    [ 65+ Feature Engineering Pipeline ]
                                                   │
                                   ┌───────────────┴───────────────┐
                                   ▼                               ▼
                      [ Level 1 Direction Classifier ]   [ Level 2 OHLC Regressors ]
                                   │                               │
                                   └───────────────┬───────────────┘
                                                   │
                                                   ▼
                                 [ Level 3 AI Trading Decision Engine ]
                                 (Combines Predictions + ATR Risk + NLP Sentiment)
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
             [ Event-Driven Backtesting ]                         [ Automated Broker Gateways ]
             - Equity Curve vs Benchmark                          - Master Boolean Button Switch
             - Sharpe / Sortino / Drawdown                        - Angel One & INDmoney Routing
```

---

## 🚀 Quickstart

### 1. Prerequisites & Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/vasanthakumarBanu/StockMarketAI.git
cd StockMarketAI
pip install -r requirements.txt
```

### 2. Launch the Application

#### Option A: One-Click Launcher (Windows)
Double-click `start_stockmarketai.bat` to automatically start the backend and open the web dashboard in your default browser!

#### Option B: Terminal Command
```bash
python run_api.py --host 127.0.0.1 --port 8000
```
Open your browser to:
- **Interactive Web Terminal**: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
- **Interactive OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Command Line Interface (CLI)

```bash
# Instant multi-market AI prediction & technical analysis
python -m src.stockmarketai.cli.predict_any_stock MSFT
python -m src.stockmarketai.cli.predict_any_stock "SRM Energy"

# Event-driven backtesting with Sharpe, Sortino & Max Drawdown
python run_backtest.py MSFT --strategy enhanced --initial-capital 10000
```

---

## 📱 Cross-Platform Flutter App

A native Flutter client is located in [`flutter_app/`](flutter_app/):

```bash
cd flutter_app
flutter pub get

# Run on Android Phone / Emulator:
flutter run -d android

# Run on Windows Desktop:
flutter run -d windows

# Run on iOS (macOS required):
flutter run -d ios
```

---

## 🧪 Testing

The repository includes a comprehensive automated test suite with **24 unit and integration tests**:

```bash
python -m unittest discover tests
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.(Initial commit: StockMarketAI Production AI & Trading Platform)
