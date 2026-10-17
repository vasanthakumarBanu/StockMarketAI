# StockMarketAI — Complete Application & Feature Guide

> **An End-to-End Artificial Intelligence Trading, Quantitative Modeling, Backtesting & Execution System**  
> *Version: 1.0.0 | Status: Production Ready | Platform: Python 3.10+ / FastAPI / .NET Standard*

---

## 📖 Table of Contents
1. [Executive Overview](#1-executive-overview)
2. [What Can You Do With StockMarketAI?](#2-what-can-you-do-with-stockmarketai)
3. [Deep-Dive Feature Breakdown](#3-deep-dive-feature-breakdown)
   - [Pillar 1: Universal Multi-Market Data Engine](#pillar-1-universal-multi-market-data-engine)
   - [Pillar 2: 65+ Quantitative Feature Engineering Pipeline](#pillar-2-65-quantitative-feature-engineering-pipeline)
   - [Pillar 3: Multi-Level Machine Learning Inference](#pillar-3-multi-level-machine-learning-inference)
   - [Pillar 4: Event-Driven Backtesting Framework](#pillar-4-event-driven-backtesting-framework)
   - [Pillar 5: Production REST API Microservice](#pillar-5-production-rest-api-microservice)
   - [Pillar 6: Interactive Financial Web Dashboard](#pillar-6-interactive-financial-web-dashboard)
   - [Pillar 7: Broker Gateways & Risk Circuit Breakers](#pillar-7-broker-gateways--risk-circuit-breakers)
   - [Pillar 8: C# .NET Wrapper & C-Compatible DLL Service](#pillar-8-c-net-wrapper--c-compatible-dll-service)
   - [Pillar 9: Developer Python SDK](#pillar-9-developer-python-sdk)
   - [Pillar 10: Automated Test Suite](#pillar-10-automated-test-suite)
4. [Step-by-Step User Guide (How to Run Everything)](#4-step-by-step-user-guide)
5. [API Reference & Sample Requests](#5-api-reference--sample-requests)
6. [Supported Assets & Markets](#6-supported-assets--markets)
7. [System Architecture & File Directory](#7-system-architecture--file-directory)
8. [How to Customize & Extend](#8-how-to-customize--extend)

---

## 1. Executive Overview

**StockMarketAI** is a comprehensive, institutional-grade quantitative finance platform designed to take market data from raw price bars all the way to predictive machine learning, risk-managed trading signals, historical backtesting simulation, and automated broker order execution.

Unlike standard single-indicator scripts or black-box predictors, StockMarketAI utilizes a **3-Level Hierarchical AI Architecture**:
1. **Level 1 (Direction Classification)**: Predicts whether tomorrow will close higher or lower than today.
2. **Level 2 (Price Regression)**: Forecasts tomorrow's exact **Open, High, Low, and Close (OHLC)** prices with uncertainty intervals.
3. **Level 3 (AI Trading Decision Engine)**: Combines directional probability and expected price returns with volatility-based risk limits to issue actionable trading signals (`STRONG BUY`, `BUY`, `HOLD`, `SELL`, `STRONG SELL`) with dynamic stop-loss and take-profit targets.

---

## 2. What Can You Do With StockMarketAI?

| User Role | What You Can Achieve |
| :--- | :--- |
| **Active Traders & Investors** | Input any stock ticker (`MSFT`, `AAPL`, `NVDA`, `RELIANCE.NS`) to immediately receive tomorrow's forecasted prices, trend probability, and exact entry, stop-loss, and take-profit levels. |
| **Quantitative Researchers** | Backtest custom trading strategies historically with zero lookahead bias, realistic slippage, broker commissions, and dynamic ATR exits. |
| **Algorithmic Traders** | Connect directly to broker gateways (**Angel One SmartAPI** for Indian equities, **INDmoney** for US equities) for automated order routing with built-in daily loss circuit breakers. |
| **Software Developers** | Consume prediction models via clean REST API endpoints (`FastAPI`), integrate with Windows desktop trading terminals using the **C# .NET Wrapper**, or embed into custom bots via the **Python SDK**. |
| **Portfolio Managers** | Batch scan entire watchlists in seconds to rank equities by signal confidence and expected return percentage. |

---

## 3. Deep-Dive Feature Breakdown

### Pillar 1: Universal Multi-Market Data Engine
- **Stock-Agnostic Ingestion**: Supports any equity, ETF, or index available through Yahoo Finance without requiring manual configuration.
- **US & Global Equities**: Apple (`AAPL`), Microsoft (`MSFT`), Tesla (`TSLA`), Nvidia (`NVDA`), Agilysys (`AGYS`), S&P 500 ETF (`SPY`), etc.
- **Indian Equities (NSE & BSE)**: Reliance (`RELIANCE.NS`), TCS (`TCS.NS`), Infosys (`INFY.NS`), Tata Motors (`TATAMOTORS.NS`), Nifty 50 (`^NSEI`).
- **Automated Data Cleaning**: Resolves timezone offsets to UTC, validates data types, handles stock splits, and chronologically orders bars.
- **High-Speed Caching**: Local CSV archives and in-memory caching prevent redundant network calls and enable sub-second responses.

---

### Pillar 2: 65+ Quantitative Feature Engineering Pipeline
The feature engine translates raw OHLCV price action into **65+ mathematically rigorous quantitative factors**:

1. **Trend Indicators**:
   - Simple Moving Averages: `MA_20`, `MA_50`, `MA_200`
   - Exponential Moving Averages: `EMA_20`, `EMA_50`, `EMA_200`
2. **Momentum Indicators**:
   - 14-period Relative Strength Index (`RSI_14`)
   - Moving Average Convergence Divergence (`MACD`, `MACD_Signal`, `MACD_Hist`)
3. **Volatility Indicators**:
   - Bollinger Bands (`BB_Upper`, `BB_Middle`, `BB_Lower`)
   - Average True Range (`ATR_14`)
   - 20-Day Rolling Daily Return Volatility (`Volatility_20`)
4. **Candlestick Geometry & Price Action**:
   - Candle body size (`Body_Size`)
   - Upper and Lower wick lengths (`Upper_Wick`, `Lower_Wick`)
   - Body-to-Range and Wick-to-Range ratios
   - Direction flag (`Is_Bullish`)
5. **Volume Dynamics**:
   - 5-day and 20-day Volume Moving Averages (`Volume_MA_5`, `Volume_MA_20`)
   - Volume Ratio (`Volume_Ratio_5_20`)
   - Price-Volume trend interactions (`PV_Trend`)
   - Price-Volume rolling correlation and volume volatility
6. **Time-Series Lag Features**:
   - 1, 2, and 3-period lags of Open, High, Low, Close, and Volume
   - 1, 2, 3, 5, and 10-period lags of Daily Returns
   - Lagged RSI, MACD, and Volatility states

---

### Pillar 3: Multi-Level Machine Learning Inference
StockMarketAI uses an integrated multi-model architecture:

```
[Raw Stock Data] ──► [Feature Engineering (65+ Indicators)]
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     [Level 1: Classifier]          [Level 2: Regressors]
     Random Forest / Logistic       Linear Regression & Random Forest
     Predicts: Tomorrow UP/DOWN     Predicts: Tomorrow Open, High, Low, Close
              │                               │
              └───────────────┬───────────────┘
                              ▼
            [Level 3: AI Decision & Risk Engine]
            Synthesizes Probability + Expected Return %
            Enforces ATR Stop-Loss & Take-Profit Targets
            Factored by News Sentiment (VADER/FinViz)
                              ▼
      [Actionable Recommendation: STRONG BUY / BUY / HOLD / SELL]
```

- **Level 1 (Direction)**: Probability of upward movement ($P(\text{Up}) \in [0.0, 1.0]$) and direction confidence.
- **Level 2 (Price Forecasting)**: Independent models trained per target (Tomorrow Open, High, Low, Close) with automatic physical price boundary validation ($\text{High} \ge \max(\text{Open}, \text{Close})$ and $\text{Low} \le \min(\text{Open}, \text{Close})$).
- **Level 3 (Trading Signal & Risk Controls)**:
  - `STRONG BUY`: Direction Prob $\ge 60\%$ & Expected Return $\ge +1.5\%$
  - `BUY`: Direction Prob $\ge 52\%$ & Expected Return $> +0.3\%$
  - `STRONG SELL`: Direction Prob $\le 40\%$ & Expected Return $\le -1.5\%$
  - `SELL`: Direction Prob $\le 48\%$ & Expected Return $< -0.3\%$
  - `HOLD`: Conflicting or low-conviction signals
  - **Dynamic ATR Stops**:
    $$\text{Stop-Loss} = \text{Current Price} - (1.5 \times \text{ATR})$$
    $$\text{Take-Profit} = \text{Current Price} + (2.5 \times \text{ATR})$$
    $$\text{Risk/Reward Ratio} = 1.67 \text{ to } 2.0$$

---

### Pillar 4: Event-Driven Backtesting Framework
A realistic trading simulator with institutional execution rules:
- **Zero Lookahead Bias**: Signals generated at Close of Day $t$; orders filled at Open of Day $t+1$.
- **Friction Modeling**: 0.05% slippage and 0.1% round-trip commission fees.
- **Intra-Day Exit Monitoring**: Stop-loss and take-profit checked against High and Low prices of Day $t+1$.
- **Strategy Implementations**:
  - `EnhancedHybridStrategy`: Consensus between Level 1 direction, Level 2 price target, and volatility filters.
  - `DirectionStrategy`: Pure probability threshold execution.
  - `RegressionExpectedReturnStrategy`: Traded on expected percentage return.
- **Performance Metrics Computed**: Total Return %, Benchmark Return %, CAGR %, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Maximum Drawdown %, Win Rate %, Profit Factor, and Alpha/Beta.
- **Automated Visualization**: Generates 3-panel performance charts saved to `src/backtesting/results/`.

---

### Pillar 5: Production REST API Microservice
Built with **FastAPI** and **Pydantic V2**:
- Pre-warmed models loaded in memory on startup.
- Full CORS support for web and mobile frontends.
- Interactive documentation automatically available at `/docs` (Swagger UI) and `/redoc`.
- High-performance asynchronous routes for predictions, backtests, status checks, and watchlist scanning.

---

### Pillar 6: Interactive Financial Web Dashboard
Served directly at `http://localhost:8000/dashboard`:
- **Dark Mode Financial Styling**: Glassmorphism cards, glowing signal badges, and Inter typography.
- **Live Prediction View**: Hero signal banner, probability meter, and tomorrow's OHLC forecast.
- **Interactive Backtester Tab**: Customize capital and strategy; renders real-time **Chart.js** equity curves and recent trade logs.
- **Market Watchlist Scanner Tab**: One-click scan of multiple equities with direct "Inspect" action buttons.
- **Model Registry Tab**: Real-time inspection of active model weights, features, and endpoints.

---

### Pillar 7: Broker Gateways, Master Boolean Switch & Risk Circuit Breakers
- **Master Boolean Auto-Execution Switch (`auto_execution_enabled: bool`)**:
  - Controlled by the **Master Toggle Button** on the Web Dashboard and Top Navigation Bar (`#autoExecutionToggleBtn` & `#navAutoTradeBtn`).
  - **Safe Default (`False`)**: When disabled, all live or automated broker orders are strictly blocked with explicit safety logs.
  - **Active State (`True`)**: When enabled via button click, the system autonomously evaluates AI signals and routes qualified orders to the active broker.
  - **API Control**: Can be toggled programmatically via `POST /api/v1/brokers/toggle-execution` or checked via `GET /api/v1/brokers/execution-status`.
- **Angel One SmartAPI (`src/stockmarketai/brokers/angel_one.py`)**: Indian equity connectivity (NSE/BSE) for live and paper order execution.
- **INDmoney Connector (`src/stockmarketai/brokers/indmoney.py`)**: US equities integration with fractional share allocation.
- **Automated Execution Engine (`src/stockmarketai/brokers/execution_engine.py`)**: Converts AI signals to broker orders with ATR position sizing.
- **Risk Circuit Breakers**: Halts trading automatically if portfolio daily drawdown exceeds 3% to 5% or if market volatility spikes abnormally.

---

### Pillar 8: C# .NET Wrapper & C-Compatible DLL Service
- **C-ABI Function Exports (`src/stockmarketai/integration/dll_service.py`)**: Functions `c_predict`, `c_backtest`, and `c_status` callable from C, C++, or desktop software like MetaTrader and NinjaTrader.
- **C# .NET Class Library (`StockPredictionWrapper.cs`)**: Native .NET Standard 2.0 wrapper implementing `IStockPredictionService` with typed models and async methods for C# WPF/WinForms trading applications.

---

### Pillar 9: Developer Python SDK
A clean Python client (`StockMarketAIClient`) for developers and Jupyter users:
```python
from src.stockmarketai.client import StockMarketAIClient

# Works in local in-process mode or remote HTTP mode
client = StockMarketAIClient()

prediction = client.predict("MSFT")
backtest_res = client.backtest("AAPL")
ranked_signals = client.scan(["MSFT", "AAPL", "NVDA", "TSLA"])
```

---

### Pillar 10: Automated Test Suite
- Comprehensive suite of **21 unit and integration tests** located in [`tests/`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/tests).
- 100% pass rate covering services, API routes, SDK methods, DLL function exports, and broker gateways.

---

## 4. Step-by-Step User Guide

### 🚀 1. How to Launch the Web Dashboard & API Server
```powershell
python run_api.py
```
Open your browser to:
- **Interactive Web Dashboard**: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 📈 2. How to Run an Event-Driven Backtest via CLI
```powershell
# Run backtest on MSFT with dynamic ATR stops
python run_backtest.py MSFT --use-atr

# Run backtest with custom stops
python run_backtest.py AAPL --strategy direction --stop-loss 0.02 --take-profit 0.05
```

---

### 🔮 3. How to Train Models & Forecast Any Stock
```powershell
# Interactive & symbol prediction CLI (instant multi-market forecasting)
python -m src.stockmarketai.cli.predict_any_stock MSFT
python -m src.stockmarketai.cli.predict_any_stock "SRM Energy"
python -m src.stockmarketai.cli.predict_any_stock RELIANCE.NS

# Or train full workflow pipeline directly
python -m src.stockmarketai.pipelines.complete_workflow AGYS
```


---

### 🧪 4. How to Run the Automated Test Suite
```powershell
python -m unittest discover tests
```

---

## 5. API Reference & Sample Requests

### `GET /api/v1/predict/{symbol}`
Generates complete 3-Level predictions, price targets, and trade recommendations.

**Sample Request (cURL):**
```bash
curl -X GET "http://localhost:8000/api/v1/predict/MSFT?use_sentiment=true"
```

**Sample JSON Response:**
```json
{
  "symbol": "MSFT",
  "prediction_date": "2026-10-17",
  "current_market": {
    "open": 519.09,
    "high": 522.50,
    "low": 513.66,
    "close": 514.02,
    "volume": 18400000,
    "atr": 7.25,
    "rsi_14": 52.8,
    "macd": 1.24
  },
  "level1_direction": {
    "prediction": "UP",
    "probability_pct": 68.5,
    "confidence": 0.37
  },
  "level2_price_targets": {
    "tomorrow_open": 514.76,
    "tomorrow_high": 518.20,
    "tomorrow_low": 512.10,
    "tomorrow_close": 517.16,
    "expected_change_pct": 0.61
  },
  "level3_trading_signal": {
    "signal": "BUY",
    "confidence_pct": 74.0,
    "expected_return_pct": 0.61,
    "stop_loss_price": 503.14,
    "take_profit_price": 532.14,
    "risk_reward_ratio": 1.75,
    "recommended_action": "BUY with stop-loss at $503.14 and take-profit at $532.14"
  }
}
```

---

### `POST /api/v1/backtest`
Simulates an out-of-sample trading strategy.

**Sample Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/backtest" \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "MSFT",
       "strategy": "enhanced",
       "initial_capital": 10000.0,
       "use_atr": true
     }'
```

---

## 6. Supported Assets & Markets

StockMarketAI supports all ticker formats available through Yahoo Finance:

| Asset Class | Market | Example Tickers |
| :--- | :--- | :--- |
| **US Large-Cap Equities** | NASDAQ / NYSE | `AAPL`, `MSFT`, `NVDA`, `TSLA`, `AMZN`, `GOOGL`, `META` |
| **US Mid/Small-Cap Equities** | NASDAQ / Russell | `AGYS`, `PLTR`, `AMD`, `CRWD` |
| **Indian Equities (NSE)** | National Stock Exchange | `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `TATAMOTORS.NS` |
| **Indian Equities (BSE)** | Bombay Stock Exchange | `RELIANCE.BO`, `TCS.BO`, `INFY.BO` |
| **ETFs & Indices** | Global | `SPY`, `QQQ`, `VTI`, `^GSPC` (S&P 500), `^NSEI` (Nifty 50) |

---

## 7. System Architecture & File Directory

```
StockMarketAI/
├── run_api.py                          # Unified launcher for REST API & Web Dashboard
├── run_backtest.py                     # CLI launcher for Phase 4 Backtesting
├── start_stockmarketai.bat             # 1-Click Windows launcher
├── ABOUT_STOCKMARKETAI.md              # Comprehensive architectural guide
├── README.md                           # GitHub repository overview
├── PHASE4_COMPLETED.md                 # Technical specification of Phase 4 Backtesting
├── PHASE5_COMPLETED.md                 # Technical specification of Phase 5 REST API & Dashboard
├── PHASE6_COMPLETED.md                 # Technical specification of Phase 6 DLL & Brokers
├── FINAL_SUMMARY.md                    # Multi-phase summary
├── requirements.txt                    # Python project dependencies
│
├── config/                             # Market configurations & stock lists
│   └── stocks.yaml
│
├── models/                             # Classification models & scalers
│   ├── stock_rf_model.pkl              # Direction classifier (Random Forest)
│   ├── stock_lr_model.pkl              # Direction classifier (Logistic Regression)
│   ├── feature_scaler.pkl              # Input standard scaler
│   └── feature_columns.pkl             # Active feature names
│
├── models_complete/                    # Regression models & scalers
│   ├── tomorrow_open_*.pkl             # Tomorrow Open price regressors
│   ├── tomorrow_high_*.pkl             # Tomorrow High price regressors
│   ├── tomorrow_low_*.pkl              # Tomorrow Low price regressors
│   ├── tomorrow_close_*.pkl            # Tomorrow Close price regressors
│   ├── feature_scaler.pkl              # Input standard scaler
│   ├── feature_columns.pkl             # Active feature names
│   └── results_summary.pkl             # Cross-validation performance metrics
│
├── notebooks/                          # Production data science Jupyter notebooks
│   ├── 01_data_collection.ipynb        # Data ingestion & Yahoo Finance download
│   └── 02_model_baseline.ipynb         # Model baseline & training exploration
│
├── src/
│   ├── api/                            # Production FastAPI REST microservice
│   │   ├── app.py                      # App factory, CORS, static mounts
│   │   ├── schemas.py                  # Pydantic v2 schemas
│   │   ├── main.py                     # FastAPI entrypoint
│   │   └── routes/                     # Health, predict, backtest, and scanner endpoints
│   │
│   ├── backtesting/                    # Backtesting framework
│   │   ├── backtest.py                 # Core event-driven simulator
│   │   ├── strategy.py                 # Hybrid, Direction, and Regression strategies
│   │   └── metrics.py                  # Sharpe, Sortino, Drawdown, Profit Factor
│   │
│   └── stockmarketai/                  # Core library & engine
│       ├── service.py                  # Reusable ML Service (singleton, cache, inference)
│       ├── sentiment.py                # VADER news sentiment scoring engine
│       ├── client.py                   # Python Developer SDK
│       ├── data/                       # Ingestion, caching, and company search resolver
│       ├── features/                   # 72 quantitative, candlestick & volume features
│       ├── pipelines/                  # Complete workflow training pipelines
│       ├── trading/                    # Enhanced decision engine
│       ├── cli/                        # Interactive CLI prediction tools
│       ├── frontend/                   # Interactive Web Dashboard (HTML/CSS/JS)
│       ├── brokers/                    # Angel One & INDmoney connectors, circuit breaker
│       └── integration/                # DLL service & C# .NET wrapper
│
├── flutter_app/                        # Multi-platform Flutter App (Android, iOS, Web, Windows)
│   ├── lib/
│   │   ├── main.dart                   # Application root
│   │   ├── screens/                    # Dashboard, Search, Backtest, Trade screens
│   │   ├── models/                     # Stock, Trade, Portfolio data models
│   │   └── services/                   # REST API client & state management
│   └── pubspec.yaml
│
└── tests/                              # Automated test suite (24 passing tests)
    ├── test_service.py                 # ML service unit tests
    ├── test_api.py                     # FastAPI REST endpoint tests
    ├── test_client.py                  # Python SDK tests
    └── test_integration.py            # C# DLL integration & broker tests
```

---

## 8. How to Customize & Extend

This project is built modularly. You can edit and extend it in several straightforward ways:

1. **Add Custom Features**: Open [`src/stockmarketai/features/feature_engineering.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/features/feature_engineering.py) to calculate new indicators (e.g., Stochastic RSI, Fibonacci retracements).
2. **Adjust Strategy Rules & Stop Limits**: Open [`src/backtesting/strategy.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/backtesting/strategy.py) to tune the ATR multiplier stops (`atr_multiplier_stop` or `atr_multiplier_profit`).
3. **Change Risk Circuit Breakers**: Open [`src/stockmarketai/brokers/base.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/brokers/base.py) to configure maximum daily loss thresholds (`max_daily_loss_pct`) or maximum concurrent positions.
4. **Edit the Web UI Dashboard**: Modify [`src/stockmarketai/frontend/index.html`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/frontend/index.html) or [`styles.css`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/frontend/styles.css) to add new widgets, watchlists, or custom charts.

