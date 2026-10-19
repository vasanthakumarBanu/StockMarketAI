# 📈 StockMarketAI: Production Quantitative ML & Trading Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Flutter](https://img.shields.io/badge/Flutter-iOS%20|%20Android%20|%20Desktop-02569B.svg)](https://flutter.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-24%20Passing-brightgreen.svg)]()

> **An Institutional-Grade, End-to-End Quantitative Finance, Machine Learning Modeling, Strategy Backtesting & Automated Broker Execution Platform for Global & Indian Equities.**  
> *Version: 1.2.0 | Platform: Python 3.10+ / FastAPI / .NET Standard 2.0 / Flutter*

---

## 📖 Table of Contents

1. [Executive Overview & Key Highlights](#1-executive-overview--key-highlights)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Comprehensive Project Phases](#3-comprehensive-project-phases)
   - [Phase 1: Universal Data Ingestion & Preprocessing Engine](#phase-1-universal-data-ingestion--preprocessing-engine)
   - [Phase 2: 65+ Quantitative Feature Engineering Pipeline](#phase-2-65-quantitative-feature-engineering-pipeline)
   - [Phase 3: Multi-Level Machine Learning Inference & AI Signals](#phase-3-multi-level-machine-learning-inference--ai-signals)
   - [Phase 4: Realistic Event-Driven Backtesting & Strategy Engine](#phase-4-realistic-event-driven-backtesting--strategy-engine)
   - [Phase 5: Reusable ML Service, Production REST API & Interactive Dashboard](#phase-5-reusable-ml-service-production-rest-api--interactive-dashboard)
   - [Phase 6: Platform Integration, C-Compatible DLL & Broker Gateways](#phase-6-platform-integration-c-compatible-dll--broker-gateways)
   - [Phase 7: Cross-Platform Native Client (Flutter App)](#phase-7-cross-platform-native-client-flutter-app)
   - [Phase 8: Automated Verification & Test Suite Matrix](#phase-8-automated-verification--test-suite-matrix)
4. [Quickstart & Installation](#4-quickstart--installation)
5. [CLI Reference & Commands](#5-cli-reference--commands)
6. [REST API Reference & Sample Requests](#6-rest-api-reference--sample-requests)
7. [Supported Assets & Markets](#7-supported-assets--markets)
8. [File & Directory Structure](#8-file--directory-structure)
9. [License](#9-license)

---

## 1. Executive Overview & Key Highlights

**StockMarketAI** bridges the entire lifecycle of algorithmic trading: from dynamic multi-market ingestion and quantitative factor engineering to multi-target machine learning forecasting, event-driven backtesting, and automated order execution.

### 🌟 Core Highlights
- **3-Level Hierarchical AI Architecture**:
  - **Level 1 (Trend Classifier)**: Calibrated Logistic Regression / Random Forest predicting directional probability ($P(\text{UP})$ vs $P(\text{DOWN})$).
  - **Level 2 (Multi-Target OHLC Regressors)**: Independent high-precision models predicting tomorrow's exact **Open, High, Low, and Close** targets ($R^2 > 0.90$).
  - **Level 3 (AI Trading Decision Engine)**: Synthesizes direction, price differentials, dynamic ATR risk bounds, and news sentiment into actionable recommendations (`STRONG BUY`, `BUY`, `HOLD`, `SELL`, `STRONG SELL`).
- **Universal Multi-Market Coverage**:
  - Dynamically fetches any US stock (`MSFT`, `AAPL`, `NVDA`, `TSLA`) and Indian stock (`RELIANCE.NS`, `TATAMOTORS.NS`, `SRMENERGY.BO`).
  - **Smart Company Resolver & Fuzzy Search**: Autocompletes natural business names (e.g., `"Tata Motors"` $\rightarrow$ `TATAMOTORS.NS`, `"SRM Energy"` $\rightarrow$ `SRMENERGY.BO`).
- **65+ Quantitative Features Pipeline**:
  - Trend (MAs, EMAs), Momentum (RSI 14, MACD), Volatility (Bollinger Bands, ATR 14), Candlestick Geometry, Volume Dynamics, and Time Lags.
- **Event-Driven Backtesting Simulator**:
  - Realistic next-day Open fills, slippage (0.05%), broker commissions (0.1%), dynamic ATR trailing stops, Sharpe/Sortino ratios, and interactive Chart.js equity curves.
- **Automated Broker Execution Gateways**:
  - **Angel One SmartAPI** for Indian equities (NSE/BSE).
  - **INDmoney** for US equities and fractional share allocation.
  - **Master Safety Switch (`auto_execution_enabled: bool`)**: Requires explicit operator activation before real trades can be dispatched.
  - **Automated Circuit Breaker**: Trips immediately on excessive daily loss (-5.0%).
- **Multi-Platform Access**:
  - **FastAPI REST API & Dark-Mode Financial Terminal**: Real-time web workspace at `http://localhost:8000/dashboard`.
  - **Cross-Platform Flutter Client**: Native iOS, Android, macOS, and Windows desktop terminal.
  - **C# .NET Standard 2.0 Wrapper & C DLL**: Embeddable in MetaTrader, NinjaTrader, or Windows trading apps.

---

## 2. End-to-End System Architecture

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
                                  (Signal + ATR SL/TP + News Sentiment)
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
             [ Event-Driven Backtesting ]                         [ Automated Broker Gateways ]
             - Equity Curve vs Benchmark                          - Master Safety Switch (Boolean)
             - Sharpe / Sortino / MDD                             - Angel One (India) & INDmoney (US)
                         │                                                   │
                         └─────────────────────────┬─────────────────────────┘
                                                   │
                                   ┌───────────────┴───────────────┐
                                   ▼                               ▼
                      [ Web Terminal & REST API ]      [ Cross-Platform Flutter Client ]
                      - FastAPI & Chart.js             - Android, iOS, Windows, macOS
```

---

## 3. Comprehensive Project Phases

### Phase 1: Universal Data Ingestion & Preprocessing Engine
- **Objective**: Ingest daily and intraday OHLCV market data for any US or Indian equity on demand without static CSV dependencies.
- **Key Modules**:
  - [`src/stockmarketai/data/load_stock_data.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/data/load_stock_data.py): Automated Yahoo Finance daily ingestion with configurable lookback.
  - [`src/stockmarketai/data/search.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/data/search.py): Intelligent company name resolver and autocomplete supporting US equities, NSE (`.NS`), and BSE (`.BO`) stocks.
  - [`src/stockmarketai/data/preprocessing.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/data/preprocessing.py): UTC timezone alignment, missing data imputation, and chronological sorting.
- **Capabilities**:
  - 100% dynamic data fetching for any ticker (`MSFT`, `AAPL`, `NVDA`, `TSLA`, `RELIANCE.NS`, `SRMENERGY.BO`).
  - High-speed local disk caching (`src/stockmarketai/data/raw/{SYMBOL}.csv`) and thread-safe in-memory caching with 5-minute TTL.

---

### Phase 2: 65+ Quantitative Feature Engineering Pipeline
- **Objective**: Transform raw price series into a high-dimensional quantitative feature space.
- **Key Modules**:
  - [`src/stockmarketai/features/feature_engineering.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/features/feature_engineering.py): Primary mathematical feature pipeline.
- **Feature Breakdown**:
  - **Trend (6)**: `MA_20`, `MA_50`, `MA_200`, `EMA_20`, `EMA_50`, `EMA_200`.
  - **Momentum (4)**: `RSI_14`, `MACD`, `MACD_Signal`, `MACD_Hist`.
  - **Volatility (5)**: `BB_Upper`, `BB_Middle`, `BB_Lower`, `ATR_14`, `Volatility_20`.
  - **Candlestick Geometry (8)**: `Body_Size`, `Upper_Wick`, `Lower_Wick`, `Body_Range_Ratio`, `Is_Bullish`, `Body_Size_Pct_Change`.
  - **Volume Dynamics (6)**: `Volume_MA_5`, `Volume_MA_20`, `Volume_Ratio_5_20`, `PV_Trend`, `Volume_Price_Correlation`, `Volume_Volatility`.
  - **Time Lags (36+)**: $t-1, t-2, t-3$ of OHLCV; $t-1, t-2, t-3, t-5, t-10$ daily returns; lagged technical indicators.
- **Leakage Prevention**: Strict 80/20 chronological train/test split. `StandardScaler` fitted strictly on training data.

---

### Phase 3: Multi-Level Machine Learning Inference & AI Signals
- **Objective**: Build a 3-tier prediction hierarchy translating quantitative features into price targets and risk-controlled trade signals.
- **Architecture**:
  - **Level 1 (Direction Classification)**: Logistic Regression & Random Forest models predicting trend probability $P(\text{UP})$.
  - **Level 2 (Multi-Target OHLC Regression)**: Independent regression models forecasting tomorrow's exact Open, High, Low, and Close prices with 95% confidence intervals.
  - **Level 3 (AI Trading Decision Engine)**: Computes trading signals (`STRONG BUY`, `BUY`, `HOLD`, `SELL`, `STRONG SELL`) and sets dynamic ATR risk limits.
- **Verified Regression Accuracy**:
  - **MSFT**: Open $R^2 = 0.9794$, High $R^2 = 0.9608$, Low $R^2 = 0.9730$, Close $R^2 = 0.9551$
  - **AGYS**: Open $R^2 = 0.9540$, High $R^2 = 0.9164$, Low $R^2 = 0.9467$, Close $R^2 = 0.9017$
- **Dynamic ATR Risk Formulas**:
  $$\text{Stop-Loss} = \text{Entry Price} - (1.5 \times \text{ATR}_{14})$$
  $$\text{Take-Profit} = \text{Entry Price} + (2.5 \times \text{ATR}_{14})$$
- **Sentiment Scoring**: Integrated financial news NLP analysis via [`src/stockmarketai/sentiment.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/sentiment.py).

---

### Phase 4: Realistic Event-Driven Backtesting & Strategy Engine
- **Objective**: Conduct zero-lookahead, transaction-cost-aware simulations with dynamic risk management.
- **Key Modules**:
  - [`src/backtesting/backtest.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/backtesting/backtest.py): Event-driven simulation engine.
  - [`src/backtesting/strategy.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/backtesting/strategy.py): Strategy library (`DirectionStrategy`, `RegressionExpectedReturnStrategy`, `EnhancedHybridStrategy`).
  - [`src/backtesting/metrics.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/backtesting/metrics.py): Quantitative calculations (Sharpe, Sortino, Calmar, MDD, Alpha, Beta).
- **Execution Semantics**: Signals generated at close of Day $t$; orders filled at open of Day $t+1$ with 0.05% slippage and 0.1% commission.
- **Verification Results**:
  | Metric | MSFT (Enhanced Hybrid) | AGYS (ATR Exits) | Benchmark (Buy & Hold) |
  | :--- | :--- | :--- | :--- |
  | **Initial Capital** | $10,000.00 | $10,000.00 | $10,000.00 |
  | **Final Equity** | **$11,017.69 (+10.18%)** | **$11,372.76 (+13.73%)** | +29.91% |
  | **Max Drawdown (MDD)** | **-3.39%** | **-4.12%** | **-23.38%** (~7x drawdown reduction) |
  | **Sharpe Ratio** | **1.71** | **1.45** | 0.88 |
  | **Sortino Ratio** | **2.21** | **1.92** | 1.05 |
  | **Win Rate** | **66.7%** (4 wins / 2 losses) | **50.0%** (5 wins / 5 losses) | N/A |
  | **Profit Factor** | **4.05** | **1.46** | N/A |

---

### Phase 5: Reusable ML Service, Production REST API & Interactive Dashboard
- **Objective**: Provide an in-memory singleton service layer, high-performance REST microservice, dark-mode financial terminal, and developer SDK.
- **Key Modules**:
  - [`src/stockmarketai/service.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/service.py): Reusable service with in-memory caching and thread safety.
  - [`src/api/app.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/api/app.py) & [`src/api/routes/`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/api/routes/): FastAPI endpoints with Pydantic V2 validation.
  - [`src/stockmarketai/frontend/`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/frontend/): Interactive web dashboard with Chart.js equity visualization.
  - [`src/stockmarketai/client.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/client.py): Python SDK (`StockMarketAIClient`) for both local in-process and remote HTTP modes.
- **Web Terminal**: Accessible at `http://localhost:8000/dashboard` featuring glowing recommendation badges, price forecast cards, interactive backtest simulator, and watchlist opportunity scanner.

---

### Phase 6: Platform Integration, C-Compatible DLL & Broker Gateways
- **Objective**: Expose native DLL exports for desktop trading terminals and connect live broker order routing.
- **Key Modules**:
  - [`src/stockmarketai/integration/dll_service.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/integration/dll_service.py): C-ABI Python DLL exporting `c_predict`, `c_backtest`, and `c_status`.
  - [`src/stockmarketai/integration/csharp/StockPredictionWrapper.cs`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/integration/csharp/StockPredictionWrapper.cs): .NET Standard 2.0 C# wrapper implementing `IStockPredictionService`.
  - [`src/stockmarketai/brokers/angel_one.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/brokers/angel_one.py): Angel One SmartAPI connector for Indian equities (NSE/BSE).
  - [`src/stockmarketai/brokers/indmoney.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/brokers/indmoney.py): INDmoney connector for US equities & fractional shares.
  - [`src/stockmarketai/brokers/base.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/brokers/base.py) & [`execution_engine.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/brokers/execution_engine.py): Trade execution engine with safety switch and loss circuit breaker.
- **Master Boolean Switch**: `auto_execution_enabled: bool` defaults to `False`. Live trades cannot execute without manual operator toggle.
- **Circuit Breaker**: Auto-trips when daily portfolio loss exceeds -5.0%.

---

### Phase 7: Cross-Platform Native Client (Flutter App)
- **Objective**: Deliver a native mobile and desktop client terminal located in [`flutter_app/`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/flutter_app/).
- **Key Screens & Features**:
  - **Dashboard**: Live signal badges, predicted OHLC cards, and real-time stock search.
  - **Backtester**: Custom capital and strategy inputs with interactive `fl_chart` equity curves.
  - **Watchlist & Broker Controls**: Opportunity ranking and master automated trade switch toggle.
- **Platform Support**: Android, iOS, Windows desktop, and macOS.

---

### Phase 8: Automated Verification & Test Suite Matrix
- **Objective**: Maintain 100% test pass rate across all services, API routes, SDK clients, and broker connectors.
- **Command**: `python -m unittest discover tests`
- **Results**: **24 passing tests out of 24** (0 failures, 0 errors).
  | Test Module | Coverage Area | Status |
  | :--- | :--- | :---: |
  | [`test_service.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/tests/test_service.py) | In-memory singleton service, TTL caching, predictions | ✅ PASS |
  | [`test_api.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/tests/test_api.py) | FastAPI endpoints (`/health`, `/predict`, `/backtest`, `/scan`) | ✅ PASS |
  | [`test_client.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/tests/test_client.py) | In-process and HTTP developer SDK client | ✅ PASS |
  | [`test_integration.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/tests/test_integration.py) | C-ABI exports, C# wrapper, Angel One, INDmoney, Circuit Breaker | ✅ PASS |

---

## 4. Quickstart & Installation

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/vasanthakumarBanu/StockMarketAI.git
cd StockMarketAI
pip install -r requirements.txt
```

### 2. Launch the Application

#### Option A: One-Click Windows Launcher
Double-click `start_stockmarketai.bat` to launch the API server and automatically open the financial dashboard in your browser.

#### Option B: Terminal Command
```bash
python run_api.py --host 127.0.0.1 --port 8000
```
Access points:
- **Interactive Web Terminal**: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
- **Interactive OpenAPI / Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 5. CLI Reference & Commands

```bash
# Instant AI prediction for US equities
python -m src.stockmarketai.cli.predict_any_stock MSFT

# Instant AI prediction for Indian equities (via company name or ticker)
python -m src.stockmarketai.cli.predict_any_stock "Tata Motors"
python -m src.stockmarketai.cli.predict_any_stock "SRM Energy"
python -m src.stockmarketai.cli.predict_any_stock RELIANCE.NS

# Event-driven backtesting with dynamic ATR stops
python run_backtest.py MSFT --use-atr

# Event-driven backtesting with custom strategy & capital
python run_backtest.py AAPL --strategy direction --initial-capital 25000 --stop-loss 0.02 --take-profit 0.05

# Retrain full model pipeline for any stock
python -m src.stockmarketai.pipelines.complete_workflow AGYS

# Run the complete automated test suite
python -m unittest discover tests
```

---

## 6. REST API Reference & Sample Requests

### Endpoint Summary Table
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Instant health check |
| `GET` | `/api/v1/status` | System uptime, cached symbols, loaded models, broker status |
| `GET` | `/api/v1/models` | Active model architectures, 65+ feature columns, test metrics |
| `GET` | `/api/v1/predict/{symbol}` | 3-Level prediction, price targets, signal & ATR risk bounds |
| `POST` | `/api/v1/predict` | JSON POST prediction request |
| `GET` | `/api/v1/backtest/{symbol}` | Run backtest via GET query parameters |
| `POST` | `/api/v1/backtest` | Run backtest with custom capital, strategy, and risk limits |
| `GET` | `/api/v1/stocks` | List configured / popular equities |
| `POST` | `/api/v1/stocks/scan` | Batch scan multiple tickers and rank trading opportunities |
| `POST` | `/api/v1/broker/toggle-execution` | Live master boolean switch toggling trade execution permission |
| `GET` | `/api/v1/broker/positions` | Live broker portfolio holdings and unrealized P&L |

### Sample cURL Request & JSON Response

```bash
curl -X GET "http://localhost:8000/api/v1/predict/MSFT?use_sentiment=true"
```

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

## 7. Supported Assets & Markets

StockMarketAI supports all ticker formats accessible through Yahoo Finance:

| Asset Class | Market | Example Tickers |
| :--- | :--- | :--- |
| **US Large-Cap Equities** | NASDAQ / NYSE | `AAPL`, `MSFT`, `NVDA`, `TSLA`, `AMZN`, `GOOGL`, `META` |
| **US Mid/Small-Cap Equities** | NASDAQ / Russell | `AGYS`, `PLTR`, `AMD`, `CRWD` |
| **Indian Equities (NSE)** | National Stock Exchange | `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `TATAMOTORS.NS` |
| **Indian Equities (BSE)** | Bombay Stock Exchange | `RELIANCE.BO`, `TCS.BO`, `INFY.BO`, `SRMENERGY.BO` |
| **ETFs & Indices** | Global | `SPY`, `QQQ`, `VTI`, `^GSPC` (S&P 500), `^NSEI` (Nifty 50) |

---

## 8. File & Directory Structure

```
StockMarketAI/
├── README.md                           # Single authoritative project & phases documentation
├── requirements.txt                    # Python dependencies
├── run_api.py                          # Unified launcher for REST API & Web Dashboard
├── run_backtest.py                     # CLI launcher for event-driven backtesting
├── start_stockmarketai.bat             # One-click Windows application launcher
│
├── models/                             # Trained Level 1 direction classification models
│   ├── stock_lr_model.pkl
│   ├── stock_rf_model.pkl
│   ├── feature_scaler.pkl
│   └── feature_columns.pkl
│
├── models_complete/                    # Trained Level 2 OHLC regression models & scalers
│   ├── tomorrow_open_*.pkl
│   ├── tomorrow_high_*.pkl
│   ├── tomorrow_low_*.pkl
│   ├── tomorrow_close_*.pkl
│   ├── feature_scaler.pkl
│   └── feature_columns.pkl
│
├── src/
│   ├── api/                            # Production FastAPI REST microservice
│   │   ├── app.py                      # App factory, CORS, static mounts
│   │   ├── schemas.py                  # Pydantic V2 request & response schemas
│   │   └── routes/                     # Health, predict, backtest, and scanner endpoints
│   │
│   ├── backtesting/                    # Event-driven backtesting & metrics engine
│   │   ├── backtest.py                 # Core backtesting simulator
│   │   ├── strategy.py                 # Direction, regression, and hybrid strategies
│   │   └── metrics.py                  # Sharpe, Sortino, Calmar, MDD calculations
│   │
│   └── stockmarketai/                  # Core package
│       ├── service.py                  # Singleton Reusable ML Service with TTL caching
│       ├── client.py                   # Python Developer SDK (in-process & HTTP modes)
│       ├── sentiment.py                # Financial news NLP sentiment engine
│       ├── data/                       # Ingestion, company search, and preprocessing
│       ├── features/                   # 65+ Quantitative feature engineering pipeline
│       ├── brokers/                    # Angel One & INDmoney connectors, circuit breaker
│       ├── integration/                # C-ABI DLL service & C# .NET wrapper
│       └── frontend/                   # Modern interactive dark-mode web terminal
│
├── flutter_app/                        # Native cross-platform client (iOS, Android, Desktop)
│   ├── pubspec.yaml
│   └── lib/
│
└── tests/                              # Automated test suite (24 passing unit/integration tests)
    ├── test_service.py
    ├── test_api.py
    ├── test_client.py
    └── test_integration.py
```

---

## 9. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
