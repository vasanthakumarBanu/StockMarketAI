# StockMarketAI: Comprehensive Project Phases (Phases 1 to 7)

> **Universal Multi-Market Quantitative AI, Predictive Modeling, Strategy Backtesting, Broker Execution & Cross-Platform Client Architecture**  
> *Status: All Phases Implemented, Tested & Verified (24/24 Automated Tests Passing)*

---

## 📑 Table of Contents

1. [Architectural Overview & Roadmap](#1-architectural-overview--roadmap)
2. [Phase 1: Universal Data Ingestion & Preprocessing Engine](#2-phase-1-universal-data-ingestion--preprocessing-engine)
3. [Phase 2: 65+ Quantitative Feature Engineering Pipeline](#3-phase-2-65-quantitative-feature-engineering-pipeline)
4. [Phase 3: Multi-Level Machine Learning Modeling & Signals](#4-phase-3-multi-level-machine-learning-modeling--signals)
5. [Phase 4: Realistic Event-Driven Backtesting & Strategy Engine](#5-phase-4-realistic-event-driven-backtesting--strategy-engine)
6. [Phase 5: Reusable ML Service, Production REST API & Interactive Dashboard](#6-phase-5-reusable-ml-service-production-rest-api--interactive-dashboard)
7. [Phase 6: Platform Integration, C-Compatible DLL & Broker Gateways](#7-phase-6-platform-integration-c-compatible-dll--broker-gateways)
8. [Phase 7: Cross-Platform Native Client (Flutter App)](#8-phase-7-cross-platform-native-client-flutter-app)
9. [Phase 8: Automated Verification & Test Suite Matrix](#9-phase-8-automated-verification--test-suite-matrix)
10. [Quickstart Command Reference](#10-quickstart-command-reference)

---

## 1. Architectural Overview & Roadmap

StockMarketAI is an institutional-grade, end-to-end quantitative trading and AI forecasting system designed to bridge raw market feeds, machine learning inference, risk-managed backtesting, and live broker execution across global and Indian markets.

### End-to-End System Pipeline

```
[ Stock Search / Autocomplete ] ──► [ Universal Data Ingestion (US / NSE / BSE) ]
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

## 2. Phase 1: Universal Data Ingestion & Preprocessing Engine

### Objective
Build a dynamic, stock-agnostic market ingestion layer capable of loading daily and intraday OHLCV (Open, High, Low, Close, Volume) data for any global or Indian ticker, complete with company name resolution, timezone normalization, and local caching.

### Implemented Modules
- [`src/stockmarketai/data/load_stock_data.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/data/load_stock_data.py): Automated Yahoo Finance daily ingestion with configurable lookback periods.
- [`src/stockmarketai/data/search.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/data/search.py): Intelligent company name resolver and autocomplete supporting US equities, NSE (`.NS`), and BSE (`.BO`) stocks.
- [`src/stockmarketai/data/preprocessing.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/data/preprocessing.py): Data cleaning, UTC timezone conversion, NaN/Infinite value cleansing, and chronological validation.

### Key Capabilities
1. **Dynamic Universal Fetching**: Fully dynamic fetching with zero hardcoded CSV requirements. Any ticker (e.g., `MSFT`, `AAPL`, `NVDA`, `TSLA`, `RELIANCE.NS`, `SRMENERGY.BO`) downloads automatically on demand.
2. **Company Name Resolution**: Translates natural company queries into precise ticker symbols:
   - `"Tata Motors"` $\rightarrow$ `TATAMOTORS.NS`
   - `"SRM Energy"` $\rightarrow$ `SRMENERGY.BO`
   - `"Microsoft"` $\rightarrow$ `MSFT`
3. **Robust Sanitization**: Handles dividend/split adjustments, timezone alignment, forward/backward filling for missing ticks, and infinite ratio sanitization.
4. **Caching Layer**: Persists raw data into `src/stockmarketai/data/raw/{SYMBOL}.csv` alongside high-speed in-memory TTL caching (5-minute expiration) to prevent rate limits.

---

## 3. Phase 2: 65+ Quantitative Feature Engineering Pipeline

### Objective
Transform raw price bars into a comprehensive, multi-dimensional feature space covering momentum, trend, volatility, volume interactions, candlestick patterns, and time lags.

### Implemented Modules
- [`src/stockmarketai/features/feature_engineering.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/features/feature_engineering.py): Primary 65+ feature generator.
- [`Complete Feature Matrix`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/models_complete/feature_columns.pkl): Serialized list of calibrated feature inputs.

### Feature Architecture

| Category | Indicators & Factors | Quantitative Purpose |
| :--- | :--- | :--- |
| **Trend Indicators** | `MA_20`, `MA_50`, `MA_200`, `EMA_20`, `EMA_50`, `EMA_200` | Identify short-, medium-, and long-term market trends and dynamic support/resistance. |
| **Momentum Indicators** | `RSI_14`, `MACD`, `MACD_Signal`, `MACD_Hist` | Quantify overbought/oversold levels and velocity divergences. |
| **Volatility Indicators** | `BB_Upper`, `BB_Middle`, `BB_Lower`, `ATR_14`, `Volatility_20` | Measure price dispersion, market expansion/compression, and dynamic risk boundaries. |
| **Candlestick Geometry** | `Body_Size`, `Upper_Wick`, `Lower_Wick`, `Body_Range_Ratio`, `Is_Bullish` | Capture intraday buying vs. selling pressure and exhaustion wicks. |
| **Volume Dynamics** | `Volume_MA_5`, `Volume_MA_20`, `Volume_Ratio_5_20`, `PV_Trend`, `PV_Correlation` | Detect institutional accumulation or distribution preceding price moves. |
| **Time-Series Lags** | $t-1, t-2, t-3$ of OHLCV; $t-1, t-2, t-3, t-5, t-10$ Returns; Lagged RSI & MACD | Feed temporal history to models without future lookahead bias. |

### Data Leakage Prevention
- **Temporal Splitting**: Strict 80/20 train/test chronological split (no shuffling).
- **Scale Isolation**: `StandardScaler` fitted strictly on training slices before transforming test evaluation sets.

---

## 4. Phase 3: Multi-Level Machine Learning Modeling & Signals

### Objective
Implement a multi-tier prediction system combining directional classification, multi-target OHLC price regression, and actionable trading signals with dynamic risk boundaries.

### Architecture

```
[ 65+ Engineered Features ]
           │
           ├───────────────────────────────┐
           ▼                               ▼
[ Level 1: Direction Classifier ]  [ Level 2: Multi-Target OHLC Regressors ]
- Logistic Regression / RF         - Tomorrow Open Regressor
- Output: P(UP), P(DOWN)           - Tomorrow High Regressor
- Direction Confidence Score       - Tomorrow Low Regressor
                                   - Tomorrow Close Regressor
                                   - 95% Confidence Intervals
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
             [ Level 3: AI Trading Decision Engine ]
             - Signal: STRONG BUY, BUY, HOLD, SELL, STRONG SELL
             - Dynamic ATR-Based Stop-Loss (1.5x - 2.0x ATR)
             - Dynamic Take-Profit (2.5x - 3.0x ATR)
             - Risk / Reward Ratio Calculation
             - Integrated Financial News NLP Sentiment
```

### Level Breakdown & Verified Accuracy

#### Level 1: Direction Classification
- **Models**: Logistic Regression (`models/stock_lr_model.pkl`) and Random Forest (`models/stock_rf_model.pkl`).
- **Target**: Direction flag: $1$ if $Close_{t+1} > Close_t$ else $0$.
- **Performance**: Upward prediction probability calibrated with probability distributions.

#### Level 2: Multi-Target OHLC Price Regression
- **Models**: Independent Linear Regression & Random Forest regressors per OHLC target.
- **Targets**: $Open_{t+1}, High_{t+1}, Low_{t+1}, Close_{t+1}$.
- **Performance Metrics Verified Across Equities**:
  - **MSFT**: Open $R^2 = 0.9794$, High $R^2 = 0.9608$, Low $R^2 = 0.9730$, Close $R^2 = 0.9551$
  - **AGYS**: Open $R^2 = 0.9540$, High $R^2 = 0.9164$, Low $R^2 = 0.9467$, Close $R^2 = 0.9017$

#### Level 3: AI Trading Decision Engine & NLP Sentiment
- **Signal Logic**: Synthesizes directional probability threshold ($\ge 0.55$ for Buy, $\le 0.45$ for Sell) with expected percentage return from regression targets.
- **Dynamic ATR Risk Bounds**:
  $$\text{Stop-Loss} = \text{Entry Price} - (1.5 \times \text{ATR}_{14})$$
  $$\text{Take-Profit} = \text{Entry Price} + (2.5 \times \text{ATR}_{14})$$
- **NLP Sentiment**: Integrated financial news sentiment scoring via [`src/stockmarketai/sentiment.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/sentiment.py) adjusts confidence score.

---

## 5. Phase 4: Realistic Event-Driven Backtesting & Strategy Engine

### Objective
Provide realistic, institutional-grade simulation eliminating lookahead bias, accounting for execution slippage and commissions, and calculating complete quantitative risk metrics.

### Implemented Modules
- [`src/backtesting/backtest.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/backtesting/backtest.py): Event-driven backtesting execution engine.
- [`src/backtesting/strategy.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/backtesting/strategy.py): Pluggable strategies (`DirectionStrategy`, `RegressionExpectedReturnStrategy`, `EnhancedHybridStrategy`).
- [`src/backtesting/metrics.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/backtesting/metrics.py): Quantitative performance calculations.
- [`run_backtest.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/run_backtest.py): CLI interface for custom parameters.

### Realism Constraints
1. **Zero Lookahead Bias**: Signals calculated at Day $t$ Close; trades filled at Day $t+1$ Open.
2. **Transaction Costs**: Default 0.05% slippage and 0.10% commission fees deducted per fill.
3. **Intraday Stop-Loss / Take-Profit**: Evaluated against Day $t+1$ High and Low bars.
4. **Validation Modes**: Out-of-sample temporal split (80/20) and rolling walk-forward validation.

### Backtest Benchmark Results

| Metric | MSFT (Enhanced Hybrid) | AGYS (ATR Exits) | Benchmark (Buy & Hold) |
| :--- | :--- | :--- | :--- |
| **Initial Capital** | $10,000.00 | $10,000.00 | $10,000.00 |
| **Final Equity** | **$11,017.69 (+10.18%)** | **$11,372.76 (+13.73%)** | +29.91% |
| **Max Drawdown (MDD)** | **-3.39%** | **-4.12%** | **-23.38%** (~7x lower drawdown) |
| **Sharpe Ratio** | **1.71** | **1.45** | 0.88 |
| **Sortino Ratio** | **2.21** | **1.92** | 1.05 |
| **Win Rate** | **66.7%** (4 wins / 2 losses) | **50.0%** (5 wins / 5 losses) | N/A |
| **Profit Factor** | **4.05** | **1.46** | N/A |
| **Win/Loss Ratio** | **2.03** | **1.41** | N/A |

---

## 6. Phase 5: Reusable ML Service, Production REST API & Interactive Dashboard

### Objective
Create a unified in-memory service layer, production-grade FastAPI microservice, dark-mode financial web terminal, and developer Python SDK.

### Implemented Modules
- [`src/stockmarketai/service.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/service.py): Singleton service with TTL caching and inference orchestration.
- [`src/api/app.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/api/app.py) & [`src/api/routes/`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/api/routes/): REST API endpoints with Pydantic V2 schemas.
- [`src/stockmarketai/frontend/`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/frontend/): Modern web terminal (HTML5/CSS3/Vanilla JS + Chart.js).
- [`src/stockmarketai/client.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/client.py): Developer SDK for in-process and remote HTTP calls.
- [`run_api.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/run_api.py): FastAPI/Uvicorn server launcher.

### Production REST API Endpoint Catalog

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

### Interactive Financial Dashboard Features
- **Signal Hero Banner**: Glowing color-coded recommendation card (STRONG BUY, BUY, HOLD, SELL, STRONG SELL) with confidence ring and ATR risk-reward ratio.
- **Interactive Price Cards**: Today's Actual OHLC vs. Tomorrow's Predicted OHLC with expected change % and dollar differentials.
- **Interactive Backtester Tab**: Run custom backtests; render Chart.js interactive equity curves vs. Buy & Hold benchmark with trade markers and metrics cards.
- **Multi-Stock Scanner Tab**: One-click batch scanner ranking opportunities by confidence and return.
- **Master Live Safety Toggle**: UI toggle switch controlling automated broker order transmission.

---

## 7. Phase 6: Platform Integration, C-Compatible DLL & Broker Gateways

### Objective
Enable integration with native desktop platforms (MetaTrader, NinjaTrader, custom WPF/.NET terminals) and provide direct order execution gateways to Indian and US brokers.

### Implemented Modules
- [`src/stockmarketai/integration/dll_service.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/integration/dll_service.py): C-ABI Python DLL Service exporting `c_predict`, `c_backtest`, and `c_status`.
- [`src/stockmarketai/integration/csharp/StockPredictionWrapper.cs`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/integration/csharp/StockPredictionWrapper.cs): .NET Standard 2.0 C# wrapper implementing `IStockPredictionService`.
- [`src/stockmarketai/brokers/angel_one.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/brokers/angel_one.py): Angel One SmartAPI connector for Indian equities (NSE/BSE).
- [`src/stockmarketai/brokers/indmoney.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/brokers/indmoney.py): INDmoney connector for US equities & fractional shares.
- [`src/stockmarketai/brokers/base.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/brokers/base.py) & [`execution_engine.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/src/stockmarketai/brokers/execution_engine.py): Automated trade execution engine with risk circuit breakers.

### Core Capabilities
1. **C# .NET Platform Wrapper**:
   - Strongly typed models: `PredictionResult`, `TradingSignal`, `PriceTargets`, `BacktestResults`.
   - Synchronous and asynchronous methods (`PredictAsync`, `BacktestAsync`).
   - Compatible with Windows .NET Framework 4.7+, .NET Core 3.1+, and .NET 6/7/8.
2. **Broker Connectors**:
   - **Angel One SmartAPI**: TOTP authentication, live market quote polling, order routing (LIMIT, MARKET, STOPLOSS), position tracking.
   - **INDmoney**: US stock allocation, fractional share orders, portfolio tracking.
3. **Master Safety Boolean Switch**:
   - Live automated order execution is governed by `auto_execution_enabled: bool` (defaults to `False`).
   - Requires explicit operator activation before real trades can be dispatched.
4. **Automated Risk Engine & Circuit Breaker**:
   - Automated portfolio circuit breaker automatically trips if portfolio daily loss exceeds the defined limit (e.g. -5.0%).
   - Position sizing constraints, maximum concurrent positions, and minimum signal confidence filters.

---

## 8. Phase 7: Cross-Platform Native Client (Flutter App)

### Objective
Deliver a responsive, native multi-platform client application for mobile (iOS, Android) and desktop (Windows, macOS) in [`flutter_app/`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/flutter_app/).

### Structure & Modules
```
flutter_app/
├── pubspec.yaml                        # Dependencies (fl_chart, http, provider, google_fonts)
├── README.md                           # Flutter setup & launch documentation
└── lib/
    ├── main.dart                       # App entry point & dark theme configuration
    ├── models/
    │   ├── prediction.dart             # Typed Dart models for 3-Level predictions
    │   └── backtest_result.dart        # Typed models for backtest performance & equity curves
    ├── services/
    │   └── api_service.dart            # HTTP client connecting to StockMarketAI REST API
    └── screens/
        ├── dashboard_screen.dart       # Live signal cards, OHLC forecast, and stock search
        ├── backtest_screen.dart        # Interactive strategy runner with fl_chart equity curves
        └── watchlist_screen.dart       # Multi-stock opportunity scanner & broker controls
```

### Features
- **Fluid Dark Financial Theme**: Glassmorphic styling, neon signal badges, and typography matching the web terminal.
- **Live Search & Autocomplete**: Search US and Indian companies with instant ticker resolution.
- **Interactive Charts**: Render equity curves and price forecasts natively via `fl_chart`.
- **Master Execution Toggle**: Visual switch to activate/deactivate live automated broker trade routing.

---

## 9. Phase 8: Automated Verification & Test Suite Matrix

### Objective
Ensure 100% test pass rate across all components, verifying service layer logic, API endpoints, SDK clients, and broker integration.

### Test Execution Command
```powershell
python -m unittest discover tests
```

### Test Verification Matrix (24/24 Tests Passing)

| Test Module | Coverage Area | Assertions & Verified Behaviors | Status |
| :--- | :--- | :--- | :---: |
| [`test_service.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/tests/test_service.py) | `StockMarketAIService` | - Multi-level inference schema validation<br>- Direction probability & confidence scores<br>- OHLC price targets consistency<br>- In-memory caching TTL verification | ✅ PASS |
| [`test_api.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/tests/test_api.py) | FastAPI REST Endpoints | - `/health` 200 OK<br>- `/api/v1/status` system metadata<br>- `/api/v1/predict/{symbol}` for US & Indian equities<br>- `/api/v1/backtest` execution<br>- `/api/v1/stocks/scan` watchlist ranking | ✅ PASS |
| [`test_client.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/tests/test_client.py) | `StockMarketAIClient` | - In-process execution mode<br>- Remote HTTP execution mode<br>- Prediction, backtest, and scan methods | ✅ PASS |
| [`test_integration.py`](file:///f:/IT/AI%20and%20ML/Data%20Science/Projects/StockMarketAI/tests/test_integration.py) | DLL, C# ABI & Brokers | - `c_predict`, `c_backtest`, `c_status` JSON ABI<br>- C# model structure compatibility<br>- Angel One & INDmoney connector initialization<br>- Circuit breaker trip on -5% loss<br>- Master safety switch enforcement | ✅ PASS |

---

## 10. Quickstart Command Reference

### Launching the Application
```powershell
# Option A: Start REST API & Web Dashboard via batch launcher
.\start_stockmarketai.bat

# Option B: Run API server directly
python run_api.py --host 127.0.0.1 --port 8000
```
- **Web Terminal**: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
- **OpenAPI / Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

### CLI Predictions & Analysis
```powershell
# Instant AI prediction for US equities
python -m src.stockmarketai.cli.predict_any_stock MSFT

# Instant AI prediction for Indian equities via company name
python -m src.stockmarketai.cli.predict_any_stock "Tata Motors"
python -m src.stockmarketai.cli.predict_any_stock "SRM Energy"
python -m src.stockmarketai.cli.predict_any_stock RELIANCE.NS
```

### Event-Driven Backtesting CLI
```powershell
# Default backtest (MSFT, Enhanced Hybrid Strategy)
python run_backtest.py MSFT

# Backtest with dynamic ATR stop-loss and take-profit
python run_backtest.py AGYS --use-atr

# Backtest with custom capital and risk limits
python run_backtest.py AAPL --strategy direction --initial-capital 25000 --stop-loss 0.02 --take-profit 0.05
```

### Cross-Platform Flutter App
```powershell
cd flutter_app
flutter pub get

# Launch on Windows Desktop
flutter run -d windows

# Launch on Android Phone / Emulator
flutter run -d android
```

### Running Test Verification Suite
```powershell
python -m unittest discover tests
```
