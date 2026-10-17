# StockMarketAI — Cross-Platform Flutter Client

> **Native Cross-Platform Financial Trading & AI Inference App for iOS, Android, macOS, Windows, and Web.**

---

## 📱 Features Included in the Flutter App

1. **Multi-Level Predictions (`DashboardScreen`)**:
   - Quick ticker chip selector (`MSFT`, `AAPL`, `NVDA`, `TSLA`, `AGYS`, `RELIANCE.NS`) or any custom ticker search.
   - **Level 3 AI Trading Recommendation Hero Card** (`STRONG BUY`, `BUY`, `HOLD`, `SELL`, `STRONG SELL`) with dynamic signal glow and confidence percentage.
   - **Master Auto-Execution Boolean Switch**: Toggle button for live broker execution safety (`False` = Safe Lock, `True` = Active Trading).
   - Instant order routing to **Angel One SmartAPI** or **INDmoney**.
   - Level 2 Tomorrow's Forecasted OHLC (Open, High, Low, Close) targets.
   - 65+ Technical indicator snapshot (RSI, MACD, MA20, ATR, Volatility).
   - Real-time VADER news sentiment polarity badge.

2. **Phase 4 Strategy Backtesting (`BacktestScreen`)**:
   - Strategy selector (`Enhanced Hybrid`, `ML Directional`, `Momentum`, `Mean Reversion`).
   - Interactive, smooth touch-enabled **Equity Curve Chart** using `fl_chart`.
   - Comprehensive performance metrics (Total Return %, Sharpe Ratio, Max Drawdown %, Win Rate).

3. **Market Opportunity Scanner (`ScannerScreen`)**:
   - Scans full watchlists simultaneously and ranks equities by signal confidence and expected return.
   - One-tap "Inspect" to load any stock into the prediction engine.

4. **Dynamic Backend Configuration (`SettingsScreen`)**:
   - Configure the FastAPI backend URL (e.g., `http://10.0.2.2:8000` for Android emulator, `http://localhost:8000` for desktop/iOS, or your cloud VPS URL `https://api.yourdomain.com`).
   - One-tap connection test.

---

## 🚀 How to Run the Flutter App

### Prerequisites
1. Download and install Flutter: [https://docs.flutter.dev/get-started/install](https://docs.flutter.dev/get-started/install)
2. Ensure Flutter is added to your system `PATH`.
3. Verify installation in your terminal:
   ```bash
   flutter doctor
   ```

---

### Step 1: Start the Python Backend Server
Before launching the Flutter app, ensure the StockMarketAI FastAPI server is running:

```powershell
# From the project root:
python run_api.py --host 0.0.0.0 --port 8000
```

---

### Step 2: Install Flutter Dependencies
Navigate into the `flutter_app` folder:

```bash
cd flutter_app
flutter pub get
```

---

### Step 3: Run the App

#### A. Run on Android Emulator or Physical Android Phone
```bash
flutter run -d android
```
*(Note: Android emulators connect to your local PC via `http://10.0.2.2:8000`, which is configured as the default.)*

#### B. Run on iPhone Simulator or Physical iPhone (macOS required)
```bash
flutter run -d ios
```

#### C. Run on Windows Native Desktop App
```bash
flutter run -d windows
```

#### D. Run on macOS Native Desktop App
```bash
flutter run -d macos
```

#### E. Run in Chrome Browser
```bash
flutter run -d chrome
```

---

## 📦 How to Build Production Releases

### 1. Build Android APK
```bash
flutter build apk --release
```
The output file will be located at:
`flutter_app/build/app/outputs/flutter-apk/app-release.apk`
You can transfer this `.apk` to any Android phone and install it directly!

### 2. Build iOS Archive (App Store / TestFlight)
```bash
flutter build ipa --release
```

### 3. Build Windows Executable (.exe)
```bash
flutter build windows --release
```
The standalone `.exe` and DLL dependencies will be in:
`flutter_app/build/windows/runner/Release/`

### 4. Build macOS App (.app)
```bash
flutter build macos --release
```

---

## 📁 Flutter Project Architecture

```
flutter_app/
├── pubspec.yaml                 # Dependencies (http, fl_chart, google_fonts, shared_preferences)
├── README.md                    # Setup and build manual
└── lib/
    ├── main.dart                # App entry point, dark theme & bottom navigation
    ├── models/
    │   ├── prediction_model.dart # 3-Level prediction data contract
    │   ├── backtest_model.dart   # Equity curve & performance metrics
    │   └── broker_model.dart     # Auto-execution boolean toggle & order models
    ├── services/
    │   └── api_service.dart      # REST API client connecting to FastAPI
    ├── widgets/
    │   ├── hero_signal_card.dart # Level 3 Signal hero badge, confidence & stops
    │   └── auto_execution_toggle.dart # Master boolean button switch & execution
    └── screens/
        ├── dashboard_screen.dart # Predictions, indicators & quick tickers
        ├── backtest_screen.dart  # Interactive charts & backtesting
        ├── scanner_screen.dart   # Watchlist scanner & ranker
        └── settings_screen.dart  # Server IP configuration & health check
```
