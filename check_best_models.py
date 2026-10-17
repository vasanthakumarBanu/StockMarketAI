import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Recreate the best models selection process to see what's happening
data_path = "src/stockmarketai/data/processed/reliance_ml_ready.csv"
df = pd.read_csv(data_path, index_col="Date", parse_dates=True)

# Create tomorrow's OHLC targets (shift by -1)
df['Tomorrow_Open'] = df['Open'].shift(-1)
df['Tomorrow_High'] = df['High'].shift(-1)
df['Tomorrow_Low'] = df['Low'].shift(-1)
df['Tomorrow_Close'] = df['Close'].shift(-1)

# Remove last row (will have NaN targets)
df = df[:-1]

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
X = X.ffill().bfill()  # Forward fill then backward fill

# Define targets
targets = ['Tomorrow_Open', 'Tomorrow_High', 'Tomorrow_Low', 'Tomorrow_Close']
y = df[targets].copy()

# Split data (temporal split - important for time series)
split_idx = int(len(X) * 0.8)  # 80% for training, 20% for testing
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
        # Train model
        model.fit(X_train_scaled, y_train_target)

        # Predictions
        y_pred_test = model.predict(X_test_scaled)

        # Metrics
        test_mae = mean_absolute_error(y_test_target, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test_target, y_pred_test))
        test_r2 = r2_score(y_test_target, y_pred_test)

        target_results[model_name] = {
            'model': model,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'y_pred_test': y_pred_test
        }

        print(f"{model_name} Results for {target_name}:")
        print(f"   Test MAE:  {test_mae:.4f}")
        print(f"   Test RMSE: {test_rmse:.4f}")
        print(f"   Test R2:   {test_r2:.4f}")

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

# Now check what models we actually have
print(f"\n{'='*70}")
print("CHECKING BEST MODELS DICTIONARY")
print(f"{'='*70}")

for target_name in targets:
    model_info = best_models[target_name]
    model = model_info['model']
    print(f"\n{target_name} ({model_info['model_name']}):")
    if hasattr(model, 'coef_'):
        print(f"  First 5 coefficients: {model.coef_[:5]}")
        print(f"  Intercept: {model.intercept_}")