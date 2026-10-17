import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
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

# Prepare features and targets
X = df[feature_columns].copy()
X = X.ffill().bfill()

targets = ['Tomorrow_Open', 'Tomorrow_High', 'Tomorrow_Low', 'Tomorrow_Close']
y = df[targets].copy()

# Split data
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a separate model for each target
print("Training separate models for each target:")
print("="*60)

models = {}
for target_name in targets:
    print(f"\nTraining model for {target_name}")

    y_train_target = y_train[target_name]
    y_test_target = y_test[target_name]

    # Create and train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_target)

    # Test
    y_pred = model.predict(X_test_scaled)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test_target, y_pred)

    models[target_name] = model

    print(f"  R2: {r2:.6f}")
    if hasattr(model, 'coef_'):
        print(f"  Coefficients: {model.coef_}")
        print(f"  Intercept: {model.intercept_}")

# Now compare the models
print("\n" + "="*60)
print("COMPARING MODELS")
print("="*60)

target_names = list(models.keys())
for i in range(len(target_names)):
    for j in range(i+1, len(target_names)):
        target1 = target_names[i]
        target2 = target_names[j]
        model1 = models[target1]
        model2 = models[target2]

        # Check if coefficients are identical
        coef_equal = np.allclose(model1.coef_, model2.coef_)
        intercept_equal = np.allclose(model1.intercept_, model2.intercept_)

        print(f"\n{target1} vs {target2}:")
        print(f"  Coefficients equal: {coef_equal}")
        print(f"  Intercept equal: {intercept_equal}")

        if not coef_equal:
            diff = np.abs(model1.coef_ - model2.coef_)
            print(f"  Max coefficient difference: {np.max(diff):.10f}")
            print(f"  Mean coefficient difference: {np.mean(diff):.10f}")

# Let's also check what the actual y values look like
print("\n" + "="*60)
print("CHECKING TARGET VARIABLES")
print("="*60)

for target_name in targets:
    y_values = y[target_name]
    print(f"{target_name}:")
    print(f"  Min: {y_values.min():.6f}")
    print(f"  Max: {y_values.max():.6f}")
    print(f"  Mean: {y_values.mean():.6f}")
    print(f"  Std: {y_values.std():.6f}")

# Check correlation between targets
print("\nCorrelation matrix of target variables:")
corr_matrix = y.corr()
print(corr_matrix.to_string())