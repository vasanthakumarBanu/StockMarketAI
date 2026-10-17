import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings
import joblib
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up directories
os.makedirs('models', exist_ok=True)

# 1. Load your processed data
print('\\n1?? Loading processed data...')
df = pd.read_csv('data/processed/reliance_ml_ready.csv', index_col='Date', parse_dates=True)
print(f'? Loaded: {df.shape[0]} rows, {df.shape[1]} columns')
print(f'?? Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}')

# 2. Explore target variable
print('\\n2?? Analyzing target variable...')
target_balance = df['Target'].mean()
print(f'?? Target distribution:')
print(f'   - Class 0 (Down/No move): {round((1-target_balance)*100, 1)}%')
print(f'   - Class 1 (Up): {round(target_balance*100, 1)}%')

# Check for class imbalance
if abs(target_balance - 0.5) > 0.1:
    print('??  Significant class imbalance detected - will use SMOTE')
else:
    print('? Reasonable class balance')

# 3. Prepare features and target
print('\\n3?? Preparing features and target...')
feature_cols = [col for col in df.columns if col != 'Target']
X = df[feature_cols]
y = df['Target']

print(f'?? Number of features: {len(feature_cols)}')
print(f'?? Feature names: {feature_cols[:5]}... (total: {len(feature_cols)})')

# Handle NaN values
missing_before = X.isnull().sum().sum()
print(f'?? Handling {missing_before} missing values...')
X = X.fillna(method='ffill').bfill()
missing_after = X.isnull().sum().sum()
print(f'? Missing values after processing: {missing_after}')

# 4. Create temporal train/test split
print('\\n4?? Creating temporal train/test split...')
split_idx = int(len(df) * 0.8)  # 80% for training, 20% for testing
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f'?? Training set:  {X_train.shape[0]} samples ({X_train.index.min().strftime('%Y-%m-%d')} to {X_train.index.max().strftime('%Y-%m-%d')})')
print(f'?? Test set:      {X_test.shape[0]} samples ({X_test.index.min().strftime('%Y-%m-%d')} to {X_test.index.max().strftime('%Y-%m-%d')})')

# 5. Feature scaling
print('\\n5?? Scaling features...')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('? Features standardized (mean=0, std=1)')

# 6. Handle class imbalance if needed
print('\\n6?? Checking for class imbalance...')
train_balance = y_train.mean()
print(f'?? Training set target balance: {train_balance:.3f} ({train_balance*100:.1f}% Class 1)')

if abs(train_balance - 0.5) > 0.1:
    print('??  Applying SMOTE to balance training data...')
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f'?? Balanced training set: {X_train_balanced.shape[0]} samples')
    print(f'   Original: {X_train_scaled.shape[0]} samples')
    print(f'   Synthetic samples added: {X_train_balanced.shape[0] - X_train_scaled.shape[0]}')
else:
    X_train_balanced, y_train_balanced = X_train_scaled, y_train
    print('? No significant imbalance - using original training data')

# 7. Train Logistic Regression
print('\\n7?? Training Logistic Regression...')
lr = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced' if abs(y_train.mean() - 0.5) > 0.1 else None,
    n_jobs=-1
)
lr.fit(X_train_balanced, y_train_balanced)

# Predictions
lr_pred = lr.predict(X_test_scaled)
lr_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]

# Metrics
lr_f1 = f1_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_pred_proba)

print(f'?? Logistic Regression Results:')
print(f'   - F1-Score: {lr_f1:.4f}')
print(f'   - ROC AUC:  {lr_auc:.4f}')

# 8. Train Random Forest
print('\\n8?? Training Random Forest...')
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced' if abs(y_train.mean() - 0.5) > 0.1 else None,
    n_jobs=-1
)
rf.fit(X_train_balanced, y_train_balanced)

# Predictions
rf_pred = rf.predict(X_test_scaled)
rf_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]

# Metrics
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_pred_proba)

print(f'?? Random Forest Results:')
print(f'   - F1-Score: {rf_f1:.4f}')
print(f'   - ROC AUC:  {rf_auc:.4f}')

# 9. Model comparison
print('\\n9?? Model Comparison...')
results_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'F1-Score': [lr_f1, rf_f1],
    'ROC AUC': [lr_auc, rf_auc]
})
print(results_df.to_string(index=False))

# Determine best model
best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
print(f'\\n?? Best model based on F1-Score: {best_model_name}')

# 10. Save model and preprocessing objects
print('\\n1??0?? Saving model and preprocessing objects...')

# Save the best model
if best_model_name == 'Random Forest':
    model_filename = 'stock_rf_model.pkl'
    joblib.dump(rf, f'models/{model_filename}')
else:
    model_filename = 'stock_lr_model.pkl'
    joblib.dump(lr, f'models/{model_filename}')

joblib.dump(scaler, 'models/feature_scaler.pkl')
joblib.dump(feature_cols, 'models/feature_columns.pkl')

print(f'?? Saved model: models/{model_filename}')
print('?? Saved scaler: models/feature_scaler.pkl')
print('?? Saved feature list: models/feature_columns.pkl')
print('? Modeling baseline complete!')

# Verify files were saved
print('\\n?? Verifying saved files:')
for file in ['models/stock_rf_model.pkl', 'models/stock_lr_model.pkl', 'models/feature_scaler.pkl', 'models/feature_columns.pkl']:
    if os.path.exists(file):
        print(f'   ? {file}')
    else:
        print(f'   ? {file}')
