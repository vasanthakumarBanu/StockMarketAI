import joblib

# Check what's in each model file
models = [
    'tomorrow_open_linear_regression.pkl',
    'tomorrow_high_linear_regression.pkl',
    'tomorrow_low_linear_regression.pkl',
    'tomorrow_close_linear_regression.pkl'
]

for model_file in models:
    print(f"\n{model_file}:")
    try:
        model = joblib.load(f"models_regression/{model_file}")
        print(f"  Model type: {type(model)}")
        if hasattr(model, 'coef_'):
            print(f"  Coefficients shape: {model.coef_.shape}")
            print(f"  First 5 coefficients: {model.coef_[:5]}")
            print(f"  Intercept: {model.intercept_}")
        if hasattr(model, 'feature_names_in_'):
            print(f"  Feature names: {model.feature_names_in_}")
    except Exception as e:
        print(f"  Error loading: {e}")