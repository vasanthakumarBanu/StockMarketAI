"""
StockMarketAI: High-Precision Ensemble & Target Predictor Framework
===================================================================
Provides:
1. WeightedEnsembleRegressor: Blends Linear Regression, Random Forest, and ANN
   using inverse-variance / inverse-MSE optimization weights.
2. PriceTargetPredictor: Reconstructs exact next-day dollar price targets from
   statistically stationary percentage return forecasts:
       Tomorrow_Price = Today_Close * (1.0 + Predicted_Return)
"""

import numpy as np
import pandas as pd


class WeightedEnsembleRegressor:
    """
    Blends multiple regression models (e.g. Ridge, Random Forest, MLP)
    using normalized performance weights.
    """
    def __init__(self, models_dict, weights_dict=None):
        self.models = models_dict
        if weights_dict is None:
            # Default to equal weighting
            n = len(models_dict)
            self.weights = {k: 1.0 / n for k in models_dict.keys()}
        else:
            self.weights = weights_dict

    def predict(self, X):
        """Compute weighted blend prediction."""
        if hasattr(X, "shape"):
            n_samples = X.shape[0]
        else:
            n_samples = len(X)

        blended = np.zeros(n_samples, dtype=float)
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w > 0:
                blended += w * np.asarray(model.predict(X), dtype=float)
        return blended


class PriceTargetPredictor:
    """
    Wraps any percentage-return model (individual or ensemble) to forecast
    the exact tomorrow dollar price targets.
    
    Mathematical Formulation:
        1. Reconstruct raw today close: Close = (z_close * scale) + mean
        2. Predict percentage return:   r = Model.predict(X_scaled)
        3. Project tomorrow's price:    Tomorrow_Target = Close * (1.0 + r)
    """
    def __init__(self, pct_model, close_mean, close_scale, close_col_idx=0, model_name="Ensemble"):
        self.pct_model = pct_model
        self.close_mean = float(close_mean)
        self.close_scale = float(close_scale)
        self.close_col_idx = int(close_col_idx)
        self.model_name = str(model_name)

    def predict(self, X_scaled):
        """Predict exact tomorrow dollar price."""
        arr = np.asarray(X_scaled)
        if arr.ndim == 1:
            arr_2d = arr.reshape(1, -1)
            z_close = float(arr[self.close_col_idx])
            raw_close = (z_close * self.close_scale) + self.close_mean
            pred_pct = float(np.asarray(self.pct_model.predict(arr_2d)).ravel()[0])
            return float(raw_close * (1.0 + pred_pct))

        if hasattr(X_scaled, "values"):
            z_close = X_scaled.values[:, self.close_col_idx]
        elif isinstance(X_scaled, pd.DataFrame):
            z_close = X_scaled.iloc[:, self.close_col_idx].values
        else:
            z_close = arr[:, self.close_col_idx]

        # Exact unscaling to get raw Today_Close
        raw_close = (z_close * self.close_scale) + self.close_mean
        pred_pct = np.asarray(self.pct_model.predict(X_scaled), dtype=float)
        return raw_close * (1.0 + pred_pct)

