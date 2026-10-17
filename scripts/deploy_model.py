#!/usr/bin/env python3
"""
Script to deploy the best-performing model as a REST API using Flask.
"""

from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the model and feature scaler
model_path = 'models_complete/tomorrow_open_linear_regression.pkl'
feature_scaler_path = 'models_complete/feature_scaler.pkl'
feature_columns_path = 'models_complete/feature_columns.pkl'

try:
    model = pickle.load(open(model_path, 'rb'))
    feature_scaler = pickle.load(open(feature_scaler_path, 'rb'))
    feature_columns = pickle.load(open(feature_columns_path, 'rb'))
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Convert data to DataFrame
        df = pd.DataFrame([data])

        # Ensure the input data has the same features as the model expects
        if not all(col in df.columns for col in feature_columns):
            missing_cols = [col for col in feature_columns if col not in df.columns]
            return jsonify({"error": f"Missing columns: {missing_cols}"}), 400

        # Scale the input data
        scaled_data = feature_scaler.transform(df[feature_columns])

        # Make prediction
        prediction = model.predict(scaled_data)

        # Return prediction
        return jsonify({"prediction": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    print("Model API is running on http://0.0.0.0:5000")