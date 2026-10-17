#!/usr/bin/env python3
"""
Script to diagnose the model file integrity and loading issues.
"""

import pickle
import sys

# Define the path to the model file
model_path = 'models_complete/tomorrow_open_linear_regression.pkl'

try:
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    print('Model loaded successfully!')
    print(f"Model type: {type(data)}")
except Exception as e:
    print(f"Error details: {type(e).__name__}: {e}")

    # Additional diagnostics
    try:
        with open(model_path, 'rb') as f:
            first_bytes = f.read(100)  # Read first 100 bytes
        print(f"First 100 bytes of the file: {first_bytes}")
    except Exception as read_error:
        print(f"Error reading first bytes: {type(read_error).__name__}: {read_error}")

print(f"Path to model file: {model_path}")