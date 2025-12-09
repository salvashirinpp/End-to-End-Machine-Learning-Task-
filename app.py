from flask import Flask, request, jsonify, render_template # <-- REQUIRED Flask Imports
from tensorflow import keras
import numpy as np
import joblib 
import os
import sys

# --- Configuration ---
MODEL_PATH = 'lstm_model.keras'
SCALER_PATH = 'minmax_scaler.joblib'
SEQUENCE_LENGTH = 4

# --- Load Assets ---
try:
    lstm_model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and Scaler loaded successfully.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to load assets: {e}")
    lstm_model = None # Set to None so the prediction route can fail gracefully
    scaler = None
    # We remove sys.exit(1) here so the server starts, allowing us to see the error in the browser

# --- Flask App Initialization ---
app = Flask(__name__)

# ----------------------------------------------------
# --- WEB ROUTES (Using JSON) ---
# ----------------------------------------------------

# Route 1: Serve the HTML Page
@app.route("/")
def home():
    return render_template("index.html")

# Route 2: Handle Predictions
@app.route("/predict", methods=["POST"])
def predict():
    # 1. Check if model is loaded (Handles initial loading failure)
    if lstm_model is None or scaler is None:
        return jsonify({"detail": "Model not loaded"}), 500

    # 2. Get JSON data from the browser
    data = request.get_json()
    
    # 3. Validation
    if not data or "recent_prices" not in data:
        return jsonify({"detail": "Missing 'recent_prices' in input"}), 400
    
    prices = data["recent_prices"]
    
    if len(prices) != SEQUENCE_LENGTH:
        return jsonify({"detail": f"Expected {SEQUENCE_LENGTH} prices, got {len(prices)}"}), 400

    # 4. Processing (This is where the CLI logic is now)
    try:
        input_array = np.array(prices).reshape(-1, 1)
        scaled_input = scaler.transform(input_array)
        lstm_input = scaled_input.reshape(1, SEQUENCE_LENGTH, 1)

        # Prediction
        predicted_scaled = lstm_model.predict(lstm_input, verbose=0)
        predicted_original = scaler.inverse_transform(predicted_scaled)
        
        forecast = predicted_original.item() 

        return jsonify({
            "status": "success",
            "next month price forecast USD": round(forecast, 2)
        })
        
    except Exception as e:
        # Catch errors during scaling or prediction
        print(f"Prediction Error: {e}")
        return jsonify({"detail": f"Processing failed: {str(e)}"}), 500


# --- Main Block to Run the App (No CLI logic here) ---
if __name__ == "__main__":
    app.run(debug=True, port=8000)