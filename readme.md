# **LSTM Price Prediction Web App**

This project is a web-based application that predicts the next month's price using a trained LSTM (Long Short-Term Memory) model. The application uses Flask as the backend and provides both a user-friendly web interface and a JSON API endpoint for predictions.

## Project Overview

The app allows users to enter the last four months of prices, processes them using a saved MinMaxScaler, and generates a forecast using a pre-trained LSTM model. This makes it ideal for time-series forecasting tasks such as commodity prices, stock trends, or monthly sales predictions.

## Key Features

LSTM-powered forecasting for next-month price prediction

Flask web app with a simple front-end (index.html)

REST API endpoint for programmatic predictions

Scalable architecture using separate saved model and scaler files

Validations and error handling built into the API

## Files Included

app.py — Flask backend for serving the web page and handling predictions

lstm_model.keras — Trained LSTM model

minmax_scaler.joblib — Scaler used during training

templates/index.html — Web interface for user input

## How It Works

User enters the most recent 4 monthly prices

Data is scaled with the pre-trained MinMaxScaler

The LSTM model predicts the next month's value

Output is returned in the original scale for easy interpretation

## Available Routes

**Home Page:**
Displays the UI where users can enter price values.

**Prediction Endpoint:**
Accepts JSON input and returns the forecast in JSON format.

## Use Cases

Commodity price forecasting

Energy and fuel price prediction

Financial time-series estimation

Retail demand or monthly sales prediction

## Requirements

Requires Python and common ML deployment libraries such as Flask, TensorFlow/Keras, and Joblib.

**Link:** http://127.0.0.1:8000

<img width="580" height="643" alt="Screenshot 2025-12-09 104339" src="https://github.com/user-attachments/assets/0c04a89b-c3d0-4f24-830b-1b80d6a72545" />

