import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the models and scaler
close_model = joblib.load("models/Close_Linear Regression.joblib")
rsi_model = joblib.load("models/rsi_Linear Regression.joblib")
macd_histogram_model = joblib.load("models/macd_histogram_Linear Regression.joblib")
scaler = joblib.load("models/scaler.joblib")

# Assuming you have the trained scalers for close and rsi
close_scaler = MinMaxScaler()
rsi_scaler = MinMaxScaler()

# Fit scalers on the original training data
# close_scaler.fit(training_data_close)
# rsi_scaler.fit(training_data_rsi)

def predict_next_day(close, rsi, macd_histogram):
    # Create and fit scalers for the single input value
    close_scaler = MinMaxScaler()
    rsi_scaler = MinMaxScaler()

    scaled_close = close_scaler.fit_transform([[close]])[0][0]
    scaled_rsi = rsi_scaler.fit_transform([[rsi]])[0][0]

    # Make predictions
    next_day_close_scaled = close_model.predict([[scaled_close]])[0]
    next_day_rsi_scaled = rsi_model.predict([[scaled_rsi]])[0]
    next_day_macd_histogram = macd_histogram_model.predict([[macd_histogram]])[0]  # No scaling for MACD histogram

    # Inverse transform the predictions using the same scalers
    next_day_close = close_scaler.inverse_transform([[next_day_close_scaled]])[0][0]
    next_day_rsi = rsi_scaler.inverse_transform([[next_day_rsi_scaled]])[0][0]

    return next_day_close, next_day_rsi, next_day_macd_histogram



# Streamlit app
st.title("Stock Prediction App")

st.write("Enter today's stock data to predict tomorrow's values:")

close = int(st.number_input("Close Price", value=0.0))
rsi = st.number_input("RSI", value=0.0)
macd_histogram = st.number_input("MACD Histogram", value=0.0)

if st.button("Predict"):
    next_day_close, next_day_rsi, next_day_macd_histogram = predict_next_day(close, rsi, macd_histogram)
    
    st.write(f"Predicted Close Price for next day: {next_day_close:.2f}")
    st.write(f"Predicted RSI for next day: {next_day_rsi:.2f}")
    st.write(f"Predicted MACD Histogram for next day: {next_day_macd_histogram:.2f}")
