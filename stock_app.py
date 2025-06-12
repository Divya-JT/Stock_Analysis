import streamlit as st
import numpy as np
import pickle

# Load model and scaler using pickle
with open("stock_price_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("📈 Stock Price Predictor")
st.markdown("Enter Open, High, Low, and Volume to predict the Close price.")

# Input fields
open_price = st.number_input("Open Price", min_value=0.0, format="%.2f")
high_price = st.number_input("High Price", min_value=0.0, format="%.2f")
low_price = st.number_input("Low Price", min_value=0.0, format="%.2f")
volume = st.number_input("Volume", min_value=0)

# Predict on button click
if st.button("Predict Closing Price"):
    input_data = np.array([[open_price, high_price, low_price, volume]])
    scaled_input = scaler.transform(input_data)
    predicted_price = model.predict(scaled_input)[0]
    st.success(f"📊 Predicted Closing Price: ${predicted_price:.2f}")
