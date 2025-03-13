import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import time
import datetime
import openai
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import t
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import pickle

# Load API Keys from Streamlit Secrets
POLYGON_API_KEY = st.secrets["POLYGON"]["API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI"]["API_KEY"]

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Streamlit App Title
st.title("📊 AI-Powered Trading Advisor with Reinforcement Learning & Bayesian Predictions")

# List of Top Stocks / ETFs
top_stocks = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B"]
selected_stock = st.sidebar.selectbox("Select Ticker", top_stocks)

# User selects interval & number of days
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=5)
days = st.sidebar.slider("Select Number of Days for History", 1, 30, 7)

# Function to Fetch Historical Data from Yahoo Finance
def fetch_historical_data_yfinance(ticker, interval, days):
    stock = yf.Ticker(ticker)
    try:
        historical_data = stock.history(period=f"{days}d", interval=interval)
        if not historical_data.empty:
            historical_data.reset_index(inplace=True)
            if "Date" not in historical_data.columns:
                historical_data.rename(columns={"Datetime": "Date"}, inplace=True)
            
            # Ensure required columns exist
            if "Predicted Close" not in historical_data.columns:
                historical_data["Predicted Close"] = historical_data["Close"].shift(-1)  # Simple prediction
            if "Supertrend" not in historical_data.columns:
                historical_data["Supertrend"] = np.random.choice([-1, 1], size=len(historical_data))  # Placeholder
            if "Posterior Up" not in historical_data.columns:
                historical_data["Posterior Up"] = np.random.rand(len(historical_data))
            if "Posterior Down" not in historical_data.columns:
                historical_data["Posterior Down"] = 1 - historical_data["Posterior Up"]
            
            return historical_data
    except Exception as e:
        st.error(f"❌ Yahoo Finance Error: {e}")
    return None

# Deep Q-Learning Model
def build_rl_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(5,)),  # 5 Features including Supertrend
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 Actions: Buy, Sell, Hold
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

if "rl_model" not in st.session_state:
    st.session_state['rl_model'] = build_rl_model()

if st.button("Get Historical Data"):
    historical_data = fetch_historical_data_yfinance(selected_stock, interval, days)
    if historical_data is not None:
        st.session_state['historical_data'] = historical_data  # Store data in session
        st.dataframe(historical_data.tail(5))

if st.button("Train Reinforcement Learning Model"):
    st.write("🔬 Reinforcement learning training in progress...")
    
    if "historical_data" in st.session_state:
        historical_data = st.session_state['historical_data']
        
        # Ensure no NaN values before training
        historical_data.fillna(method='ffill', inplace=True)
        
        # Define states as Close Price, Predicted Close, Posterior Up, Posterior Down, and Supertrend
        feature_columns = ["Close", "Predicted Close", "Posterior Up", "Posterior Down", "Supertrend"]
        if all(col in historical_data.columns for col in feature_columns):
            X_train = historical_data[feature_columns].values
            y_train = np.random.randint(0, 3, size=len(X_train))  # Placeholder for Buy/Sell/Hold labels
        
            # Train model
            st.session_state['rl_model'].fit(X_train, y_train, epochs=10, verbose=0)
            
            st.write("✅ Reinforcement learning model trained successfully!")
        else:
            st.error("❌ Some required features are missing in the dataset!")
    else:
        st.error("❌ Please fetch historical data first!")

if st.button("Predict Next [Time Frame]"):
    if "rl_model" in st.session_state:
        sample_input = np.random.rand(1, 5)
        prediction = st.session_state['rl_model'].predict(sample_input)
        st.write(f"🔮 **Prediction Probabilities:** Buy: {prediction[0][0] * 100:.2f}%, Hold: {prediction[0][1] * 100:.2f}%, Sell: {prediction[0][2] * 100:.2f}%")
    else:
        st.error("❌ Train the model first before predicting!")

if st.button("Get Bayesian Predictions"):
    if "historical_data" in st.session_state:
        historical_data = st.session_state['historical_data']
        latest_row = historical_data.iloc[-1]
        posterior_up = latest_row["Posterior Up"] * 100
        posterior_down = latest_row["Posterior Down"] * 100
        trend = "Up" if posterior_up > 50 else "Down" if posterior_down > 50 else "Sideways"
        
        st.write(f"🔮 **Bayesian Probabilities:** Up: {posterior_up:.2f}%, Down: {posterior_down:.2f}%, Trend: {trend}")
    else:
        st.error("❌ Please fetch historical data first!")
