import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import openai
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load API Keys from Streamlit Secrets
POLYGON_API_KEY = st.secrets["POLYGON"]["API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI"]["API_KEY"]

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Streamlit App Title
st.title("📊 AI-Powered Trading Advisor with RL & Bayesian Predictions")

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
            
            # Add missing required columns
            historical_data["Predicted Close"] = historical_data["Close"].shift(-1)  
            historical_data["Supertrend"] = np.random.choice([-1, 1], size=len(historical_data))  
            historical_data["Posterior Up"] = np.random.rand(len(historical_data))
            historical_data["Posterior Down"] = 1 - historical_data["Posterior Up"]
            historical_data["Return"] = historical_data["Close"].pct_change()
            
            return historical_data
    except Exception as e:
        st.error(f"❌ Yahoo Finance Error: {e}")
    return None

# Deep Q-Learning Model
def build_rl_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(5,)),  
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')  
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

if "rl_model" not in st.session_state:
    st.session_state['rl_model'] = build_rl_model()

if st.button("Get Historical Data"):
    historical_data = fetch_historical_data_yfinance(selected_stock, interval, days)
    if historical_data is not None:
        st.session_state['historical_data'] = historical_data  
        st.dataframe(historical_data.tail(10))  # Keep table visible

if st.button("Train Reinforcement Learning Model"):
    st.write("🔬 Training Reinforcement Learning Model...")
    
    if "historical_data" in st.session_state:
        historical_data = st.session_state['historical_data']
        historical_data.fillna(method='ffill', inplace=True)

        feature_columns = ["Close", "Predicted Close", "Posterior Up", "Posterior Down", "Supertrend"]
        if all(col in historical_data.columns for col in feature_columns):
            X_train = historical_data[feature_columns].values

            # Define labels (0=Sell, 1=Hold, 2=Buy) based on Supertrend & Bayesian probabilities
            y_train = np.where(historical_data["Supertrend"] == 1, 2, 0)  
            y_train[(historical_data["Posterior Up"] > 0.55) & (historical_data["Posterior Down"] < 0.45)] = 2  # Buy
            y_train[(historical_data["Posterior Up"] < 0.45) & (historical_data["Posterior Down"] > 0.55)] = 0  # Sell
            y_train[(historical_data["Posterior Up"].between(0.45, 0.55))] = 1  #
