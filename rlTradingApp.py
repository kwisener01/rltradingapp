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
st.title("📊 AI-Powered Trading Advisor with Reinforcement Learning")

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
            return historical_data
    except Exception as e:
        st.error(f"❌ Yahoo Finance Error: {e}")
    return None

# Function to Perform Bayesian Forecasting
def bayesian_forecast(df):
    if df is None or df.empty:
        return None

    df["Returns"] = df["Close"].pct_change().dropna()
    prior_mean = df["Returns"].mean()
    prior_std = df["Returns"].std()

    predicted_prices = []
    posterior_up = []
    posterior_down = []
    trend_directions = []

    for i in range(1, len(df)):
        observed_return = df["Returns"].iloc[i-1]
        posterior_mean = (prior_mean + observed_return) / 2
        posterior_std = np.sqrt((prior_std ** 2 + observed_return ** 2) / 2)
        predicted_price = df["Close"].iloc[i-1] * (1 + posterior_mean)

        predicted_prices.append(predicted_price)
        posterior_up.append(norm.cdf(0.01, loc=posterior_mean, scale=posterior_std))  # Up probability
        posterior_down.append(norm.cdf(-0.01, loc=posterior_mean, scale=posterior_std))  # Down probability
        trend_directions.append("Up" if posterior_mean > 0 else "Down" if posterior_mean < 0 else "Sideways")

    df = df.iloc[1:].copy()
    df["Predicted Close"] = predicted_prices
    df["Posterior Up"] = posterior_up
    df["Posterior Down"] = posterior_down
    df["Trend Direction"] = trend_directions

    last_close = df["Close"].iloc[-1]
    next_predicted_price = last_close * (1 + posterior_mean)

    return df, {
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "predicted_price": next_predicted_price,
        "last_close": last_close
    }

# Deep Q-Learning Model
def build_rl_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(5,)),
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
        predicted_data, bayesian_results = bayesian_forecast(historical_data)

        if bayesian_results:
            st.session_state['predicted_data'] = predicted_data
            st.subheader(f"📈 {selected_stock} Historical Price Chart ({days} Days)")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(predicted_data["Date"], predicted_data["Close"], label="Actual Close Price", color="blue")
            ax.plot(predicted_data["Date"], predicted_data["Predicted Close"], linestyle="dashed", label="Predicted Close", color="red")
            ax.legend()
            ax.set_title(f"{selected_stock} Price Chart & Bayesian Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            st.pyplot(fig)

            st.subheader("📋 Historical Data with Predictions")
            st.dataframe(predicted_data.tail(150))

if st.button("Train Reinforcement Learning Model"):
    st.write("🔬 Reinforcement learning training in progress...")
    
    if "predicted_data" in st.session_state:
        predicted_data = st.session_state['predicted_data']
        X_train = predicted_data[['Close', 'Predicted Close', 'Posterior Up', 'Posterior Down']].values
        y_train = np.random.randint(0, 3, size=len(X_train))  # Placeholder for Buy/Sell/Hold labels
        
        st.session_state['rl_model'].fit(X_train, y_train, epochs=10, verbose=0)
        st.write("✅ Reinforcement learning model trained successfully!")
    else:
        st.error("❌ Please fetch historical data first!")

if st.button("Predict Next [Time Frame]"):
    if "rl_model" in st.session_state:
        sample_input = np.random.rand(1, 5)
        prediction = st.session_state['rl_model'].predict(sample_input)
        st.write(f"🔮 **Prediction Probabilities:** Buy: {prediction[0][0] * 100:.2f}%, Hold: {prediction[0][1] * 100:.2f}%, Sell: {prediction[0][2] * 100:.2f}%")
    else:
        st.error("❌ Train the model first before predicting!")

if st.button("Get AI Trade Plan"):
    st.write("🧠 Generating AI Trade Plan...")
    if "predicted_data" in st.session_state:
        df = st.session_state['predicted_data']
        prompt = f"""
        You are an AI trading assistant analyzing {selected_stock}.
        Last 5 records:
        {df.tail(5).to_string(index=False)}

        Bayesian Forecasting:
        - **Predicted Next Closing Price:** ${round(df['Predicted Close'].iloc[-1], 2)}
        - **Posterior Up:** {df['Posterior Up'].iloc[-1] * 100:.2f}%
        - **Posterior Down:** {df['Posterior Down'].iloc[-1] * 100:.2f}%
        - **Trend Direction:** {df['Trend Direction'].iloc[-1]}

        Provide a technical strategy including:
        - Market trend (bullish, bearish, sideways)
        - Key support & resistance levels
        - Volume-based insights
        - Recommended entry/exit points, stop-loss, and take-profit levels
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert trading assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )

        st.write(response.choices[0].message.content.strip())
    else:
        st.error("❌ Please fetch historical data first!")
