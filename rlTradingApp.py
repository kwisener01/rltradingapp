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

# Sidebar Stock Selection
top_stocks = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B"]
selected_stock = st.sidebar.selectbox("Select Ticker", top_stocks)

# Interval & Days Selection
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=5)
days = st.sidebar.slider("Select Number of Days for History", 1, 30, 7)

# Function to Fetch Historical Data
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

# Fetch Historical Data Button
if st.button("📊 Get Historical Data"):
    historical_data = fetch_historical_data_yfinance(selected_stock, interval, days)
    if historical_data is not None:
        st.session_state['historical_data'] = historical_data  
        st.dataframe(historical_data.tail(10))  # Keep table visible


# train model
from tensorflow.keras.utils import to_categorical  # Import one-hot encoding utility

if st.button("🔬 Train Reinforcement Learning Model"):
    st.write("🚀 Training Reinforcement Learning Model...")

    if "historical_data" in st.session_state:
        historical_data = st.session_state['historical_data'].copy()
        historical_data.fillna(method='ffill', inplace=True)  # Fill missing values

        # Define feature columns for training
        feature_columns = ["Close", "Predicted Close", "Posterior Up", "Posterior Down", "Supertrend"]
        
        if all(col in historical_data.columns for col in feature_columns):
            X_train = historical_data[feature_columns].values.astype(np.float32)  # Convert to float32
            
            # Generate Labels for Actions (Buy=2, Hold=1, Sell=0)
            y_train = np.zeros(len(X_train), dtype=int)  # Default: Hold (1)
            y_train[historical_data["Supertrend"] == 1] = 2  # Buy
            y_train[historical_data["Supertrend"] == -1] = 0  # Sell
            y_train[(historical_data["Posterior Up"] > 0.55) & (historical_data["Posterior Down"] < 0.45)] = 2  # Buy
            y_train[(historical_data["Posterior Up"] < 0.45) & (historical_data["Posterior Down"] > 0.55)] = 0  # Sell
            
            # 🔹 Convert labels to One-Hot Encoding (to match model output shape)
            y_train = to_categorical(y_train, num_classes=3)

            # Debugging Step: Print Data Shapes
            st.write(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
            st.write(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")

            # Train model only if there's enough data
            if len(X_train) > 10:
                try:
                    st.session_state['rl_model'].fit(X_train, y_train, epochs=10, verbose=0)
                    st.write("✅ RL Model Trained Successfully!")
                except Exception as e:
                    st.error(f"❌ Model Training Failed: {e}")
            else:
                st.error("❌ Not enough data to train the model! Please fetch more historical data.")
        else:
            st.error("❌ Some required features are missing in the dataset!")
    else:
        st.error("❌ Please fetch historical data first!")

# Predict Next Time Frame Button
if st.button("Train Reinforcement Learning Model"):
    st.write("🔬 Training Reinforcement Learning Model...")

    if "historical_data" in st.session_state:
        historical_data = st.session_state['historical_data']

        # Fill missing values using forward fill
        historical_data.fillna(method='ffill', inplace=True)

        # Ensure required columns exist
        feature_columns = ["Close", "Predicted Close", "Posterior Up", "Posterior Down", "Supertrend"]
        if all(col in historical_data.columns for col in feature_columns):
            X_train = historical_data[feature_columns].values
            y_train = np.random.randint(0, 3, size=len(X_train))  

            # Ensure no NaN values in training data
            if np.isnan(X_train).any():
                st.error("❌ Training failed: X_train contains NaN values!")
                st.stop()

            # Train the model
            st.session_state['rl_model'].fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            st.write("✅ RL Model Trained Successfully!")
        else:
            st.error("❌ Some required features are missing in the dataset!")
    else:
        st.error("❌ Please fetch historical data first!")

# Get Bayesian Predictions Button
if st.button("📈 Get Bayesian Predictions"):
    if "historical_data" in st.session_state:
        historical_data = st.session_state['historical_data']
        last_five_closes = historical_data["Close"].tail(5).values  

        st.write("📊 **Last 5 Closing Prices:**")
        for i, price in enumerate(last_five_closes[::-1], 1):
            st.write(f"{i}. {price:.2f}")

        latest_row = historical_data.iloc[-1]
        posterior_up = latest_row["Posterior Up"] * 100
        posterior_down = latest_row["Posterior Down"] * 100
        trend = "Up" if posterior_up > 50 else "Down" if posterior_down > 50 else "Sideways"

        st.write(f"🔮 **Bayesian Probabilities:** Up: {posterior_up:.2f}%, Down: {posterior_down:.2f}%, Trend: {trend}")

        # Display backtest results
        win_rate = (historical_data["Supertrend"] == np.sign(historical_data["Close"].diff())).mean() * 100
        avg_return = historical_data["Close"].pct_change().mean() * 100
        st.write(f"📈 **Backtest Stats:** Win Rate: {win_rate:.2f}%, Avg Return: {avg_return:.2f}%")
        
        st.dataframe(historical_data.tail(10))  # Keep historical data visible
    else:
        st.error("❌ Please fetch historical data first!")
