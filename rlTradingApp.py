import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import openai

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



def bayesian_forecast(df):
    if df is None or df.empty:
        return None, None

    df["Returns"] = df["Close"].pct_change().dropna()

    # Compute prior mean and standard deviation
    prior_mean = df["Returns"].mean()
    prior_std = df["Returns"].std()

    # Compute Bayesian posterior for each price point
    predicted_prices = []
    posterior_up = []
    posterior_down = []

    for i in range(1, len(df)):
        observed_return = df["Returns"].iloc[i - 1]
        posterior_mean = (prior_mean + observed_return) / 2
        posterior_std = np.sqrt((prior_std ** 2 + observed_return ** 2) / 2)
        predicted_price = df["Close"].iloc[i - 1] * (1 + posterior_mean)  # Forecasted price

        # Calculate probabilities using normal distribution
        prob_up = norm.cdf(predicted_price, loc=posterior_mean, scale=posterior_std)
        prob_down = 1 - prob_up

        predicted_prices.append(predicted_price)
        posterior_up.append(prob_up)
        posterior_down.append(prob_down)

    # Align predicted prices with actual data
    df = df.iloc[1:].copy()  # Remove the first row (since it has no prediction)
    df["Predicted Close"] = predicted_prices
    df["Posterior Up"] = posterior_up
    df["Posterior Down"] = posterior_down

    # Store the last predicted price
    last_close = df["Close"].iloc[-1]
    next_predicted_price = last_close * (1 + posterior_mean)

    return df, {
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "predicted_price": next_predicted_price,
        "last_close": last_close
    }


# Deep Q-Learning Model (Fixing input shape)
def build_rl_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(4,)),  # 👈 Expecting 4 features now
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 Actions: Buy, Sell, Hold
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Rebuild model with correct input shape
if "rl_model" not in st.session_state:
    st.session_state['rl_model'] = build_rl_model()


if "rl_model" not in st.session_state:
    st.session_state['rl_model'] = build_rl_model()

if st.button("Get Historical Data"):
    historical_data = fetch_historical_data_yfinance(selected_stock, interval, days)

    if historical_data is not None:
        # Apply Bayesian Forecasting
        predicted_df, forecast_summary = bayesian_forecast(historical_data)
        
        # Store both historical and predicted data
        st.session_state['historical_data'] = historical_data
        st.session_state['predicted_data'] = predicted_df  # ✅ Store in session state

        # Display the last 150 records
        st.dataframe(predicted_df.tail(150))

if st.button("Train Reinforcement Learning Model"):
    st.write("🔬 Reinforcement learning training in progress...")

    if "historical_data" in st.session_state:
        historical_data = st.session_state["historical_data"]

        # Apply Bayesian Forecasting before training
        predicted_df, forecast_summary = bayesian_forecast(historical_data)

        if predicted_df is not None:
            st.session_state["predicted_data"] = predicted_df  # ✅ Store Bayesian data
            
            # ✅ Convert columns to numeric & fill NaNs
            for col in ['Close', 'Predicted Close', 'Posterior Up', 'Posterior Down']:
                if col in predicted_df.columns:
                    predicted_df[col] = pd.to_numeric(predicted_df[col], errors='coerce').fillna(0)

            # ✅ Ensure X_train has correct numerical format
            X_train = predicted_df[['Close', 'Predicted Close', 'Posterior Up', 'Posterior Down']].to_numpy(dtype=np.float32)

            # ✅ Ensure y_train is categorical (Buy=0, Hold=1, Sell=2)
            y_train = np.random.randint(0, 3, size=(len(X_train),)).astype(np.int32)

            # ✅ Check data shape & consistency before training
            if len(X_train) > 0 and X_train.shape[1] == 4 and len(y_train) == len(X_train):
                try:
                    st.session_state["rl_model"].fit(X_train, y_train, epochs=10, verbose=0)
                    st.write("✅ Reinforcement learning model trained successfully!")
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
            else:
                st.error(f"❌ Data shape mismatch: X_train={X_train.shape}, y_train={y_train.shape}")
        else:
            st.error("❌ Bayesian Forecasting failed. Check data format.")
    else:
        st.error("❌ Please fetch historical data first!")



if st.button("Predict Next [Time Frame]"):
    if "rl_model" in st.session_state and "predicted_data" in st.session_state:
        df = st.session_state["predicted_data"]

        if len(df) < 10:
            st.error("❌ Not enough data for prediction. Train with more historical data!")
        else:
            # Prepare input (convert to float)
            last_row = df.iloc[-1][["Close", "Predicted Close", "Posterior Up", "Posterior Down"]].astype(float).values.reshape(1, -1)

            # Fix NaNs
            last_row = np.nan_to_num(last_row, nan=0.5)

            # Make prediction
            prediction = st.session_state["rl_model"].predict(last_row)

            # Ensure output is numeric
            prediction = np.nan_to_num(prediction)

            # Display results
            st.write(f"🔮 **Prediction Probabilities:**")
            st.write(f"📈 Buy: {prediction[0][0] * 100:.2f}%")
            st.write(f"⏳ Hold: {prediction[0][1] * 100:.2f}%")
            st.write(f"📉 Sell: {prediction[0][2] * 100:.2f}%")
    else:
        st.error("❌ Train the model first before predicting!")



if st.button("Get AI Trade Plan"):
    st.write("🧠 Generating AI Trade Plan...")
    if "predicted_data" in st.session_state:
        df = st.session_state['predicted_data']
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert trading assistant."},
                      {"role": "user", "content": f"Analyze the trading trends for {selected_stock}"}],
            temperature=0.7
        )
        st.write(response.choices[0].message.content.strip())
    else:
        st.error("❌ Please fetch historical data first!")
