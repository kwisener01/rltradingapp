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

# Load API Keys from Streamlit Secrets
POLYGON_API_KEY = st.secrets["POLYGON"]["API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI"]["API_KEY"]

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Streamlit App Title
st.title("📊 AI-Powered Trading Advisor with Bayesian Forecasting")

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

            # Standardizing Date Column
            if "Date" not in historical_data.columns:
                historical_data.rename(columns={"Datetime": "Date"}, inplace=True)

            return historical_data
    except Exception as e:
        st.error(f"❌ Yahoo Finance Error: {e}")
    return None

# Function to Perform Bayesian Forecasting on Historical Data
def bayesian_forecast(df):
    if df is None or df.empty:
        return None

    df["Returns"] = df["Close"].pct_change().dropna()

    # Compute prior mean and standard deviation
    prior_mean = df["Returns"].mean()
    prior_std = df["Returns"].std()

    # Compute Bayesian posterior for each price point
    predicted_prices = []
    for i in range(1, len(df)):
        observed_return = df["Returns"].iloc[i-1]
        posterior_mean = (prior_mean + observed_return) / 2
        posterior_std = np.sqrt((prior_std ** 2 + observed_return ** 2) / 2)
        predicted_price = df["Close"].iloc[i-1] * (1 + posterior_mean)  # Forecasted price

        predicted_prices.append(predicted_price)

    # Align predicted prices with actual data
    df = df.iloc[1:].copy()  # Remove the first row (since it has no prediction)
    df["Predicted Close"] = predicted_prices

    # Get next predicted price for future reference
    last_close = df["Close"].iloc[-1]
    next_predicted_price = last_close * (1 + posterior_mean)

    return df, {
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "predicted_price": next_predicted_price,
        "last_close": last_close
    }

# 📊 BUTTON 1: Get Historical Data & Predictions
if st.button("Get Historical Data"):
    with st.spinner(f"Fetching {selected_stock} data..."):
        historical_data = fetch_historical_data_yfinance(selected_stock, interval, days)

    if historical_data is not None and not historical_data.empty:
        st.subheader(f"📈 {selected_stock} Historical Price Chart ({days} Days)")

        # Perform Bayesian Forecasting on Historical Data
        predicted_data, bayesian_results = bayesian_forecast(historical_data)

        if bayesian_results:
            # Plot actual vs. predicted prices
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(predicted_data["Date"], predicted_data["Close"], label="Actual Close Price", color="blue")
            ax.plot(predicted_data["Date"], predicted_data["Predicted Close"], linestyle="dashed", label="Predicted Close Price", color="red")
            ax.scatter(predicted_data["Date"].iloc[-1], bayesian_results["predicted_price"], color="green", label="Next Predicted Price", zorder=3)
            ax.legend()
            ax.set_title(f"{selected_stock} Price Chart & Bayesian Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            st.pyplot(fig)

            st.subheader("📋 Historical Data with Predictions (Last 150 Entries)")
            st.dataframe(predicted_data.tail(150))  # Show last 150 entries with predicted values

            st.subheader("📊 Bayesian Forecasting Results")
            st.write(f"🔹 **Predicted Next Closing Price:** ${round(bayesian_results['predicted_price'], 2)}")
            st.write(f"🔹 **Posterior Mean (Expected Return):** {bayesian_results['posterior_mean']:.5f}")
            st.write(f"🔹 **Posterior Std Dev (Market Volatility):** {bayesian_results['posterior_std']:.5f}")

            # Store data for AI analysis
            st.session_state["historical_data"] = predicted_data
            st.session_state["bayesian_results"] = bayesian_results
    else:
        st.error("❌ No historical data found. Try a different time period.")

# 🤖 BUTTON 2: Get AI Trading Strategy
def generate_ai_strategy(df, ticker, bayesian_results):
    if df is None or df.empty:
        return "No sufficient data to generate a strategy."

    latest_data = df.tail(50).to_string(index=False)  # AI learns from last 50 market data points

    prompt = f"""
    You are an AI trading expert analyzing {ticker} stock using the last {len(df)} historical data points.

    Recent market data (last {len(df)} candles):
    {latest_data}

    Bayesian Forecasting indicates:
    - **Predicted Next Closing Price:** ${round(bayesian_results['predicted_price'], 2)}
    - **Expected return (posterior mean):** {bayesian_results['posterior_mean']}
    - **Market volatility (posterior std dev):** {bayesian_results['posterior_std']}

    Provide a **detailed technical strategy** including:
    - **Current market trend (bullish, bearish, sideways)**
    - **Key support & resistance levels**
    - **Volume-based insights (institutional buying/selling)**
    - **Momentum & volatility analysis**
    - **Trade setup (scalping, day trading, or swing trading)**
    - **Recommended entry/exit points, stop-loss, and take-profit levels**
    
    The strategy must be actionable and precise.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an advanced trading assistant providing expert market analysis."},
                  {"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

if st.button("Get AI Trading Strategy"):
    if "historical_data" in st.session_state and not st.session_state["historical_data"].empty:
        with st.spinner("Generating AI trading strategy..."):
            ai_strategy = generate_ai_strategy(st.session_state["historical_data"], selected_stock, st.session_state["bayesian_results"])

        st.subheader(f"🤖 AI-Generated {selected_stock} Trading Strategy")
        st.write(ai_strategy)
    else:
        st.error("❗ Please fetch stock data first before requesting AI analysis.")
