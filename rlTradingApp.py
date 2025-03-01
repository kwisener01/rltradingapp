import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import time
import datetime
import openai
import numpy as np
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
            return historical_data
    except Exception as e:
        st.error(f"❌ Yahoo Finance Error: {e}")
    return None

# Function to Perform Bayesian Forecasting
def bayesian_forecast(df):
    if df is None or df.empty:
        return None

    df["Returns"] = df["Close"].pct_change().dropna()

    # Compute prior mean and standard deviation
    prior_mean = df["Returns"].mean()
    prior_std = df["Returns"].std()

    # Assume new observed return is the last return
    observed_return = df["Returns"].iloc[-1]

    # Compute Bayesian posterior mean and standard deviation
    posterior_mean = (prior_mean + observed_return) / 2
    posterior_std = np.sqrt((prior_std ** 2 + observed_return ** 2) / 2)

    # Predict probability of price moving up or down
    up_prob = norm.cdf(0, loc=posterior_mean, scale=posterior_std)
    down_prob = 1 - up_prob

    return {"posterior_mean": posterior_mean, "posterior_std": posterior_std, "up_prob": up_prob, "down_prob": down_prob}

# Function to Generate AI Trading Strategy with Bayesian Forecasting
def generate_ai_strategy(df, ticker, bayesian_results):
    if df is None or df.empty:
        return "No sufficient data to generate a strategy."

    latest_data = df.tail(50).to_string(index=False)  # AI learns from last 50 market data points

    # Bayesian Forecasting Results
    up_prob = round(bayesian_results["up_prob"] * 100, 2)
    down_prob = round(bayesian_results["down_prob"] * 100, 2)

    prompt = f"""
    You are an AI trading expert analyzing {ticker} stock using the last {len(df)} historical data points.

    Recent market data (last {len(df)} candles):
    {latest_data}

    Bayesian Forecasting indicates:
    - **Probability of price increase:** {up_prob}%
    - **Probability of price decrease:** {down_prob}%
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

# 📊 BUTTON 1: Get Historical Data
if st.button("Get Historical Data"):
    with st.spinner(f"Fetching {selected_stock} data..."):
        historical_data = fetch_historical_data_yfinance(selected_stock, interval, days)

    if historical_data is not None and not historical_data.empty:
        st.subheader(f"📈 {selected_stock} Historical Price Chart ({days} Days)")
        st.line_chart(historical_data.set_index(historical_data.columns[0])["Close"])
        st.subheader("📋 Historical Data (Last 150 Entries)")
        st.dataframe(historical_data.tail(150))  # Show last 50 entries for better insights

        # Perform Bayesian Forecasting
        bayesian_results = bayesian_forecast(historical_data)

        if bayesian_results:
            st.subheader("📊 Bayesian Forecasting Results")
            st.write(f"🔹 **Probability of Price Increase:** {round(bayesian_results['up_prob'] * 100, 2)}%")
            st.write(f"🔹 **Probability of Price Decrease:** {round(bayesian_results['down_prob'] * 100, 2)}%")
            st.write(f"🔹 **Expected Return (Posterior Mean):** {bayesian_results['posterior_mean']:.5f}")
            st.write(f"🔹 **Market Volatility (Posterior Std Dev):** {bayesian_results['posterior_std']:.5f}")

            # Store data for AI analysis
            st.session_state["historical_data"] = historical_data
            st.session_state["bayesian_results"] = bayesian_results
    else:
        st.error("❌ No historical data found. Try a different time period.")

# 🤖 BUTTON 2: Get AI Trading Strategy
if st.button("Get AI Trading Strategy"):
    if "historical_data" in st.session_state and not st.session_state["historical_data"].empty:
        with st.spinner("Generating AI trading strategy..."):
            ai_strategy = generate_ai_strategy(st.session_state["historical_data"], selected_stock, st.session_state["bayesian_results"])

        st.subheader(f"🤖 AI-Generated {selected_stock} Trading Strategy")
        st.write(ai_strategy)
    else:
        st.error("❗ Please fetch stock data first before requesting AI analysis.")
