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
            if "Date" not in historical_data.columns:
                historical_data.rename(columns={"Datetime": "Date"}, inplace=True)
            return historical_data
    except Exception as e:
        st.error(f"❌ Yahoo Finance Error: {e}")
    return None

# Function to Compute Supertrend
def supertrend(df, period=14, multiplier=1.5):
    atr = df['High'] - df['Low']
    atr = atr.rolling(window=period).mean()
    basic_ub = (df['High'] + df['Low']) / 2 + multiplier * atr
    basic_lb = (df['High'] + df['Low']) / 2 - multiplier * atr
    
    df['Supertrend'] = np.nan
    df['Direction'] = np.nan
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > basic_ub.iloc[i-1]:
            df['Supertrend'].iloc[i] = basic_lb.iloc[i]
            df['Direction'].iloc[i] = 1
        elif df['Close'].iloc[i] < basic_lb.iloc[i-1]:
            df['Supertrend'].iloc[i] = basic_ub.iloc[i]
            df['Direction'].iloc[i] = -1
        else:
            df['Supertrend'].iloc[i] = df['Supertrend'].iloc[i-1]
            df['Direction'].iloc[i] = df['Direction'].iloc[i-1]
    return df

# Function to Perform Bayesian Forecasting with Posterior Probabilities
def bayesian_forecast(df):
    if df is None or df.empty:
        return None
    
    df['Returns'] = df['Close'].pct_change().dropna()
    prior_mean = df['Returns'].mean()
    prior_std = df['Returns'].std()
    
    posterior_up = []
    posterior_down = []
    predicted_prices = []
    
    for i in range(1, len(df)):
        observed_return = df['Returns'].iloc[i-1]
        posterior_mean = (prior_mean + observed_return) / 2
        posterior_std = np.sqrt((prior_std ** 2 + observed_return ** 2) / 2)
        predicted_price = df['Close'].iloc[i-1] * (1 + posterior_mean)
        
        # Bayesian posterior probabilities
        prob_up = norm.cdf(observed_return, prior_mean, prior_std)
        prob_down = 1 - prob_up
        posterior_up.append(prob_up)
        posterior_down.append(prob_down)
        predicted_prices.append(predicted_price)
    
    df = df.iloc[1:].copy()
    df['Predicted Close'] = predicted_prices
    df['Posterior Up'] = posterior_up
    df['Posterior Down'] = posterior_down
    
    last_close = df['Close'].iloc[-1]
    next_predicted_price = last_close * (1 + posterior_mean)
    
    return df, {
        'posterior_mean': posterior_mean,
        'posterior_std': posterior_std,
        'predicted_price': next_predicted_price,
        'last_close': last_close
    }

# Function to Backtest Strategy
def backtest_strategy(df):
    df['Buy Signal'] = (df['Posterior Up'] > 0.6) & (df['Direction'] == 1)
    df['Sell Signal'] = (df['Posterior Down'] > 0.6) & (df['Direction'] == -1)
    df['Strategy Returns'] = df['Returns'] * df['Buy Signal'].shift(1) - df['Returns'] * df['Sell Signal'].shift(1)
    df['Cumulative Returns'] = (1 + df['Strategy Returns']).cumprod()
    return df

# Function to Generate AI Trade Strategy
def generate_ai_strategy(df, bayesian_results, stock):
    prompt = f"""
    You are a high-probability trading AI.
    Analyze {stock} market data and Bayesian forecasting.
    Use historical patterns to predict the best strategy.
    Bayesian results:
    - Predicted Next Closing Price: {bayesian_results['predicted_price']}
    - Posterior Mean: {bayesian_results['posterior_mean']}
    - Posterior Std Dev: {bayesian_results['posterior_std']}
    Provide a detailed, realistic trade plan.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a trading AI providing advanced market analysis."},
                  {"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Buttons to Fetch Data & Get AI Strategy
if st.button("Get Historical Data"):
    with st.spinner(f"Fetching {selected_stock} data..."):
        historical_data = fetch_historical_data_yfinance(selected_stock, interval, days)
    if historical_data is not None:
        historical_data = supertrend(historical_data)
        predicted_data, bayesian_results = bayesian_forecast(historical_data)
        st.subheader("📊 Data Table with Backtest Performance")
        st.dataframe(predicted_data.tail(150))
        st.subheader("🔢 Bayesian Forecast Results")
        st.write(f"**Predicted Next Closing Price:** ${round(bayesian_results['predicted_price'], 2)}")
        st.write(f"**Posterior Up Probability:** {predicted_data['Posterior Up'].iloc[-1]:.5f}")
        st.write(f"**Posterior Down Probability:** {predicted_data['Posterior Down'].iloc[-1]:.5f}")
        st.session_state['historical_data'] = predicted_data
        st.session_state['bayesian_results'] = bayesian_results


if st.button("Get AI Trading Strategy"):
    if 'historical_data' in st.session_state and 'bayesian_results' in st.session_state:
        ai_strategy = generate_ai_strategy(st.session_state['historical_data'], st.session_state['bayesian_results'], selected_stock)
        st.subheader("🤖 AI Trading Strategy")
        st.write(ai_strategy)
    else:
        st.error("❗ Please fetch historical data first.")
