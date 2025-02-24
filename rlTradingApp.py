import streamlit as st
import yfinance as yf
import pandas as pd
import openai

# Load OpenAI API Key from secrets.toml
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit App Title
st.title("📊 SPY Trading Advisor with AI Strategy")

# Sidebar for User Inputs
st.sidebar.header("SPY Data Input")
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h", "1d"])
period = st.sidebar.selectbox("Select Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])

# Function to Fetch SPY Data from Yahoo Finance
def fetch_spy_data(interval, period):
    spy_data = yf.download(tickers="SPY", interval=interval, period=period)
    return spy_data

# Function to Query OpenAI for Strategy
def ask_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a trading expert."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"

# Main App Logic
if st.sidebar.button("Get SPY Data & AI Strategy"):
    with st.spinner("Fetching SPY data..."):
        spy_data = fetch_spy_data(interval, period)

    if not spy_data.empty:
        st.subheader("📊 SPY Market Data")
        st.line_chart(spy_data["Close"])

        st.subheader("📋 Raw Data Preview")
        st.write(spy_data.tail(10))

        # Prepare data for OpenAI prompt
        prompt = f"""Given the following SPY market data:
        {spy_data.tail(10).to_string()}
        
        Please suggest a trading strategy based on the trends, including buy/sell signals, risk management, and optimal timeframes for trading.
        """

        # Get AI-Generated Strategy
        with st.spinner("Generating AI Trading Strategy..."):
            ai_strategy = ask_openai(prompt)

        st.subheader("🤖 AI-Generated SPY Trading Strategy")
        st.write(ai_strategy)

    else:
        st.error("❌ No data found. Please try again later.")
