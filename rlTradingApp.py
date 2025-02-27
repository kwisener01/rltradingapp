import streamlit as st
import requests
import pandas as pd
import datetime
import time

# Load Finnhub API Key
FINNHUB_API_KEY = st.secrets["FINNHUB"]["API_KEY"]

# Streamlit App Title
st.title("📊 AI-Powered Trading Advisor (Real-Time & Historical)")

# List of Top 20 Stocks + SPY and QQQ
top_stocks = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
              "UNH", "V", "JNJ", "WMT", "JPM", "PG", "MA", "HD", "DIS", "BAC", "XOM", "PFE"]

# Sidebar for User Inputs
st.sidebar.header("Select Stock/ETF & Timeframe")
selected_stock = st.sidebar.selectbox("Select Ticker", top_stocks)
interval = st.sidebar.selectbox("Select Interval", ["1", "5", "15", "30", "60", "D", "W", "M"], index=4)  # Finnhub uses 1, 5, 15, 30, 60, D, W, M
days = st.sidebar.slider("Select Number of Days for History", 1, 30, 7)

# Function to Fetch Real-Time Data from Finnhub
def fetch_real_time_price(ticker):
    url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "c" in data and data["c"] != 0:
        return {
            "Symbol": ticker,
            "Latest Price": f"${data['c']:.2f}",
            "Previous Close": f"${data['pc']:.2f}",
            "Open Price": f"${data['o']:.2f}",
            "High Price": f"${data['h']:.2f}",
            "Low Price": f"${data['l']:.2f}",
            "Change": f"${data['d']:.2f}",
            "Change %": f"{data['dp']:.2f}%",
            "Timestamp": datetime.datetime.fromtimestamp(data["t"]).strftime('%Y-%m-%d %H:%M:%S')
        }
    else:
        return None

# Function to Fetch Historical Data from Finnhub
def fetch_historical_data(ticker, interval, days):
    end_time = int(time.time())  # Current time in Unix timestamp
    start_time = end_time - (days * 24 * 60 * 60)  # Subtract number of days

    url = f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution={interval}&from={start_time}&to={end_time}&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "s" in data and data["s"] == "ok":
        df = pd.DataFrame({
            "Timestamp": [datetime.datetime.fromtimestamp(ts) for ts in data["t"]],
            "Open": data["o"],
            "High": data["h"],
            "Low": data["l"],
            "Close": data["c"],
            "Volume": data["v"]
        })
        return df
    else:
        return None

# 📊 BUTTON 1: Get Real-Time & Historical Stock Data
if st.button("Get Stock Data"):
    with st.spinner(f"Fetching real-time and historical data for {selected_stock}..."):
        stock_data = fetch_real_time_price(selected_stock)
        historical_data = fetch_historical_data(selected_stock, interval, days)

        if stock_data is not None:
            st.subheader(f"📊 {selected_stock} Market Data (Live)")
            st.write(stock_data)

        if historical_data is not None:
            st.subheader(f"📈 {selected_stock} Historical Price Chart ({days} Days)")
            st.line_chart(historical_data.set_index("Timestamp")["Close"])
            st.subheader("📋 Historical Data (Last 10 Entries)")
            st.dataframe(historical_data.tail(10))
        else:
            st.error("❌ No historical data found. Try a different time period.")

# 🤖 BUTTON 2: Get AI Trading Strategy
if st.button("Get AI Trading Strategy"):
    if st.session_state.get('stock_data') is not None:
        prompt = f"""Given the following real-time and historical data for {selected_stock}:

        - Latest Price: {st.session_state['stock_data']['Latest Price']}
        - Previous Close: {st.session_state['stock_data']['Previous Close']}
        - Open Price: {st.session_state['stock_data']['Open Price']}
        - High Price: {st.session_state['stock_data']['High Price']}
        - Low Price: {st.session_state['stock_data']['Low Price']}
        - Change: {st.session_state['stock_data']['Change']} ({st.session_state['stock_data']['Change %']})

        **Last {days} days of price data:**
        {st.session_state['historical_data'].tail(10).to_string(index=False)}

        Provide an **intraday and swing trading strategy** based on:
        - Trend analysis (bullish or bearish)
        - Support & resistance levels
        - Institutional buying activity
        - Entry & exit signals with stop-loss targets

        Format the response clearly with actionable insights.
        """

        with st.spinner("Generating AI Trading Strategy..."):
            ai_strategy = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a trading expert providing real-time analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            st.subheader(f"🤖 AI-Generated {selected_stock} Trading Strategy")
            st.write(ai_strategy.choices[0].message.content.strip())
    else:
        st.error("❗ Please fetch stock data first before requesting AI analysis.")
