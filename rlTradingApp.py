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
interval = st.sidebar.selectbox(
    "Select Interval", 
    ["D", "W", "M"],  # Removed intraday options since free plan doesn't support them
    index=0
)
days = st.sidebar.slider("Select Number of Days for History", 1, 30, 7)

# Function to Fetch Historical Data from Finnhub
def fetch_historical_data(ticker, interval, days):
    end_time = int(time.time())  # Current time in Unix timestamp
    start_time = end_time - (days * 24 * 60 * 60)  # Subtract number of days

    url = f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution={interval}&from={start_time}&to={end_time}&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    data = response.json()

    # Debugging: Show Raw API Response
    st.write("📡 **Raw API Response:**", data)

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

# 📊 BUTTON 1: Get Historical Stock Data
if st.button("Get Historical Data"):
    with st.spinner(f"Fetching historical data for {selected_stock}..."):
        historical_data = fetch_historical_data(selected_stock, interval, days)

        if historical_data is not None and not historical_data.empty:
            st.subheader(f"📈 {selected_stock} Historical Price Chart ({days} Days)")
            st.line_chart(historical_data.set_index("Timestamp")["Close"])
            st.subheader("📋 Historical Data (Last 10 Entries)")
            st.dataframe(historical_data.tail(10))
        else:
            st.error("❌ No historical data found. Try a different time period.")
