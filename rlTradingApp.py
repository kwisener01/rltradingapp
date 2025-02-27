import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import time
import datetime

# Load Polygon.io API Key
POLYGON_API_KEY = st.secrets["POLYGON"]["API_KEY"]

# Streamlit App Title
st.title("📊 AI-Powered Trading Advisor (Rate Limit Fix & Free Data Fallback)")

# List of Top Stocks / ETFs
top_stocks = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B"]
selected_stock = st.sidebar.selectbox("Select Ticker", top_stocks)

# User selects the interval
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=5)  # Default: 1 Day
days = st.sidebar.slider("Select Number of Days for History", 1, 30, 7)

# Function to Fetch Historical Data from Polygon.io
def fetch_historical_data_polygon(ticker, interval, days):
    end_date = datetime.datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.today() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{interval}/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY}

    for attempt in range(3):  # Retry up to 3 times
        response = requests.get(url, params=params)
        data = response.json()

        if "status" in data and data["status"] == "ERROR":
            if "exceeded the maximum requests per minute" in data.get("error", ""):
                st.warning("🚨 Rate limit exceeded. Waiting 10 seconds before retrying...")
                time.sleep(10)  # Wait and retry
            else:
                st.error(f"❌ Polygon.io Error: {data.get('error', 'Unknown error')}")
                return None
        elif "results" in data:
            df = pd.DataFrame(data["results"])
            df["Timestamp"] = pd.to_datetime(df["t"], unit="ms")  # Convert timestamps
            df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
            return df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]
    
    return None  # Failed after retries

# Function to Fetch Data from Yahoo Finance as Backup
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

# 📊 BUTTON: Get Historical Data
if st.button("Get Historical Data"):
    with st.spinner(f"Fetching {selected_stock} data..."):
        historical_data = fetch_historical_data_polygon(selected_stock, interval, days)
        if historical_data is None:
            st.warning("⚠️ Switching to Yahoo Finance due to API limit...")
            historical_data = fetch_historical_data_yfinance(selected_stock, interval, days)

    if historical_data is not None and not historical_data.empty:
        st.subheader(f"📈 {selected_stock} Historical Price Chart ({days} Days)")
        st.line_chart(historical_data.set_index(historical_data.columns[0])["Close"])
        st.subheader("📋 Historical Data (Last 10 Entries)")
        st.dataframe(historical_data.tail(10))
    else:
        st.error("❌ No historical data found. Try a different time period.")
