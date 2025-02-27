import streamlit as st
import requests
import pandas as pd
import datetime

# Load Polygon.io API Key
POLYGON_API_KEY = st.secrets["POLYGON"]["API_KEY"]

# Streamlit App Title
st.title("📊 AI-Powered Trading Advisor (Real-Time & Historical)")

# List of Top 20 Stocks + SPY and QQQ
top_stocks = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
              "UNH", "V", "JNJ", "WMT", "JPM", "PG", "MA", "HD", "DIS", "BAC", "XOM", "PFE"]

# Sidebar for User Inputs
st.sidebar.header("Select Stock/ETF & Timeframe")
selected_stock = st.sidebar.selectbox("Select Ticker", top_stocks)

# User selects interval
interval = st.sidebar.selectbox("Select Interval", ["minute", "hour", "day"], index=2)  # Default: Daily
days = st.sidebar.slider("Select Number of Days for History", 1, 30, 7)

# Function to Fetch Historical Data from Polygon.io
def fetch_historical_data(ticker, interval, days):
    end_date = datetime.datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.today() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")

    # Polygon.io API URL for historical prices
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{interval}/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
    
    response = requests.get(url)
    data = response.json()

    # Debugging: Show Raw API Response
    st.write("📡 **Raw API Response:**", data)

    if "results" in data:
        df = pd.DataFrame(data["results"])
        df["Timestamp"] = pd.to_datetime(df["t"], unit="ms")  # Convert Unix timestamp to readable date
        df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
        return df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]
    else:
        return None

# 📊 BUTTON: Get Historical Data
if st.button("Get Historical Data"):
    with st.spinner(f"Fetching historical data for {selected_stock}..."):
        historical_data = fetch_historical_data(selected_stock, interval, days)

        if historical_data is not None:
            st.subheader(f"📈 {selected_stock} Historical Price Chart ({days} Days)")
            st.line_chart(historical_data.set_index("Timestamp")["Close"])
            st.subheader("📋 Historical Data (Last 10 Entries)")
            st.dataframe(historical_data.tail(10))
        else:
            st.error("❌ No historical data found. Try a different time period.")
