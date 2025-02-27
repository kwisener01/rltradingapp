import streamlit as st
import requests
import pandas as pd
import datetime

# Load Polygon.io API Key from secrets
POLYGON_API_KEY = st.secrets["POLYGON"]["API_KEY"]

# Streamlit App Title
st.title("📊 Historical Data from Polygon.io")

# List of Top Stocks / ETFs (sample list)
top_stocks = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B"]
selected_stock = st.sidebar.selectbox("Select Ticker", top_stocks)

# User selects interval and number of days
# Polygon.io expects interval as "minute", "hour", or "day"
interval = st.sidebar.selectbox("Select Interval", ["minute", "hour", "day"], index=2)  # Default daily
days = st.sidebar.slider("Select Number of Days for History", 1, 30, 7)

# Function to fetch historical data from Polygon.io
def fetch_historical_data(ticker, interval, days):
    # Define date range: use today's date and subtract number of days for start date
    end_date = datetime.datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.today() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    
    # Polygon.io endpoint for aggregates (candles)
    # Using a 1 unit range per candle; adjust as needed
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{interval}/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "apiKey": POLYGON_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Debug: Display the raw API response for troubleshooting
    st.write("📡 **Raw API Response:**", data)
    
    if "results" in data:
        results = data["results"]
        # Create DataFrame from the results
        df = pd.DataFrame(results)
        # Convert Unix timestamp in 't' to a datetime column
        df["Timestamp"] = pd.to_datetime(df["t"], unit="ms")
        # Rename columns for clarity
        df.rename(columns={
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume"
        }, inplace=True)
        # Select and order desired columns
        df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]
        return df
    else:
        return None

# BUTTON: Get Historical Data
if st.button("Get Historical Data"):
    with st.spinner(f"Fetching historical data for {selected_stock}..."):
        hist_data = fetch_historical_data(selected_stock, interval, days)
    
    if hist_data is not None and not hist_data.empty:
        st.subheader(f"📈 {selected_stock} Historical Price Chart ({days} Days)")
        # Plot the close price using Streamlit's line_chart
        st.line_chart(hist_data.set_index("Timestamp")["Close"])
        st.subheader("📋 Historical Data (Last 10 Entries)")
        st.dataframe(hist_data.tail(10))
    else:
        st.error("❌ No historical data found. Try a different time period or interval.")
