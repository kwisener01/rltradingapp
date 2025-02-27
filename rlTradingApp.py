import streamlit as st
import requests
import pandas as pd
import datetime
import pytz
import openai

# Load API Keys from Streamlit Secrets
POLYGON_API_KEY = st.secrets["POLYGON"]["API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI"]["API_KEY"]

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Define EST timezone
EST = pytz.timezone("US/Eastern")

# Streamlit App Title
st.title("📊 AI-Powered Trading Advisor (EST Time & Smarter AI Learning)")

# List of Top Stocks / ETFs
top_stocks = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B"]
selected_stock = st.sidebar.selectbox("Select Ticker", top_stocks)

# User selects interval and number of days
interval = st.sidebar.selectbox("Select Interval", ["minute", "hour", "day"], index=0)  # Default to "minute" for short-term analysis
hours_to_analyze = st.sidebar.slider("Select Number of Hours for AI Strategy", 1, 12, 1)  # Analyze last 1-12 hours

# Function to Fetch Historical Data from Polygon.io
def fetch_historical_data(ticker, interval, hours):
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(hours=hours)

    # Convert time to required format
    start_date = start_time.strftime("%Y-%m-%d")
    end_date = end_time.strftime("%Y-%m-%d")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{interval}/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY}
    response = requests.get(url, params=params)
    data = response.json()

    # Debugging: Show Raw API Response
    st.write("📡 **Raw API Response:**", data)

    if "results" in data:
        df = pd.DataFrame(data["results"])
        df["Timestamp"] = pd.to_datetime(df["t"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(EST)  # Convert to EST
        df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
        df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]
        return df
    else:
        return None

# Function to Generate AI Trading Strategy
def generate_ai_strategy(df, ticker):
    if df is None or df.empty:
        return "No sufficient data to generate a strategy."

    latest_data = df.tail(30).to_string(index=False)  # Use last 30 candles (~1 hour for minute data)

    prompt = f"""
    You are an AI trading expert analyzing {ticker} stock based on the last {len(df)} market data points.

    Recent market data (last 1 hour in EST):
    {latest_data}

    Provide a **detailed but straightforward trading strategy** that includes:
    - **Current market trend (bullish, bearish, sideways)**
    - **Key support & resistance levels**
    - **Optimal entry & exit points**
    - **Stop-loss & take-profit levels**
    - **Volume impact (institutional buying/selling detection)**
    - **Trade recommendation (scalp, day trade, or swing trade)**

    Be technical, clear, and concise.
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
    with st.spinner(f"Fetching historical data for {selected_stock}..."):
        historical_data = fetch_historical_data(selected_stock, interval, hours_to_analyze)

    if historical_data is not None and not historical_data.empty:
        st.subheader(f"📈 {selected_stock} Historical Price Chart ({hours_to_analyze} Hours, EST)")
        st.line_chart(historical_data.set_index("Timestamp")["Close"])
        st.subheader("📋 Historical Data (Last 10 Entries in EST)")
        st.dataframe(historical_data.tail(10))

        # Store historical data for AI analysis
        st.session_state["historical_data"] = historical_data
    else:
        st.error("❌ No historical data found. Try a different time period or interval.")

# 🤖 BUTTON 2: Get AI Trading Strategy
if st.button("Get AI Trading Strategy"):
    if "historical_data" in st.session_state and not st.session_state["historical_data"].empty:
        with st.spinner("Generating AI trading strategy..."):
            ai_strategy = generate_ai_strategy(st.session_state["historical_data"], selected_stock)

        st.subheader(f"🤖 AI-Generated {selected_stock} Trading Strategy")
        st.write(ai_strategy)
    else:
        st.error("❗ Please fetch stock data first before requesting AI analysis.")
