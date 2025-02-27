import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import time
import datetime
import openai

# Load API Keys from Streamlit Secrets
POLYGON_API_KEY = st.secrets["POLYGON"]["API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI"]["API_KEY"]

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Streamlit App Title
st.title("📊 AI-Powered Trading Advisor (Smarter Market Learning)")

# List of Top Stocks / ETFs
top_stocks = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B"]
selected_stock = st.sidebar.selectbox("Select Ticker", top_stocks)

# User selects interval & number of days
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=5)
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

# Function to Generate AI Trading Strategy (Uses 50+ Data Points)
def generate_ai_strategy(df, ticker):
    if df is None or df.empty:
        return "No sufficient data to generate a strategy."

    latest_data = df.tail(50).to_string(index=False)  # AI learns from last 50 market data points

    prompt = f"""
    You are an AI trading expert analyzing {ticker} stock using the last {len(df)} historical data points.

    Recent market data (last {len(df)} candles):
    {latest_data}

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
        historical_data = fetch_historical_data_polygon(selected_stock, interval, days)
        if historical_data is None:
            st.warning("⚠️ Switching to Yahoo Finance due to API limit...")
            historical_data = fetch_historical_data_yfinance(selected_stock, interval, days)

    if historical_data is not None and not historical_data.empty:
        st.subheader(f"📈 {selected_stock} Historical Price Chart ({days} Days)")
        st.line_chart(historical_data.set_index(historical_data.columns[0])["Close"])
        st.subheader("📋 Historical Data (Last 50 Entries)")
        st.dataframe(historical_data.tail(50))  # Show last 50 entries for better insights

        # Store historical data for AI analysis
        st.session_state["historical_data"] = historical_data
    else:
        st.error("❌ No historical data found. Try a different time period.")

# 🤖 BUTTON 2: Get AI Trading Strategy
if st.button("Get AI Trading Strategy"):
    if "historical_data" in st.session_state and not st.session_state["historical_data"].empty:
        with st.spinner("Generating AI trading strategy..."):
            ai_strategy = generate_ai_strategy(st.session_state["historical_data"], selected_stock)

        st.subheader(f"🤖 AI-Generated {selected_stock} Trading Strategy")
        st.write(ai_strategy)
    else:
        st.error("❗ Please fetch stock data first before requesting AI analysis.")
