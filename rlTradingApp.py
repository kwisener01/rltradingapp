import streamlit as st
import requests
import pandas as pd
import datetime
import openai

# Load API Keys from Streamlit Secrets
POLYGON_API_KEY = st.secrets["POLYGON"]["API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI"]["API_KEY"]

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Streamlit App Title
st.title("📊 AI-Powered Trading Advisor (Historical & AI Strategy)")

# List of Top Stocks / ETFs
top_stocks = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B"]
selected_stock = st.sidebar.selectbox("Select Ticker", top_stocks)

# User selects interval and number of days
interval = st.sidebar.selectbox("Select Interval", ["minute", "hour", "day"], index=2)
days = st.sidebar.slider("Select Number of Days for History", 1, 30, 7)

# Function to Fetch Historical Data from Polygon.io
def fetch_historical_data(ticker, interval, days):
    end_date = datetime.datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.today() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{interval}/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY}
    response = requests.get(url, params=params)
    data = response.json()

    # Debugging: Show Raw API Response
    st.write("📡 **Raw API Response:**", data)

    if "results" in data:
        df = pd.DataFrame(data["results"])
        df["Timestamp"] = pd.to_datetime(df["t"], unit="ms")
        df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
        df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]
        return df
    else:
        return None

# Function to Generate AI Trading Strategy
def generate_ai_strategy(df, ticker):
    if df is None or df.empty:
        return "No sufficient data to generate a strategy."

    latest_data = df.tail(10).to_string(index=False)  # Get the last 10 data points as text

    prompt = f"""
    You are an AI trading expert analyzing {ticker} stock.
    Based on the latest market data:
    
    {latest_data}

    Provide a **detailed but straightforward trading strategy** that includes:
    - **Current trend (bullish/bearish/sideways)**
    - **Key support & resistance levels**
    - **Entry points, stop-loss, and take-profit targets**
    - **Volume impact (high volume = institutional trading)**
    - **Whether to scalp, day trade, or swing trade**
    
    The response should be technical and actionable.
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
        historical_data = fetch_historical_data(selected_stock, interval, days)

    if historical_data is not None and not historical_data.empty:
        st.subheader(f"📈 {selected_stock} Historical Price Chart ({days} Days)")
        st.line_chart(historical_data.set_index("Timestamp")["Close"])
        st.subheader("📋 Historical Data (Last 10 Entries)")
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
