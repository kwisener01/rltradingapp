import streamlit as st
import pandas as pd
import openai
import requests

# Load Alpha Vantage API Key
ALPHAVANTAGE_API_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]

# Initialize OpenAI client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Streamlit App Title
st.title("📊 AI-Powered Trading Advisor (Technical Strategies)")

# List of Top 20 Stocks + SPY and QQQ
top_stocks = [
    "SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "UNH", "V", "JNJ", "WMT", "JPM", "PG", "MA", "HD", "DIS", "BAC", "XOM", "PFE"
]

# Sidebar for User Inputs
st.sidebar.header("Select Stock/ETF and Timeframe")
selected_stock = st.sidebar.selectbox("Select Ticker", top_stocks)
interval = st.sidebar.selectbox("Select Interval", ["1min", "5min", "15min", "30min", "60min"])
period = st.sidebar.selectbox("Select Period", ["1d", "5d", "1mo"])

# Session state for caching
if 'stock_data' not in st.session_state:
    st.session_state['stock_data'] = None

if 'ai_strategy' not in st.session_state:
    st.session_state['ai_strategy'] = None

# Function to Fetch Stock Data from Alpha Vantage
def fetch_stock_data(ticker, interval):
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": interval,
        "apikey": ALPHAVANTAGE_API_KEY,
        "outputsize": "compact"
    }
    
    response = requests.get(url)
    data = response.json()

    if "Time Series" in data:
        df = pd.DataFrame.from_dict(data[f"Time Series ({interval})"], orient="index")
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    else:
        return None

# Function to Fetch Technical Indicators
def fetch_technical_indicators(ticker, interval):
    indicators = {}

    # Fetch RSI
    rsi_url = f"https://www.alphavantage.co/query"
    rsi_params = {
        "function": "RSI",
        "symbol": ticker,
        "interval": interval,
        "time_period": 14,
        "series_type": "close",
        "apikey": ALPHAVANTAGE_API_KEY
    }
    rsi_data = requests.get(rsi_url, params=rsi_params).json()
    if "Technical Analysis: RSI" in rsi_data:
        indicators["RSI"] = float(list(rsi_data["Technical Analysis: RSI"].values())[0]["RSI"])

    # Fetch MACD
    macd_url = f"https://www.alphavantage.co/query"
    macd_params = {
        "function": "MACD",
        "symbol": ticker,
        "interval": interval,
        "series_type": "close",
        "apikey": ALPHAVANTAGE_API_KEY
    }
    macd_data = requests.get(macd_url, params=macd_params).json()
    if "Technical Analysis: MACD" in macd_data:
        macd_values = list(macd_data["Technical Analysis: MACD"].values())[0]
        indicators["MACD"] = float(macd_values["MACD"])
        indicators["Signal"] = float(macd_values["MACD_Signal"])

    return indicators

# Function to Query OpenAI for Strategy
def ask_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a trading expert providing real-time analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# 📊 BUTTON 1: Get Stock Data
if st.button("Get Stock Data"):
    with st.spinner(f"Fetching {selected_stock} data..."):
        stock_data = fetch_stock_data(selected_stock, interval)
        if stock_data is not None:
            st.session_state['stock_data'] = stock_data
            st.success(f"{selected_stock} data fetched successfully!")

            # 📈 Display Line Chart
            st.subheader("📈 Price Chart")
            st.line_chart(stock_data["4. close"])

            # 📋 Display Raw Data
            st.subheader(f"📊 {selected_stock} Market Data")
            st.dataframe(stock_data.tail(10), use_container_width=True)

            # 📊 Fetch and Display Technical Indicators
            st.subheader("📈 Technical Indicators")
            indicators = fetch_technical_indicators(selected_stock, interval)
            st.write(indicators)

        else:
            st.error("❌ No data found. Please try again later.")

# 🤖 BUTTON 2: Get AI Trading Strategy
if st.button("Get AI Trading Strategy"):
    if st.session_state['stock_data'] is not None:
        indicators = fetch_technical_indicators(selected_stock, interval)

        prompt = f"""Given the following {selected_stock} market data:
        {st.session_state['stock_data'].tail(10).to_string()}

        The latest technical indicators are:
        - RSI: {indicators.get('RSI', 'N/A')}
        - MACD: {indicators.get('MACD', 'N/A')}
        - MACD Signal: {indicators.get('Signal', 'N/A')}

        Please provide a **technical trading strategy** based on these conditions.
        - If RSI is above 70, suggest overbought strategies.
        - If RSI is below 30, suggest oversold strategies.
        - If MACD crosses above the signal line, suggest a bullish strategy.
        - If MACD crosses below the signal line, suggest a bearish strategy.
        - Consider volume and price action for **institutional buying/selling** trends.

        Include **entry/exit points, stop losses, and profit targets.**
        """

        with st.spinner("Generating AI Trading Strategy..."):
            ai_strategy = ask_openai(prompt)
            st.session_state['ai_strategy'] = ai_strategy

        # Show AI-Generated Strategy
        st.subheader(f"🤖 AI-Generated {selected_stock} Trading Strategy")
        st.write(ai_strategy)
    else:
        st.error("❗ Please fetch stock data first before requesting AI analysis.")
