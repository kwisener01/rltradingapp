import streamlit as st
import requests
import openai
import datetime

# Load Finnhub API Key
FINNHUB_API_KEY = st.secrets["FINNHUB"]["API_KEY"]

# Initialize OpenAI client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Streamlit App Title
st.title("📊 AI-Powered Trading Advisor (Real-Time)")

# List of Top 20 Stocks + SPY and QQQ
top_stocks = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
              "UNH", "V", "JNJ", "WMT", "JPM", "PG", "MA", "HD", "DIS", "BAC", "XOM", "PFE"]

# Sidebar for User Inputs
st.sidebar.header("Select Stock/ETF")
selected_stock = st.sidebar.selectbox("Select Ticker", top_stocks)

# Session state for caching
if 'stock_data' not in st.session_state:
    st.session_state['stock_data'] = None
if 'ai_strategy' not in st.session_state:
    st.session_state['ai_strategy'] = None

# Function to Fetch Real-Time Data from Finnhub
def fetch_real_time_price(ticker):
    url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "c" in data and data["c"] != 0:
        # Convert Unix timestamp to readable time
        timestamp = datetime.datetime.fromtimestamp(data["t"]).strftime('%Y-%m-%d %H:%M:%S')

        return {
            "Symbol": ticker,
            "Latest Price": f"${data['c']:.2f}",
            "Previous Close": f"${data['pc']:.2f}",
            "Open Price": f"${data['o']:.2f}",
            "High Price": f"${data['h']:.2f}",
            "Low Price": f"${data['l']:.2f}",
            "Change": f"${data['d']:.2f}",
            "Change %": f"{data['dp']:.2f}%",
            "Timestamp": timestamp
        }
    else:
        return None

# 📊 BUTTON 1: Get Real-Time Stock Data
if st.button("Get Real-Time Price"):
    with st.spinner(f"Fetching real-time price for {selected_stock}..."):
        stock_data = fetch_real_time_price(selected_stock)
        if stock_data is not None:
            st.session_state['stock_data'] = stock_data
            st.success(f"Real-time data for {selected_stock} fetched successfully!")

            # 📈 Display Stock Information
            st.subheader(f"📊 {selected_stock} Market Data (Live)")
            st.write(stock_data)
        else:
            st.error("❌ No real-time data found. Please try again later.")

# 🤖 BUTTON 2: Get AI Trading Strategy
if st.button("Get AI Trading Strategy"):
    if st.session_state['stock_data'] is not None:
        prompt = f"""Given the following real-time data for {selected_stock}:

        - Latest Price: {st.session_state['stock_data']['Latest Price']}
        - Previous Close: {st.session_state['stock_data']['Previous Close']}
        - Open Price: {st.session_state['stock_data']['Open Price']}
        - High Price: {st.session_state['stock_data']['High Price']}
        - Low Price: {st.session_state['stock_data']['Low Price']}
        - Change: {st.session_state['stock_data']['Change']} ({st.session_state['stock_data']['Change %']})
        - Timestamp: {st.session_state['stock_data']['Timestamp']}

        Please provide an **intraday trading strategy** based on these conditions:
        - If price is above previous close, suggest a bullish strategy.
        - If price is below previous close, suggest a bearish strategy.
        - Identify **support and resistance levels**.
        - Include **entry points, stop-loss, and profit targets**.
        - Consider **volume trends & institutional buying/selling patterns**.

        Format the response clearly with actionable insights.
        """

        with st.spinner("Generating AI Trading Strategy..."):
            ai_strategy = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a trading expert providing real-time analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            st.session_state['ai_strategy'] = ai_strategy.choices[0].message.content.strip()

        # Show AI-Generated Strategy
        st.subheader(f"🤖 AI-Generated {selected_stock} Trading Strategy")
        st.write(st.session_state['ai_strategy'])
    else:
        st.error("❗ Please fetch real-time stock data first before requesting AI analysis.")
