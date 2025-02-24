import streamlit as st
import yfinance as yf
import pandas as pd
import openai

# Initialize OpenAI client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Streamlit App Title
st.title("📊 AI-Powered Trading Advisor with Data Insights")

# List of Top 20 Stocks + SPY and QQQ
top_stocks = [
    "SPY", "QQQ",  # Index ETFs
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "UNH", "V", "JNJ", "WMT", "JPM", "PG", "MA", "HD", "DIS", "BAC", "XOM", "PFE"
]

# Sidebar for User Inputs
st.sidebar.header("Select Stock/ETF and Timeframe")

# Dropdown for Stock Selection
selected_stock = st.sidebar.selectbox("Select Ticker", top_stocks)

# Timeframe and Period Selection
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h", "1d"])
period = st.sidebar.selectbox("Select Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])

# Function to Fetch Stock Data from Yahoo Finance
def fetch_stock_data(ticker, interval, period):
    stock_data = yf.download(tickers=ticker, interval=interval, period=period)
    return stock_data

# Function to Query OpenAI for Strategy
def ask_openai(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Use gpt-3.5-turbo if preferred
            messages=[
                {"role": "system", "content": "You are a trading expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# Function to Calculate Basic Statistics
def calculate_statistics(df):
    stats = {
        "Mean Close": round(df["Close"].mean(), 2),
        "Median Close": round(df["Close"].median(), 2),
        "Std Dev Close": round(df["Close"].std(), 2),
        "Max Close": round(df["Close"].max(), 2),
        "Min Close": round(df["Close"].min(), 2),
        "Total Volume": round(df["Volume"].sum(), 2)
    }
    return pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])

# Main App Logic
if st.sidebar.button("Get Stock Data"):
    with st.spinner(f"Fetching {selected_stock} data..."):
        stock_data = fetch_stock_data(selected_stock, interval, period)

    if not stock_data.empty:
        # Layout: Two Columns
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"📊 {selected_stock} Market Data")
            st.dataframe(stock_data.tail(10))

        with col2:
            st.subheader("📈 Basic Statistics")
            stats_df = calculate_statistics(stock_data)
            st.table(stats_df)

        # Line Chart for Close Prices
        st.subheader("📈 Price Chart")
        st.line_chart(stock_data["Close"])

        # AI Analysis Button
        if st.button("Get AI Trading Strategy"):
            # Prepare data for OpenAI prompt
            prompt = f"""The market is currently closed. Given the following {selected_stock} market data:
            {stock_data.tail(10).to_string()}

            Please suggest a trading strategy for the next market open. Include technical insights, risk management, and ideal entry/exit points.
            """

            # Get AI-Generated Strategy
            with st.spinner("Generating AI Trading Strategy..."):
                ai_strategy = ask_openai(prompt)

            st.subheader(f"🤖 AI-Generated {selected_stock} Trading Strategy")
            st.write(ai_strategy)

    else:
        st.error("❌ No data found. Please try again later.")
