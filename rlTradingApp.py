import streamlit as st
import yfinance as yf
import pandas as pd
import openai

# Initialize OpenAI client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Streamlit App Title
st.title("📊 AI-Powered Trading Advisor (Buttons Reorganized)")

# List of Top 20 Stocks + SPY and QQQ
top_stocks = [
    "SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "UNH", "V", "JNJ", "WMT", "JPM", "PG", "MA", "HD", "DIS", "BAC", "XOM", "PFE"
]

# Sidebar for User Inputs
st.sidebar.header("Select Stock/ETF and Timeframe")

# Dropdown for Stock Selection
selected_stock = st.sidebar.selectbox("Select Ticker", top_stocks)
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h", "1d"])
period = st.sidebar.selectbox("Select Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])

# Session state to hold data
if 'stock_data' not in st.session_state:
    st.session_state['stock_data'] = None

if 'ai_strategy' not in st.session_state:
    st.session_state['ai_strategy'] = None

# Function to Fetch Stock Data
def fetch_stock_data(ticker, interval, period):
    stock_data = yf.download(tickers=ticker, interval=interval, period=period)
    return stock_data

# Enhanced OpenAI Call with Error Handling
def ask_openai(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a trading expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        # Log the full response for debugging
        st.write("### 🔄 Raw OpenAI Response:", response)

        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            return "⚠️ No strategy generated. Check API response."

    except openai.RateLimitError:
        return "🚫 Rate limit exceeded. Please try again later."
    except openai.AuthenticationError:
        return "🚫 Invalid API Key. Please verify your credentials."
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

# Function to Calculate Statistics
def calculate_statistics(df):
    stats = {
        "Mean Close": round(df["Close"].mean(), 2),
        "Median Close": round(df["Close"].median(), 2),
        "Std Dev Close": round(df["Close"].std(), 2),
        "Max Close": round(df["Close"].max(), 2),
        "Min Close": round(df["Close"].min(), 2),
        "Total Volume": round(df["Volume"].sum(), 2)
    }
    stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
    stats_df["Value"] = stats_df["Value"].astype(str)  # Ensure Arrow compatibility
    return stats_df

# 📊 BUTTON 1: Get Stock Data
if st.button("Get Stock Data"):
    with st.spinner(f"Fetching {selected_stock} data..."):
        stock_data = fetch_stock_data(selected_stock, interval, period)
        if not stock_data.empty:
            st.session_state['stock_data'] = stock_data
            st.success(f"{selected_stock} data fetched successfully!")

            # 📈 Display Line Chart
            st.subheader("📈 Price Chart")
            st.line_chart(stock_data["Close"])

            # 📋 Display Raw Data
            st.subheader(f"📊 {selected_stock} Market Data")
            st.dataframe(stock_data.tail(10), use_container_width=True)

            # 📊 Display Basic Statistics
            st.subheader("📈 Basic Statistics")
            stats_df = calculate_statistics(stock_data)
            st.table(stats_df)

        else:
            st.error("❌ No data found. Please try again later.")

# 🤖 BUTTON 2: Get AI Trading Strategy (Now directly below)
if st.button("Get AI Trading Strategy"):
    if st.session_state['stock_data'] is not None:
        prompt = f"""Given the following {selected_stock} market data:
        {st.session_state['stock_data'].tail(10).to_string()}

        Please suggest a trading strategy for the next market open. Include technical insights, risk management, and ideal entry/exit points.
        """

        with st.spinner("Generating AI Trading Strategy..."):
            ai_strategy = ask_openai(prompt)
            st.session_state['ai_strategy'] = ai_strategy

        # Show AI-Generated Strategy
        st.subheader(f"🤖 AI-Generated {selected_stock} Trading Strategy")
        st.write(ai_strategy)
    else:
        st.error("❗ Please fetch stock data first before requesting AI analysis.")
