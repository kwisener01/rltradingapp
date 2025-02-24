import streamlit as st
import requests
import pandas as pd
import openai

# Load API Keys from secrets.toml
ALPHAVANTAGE_API_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit App Title
st.title("AI-Powered ES Futures Trading Advisor (Alpha Vantage)")

# Sidebar for User Inputs
st.sidebar.header("ES Futures Data Input")
interval = st.sidebar.selectbox("Select Interval", ["1min", "5min", "15min", "30min", "60min"])
output_size = st.sidebar.selectbox("Select Data Size", ["compact", "full"])

# Function to Fetch ES Futures Data from Alpha Vantage
def fetch_es_futures_data(api_key, interval, output_size):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": "ES=F",  # ES Futures Symbol
        "interval": interval,
        "outputsize": output_size,
        "apikey": api_key
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    time_series_key = f"Time Series ({interval})"
    if time_series_key in data:
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    else:
        st.error("Failed to fetch data. Check API key or rate limits.")
        return pd.DataFrame()

# Function to Query OpenAI for Trading Strategy
def ask_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a futures trading expert."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"

# Main App Logic
if st.sidebar.button("Get ES Futures Data"):
    with st.spinner("Fetching ES Futures data..."):
        es_data = fetch_es_futures_data(ALPHAVANTAGE_API_KEY, interval, output_size)

    if not es_data.empty:
        st.subheader("📊 ES Futures Market Data")
        st.line_chart(es_data["Close"])

        # Prepare Data for OpenAI Analysis
        prompt = f"""Given the following ES Futures data:
        {es_data.tail(10).to_string()}
        Suggest a scalping strategy focused on volatility, momentum, and risk management.
        """

        # Get AI-Generated Trading Strategy
        with st.spinner("Analyzing with AI..."):
            ai_response = ask_openai(prompt)

        st.subheader("🤖 AI-Generated ES Futures Trading Strategy")
        st.write(ai_response)
    else:
        st.error("❌ No data found. Please try again later.")
