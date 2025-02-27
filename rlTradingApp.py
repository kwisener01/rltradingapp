import streamlit as st
import requests
import datetime

# Load Polygon.io API Key from secrets
POLYGON_API_KEY = st.secrets["POLYGON"]["API_KEY"]

# Ticker to Test
selected_stock = "SPY"

# Define start and end dates
end_date = datetime.datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

# Function to test different timeframes
def test_polygon_api(ticker, interval):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{interval}/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY}
    response = requests.get(url, params=params)
    data = response.json()

    if "status" in data and data["status"] == "NOT_AUTHORIZED":
        return f"❌ NOT AUTHORIZED for {interval} interval."
    elif "results" in data:
        return f"✅ SUCCESS: {len(data['results'])} data points received for {interval} interval."
    else:
        return f"⚠️ UNKNOWN RESPONSE for {interval}: {data}"

# Test different intervals
st.write("🔍 **Testing Polygon.io API Access for Different Intervals...**")
for interval in ["minute", "hour", "day"]:
    result = test_polygon_api(selected_stock, interval)
    st.write(result)
