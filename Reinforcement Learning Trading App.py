import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces
import os
import datetime

# ============== STREAMLIT CONFIG ==============
st.set_page_config(page_title="Reinforcement Learning Trading App", layout="wide")

# ============== ALPHA VANTAGE API CONFIG ==============
API_KEY = 'MVOSLRG0ESBSJ2IF'
BASE_URL = 'https://www.alphavantage.co/query'

# ============== TRADING ENVIRONMENT ==============
class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = self.add_indicators(df)
        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.df.columns),), dtype=np.float32)
        self.balance = 10000
        self.position = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.trade_count = 0
        self.total_profit = 0
        self.trades = []

    def add_indicators(self, df):
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(20)
        df['zscore'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
        df['momentum'] = df['close'].diff()
        df['acceleration'] = df['momentum'].diff()
        df['fractal_dim'] = df['volatility'] / df['momentum'].rolling(window=20).std()
        df['kalman_est'] = self.kalman_filter(df['close'])
        df.dropna(inplace=True)
        return df

    def kalman_filter(self, series, alpha=0.8):
        kalman_est = [series.iloc[0]]
        for price in series[1:]:
            new_est = alpha * price + (1 - alpha) * kalman_est[-1]
            kalman_est.append(new_est)
        return pd.Series(kalman_est, index=series.index)

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.trade_count = 0
        self.total_profit = 0
        self.trades = []
        return self.df.iloc[self.current_step].values

    def step(self, action):
        price = self.df.iloc[self.current_step]['close']

        if action == 1:  # Buy
            if self.balance > 0:
                self.position = self.balance / price
                self.balance = 0
                self.trade_count += 1
                self.trades.append({'step': self.current_step, 'type': 'buy', 'price': price})

        elif action == 2:  # Sell
            if self.position > 0:
                self.balance = self.position * price
                self.position = 0
                self.trade_count += 1
                self.total_profit += self.balance - 10000
                self.trades.append({'step': self.current_step, 'type': 'sell', 'price': price})

        self.net_worth = self.balance + self.position * price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # Optimized Reward Function
        reward = 0
        reward += (self.net_worth - 10000) * 0.1
        reward -= abs(self.net_worth - self.max_net_worth) * 0.05
        reward -= self.trade_count * 0.01
        reward += (self.total_profit / 10000) * 0.2

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return self.df.iloc[self.current_step].values, reward, done, {}

# ============== DATA FETCHING FUNCTION ==============
def fetch_data(symbol, interval='1min', outputsize='compact'):
    url = f"{BASE_URL}?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={API_KEY}&outputsize={outputsize}"
    response = requests.get(url)
    data = response.json()

    if f"Time Series ({interval})" in data:
        df = pd.DataFrame(data[f"Time Series ({interval})"]).T
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    else:
        st.error("Error fetching data from Alpha Vantage API.")
        return pd.DataFrame()

# ============== BACKTESTING FUNCTION ==============
def backtest(env, model):
    obs = env.reset()
    for _ in range(len(env.df)):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            break

    # Extract trade history
    trades = pd.DataFrame(env.trades)

    # Plot backtest results
    st.subheader("Backtest Results")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=env.df.index, open=env.df['open'], high=env.df['high'], low=env.df['low'], close=env.df['close'], name='Price'))

    for _, trade in trades.iterrows():
        color = 'green' if trade['type'] == 'buy' else 'red'
        fig.add_trace(go.Scatter(x=[env.df.index[trade['step']]], y=[trade['price']], mode='markers', marker=dict(color=color, size=10), name=trade['type'].capitalize()))

    st.plotly_chart(fig, use_container_width=True)

    # Performance Metrics
    st.write(f"Total Profit: ${env.total_profit:.2f}")
    st.write(f"Total Trades: {env.trade_count}")

# ============== MAIN STREAMLIT APP ==============
st.title("Reinforcement Learning Trading App with Backtesting")

# Sidebar
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Enter Stock Symbol", value='AAPL')
interval = st.sidebar.selectbox("Select Time Interval", options=['1min', '5min', '15min'])
train_model = st.sidebar.button("Train Model")
backtest_model = st.sidebar.button("Run Backtest")

# Fetch Data
data = fetch_data(symbol, interval)

if not data.empty:
    st.subheader(f"{symbol} Price Data ({interval} interval)")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'], name='Price'))
    st.plotly_chart(fig, use_container_width=True)

    # Train RL Model
    if train_model:
        st.subheader("Training Reinforcement Learning Model...")
        env = TradingEnv(data)
        check_env(env)
        env = DummyVecEnv([lambda: env])

        model = DQN("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)

        model_path = "trained_rl_model.zip"
        model.save(model_path)
        st.success("Model Trained and Saved!")

    # Load Trained Model
    if os.path.exists("trained_rl_model.zip"):
        model = DQN.load("trained_rl_model.zip")

        # Run Backtest
        if backtest_model:
            env = TradingEnv(data)
            backtest(env, model)
else:
    st.warning("No data to display.")
