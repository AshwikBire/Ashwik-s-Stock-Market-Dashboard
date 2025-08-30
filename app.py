import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
from plotly import graph_objects as go
from streamlit_option_menu import option_menu
from textblob import TextBlob
from xgboost import XGBRegressor
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
import json
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Set page config with dark theme
st.set_page_config(
    page_title="MarketMentor",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for dark theme with red accents
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 75, 75, 0.3);
    }
    .stSelectbox>div>div>select {
        background-color: #262730;
        color: white;
        border: 1px solid #444;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
        border: 1px solid #444;
    }
    .stSlider>div>div>div>div {
        background-color: #FF4B4B;
    }
    .stProgress>div>div>div {
        background-color: #FF4B4B;
    }
    .stMetric {
        background-color: #262730;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stAlert {
        background-color: #262730;
        border-left: 4px solid #FF4B4B;
    }
    .css-1d391kg {
        background-color: #0E1117;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #FAFAFA;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: #0E1117;
        font-weight: bold;
    }
    .stDataFrame {
        background-color: #262730;
    }
    .stTable {
        background-color: #262730;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    header {
        background-color: #0E1117;
    }
    .card {
        background-color: #262730;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .card h3 {
        margin-top: 0;
        color: #FF4B4B;
    }
    .positive {
        color: #00C853;
    }
    .negative {
        color: #FF4B4B;
    }
    .neutral {
        color: #FFD700;
    }
    .company-header {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    .company-header img {
        width: 60px;
        height: 60px;
        margin-right: 1rem;
        border-radius: 8px;
    }
    .company-header h1 {
        margin-bottom: 0;
    }
    .financial-metric {
        text-align: center;
        padding: 1rem;
        background-color: #1E2229;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .financial-metric h4 {
        margin: 0 0 0.5rem 0;
        color: #FF4B4B;
        font-size: 0.9rem;
    }
    .financial-metric p {
        margin: 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #0E1117;
    }
</style>
""", unsafe_allow_html=True)

# Load company data from JSON file
@st.cache_data
def load_company_data():
    # Sample company data
    company_data = {
        "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
        "MSFT": {"name": "Microsoft Corporation", "sector": "Technology"},
        "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology"},
        "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer Cyclical"},
        "TSLA": {"name": "Tesla, Inc.", "sector": "Automotive"},
        "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Financial Services"},
        "WMT": {"name": "Walmart Inc.", "sector": "Consumer Defensive"},
        "NVDA": {"name": "NVIDIA Corporation", "sector": "Technology"},
        "DIS": {"name": "The Walt Disney Company", "sector": "Entertainment"},
        "NFLX": {"name": "Netflix, Inc.", "sector": "Communication Services"}
    }
    return company_data

# Cache stock data download
@st.cache_data
def load_stock_data(symbol, period="1y"):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    return hist

# Fetch news for a company
@st.cache_data
def fetch_news(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if data['status'] == 'ok':
            return data['articles'][:5]  # Return top 5 articles
        else:
            return []
    except:
        return []

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate MACD
def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

# Sidebar menu
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-share.png", width=80)
    st.title("MarketMentor")
    
    # Navigation menu
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Company Overview", "Stock Prediction", "Market Analysis", "News Sentiment"],
        icons=["house", "building", "graph-up", "bar-chart", "newspaper"],
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#0E1117"},
            "icon": {"color": "#FF4B4B", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "color": "#FAFAFA",
                "border-radius": "5px",
                "padding": "10px",
                "--hover-color": "#262730"
            },
            "nav-link-selected": {"background-color": "#FF4B4B", "color": "#0E1117", "font-weight": "bold"},
        }
    )

# Load company data
company_data = load_company_data()

# Home - Market Overview
if selected == "Home":
    st.title("🏠 Home - Market Overview")
    
    # Display market indices
    st.subheader("Major Market Indices")
    indices = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones Industrial Average',
        '^IXIC': 'NASDAQ Composite',
        '^RUT': 'Russell 2000'
    }
    
    cols = st.columns(len(indices))
    index_data = {}
    
    for i, (symbol, name) in enumerate(indices.items()):
        data = load_stock_data(symbol, period="1d")
        if not data.empty:
            change = ((data['Close'].iloc[-1] - data['Open'].iloc[-1]) / data['Open'].iloc[-1]) * 100
            index_data[symbol] = {
                'name': name,
                'price': data['Close'].iloc[-1],
                'change': change
            }
            
            with cols[i]:
                st.metric(
                    label=name,
                    value=f"${data['Close'].iloc[-1]:.2f}",
                    delta=f"{change:.2f}%"
                )
    
    # Display popular stocks
    st.subheader("Popular Stocks")
    popular_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    popular_data = {}
    
    for symbol in popular_symbols:
        data = load_stock_data(symbol, period="1d")
        if not data.empty:
            change = ((data['Close'].iloc[-1] - data['Open'].iloc[-1]) / data['Open'].iloc[-1]) * 100
            popular_data[symbol] = {
                'name': company_data[symbol]['name'],
                'price': data['Close'].iloc[-1],
                'change': change
            }
    
    cols = st.columns(len(popular_symbols))
    for i, symbol in enumerate(popular_symbols):
        if symbol in popular_data:
            with cols[i]:
                st.metric(
                    label=company_data[symbol]['name'],
                    value=f"${popular_data[symbol]['price']:.2f}",
                    delta=f"{popular_data[symbol]['change']:.2f}%"
                )

# Company Overview - Detailed stock analysis
elif selected == "Company Overview":
    st.title("🏢 Company Overview")
    
    # Company selection
    company_symbols = list(company_data.keys())
    company_names = [f"{symbol} - {company_data[symbol]['name']}" for symbol in company_symbols]
    selected_company = st.selectbox("Select a company:", company_names)
    symbol = selected_company.split(" - ")[0]
    
    if symbol:
        company = company_data[symbol]
        st.markdown(f"<div class='company-header'><h1>{company['name']} ({symbol})</h1></div>", unsafe_allow_html=True)
        
        # Display stock data
        period = st.selectbox("Select period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        data = load_stock_data(symbol, period=period)
        
        if not data.empty:
            # Price chart
            st.subheader("Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ))
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#FAFAFA',
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators
            st.subheader("Technical Indicators")
            
            # RSI
            data['RSI'] = calculate_rsi(data)
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='#FF4B4B')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#FAFAFA',
                showlegend=True
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # MACD
            macd, signal, histogram = calculate_macd(data)
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=data.index, y=macd, name='MACD', line=dict(color='#FF4B4B')))
            fig_macd.add_trace(go.Scatter(x=data.index, y=signal, name='Signal', line=dict(color='#00C853')))
            fig_macd.add_trace(go.Bar(x=data.index, y=histogram, name='Histogram', marker_color='#FFD700'))
            fig_macd.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#FAFAFA',
                showlegend=True
            )
            st.plotly_chart(fig_macd, use_container_width=True)
            
            # Company news
            st.subheader("Latest News")
            news = fetch_news(symbol)
            
            if news:
                for article in news:
                    with st.container():
                        st.markdown(f"**{article['title']}**")
                        st.caption(f"Published: {article['publishedAt'][:10]}")
                        st.write(article['description'])
                        st.markdown(f"[Read more]({article['url']})")
                        st.divider()
            else:
                st.info("No news articles found for this company.")
        else:
            st.error("Failed to load stock data. Please try again later.")

# Placeholder for other menu options
elif selected == "Stock Prediction":
    st.title("📈 Stock Prediction")
    st.info("This feature is under development. Check back soon!")

elif selected == "Market Analysis":
    st.title("📊 Market Analysis")
    st.info("This feature is under development. Check back soon!")

elif selected == "News Sentiment":
    st.title("📰 News Sentiment Analysis")
    st.info("This feature is under development. Check back soon!")
