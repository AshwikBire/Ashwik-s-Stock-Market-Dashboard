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
import time
from datetime import date
import ta  # Technical analysis library
from newsapi import NewsApiClient
import investpy  # For global market data
import base64

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

# Apply custom CSS for dark theme with red accents and mobile responsiveness
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
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #FF3333;
        color: white;
    }
    .stSelectbox>div>div>select {
        background-color: #262730;
        color: white;
        border: 1px solid #4A4A4A;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
        border: 1px solid #4A4A4A;
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
        border: 1px solid #4A4A4A;
        border-radius: 4px;
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
        color: white;
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
    .stock-card {
        background-color: #262730;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stock-card h3 {
        margin-top: 0;
        color: #FF4B4B;
    }
    .positive-change {
        color: #00C853;
    }
    .negative-change {
        color: #FF4B4B;
    }
    .section-header {
        border-bottom: 2px solid #FF4B4B;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        color: #FAFAFA;
    }
    .feature-card {
        background-color: #262730;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-card h3 {
        color: #FF4B4B;
        margin-top: 0;
    }
    
    /* Mobile Navigation */
    .mobile-nav {
        display: none;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #262730;
        z-index: 1000;
        padding: 10px;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .mobile-nav-items {
        display: flex;
        justify-content: space-around;
        align-items: center;
    }
    
    .mobile-nav-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        color: #FAFAFA;
        text-decoration: none;
        font-size: 12px;
        padding: 5px;
    }
    
    .mobile-nav-item.active {
        color: #FF4B4B;
    }
    
    .mobile-nav-icon {
        font-size: 20px;
        margin-bottom: 4px;
    }
    
    /* Mobile Header */
    .mobile-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background-color: #0E1117;
        border-bottom: 1px solid #262730;
        position: sticky;
        top: 0;
        z-index: 999;
    }
    
    .mobile-menu-button {
        background: none;
        border: none;
        color: #FF4B4B;
        font-size: 24px;
        cursor: pointer;
    }
    
    .mobile-menu-content {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: #0E1117;
        z-index: 1001;
        padding: 20px;
        overflow-y: auto;
    }
    
    .mobile-menu-close {
        position: absolute;
        top: 20px;
        right: 20px;
        background: none;
        border: none;
        color: #FF4B4B;
        font-size: 24px;
        cursor: pointer;
    }
    
    .mobile-menu-item {
        display: block;
        padding: 15px 0;
        color: #FAFAFA;
        text-decoration: none;
        font-size: 18px;
        border-bottom: 1px solid #262730;
    }
    
    .mobile-menu-item.active {
        color: #FF4B4B;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main > div {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .stSidebar {
            display: none !important;
        }
        
        .mobile-nav {
            display: block;
        }
        
        .mobile-header {
            display: flex;
        }
        
        .desktop-only {
            display: none;
        }
    }
    
    @media (min-width: 769px) {
        .mobile-nav {
            display: none;
        }
        
        .mobile-header {
            display: none;
        }
        
        .mobile-menu-content {
            display: none !important;
        }
    }
    
    /* Desktop styles */
    .desktop-sidebar {
        display: block;
    }
</style>
""", unsafe_allow_html=True)

# Load company data from JSON file
@st.cache_data
def load_company_data():
    company_data = {
        "AAPL": {
            "name": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "founded": 1976,
            "ceo": "Tim Cook",
            "employees": 154000,
            "headquarters": "Cupertino, California",
            "website": "https://www.apple.com",
            "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
            "products": ["iPhone", "Mac", "iPad", "Apple Watch", "AirPods", "Services"],
            "competitors": ["Samsung", "Google", "Microsoft", "HP", "Dell"]
        },
        "MSFT": {
            "name": "Microsoft Corporation",
            "sector": "Technology",
            "industry": "Software—Infrastructure",
            "founded": 1975,
            "ceo": "Satya Nadella",
            "employees": 181000,
            "headquarters": "Redmond, Washington",
            "website": "https://www.microsoft.com",
            "description": "Microsoft Corporation develops, licenses, and supports software, services, devices, and solutions worldwide.",
            "products": ["Windows", "Office", "Azure", "LinkedIn", "Xbox"],
            "competitors": ["Apple", "Google", "Amazon", "Oracle", "Salesforce"]
        },
        "GOOGL": {
            "name": "Alphabet Inc.",
            "sector": "Communication Services",
            "industry": "Internet Content & Information",
            "founded": 1998,
            "ceo": "Sundar Pichai",
            "employees": 144056,
            "headquarters": "Mountain View, California",
            "website": "https://www.abc.xyz",
            "description": "Alphabet Inc. provides online advertising services, cloud computing, software, and hardware.",
            "products": ["Google Search", "YouTube", "Google Cloud", "Android", "Google Ads"],
            "competitors": ["Microsoft", "Apple", "Amazon", "Facebook", "Twitter"]
        },
        "AMZN": {
            "name": "Amazon.com Inc.",
            "sector": "Consumer Cyclical",
            "industry": "Internet Retail",
            "founded": 1994,
            "ceo": "Andy Jassy",
            "employees": 1298000,
            "headquarters": "Seattle, Washington",
            "website": "https://www.amazon.com",
            "description": "Amazon.com Inc. engages in the retail sale of consumer products and subscriptions through online and physical stores.",
            "products": ["Amazon.com", "AWS", "Prime Video", "Alexa", "Kindle"],
            "competitors": ["Walmart", "Alibaba", "eBay", "Target", "Google"]
        },
        "TSLA": {
            "name": "Tesla, Inc.",
            "sector": "Automotive",
            "industry": "Auto Manufacturers",
            "founded": 2003,
            "ceo": "Elon Musk",
            "employees": 99290,
            "headquarters": "Austin, Texas",
            "website": "https://www.tesla.com",
            "description": "Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, energy generation and storage systems.",
            "products": ["Model S", "Model 3", "Model X", "Model Y", "Solar Roof", "Powerwall"],
            "competitors": ["Ford", "General Motors", "Toyota", "NIO", "Lucid Motors"]
        }
    }
    return company_data

# Function to fetch stock data
@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist, stock

# Function to get company info
@st.cache_data(ttl=3600)
def get_company_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info

# Function to get live price
@st.cache_data(ttl=60)
def get_live_price(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d", interval="1m")
    if not data.empty:
        return data['Close'][-1]
    return None

# Function to get financial news
@st.cache_data(ttl=3600)
def get_news(query="stock market", language="en", num_articles=10):
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    try:
        news = newsapi.get_everything(q=query,
                                     language=language,
                                     sort_by='relevancy',
                                     page_size=num_articles)
        return news['articles']
    except:
        # Fallback mock data if API fails
        return [
            {"title": "Stock Market Hits Record High", "url": "#", "source": {"name": "Financial Times"}, "publishedAt": "2023-05-01T10:00:00Z"},
            {"title": "Tech Stocks Rally Amid Positive Earnings", "url": "#", "source": {"name": "Bloomberg"}, "publishedAt": "2023-05-01T09:30:00Z"},
            {"title": "Federal Reserve Holds Interest Rates Steady", "url": "#", "source": {"name": "Reuters"}, "publishedAt": "2023-05-01T08:15:00Z"}
        ]

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    df = data.copy()
    
    # Moving averages
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_high'] = bollinger.bollinger_hband()
    df['BB_mid'] = bollinger.bollinger_mavg()
    df['BB_low'] = bollinger.bollinger_lband()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_%K'] = stoch.stoch()
    df['Stoch_%D'] = stoch.stoch_signal()
    
    return df

# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to prepare data for LSTM
def prepare_lstm_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Initialize session state for navigation
if 'selected' not in st.session_state:
    st.session_state.selected = "Home"

if 'mobile_menu_open' not in st.session_state:
    st.session_state.mobile_menu_open = False

# Mobile header
st.markdown("""
<div class="mobile-header">
    <h2>MarketMentor</h2>
    <button class="mobile-menu-button" onclick="toggleMobileMenu()">☰</button>
</div>
""", unsafe_allow_html=True)

# Mobile menu
st.markdown(f"""
<div class="mobile-menu-content" id="mobileMenu" style="display: {'block' if st.session_state.mobile_menu_open else 'none'};">
    <button class="mobile-menu-close" onclick="toggleMobileMenu()">×</button>
    <h2 style="color: #FF4B4B; margin-bottom: 20px;">Navigation</h2>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'Home' else ''}" onclick="setPage('Home')">🏠 Home</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'Company Overview' else ''}" onclick="setPage('Company Overview')">🏢 Company Overview</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'Market Movers' else ''}" onclick="setPage('Market Movers')">📈 Market Movers</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'Global Markets' else ''}" onclick="setPage('Global Markets')">🌍 Global Markets</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'Mutual Funds' else ''}" onclick="setPage('Mutual Funds')">💰 Mutual Funds</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'Sectors' else ''}" onclick="setPage('Sectors')">📊 Sectors</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'News' else ''}" onclick="setPage('News')">📰 News</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'Learning' else ''}" onclick="setPage('Learning')">🎓 Learning</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'Volume Spike' else ''}" onclick="setPage('Volume Spike')">📊 Volume Spike</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'News Sentiment' else ''}" onclick="setPage('News Sentiment')">😊 News Sentiment</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'Predictions' else ''}" onclick="setPage('Predictions')">🔮 Predictions</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'Buy/Sell Predictor' else ''}" onclick="setPage('Buy/Sell Predictor')">↕️ Buy/Sell Predictor</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'Stock Screener' else ''}" onclick="setPage('Stock Screener')">🔍 Stock Screener</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'F&O' else ''}" onclick="setPage('F&O')">📊 F&O</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'SIP Calculator' else ''}" onclick="setPage('SIP Calculator')">🧮 SIP Calculator</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'IPO Tracker' else ''}" onclick="setPage('IPO Tracker')">📈 IPO Tracker</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'Watchlist' else ''}" onclick="setPage('Watchlist')">⭐ Watchlist</a>
    <a href="#" class="mobile-menu-item {'active' if st.session_state.selected == 'Options Chain' else ''}" onclick="setPage('Options Chain')">⛓️ Options Chain</a>
</div>
""", unsafe_allow_html=True)

# Mobile bottom navigation
st.markdown("""
<div class="mobile-nav">
    <div class="mobile-nav-items">
        <a href="#" class="mobile-nav-item" onclick="setPage('Home')">
            <div class="mobile-nav-icon">🏠</div>
            <span>Home</span>
        </a>
        <a href="#" class="mobile-nav-item" onclick="setPage('Market Movers')">
            <div class="mobile-nav-icon">📈</div>
            <span>Movers</span>
        </a>
        <a href="#" class="mobile-nav-item" onclick="setPage('News')">
            <div class="mobile-nav-icon">📰</div>
            <span>News</span>
        </a>
        <a href="#" class="mobile-nav-item" onclick="setPage('Predictions')">
            <div class="mobile-nav-icon">🔮</div>
            <span>Predict</span>
        </a>
        <a href="#" class="mobile-nav-item" onclick="toggleMobileMenu()">
            <div class="mobile-nav-icon">☰</div>
            <span>More</span>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# JavaScript for mobile navigation
st.markdown("""
<script>
function setPage(page) {
    window.parent.document.querySelector('iframe').contentWindow.setPage(page);
    toggleMobileMenu();
}

function toggleMobileMenu() {
    const menu = document.getElementById('mobileMenu');
    if (menu.style.display === 'block') {
        menu.style.display = 'none';
    } else {
        menu.style.display = 'block';
    }
}

// Set iframe communication
window.setPage = function(page) {
    const event = new CustomEvent('setPage', { detail: page });
    window.dispatchEvent(event);
}

window.addEventListener('setPage', function(e) {
    const page = e.detail;
    // This will be handled by Streamlit
    const data = {type: 'setPage', page: page};
    window.parent.postMessage(data, '*');
});

// Listen for messages from parent
window.addEventListener('message', function(event) {
    if (event.data.type === 'setPage') {
        setPage(event.data.page);
    }
});
</script>
""", unsafe_allow_html=True)

# Sidebar menu for desktop
with st.sidebar:
    st.markdown('<div class="desktop-sidebar">', unsafe_allow_html=True)
    st.image("https://img.icons8.com/fluency/96/stock-share.png", width=80)
    st.title("MarketMentor")
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "Company Overview", "Market Movers", "Global Markets", 
                "Mutual Funds", "Sectors", "News", "Learning", "Volume Spike", 
                "News Sentiment", "Predictions", "Buy/Sell Predictor", "Stock Screener",
                "F&O", "SIP Calculator", "IPO Tracker", "Watchlist", "Options Chain"],
        icons=["house", "building", "graph-up", "globe", "piggy-bank", "pie-chart", 
               "newspaper", "book", "activity", "bar-chart", "lightbulb", "arrow-left-right", 
               "search", "graph-up-arrow", "calculator", "clipboard-data", "list-check", "link"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#262730"},
            "icon": {"color": "#FF4B4B", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#FAFAFA"},
            "nav-link-selected": {"background-color": "#FF4B4B", "color": "#0E1117"},
        }
    )
    
    st.session_state.selected = selected
    
    st.markdown("---")
    st.subheader("Quick Stock Lookup")
    quick_ticker = st.text_input("Enter Symbol", value="AAPL").upper()
    
    if st.button("Get Quote"):
        st.session_state.selected_ticker = quick_ticker
        st.session_state.selected = "Company Overview"
        st.rerun()
    
    st.markdown("---")
    st.subheader("Market Status")
    
    # Mock market status data
    markets = {
        "S&P 500": {"price": 4567.25, "change": 0.78},
        "NASDAQ": {"price": 14346.00, "change": 1.23},
        "DOW": {"price": 35443.82, "change": -0.15},
        "RUSSELL 2000": {"price": 1925.42, "change": 0.42}
    }
    
    for market, data in markets.items():
        change_color = "positive-change" if data["change"] >= 0 else "negative-change"
        st.markdown(f"""
        <div class="stock-card">
            <h4>{market}</h4>
            <p>Price: ${data['price']}</p>
            <p class="{change_color}">Change: {data['change']}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>MarketMentor v2.0</p>
        <p>Data provided by Yahoo Finance</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Handle page navigation from JavaScript
if 'selected' in st.session_state:
    selected = st.session_state.selected

# Load company data
company_data = load_company_data()

# Home - Market Overview
if selected == "Home":
    st.title("🏠 MarketMentor - Stock Analysis Platform")
    st.markdown("""
    <div style='background-color: #262730; padding: 20px; border-radius: 8px; margin-bottom: 20px;'>
        <h3 style='color: #FF4B4B; margin-top: 0;'>Your Comprehensive Stock Market Analysis Tool</h3>
        <p>Access real-time market data, advanced analytics, and predictive insights all in one place.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Market overview columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("S&P 500", "4,567.25", "0.78%")
    
    with col2:
        st.metric("NASDAQ", "14,346.00", "1.23%")
    
    with col3:
        st.metric("DOW", "35,443.82", "-0.15%")
    
    with col4:
        st.metric("VIX", "17.25", "-2.16%")
    
    st.markdown("---")
    
    # Top gainers and losers
    st.subheader("📈 Today's Market Movers")
    
    # Mock data for gainers and losers
    gainers = [
        {"symbol": "AAPL", "name": "Apple Inc.", "price": 175.43, "change": 3.2},
        {"symbol": "MSFT", "name": "Microsoft Corp.", "price": 337.69, "change": 2.8},
        {"symbol": "NVDA", "name": "NVIDIA Corp.", "price": 455.72, "change": 5.4}
    ]
    
    losers = [
        {"symbol": "TSLA", "name": "Tesla Inc.", "price": 230.15, "change": -2.3},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "price": 131.27, "change": -1.7},
        {"symbol": "DIS", "name": "Walt Disney Co.", "price": 87.65, "change": -1.2}
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='color: #00C853;'>Top Gainers</h4>", unsafe_allow_html=True)
        for stock in gainers:
            st.markdown(f"""
            <div class="stock-card">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <strong>{stock['symbol']}</strong> - {stock['name']}
                    </div>
                    <div class="positive-change">
                        +{stock['change']}%
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                    <div>Price:</div>
                    <div>${stock['price']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h4 style='color: #FF4B4B;'>Top Losers</h4>", unsafe_allow_html=True)
        for stock in losers:
            st.markdown(f"""
            <div class="stock-card">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <strong>{stock['symbol']}</strong> - {stock['name']}
                    </div>
                    <div class="negative-change">
                        {stock['change']}%
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                    <div>Price:</div>
                    <div>${stock['price']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Latest news section
    st.subheader("📰 Latest Market News")
    news_articles = get_news(num_articles=3)
    
    for article in news_articles:
        st.markdown(f"""
        <div class="stock-card">
            <h4>{article['title']}</h4>
            <p><em>Source: {article['source']['name']}</em></p>
            <p>{article['publishedAt'][:10]}</p>
            <a href="{article['url']}" target="_blank" style="color: #FF4B4B;">Read more →</a>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features overview
    st.subheader("🚀 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Real-time Data</h3>
            <p>Access live stock prices, market indices, and financial news updated in real-time.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Advanced Analytics</h3>
            <p>Technical indicators, charting tools, and predictive analytics for informed decisions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🔮 AI Predictions</h3>
            <p>Machine learning models for price predictions and buy/sell recommendations.</p>
        </div>
        """, unsafe_allow_html=True)

# Other page content would follow here...

# Add JavaScript communication handler
components.html(
    """
    <script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'setPage') {
            // Send message to Streamlit
            const data = {type: 'setPage', page: event.data.page};
            window.parent.postMessage(data, '*');
        }
    });
    
    // Function to set page from JavaScript
    window.setPage = function(page) {
        const data = {type: 'setPage', page: page};
        window.parent.postMessage(data, '*');
    }
    </script>
    """,
    height=0
)

# Handle page navigation from JavaScript
try:
    if 'setPage' in st.session_state:
        st.session_state.selected = st.session_state.setPage
except:
    pass
