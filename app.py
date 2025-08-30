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

# Sidebar menu
with st.sidebar:
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
            "icon": {"color": #FF4B4B", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#FAFAFA"},
            "nav-link-selected": {"background-color": "#FF4B4B", "color": "#0E1117"},
        }
    )
    
    st.markdown("---")
    st.subheader("Quick Stock Lookup")
    quick_ticker = st.text_input("Enter Symbol", value="AAPL").upper()
    
    if st.button("Get Quote"):
        selected = "Company Overview"
        st.session_state.selected_ticker = quick_ticker
    
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
    
    # Latest news
    st.subheader("📰 Latest Financial News")
    news_articles = get_news(num_articles=3)
    
    for article in news_articles:
        st.markdown(f"""
        <div class="stock-card">
            <h4>{article['title']}</h4>
            <p><strong>Source:</strong> {article['source']['name']} | <strong>Published:</strong> {article['publishedAt'][:10]}</p>
            <a href="{article['url']}" target="_blank" style="color: #FF4B4B; text-decoration: none;">Read more →</a>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features overview
    st.subheader("🚀 Platform Features")
    
    features = [
        {"title": "Company Analysis", "description": "Detailed financials, technical indicators, and performance metrics for any stock.", "icon": "📊"},
        {"title": "Predictive Analytics", "description": "AI-powered price predictions and buy/sell signals using machine learning models.", "icon": "🤖"},
        {"title": "Market Overview", "description": "Real-time data on indices, sectors, and global markets with interactive charts.", "icon": "🌍"},
        {"title": "News & Sentiment", "description": "Latest financial news with sentiment analysis to gauge market mood.", "icon": "📰"},
        {"title": "Screening Tools", "description": "Advanced stock screening based on technical and fundamental criteria.", "icon": "🔍"},
        {"title": "Learning Resources", "description": "Educational content to improve your trading and investment knowledge.", "icon": "🎓"}
    ]
    
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{feature['icon']} {feature['title']}</h3>
                <p>{feature['description']}</p>
            </div>
            """, unsafe_allow_html=True)

# Company Overview - Detailed stock analysis
elif selected == "Company Overview":
    st.title("🏢 Company Overview")
    
    # Ticker input
    col1, col2 = st.columns([2, 1])
    with col1:
        if 'selected_ticker' in st.session_state:
            default_ticker = st.session_state.selected_ticker
        else:
            default_ticker = "AAPL"
        
        ticker = st.text_input("Enter Stock Symbol", value=default_ticker).upper()
    
    with col2:
        period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    
    if ticker:
        try:
            # Get stock data
            data, stock = get_stock_data(ticker, period)
            
            if data.empty:
                st.error(f"No data found for {ticker}")
            else:
                # Get company info
                info = get_company_info(ticker)
                
                # Calculate additional metrics
                current_price = data['Close'][-1]
                prev_close = data['Close'][-2] if len(data) > 1 else current_price
                price_change = current_price - prev_close
                percent_change = (price_change / prev_change) * 100
                
                # Display company header
                company_name = info.get('longName', ticker)
                st.markdown(f"<h2>{company_name} ({ticker})</h2>", unsafe_allow_html=True)
                
                # Price and change display
                change_color = "positive-change" if price_change >= 0 else "negative-change"
                change_icon = "📈" if price_change >= 0 else "📉"
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Change", f"${price_change:.2f}", f"{percent_change:.2f}%")
                with col3:
                    st.markdown(f"<h3 class='{change_color}' style='margin-top: 30px;'>{change_icon} {percent_change:.2f}%</h3>", unsafe_allow_html=True)
                
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
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Company information
                st.subheader("Company Information")
                
                # Get company data from our JSON or Yahoo Finance
                company_info = company_data.get(ticker, {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="stock-card">
                        <h3>Company Details</h3>
                    """, unsafe_allow_html=True)
                    
                    details = [
                        ("Sector", info.get('sector', company_info.get('sector', 'N/A'))),
                        ("Industry", info.get('industry', company_info.get('industry', 'N/A'))),
                        ("CEO", info.get('ceo', company_info.get('ceo', 'N/A'))),
                        ("Employees", f"{info.get('fullTimeEmployees', company_info.get('employees', 'N/A')):,}"),
                        ("Headquarters", info.get('city', '') + ", " + info.get('state', '') 
                         if info.get('city') else company_info.get('headquarters', 'N/A'))
                    ]
                    
                    for label, value in details:
                        st.markdown(f"<p><strong>{label}:</strong> {value}</p>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="stock-card">
                        <h3>Financial Metrics</h3>
                    """, unsafe_allow_html=True)
                    
                    metrics = [
                        ("Market Cap", f"${info.get('marketCap', 0):,}" if info.get('marketCap') else "N/A"),
                        ("P/E Ratio", info.get('trailingPE', 'N/A')),
                        ("EPS", info.get('trailingEps', 'N/A')),
                        ("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A"),
                        ("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}"),
                        ("52 Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
                    ]
                    
                    for label, value in metrics:
                        st.markdown(f"<p><strong>{label}:</strong> {value}</p>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Technical indicators
                st.subheader("Technical Indicators")
                tech_data = calculate_technical_indicators(data)
                
                # Display current values of key indicators
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("RSI", f"{tech_data['RSI'].iloc[-1]:.2f}")
                
                with col2:
                    st.metric("MACD", f"{tech_data['MACD'].iloc[-1]:.2f}")
                
                with col3:
                    st.metric("20-Day SMA", f"{tech_data['SMA_20'].iloc[-1]:.2f}")
                
                with col4:
                    st.metric("50-Day SMA", f"{tech_data['SMA_50'].iloc[-1]:.2f}")
                
                # Plot technical indicators
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Technical Indicators', color='white')
                fig.patch.set_facecolor('#0E1117')
                
                # RSI
                axes[0, 0].plot(tech_data.index, tech_data['RSI'], color='#FF4B4B')
                axes[0, 0].axhline(70, color='red', linestyle='--', alpha=0.3)
                axes[0, 0].axhline(30, color='green', linestyle='--', alpha=0.3)
                axes[0, 0].set_title('RSI', color='white')
                axes[0, 0].set_facecolor('#262730')
                axes[0, 0].tick_params(colors='white')
                
                # MACD
                axes[0, 1].plot(tech_data.index, tech_data['MACD'], label='MACD', color='#FF4B4B')
                axes[0, 1].plot(tech_data.index, tech_data['MACD_signal'], label='Signal', color='#00C853')
                axes[0, 1].bar(tech_data.index, tech_data['MACD_hist'], label='Histogram', alpha=0.3)
                axes[0, 1].set_title('MACD', color='white')
                axes[0, 1].legend()
                axes[0, 1].set_facecolor('#262730')
                axes[0, 1].tick_params(colors='white')
                
                # Moving Averages
                axes[1, 0].plot(tech_data.index, tech_data['Close'], label='Price', color='white', alpha=0.7)
                axes[1, 0].plot(tech_data.index, tech_data['SMA_20'], label='20-Day SMA', color='#FF4B4B')
                axes[1, 0].plot(tech_data.index, tech_data['SMA_50'], label='50-Day SMA', color='#00C853')
                axes[1, 0].set_title('Moving Averages', color='white')
                axes[1, 0].legend()
                axes[1, 0].set_facecolor('#262730')
                axes[1, 0].tick_params(colors='white')
                
                # Bollinger Bands
                axes[1, 1].plot(tech_data.index, tech_data['Close'], label='Price', color='white', alpha=0.7)
                axes[1, 1].plot(tech_data.index, tech_data['BB_mid'], label='Middle Band', color='#FF4B4B')
                axes[1, 1].plot(tech_data.index, tech_data['BB_high'], label='Upper Band', color='#00C853', alpha=0.7)
                axes[1, 1].plot(tech_data.index, tech_data['BB_low'], label='Lower Band', color='#FF4B4B', alpha=0.7)
                axes[1, 1].fill_between(tech_data.index, tech_data['BB_high'], tech_data['BB_low'], alpha=0.1)
                axes[1, 1].set_title('Bollinger Bands', color='white')
                axes[1, 1].legend()
                axes[1, 1].set_facecolor('#262730')
                axes[1, 1].tick_params(colors='white')
                
                plt.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")

# Other menu options would be implemented here
else:
    st.title(f"{selected} Section")
    st.info("This section is under development. Check back soon for updates!")

# Note: The other menu options would need to be implemented similarly
# For now, they just display a placeholder message
