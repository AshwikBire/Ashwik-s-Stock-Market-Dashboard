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
import ta
from newsapi import NewsApiClient
import investpy
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
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            width: 100%;
        }
        
        .stSidebar .stButton>button {
            padding: 0.4rem 0.8rem;
            font-size: 14px;
        }
        
        .stSidebar .stSelectbox>div>div>select {
            font-size: 14px;
        }
        
        .stSidebar .stTextInput>div>div>input {
            font-size: 14px;
        }
        
        .css-1d391kg {
            padding: 1rem 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            font-size: 12px;
            padding: 8px 12px;
        }
        
        .stock-card {
            padding: 10px;
        }
        
        .feature-card {
            padding: 15px;
        }
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

# Load company data
company_data = load_company_data()

# Update selected from session state
selected = st.session_state.selected

# Home - Market Overview
if selected == "Home":
    st.title("MarketMentor - Stock Analysis Platform")
    
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
    st.subheader("Today's Market Movers")
    
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

# Company Overview Page
elif selected == "Company Overview":
    st.title("Company Overview")
    
    # Get ticker from session state or default to AAPL
    ticker = st.session_state.get('selected_ticker', 'AAPL')
    
    # Fetch data
    data, stock = get_stock_data(ticker)
    info = get_company_info(ticker)
    live_price = get_live_price(ticker)
    
    # Display company info
    if info:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"{info.get('longName', ticker)} ({ticker})")
            if live_price:
                prev_close = info.get('regularMarketPreviousClose', 0)
                change = live_price - prev_close
                change_percent = (change / prev_close) * 100 if prev_close else 0
                change_color = "positive-change" if change >= 0 else "negative-change"
                
                st.metric("Current Price", f"${live_price:.2f}", 
                         f"{change:.2f} ({change_percent:.2f}%)")
            
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Market Cap:** ${info.get('marketCap', 0):,}")
            st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
            st.write(f"**52W High:** ${info.get('fiftyTwoWeekHigh', 'N/A')}")
            st.write(f"**52W Low:** ${info.get('fiftyTwoWeekLow', 'N/A')}")
        
        with col2:
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name='Price'))
            fig.update_layout(title=f"{ticker} Price Chart",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            template="plotly_dark",
                            height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators
    st.subheader("Technical Indicators")
    if not data.empty:
        data_with_indicators = calculate_technical_indicators(data)
        
        # Select indicator to display
        indicator_option = st.selectbox("Select Indicator", 
                                      ["RSI", "MACD", "Moving Averages", "Bollinger Bands"])
        
        if indicator_option == "RSI":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['RSI'], mode='lines', name='RSI'))
            fig.add_hline(y=70, line_dash="dash", line_color="red")
            fig.add_hline(y=30, line_dash="dash", line_color="green")
            fig.update_layout(title="RSI (14 days)",
                            xaxis_title="Date",
                            yaxis_title="RSI",
                            template="plotly_dark",
                            height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        elif indicator_option == "MACD":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['MACD'], mode='lines', name='MACD'))
            fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['MACD_signal'], mode='lines', name='Signal'))
            fig.update_layout(title="MACD",
                            xaxis_title="Date",
                            yaxis_title="Value",
                            template="plotly_dark",
                            height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        elif indicator_option == "Moving Averages":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'))
            fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['SMA_20'], mode='lines', name='SMA 20'))
            fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['SMA_50'], mode='lines', name='SMA 50'))
            fig.update_layout(title="Moving Averages",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            template="plotly_dark",
                            height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        elif indicator_option == "Bollinger Bands":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'))
            fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['BB_high'], mode='lines', name='Upper Band', line={'dash': 'dash'}))
            fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['BB_mid'], mode='lines', name='Middle Band', line={'dash': 'dash'}))
            fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['BB_low'], mode='lines', name='Lower Band', line={'dash': 'dash'}))
            fig.update_layout(title="Bollinger Bands",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            template="plotly_dark",
                            height=400)
            st.plotly_chart(fig, use_container_width=True)

# Other pages would be implemented similarly...

# For demonstration, I'll show a basic implementation of a few more pages
elif selected == "News":
    st.title("Financial News")
    articles = get_news()
    
    for article in articles:
        st.markdown(f"""
        <div class="feature-card">
            <h3>{article['title']}</h3>
            <p><strong>Source:</strong> {article['source']['name']}</p>
            <p><strong>Published:</strong> {article['publishedAt'][:10]}</p>
            <a href="{article['url']}" target="_blank">Read more</a>
        </div>
        """, unsafe_allow_html=True)

elif selected == "Predictions":
    st.title("Stock Price Predictions")
    
    ticker = st.session_state.get('selected_ticker', 'AAPL')
    st.write(f"Predicting future prices for {ticker}")
    
    # Get data
    data, stock = get_stock_data(ticker, "2y")
    
    if not data.empty:
        # Prepare data for LSTM
        dataset = data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        # Create training data
        training_data_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[0:training_data_len, :]
        
        # Create the training dataset
        x_train, y_train = [], []
        time_step = 60
        
        for i in range(time_step, len(train_data)):
            x_train.append(train_data[i-time_step:i, 0])
            y_train.append(train_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Create and fit the LSTM model
        model = create_lstm_model((x_train.shape[1], 1))
        
        with st.spinner('Training prediction model...'):
            # Just for demonstration, we'll use a small number of epochs
            model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)
        
        # Create testing dataset
        test_data = scaled_data[training_data_len - time_step:, :]
        x_test = []
        
        for i in range(time_step, len(test_data)):
            x_test.append(test_data[i-time_step:i, 0])
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        # Get predictions
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        
        # Plot the results
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Training Data'))
        fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Actual Price'))
        fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predicted Price'))
        fig.update_layout(title=f"{ticker} Price Prediction",
                         xaxis_title="Date",
                         yaxis_title="Price ($)",
                         template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction for next day
        last_60_days = dataset[-60:]
        last_60_days_scaled = scaler.transform(last_60_days)
        
        X_test = np.array([last_60_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        
        st.metric("Predicted Price for Next Day", f"${pred_price[0][0]:.2f}")

# Additional pages would follow similar patterns...

else:
    st.title(selected)
    st.info("This section is under development. Check back soon for updates!")
