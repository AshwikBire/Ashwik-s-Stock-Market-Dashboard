import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from plotly import graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
import warnings
import time
import pytz
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
import fundamentalanalysis as fa
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import json
import os
from pathlib import Path
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set page config for optimal performance
st.set_page_config(
    page_title="MarketMentor Pro - Advanced Financial Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply enhanced black and red theme
st.markdown("""
<style>
    /* Main background */
    .main, .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Sidebar */
    .stSidebar {
        background-color: #0A0A0A;
        border-right: 1px solid #2A0A0A;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        border-bottom: 1px solid #2A0A0A;
        padding-bottom: 8px;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1A0A0A;
        color: #FF0000;
        border: 1px solid #FF0000;
        border-radius: 4px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #FF0000;
        color: #000000;
        border: 1px solid #FF0000;
    }
    
    /* Input fields */
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #1A0A0A;
        color: #FF0000;
        border: 1px solid #2A0A0A;
        border-radius: 4px;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #0A0A0A;
        border-radius: 5px;
        padding: 10px;
        border-left: 3px solid #FF0000;
        box-shadow: 0 2px 4px rgba(255, 0, 0, 0.1);
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #0A0A0A;
        border: 1px solid #2A0A0A;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #0A0A0A;
        border-radius: 4px;
        padding: 8px;
        border: 1px solid #2A0A0A;
        color: #FF0000;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs {
        background-color: #000000;
    }
    
    div[data-baseweb="tab-list"] {
        background-color: #0A0A0A;
        gap: 2px;
        padding: 4px;
        border-radius: 4px;
    }
    
    div[data-baseweb="tab"] {
        background-color: #1A0A0A;
        color: #FFFFFF;
        padding: 10px 20px;
        border-radius: 4px;
        border: 1px solid #2A0A0A;
        transition: all 0.3s ease;
    }
    
    div[data-baseweb="tab"]:hover {
        background-color: #2A0A0A;
        color: #FF0000;
    }
    
    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: #FF0000;
        color: #000000;
        font-weight: 700;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: #FF0000;
    }
    
    /* Alerts */
    .stAlert {
        background-color: #1A0A0A;
        border: 1px solid #2A0A0A;
        border-radius: 4px;
    }
    
    /* Sidebar navigation */
    .css-1d391kg {
        background-color: #0A0A0A;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0A0A0A;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #FF0000;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #CC0000;
    }
    
    /* Custom selectbox dropdown */
    .stSelectbox div[data-baseweb="select"] div {
        background-color: #1A0A0A;
        color: #FF0000;
    }
    
    /* Custom number input */
    .stNumberInput div[data-baseweb="input"] {
        background-color: #1A0A0A;
    }
    
    /* Custom date input */
    .stDateInput div[data-baseweb="input"] {
        background-color: #1A0A0A;
    }
    
    /* Plotly chart customization */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly div {
        background-color: #0A0A0A !important;
    }
    
    /* Custom metric labels */
    .stMetric label {
        color: #FF0000 !important;
        font-weight: 600;
    }
    
    /* Custom success message */
    .stSuccess {
        background-color: #0A2A0A;
        border: 1px solid #00FF00;
    }
    
    /* Custom error message */
    .stError {
        background-color: #2A0A0A;
        border: 1px solid #FF0000;
    }
    
    /* Custom info message */
    .stInfo {
        background-color: #0A1A2A;
        border: 1px solid #0080FF;
    }
    
    /* Custom warning message */
    .stWarning {
        background-color: #2A2A0A;
        border: 1px solid #FFFF00;
    }
    
    /* Custom card styling */
    .card {
        background-color: #0A0A0A;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FF0000;
        box-shadow: 0 4px 6px rgba(255, 0, 0, 0.1);
    }
    
    /* Custom table styling */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    
    .data-table th, .data-table td {
        padding: 10px;
        text-align: left;
        border-bottom: 1px solid #2A0A0A;
    }
    
    .data-table th {
        background-color: #1A0A0A;
        color: #FF0000;
        font-weight: 600;
    }
    
    .data-table tr:hover {
        background-color: #1A1A1A;
    }
    
    /* Custom badge styling */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 0 5px;
    }
    
    .badge-success {
        background-color: #0A2A0A;
        color: #00FF00;
    }
    
    .badge-danger {
        background-color: #2A0A0A;
        color: #FF0000;
    }
    
    .badge-warning {
        background-color: #2A2A0A;
        color: #FFFF00;
    }
    
    .badge-info {
        background-color: #0A1A2A;
        color: #0080FF;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for caching and user data
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = {}
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Quantity', 'Purchase Price', 'Purchase Date', 'Sector', 'Currency'])
if 'user_settings' not in st.session_state:
    st.session_state.user_settings = {
        'theme': 'dark',
        'auto_refresh': True,
        'refresh_interval': 15,
        'default_currency': 'USD',
        'risk_tolerance': 'medium'
    }
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'news_preferences' not in st.session_state:
    st.session_state.news_preferences = ['stock market', 'investing', 'economy']
if 'analyze_symbol' not in st.session_state:
    st.session_state.analyze_symbol = 'AAPL'

# Format currency based on stock type
def format_currency(value, currency="USD"):
    if pd.isna(value):
        return "N/A"
    
    if currency == "INR" or currency == "₹":
        return f"₹{value:,.2f}"
    elif currency == "USD" or currency == "$":
        return f"${value:,.2f}"
    elif currency == "EUR" or currency == "€":
        return f"€{value:,.2f}"
    elif currency == "GBP" or currency == "£":
        return f"£{value:,.2f}"
    elif currency == "JPY" or currency == "¥":
        return f"¥{value:,.0f}"
    else:
        return f"{value:,.2f} {currency}"

# Format large numbers for better readability
def format_number(value):
    if pd.isna(value):
        return "N/A"
    
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value/1_000:.2f}K"
    else:
        return f"{value:,.2f}"

# Optimized data fetching with caching
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker, period="1mo", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        info = stock.info
        return hist, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_global_indices():
    indices = {
        "^GSPC": {"name": "S&P 500", "currency": "USD", "category": "Index"},
        "^DJI": {"name": "Dow Jones", "currency": "USD", "category": "Index"},
        "^IXIC": {"name": "NASDAQ", "currency": "USD", "category": "Index"},
        "^NSEI": {"name": "Nifty 50", "currency": "INR", "category": "Index"},
        "^BSESN": {"name": "Sensex", "currency": "INR", "category": "Index"},
        "^FTSE": {"name": "FTSE 100", "currency": "GBP", "category": "Index"},
        "^GDAXI": {"name": "DAX", "currency": "EUR", "category": "Index"},
        "^FCHI": {"name": "CAC 40", "currency": "EUR", "category": "Index"},
        "^N225": {"name": "Nikkei 225", "currency": "JPY", "category": "Index"},
        "^HSI": {"name": "Hang Seng", "currency": "HKD", "category": "Index"},
        "GC=F": {"name": "Gold", "currency": "USD", "category": "Commodity"},
        "SI=F": {"name": "Silver", "currency": "USD", "category": "Commodity"},
        "PL=F": {"name": "Platinum", "currency": "USD", "category": "Commodity"},
        "CL=F": {"name": "Crude Oil", "currency": "USD", "category": "Commodity"},
        "NG=F": {"name": "Natural Gas", "currency": "USD", "category": "Commodity"},
        "BTC-USD": {"name": "Bitcoin", "currency": "USD", "category": "Crypto"},
        "ETH-USD": {"name": "Ethereum", "currency": "USD", "category": "Crypto"},
        "BNB-USD": {"name": "Binance Coin", "currency": "USD", "category": "Crypto"},
        "ADA-USD": {"name": "Cardano", "currency": "USD", "category": "Crypto"},
        "XRP-USD": {"name": "Ripple", "currency": "USD", "category": "Crypto"},
    }
    
    data = []
    for symbol, info in indices.items():
        try:
            stock_data = yf.Ticker(symbol)
            hist = stock_data.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = stock_data.info.get('previousClose', current_price)
                change = current_price - prev_close
                change_percent = (change / prev_close) * 100
                
                data.append({
                    "Symbol": symbol,
                    "Name": info["name"],
                    "Price": current_price,
                    "Change": change,
                    "Change %": change_percent,
                    "Currency": info["currency"],
                    "Category": info["category"]
                })
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            continue
    
    return pd.DataFrame(data)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(query="stock market", num_articles=10):
    try:
        NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "0b08be107dca45d3be30ca7e06544408")
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&pageSize={num_articles}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            # Filter out articles with minimal content
            filtered_articles = []
            for article in articles:
                if article.get('title') and article.get('title') != '[Removed]':
                    filtered_articles.append(article)
            return filtered_articles[:num_articles]
        return []
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_technical_indicators(df):
    if df.empty:
        return df
    
    try:
        # Calculate RSI
        rsi_indicator = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi_indicator.rsi()
        
        # Calculate MACD
        macd_indicator = MACD(close=df['Close'])
        df['MACD'] = macd_indicator.macd()
        df['MACD_Signal'] = macd_indicator.macd_signal()
        df['MACD_Histogram'] = macd_indicator.macd_diff()
        
        # Calculate EMA
        ema_12 = EMAIndicator(close=df['Close'], window=12)
        df['EMA_12'] = ema_12.ema_indicator()
        
        ema_26 = EMAIndicator(close=df['Close'], window=26)
        df['EMA_26'] = ema_26.ema_indicator()
        
        # Calculate Bollinger Bands
        bb_indicator = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb_indicator.bollinger_hband()
        df['BB_Lower'] = bb_indicator.bollinger_lband()
        df['BB_Middle'] = bb_indicator.bollinger_mavg()
        
        # Calculate VWAP
        vwap_indicator = VolumeWeightedAveragePrice(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume'],
            window=14
        )
        df['VWAP'] = vwap_indicator.volume_weighted_average_price()
        
        # Calculate moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate price volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        return df
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return df

# Advanced prediction functions
@st.cache_data(ttl=3600, show_spinner=False)
def predict_stock_price(ticker, days=30, method='linear'):
    """ Advanced stock price prediction using multiple methods """
    try:
        # Get historical data
        hist, _ = fetch_stock_data(ticker, "2y")
        if hist is None or hist.empty:
            return None, None, None, None
        
        # Prepare data
        prices = hist['Close'].values
        x = np.arange(len(prices))
        
        if method == 'linear':
            # Linear regression
            z = np.polyfit(x, prices, 1)
            p = np.poly1d(z)
        elif method == 'polynomial':
            # Polynomial regression (degree 2)
            z = np.polyfit(x, prices, 2)
            p = np.poly1d(z)
        elif method == 'exponential':
            # Exponential regression
            log_prices = np.log(prices)
            z = np.polyfit(x, log_prices, 1)
            p = np.poly1d(z)
            # Convert back from log space
            future_prices = np.exp(p(np.arange(len(prices), len(prices) + days)))
            current_price = prices[-1]
            predicted_price = future_prices[-1]
            confidence = max(0, min(100, 95 - (abs(predicted_price - current_price) / current_price * 100)))
            return future_prices, predicted_price, confidence, method
        else:
            # Default to linear
            z = np.polyfit(x, prices, 1)
            p = np.poly1d(z)
        
        # Predict future prices
        future_x = np.arange(len(prices), len(prices) + days)
        future_prices = p(future_x)
        
        # Calculate confidence intervals (simplified)
        current_price = prices[-1]
        predicted_price = future_prices[-1]
        confidence = max(0, min(100, 95 - (abs(predicted_price - current_price) / current_price * 100)))
        
        return future_prices, predicted_price, confidence, method
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None, None, None

@st.cache_data(ttl=3600, show_spinner=False)
def get_mutual_funds():
    """ Get a list of popular mutual funds with their performance data """
    mutual_funds = {
        'VFIAX': {'name': 'Vanguard 500 Index Fund', 'category': 'Large Blend', 'expense_ratio': 0.04},
        'VTSAX': {'name': 'Vanguard Total Stock Market Index Fund', 'category': 'Large Blend', 'expense_ratio': 0.04},
        'VGSLX': {'name': 'Vanguard Real Estate Index Fund', 'category': 'Real Estate', 'expense_ratio': 0.12},
        'VIMAX': {'name': 'Vanguard Mid-Cap Index Fund', 'category': 'Mid-Cap Blend', 'expense_ratio': 0.05},
        'VSMAX': {'name': 'Vanguard Small-Cap Index Fund', 'category': 'Small-Cap Blend', 'expense_ratio': 0.05},
        'VTIAX': {'name': 'Vanguard Total International Stock Index Fund', 'category': 'International', 'expense_ratio': 0.11},
        'VBTLX': {'name': 'Vanguard Total Bond Market Index Fund', 'category': 'Intermediate-Term Bond', 'expense_ratio': 0.05},
        'VWELX': {'name': 'Vanguard Wellington Fund', 'category': 'Balanced', 'expense_ratio': 0.25},
        'PRGFX': {'name': 'T. Rowe Price Growth Stock Fund', 'category': 'Large Growth', 'expense_ratio': 0.65},
        'AGTHX': {'name': 'American Funds Growth Fund of America', 'category': 'Large Growth', 'expense_ratio': 0.64},
    }
    
    data = []
    for symbol, info in mutual_funds.items():
        try:
            fund_data = yf.Ticker(symbol)
            hist = fund_data.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = fund_data.info.get('previousClose', current_price)
                change = current_price - prev_close
                change_percent = (change / prev_close) * 100
                
                # Get additional info
                info_data = fund_data.info
                ytd_return = info_data.get('ytdReturn', 0) * 100 if info_data.get('ytdReturn') else 0
                
                data.append({
                    "Symbol": symbol,
                    "Name": info["name"],
                    "Category": info["category"],
                    "Expense Ratio": info["expense_ratio"],
                    "Price": current_price,
                    "Change": change,
                    "Change %": change_percent,
                    "YTD Return %": ytd_return
                })
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            continue
    
    return pd.DataFrame(data)

@st.cache_data(ttl=3600, show_spinner=False)
def get_options_chain(ticker, expiration=None):
    """ Get options chain data for a given ticker """
    try:
        stock = yf.Ticker(ticker)
        options_dates = stock.options
        
        if not options_dates:
            return None, None
        
        if not expiration:
            expiration = options_dates[0]
        
        opt_chain = stock.option_chain(expiration)
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        return calls, puts, options_dates
    except Exception as e:
        st.error(f"Error fetching options chain: {str(e)}")
        return None, None, None

@st.cache_data(ttl=3600, show_spinner=False)
def get_financial_statements(ticker):
    """ Get financial statements for a company """
    try:
        stock = yf.Ticker(ticker)
        
        # Get income statement, balance sheet, and cash flow
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        return income_stmt, balance_sheet, cash_flow
    except Exception as e:
        st.error(f"Error fetching financial statements: {str(e)}")
        return None, None, None

@st.cache_data(ttl=3600, show_spinner=False)
def get_analyst_recommendations(ticker):
    """ Get analyst recommendations for a stock """
    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        
        if recommendations is not None and not recommendations.empty:
            # Get the latest recommendation
            latest_recommendation = recommendations.iloc[-1]
            return recommendations, latest_recommendation
        else:
            return None, None
    except Exception as e:
        st.error(f"Error fetching analyst recommendations: {str(e)}")
        return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def get_earnings_calendar(ticker):
    """ Get earnings calendar for a stock """
    try:
        stock = yf.Ticker(ticker)
        earnings_calendar = stock.calendar
        
        if earnings_calendar is not None and not earnings_calendar.empty:
            return earnings_calendar
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching earnings calendar: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_insider_transactions(ticker):
    """ Get insider transactions for a stock """
    try:
        stock = yf.Ticker(ticker)
        insider_transactions = stock.insider_transactions
        
        if insider_transactions is not None and not insider_transactions.empty:
            return insider_transactions
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching insider transactions: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_institutional_holders(ticker):
    """ Get institutional holders for a stock """
    try:
        stock = yf.Ticker(ticker)
        institutional_holders = stock.institutional_holders
        
        if institutional_holders is not None and not institutional_holders.empty:
            return institutional_holders
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching institutional holders: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_major_holders(ticker):
    """ Get major holders for a stock """
    try:
        stock = yf.Ticker(ticker)
        major_holders = stock.major_holders
        
        if major_holders is not None and not major_holders.empty:
            return major_holders
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching major holders: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_sustainability_data(ticker):
    """ Get sustainability data for a stock """
    try:
        stock = yf.Ticker(ticker)
        sustainability = stock.sustainability
        
        if sustainability is not None and not sustainability.empty:
            return sustainability
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching sustainability data: {str(e)}")
        return None

# Risk analysis functions
def calculate_portfolio_risk(portfolio):
    """ Calculate risk metrics for a portfolio """
    if portfolio.empty:
        return None
    
    risk_metrics = {}
    
    # Calculate portfolio value
    total_value = 0
    for _, holding in portfolio.iterrows():
        hist, _ = fetch_stock_data(holding['Symbol'], "1d")
        if hist is not None and not hist.empty:
            current_price = hist['Close'].iloc[-1]
            total_value += current_price * holding['Quantity']
    
    risk_metrics['Total Value'] = total_value
    
    # Calculate concentration risk
    sector_concentration = {}
    for _, holding in portfolio.iterrows():
        hist, info = fetch_stock_data(holding['Symbol'], "1d")
        if hist is not None and not hist.empty and info is not None:
            current_price = hist['Close'].iloc[-1]
            holding_value = current_price * holding['Quantity']
            sector = info.get('sector', 'Unknown')
            
            if sector in sector_concentration:
                sector_concentration[sector] += holding_value
            else:
                sector_concentration[sector] = holding_value
    
    risk_metrics['Sector Concentration'] = sector_concentration
    
    # Calculate beta-weighted exposure (simplified)
    beta_exposure = 0
    for _, holding in portfolio.iterrows():
        hist, info = fetch_stock_data(holding['Symbol'], "1d")
        if hist is not None and not hist.empty and info is not None:
            current_price = hist['Close'].iloc[-1]
            holding_value = current_price * holding['Quantity']
            beta = info.get('beta', 1.0)
            beta_exposure += (holding_value / total_value) * beta
    
    risk_metrics['Beta Exposure'] = beta_exposure
    
    return risk_metrics

# Portfolio optimization functions
def optimize_portfolio(portfolio, risk_tolerance='medium'):
    """ Optimize portfolio based on risk tolerance """
    if portfolio.empty:
        return None
    
    # This is a simplified optimization approach
    # In a real application, you would use more sophisticated methods like MPT
    
    recommendations = []
    
    # Calculate current allocation
    total_value = 0
    allocation = {}
    for _, holding in portfolio.iterrows():
        hist, info = fetch_stock_data(holding['Symbol'], "1d")
        if hist is not None and not hist.empty and info is not None:
            current_price = hist['Close'].iloc[-1]
            holding_value = current_price * holding['Quantity']
            total_value += holding_value
            sector = info.get('sector', 'Unknown')
            
            if sector in allocation:
                allocation[sector] += holding_value
            else:
                allocation[sector] = holding_value
    
    # Calculate percentage allocation
    for sector in allocation:
        allocation[sector] = (allocation[sector] / total_value) * 100
    
    # Generate recommendations based on risk tolerance
    if risk_tolerance == 'low':
        # Recommend more diversified, lower risk assets
        if allocation.get('Technology', 0) > 30:
            recommendations.append("Consider reducing Technology exposure for better diversification")
        if allocation.get('Healthcare', 0) < 15:
            recommendations.append("Consider adding Healthcare sector for stability")
        if allocation.get('Consumer Defensive', 0) < 10:
            recommendations.append("Consider adding Consumer Defensive sector for stability")
    elif risk_tolerance == 'medium':
        # Balanced recommendations
        if allocation.get('Technology', 0) > 40:
            recommendations.append("Consider reducing Technology exposure for better diversification")
        if allocation.get('Healthcare', 0) < 10:
            recommendations.append("Consider adding Healthcare sector for growth and stability")
    else:  # high risk tolerance
        # Growth-oriented recommendations
        if allocation.get('Technology', 0) < 30:
            recommendations.append("Consider adding Technology sector for growth potential")
        if allocation.get('Communication Services', 0) < 15:
            recommendations.append("Consider adding Communication Services sector for growth potential")
    
    return recommendations, allocation

# Economic calendar data
@st.cache_data(ttl=3600, show_spinner=False)
def get_economic_calendar():
    """ Get economic calendar data """
    try:
        # This is a placeholder function
        # In a real application, you would use an API like Forex Factory or similar
        today = datetime.now().date()
        events = [
            {
                'date': today,
                'time': '10:00 AM',
                'event': 'Consumer Price Index (CPI)',
                'impact': 'High',
                'actual': '3.2%',
                'forecast': '3.1%',
                'previous': '3.0%'
            },
            {
                'date': today + timedelta(days=1),
                'time': '8:30 AM',
                'event': 'Initial Jobless Claims',
                'impact': 'Medium',
                'actual': None,
                'forecast': '210K',
                'previous': '205K'
            },
            {
                'date': today + timedelta(days=2),
                'time': '10:00 AM',
                'event': 'Retail Sales',
                'impact': 'High',
                'actual': None,
                'forecast': '0.5%',
                'previous': '0.3%'
            }
        ]
        return events
    except Exception as e:
        st.error(f"Error fetching economic calendar: {str(e)}")
        return []

# Sidebar navigation
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #FF0000;'>MarketMentor Pro</h1>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Dashboard", "Stock Analysis", "Technical Analysis", "Portfolio Manager", 
                 "Options Chain", "Market Overview", "Economic Calendar", "Crypto Markets", 
                 "News & Sentiment", "Learning Center", "Company Info", "Predictions", "Settings"],
        icons=["house", "graph-up", "bar-chart", "wallet", 
               "diagram-3", "globe", "calendar", "currency-bitcoin",
               "newspaper", "book", "building", "lightbulb", "gear"],
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#0A0A0A"},
            "icon": {"color": #FF0000", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#FFFFFF"},
            "nav-link-selected": {"background-color": "#FF0000", "color": "#000000", "font-weight": "bold"},
        })
     
    
    # Watchlist section in sidebar
    st.subheader("My Watchlist")
    watchlist_symbol = st.text_input("Add symbol to watchlist:", key="watchlist_input")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add to Watchlist", key="add_watchlist"):
            if watchlist_symbol and watchlist_symbol.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(watchlist_symbol.upper())
                st.success(f"Added {watchlist_symbol.upper()} to watchlist")
                st.rerun()
    with col2:
        if st.button("Clear Watchlist", key="clear_watchlist"):
            st.session_state.watchlist = []
            st.success("Watchlist cleared")
            st.rerun()
    
    if st.session_state.watchlist:
        st.write("**Current Watchlist:**")
        for symbol in st.session_state.watchlist:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(symbol)
            with col2:
                if st.button("X", key=f"remove_{symbol}"):
                    st.session_state.watchlist.remove(symbol)
                    st.rerun()
    
    # Market status
    st.subheader("Market Status")
    now = datetime.now()
    ny_time = now.astimezone(pytz.timezone('US/Eastern'))
    market_open = ny_time.hour >= 9 and ny_time.hour < 16 and ny_time.weekday() < 5
    
    if market_open:
        st.success("US Markets: OPEN")
        st.write(f"Time in NY: {ny_time.strftime('%H:%M')}")
    else:
        st.error("US Markets: CLOSED")
        st.write(f"Time in NY: {ny_time.strftime('%H:%M')}")
    
    # Quick actions
    st.subheader("Quick Actions")
    if st.button("Refresh All Data"):
        st.cache_data.clear()
        st.success("Data refreshed!")
    
    if st.button("View Portfolio Summary"):
        st.session_state.analyze_symbol = "PORTFOLIO"
        st.switch_page("Portfolio Manager")

# Dashboard Page
if selected == "Dashboard":
    st.title("📈 Market Dashboard")
    
    # Market overview with global indices
    st.subheader("Global Market Overview")
    indices_df = fetch_global_indices()
    
    if not indices_df.empty:
        # Group by category
        categories = indices_df['Category'].unique()
        
        for category in categories:
            st.write(f"**{category}**")
            category_df = indices_df[indices_df['Category'] == category]
            
            cols = st.columns(4)
            for idx, row in category_df.iterrows():
                col_idx = idx % 4
                with cols[col_idx]:
                    change_color = "normal" if row["Change"] >= 0 else "inverse"
                    st.metric(
                        label=f"{row['Name']} ({row['Symbol']})",
                        value=format_currency(row["Price"], row["Currency"]),
                        delta=f"{row['Change']:.2f} ({row['Change %']:.2f}%)",
                        delta_color=change_color
                    )
    
    # Watchlist performance
    if st.session_state.watchlist:
        st.subheader("Watchlist Performance")
        watchlist_data = []
        
        for symbol in st.session_state.watchlist:
            hist, info = fetch_stock_data(symbol, "1d")
            if hist is not None and not hist.empty and info is not None:
                current_price = hist['Close'].iloc[-1]
                prev_close = info.get('previousClose', current_price)
                change = current_price - prev_close
                change_percent = (change / prev_close) * 100
                currency = info.get('currency', 'USD')
                
                watchlist_data.append({
                    "Symbol": symbol,
                    "Price": current_price,
                    "Change": change,
                    "Change %": change_percent,
                    "Currency": currency
                })
        
        if watchlist_data:
            watchlist_df = pd.DataFrame(watchlist_data)
            for idx, row in watchlist_df.iterrows():
                col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                with col1:
                    st.write(f"**{row['Symbol']}**")
                with col2:
                    st.write(format_currency(row["Price"], row["Currency"]))
                with col3:
                    change_color = "green" if row["Change"] >= 0 else "red"
                    st.markdown(f"<span style='color:{change_color}'>{row['Change']:.2f} ({row['Change %']:.2f}%)</span>", 
                               unsafe_allow_html=True)
                with col4:
                    if st.button("Analyze", key=f"analyze_{row['Symbol']}"):
                        st.session_state.analyze_symbol = row['Symbol']
                        st.switch_page("Stock Analysis")
    
    # Portfolio snapshot
    if not st.session_state.portfolio.empty:
        st.subheader("Portfolio Snapshot")
        
        # Calculate portfolio value
        total_value = 0
        portfolio_data = []
        
        for _, holding in st.session_state.portfolio.iterrows():
            hist, info = fetch_stock_data(holding['Symbol'], "1d")
            if hist is not None and not hist.empty:
                current_price = hist['Close'].iloc[-1]
                holding_value = current_price * holding['Quantity']
                total_value += holding_value
                purchase_value = holding['Purchase Price'] * holding['Quantity']
                gain_loss = holding_value - purchase_value
                gain_loss_percent = (gain_loss / purchase_value) * 100
                currency = info.get('currency', 'USD') if info else 'USD'
                
                portfolio_data.append({
                    "Symbol": holding['Symbol'],
                    "Quantity": holding['Quantity'],
                    "Current Price": current_price,
                    "Current Value": holding_value,
                    "Gain/Loss": gain_loss,
                    "Gain/Loss %": gain_loss_percent,
                    "Currency": currency
                })
        
        if portfolio_data:
            portfolio_df = pd.DataFrame(portfolio_data)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Portfolio Value", format_currency(total_value, "USD"))
            with col2:
                total_gain_loss = portfolio_df['Gain/Loss'].sum()
                st.metric("Total Gain/Loss", format_currency(total_gain_loss, "USD"))
            with col3:
                if total_value > 0:
                    total_gain_loss_percent = (total_gain_loss / (total_value - total_gain_loss)) * 100
                    st.metric("Total Gain/Loss %", f"{total_gain_loss_percent:.2f}%")
            
            # Show top 5 holdings
            st.write("**Top Holdings:**")
            top_holdings = portfolio_df.nlargest(5, 'Current Value')
            for _, holding in top_holdings.iterrows():
                st.write(f"{holding['Symbol']}: {format_currency(holding['Current Value'], holding['Currency'])} "
                        f"({holding['Gain/Loss %']:.2f}%)")
    
    # Latest news
    st.subheader("Latest Market News")
    news_articles = fetch_news("stock market", 5)
    
    if news_articles:
        for article in news_articles:
            with st.expander(f"{article['title']} - {article['source']['name']}"):
                if article.get('urlToImage'):
                    st.image(article['urlToImage'], width=300)
                st.write(article['description'])
                if article['url']:
                    st.markdown(f"[Read more]({article['url']})")
    else:
        st.info("No news available at the moment. Check your internet connection or try again later.")

# Stock Analysis Page
elif selected == "Stock Analysis":
    st.title("📊 Stock Analysis")
    
    # Symbol input
    symbol = st.text_input("Enter stock symbol (e.g., AAPL, MSFT, RELIANCE.NS, TCS.NS):", 
                          value=st.session_state.analyze_symbol)
    
    if symbol:
        # Period selection
        period = st.selectbox("Select period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"])
        
        # Fetch data
        hist, info = fetch_stock_data(symbol, period)
        
        if hist is not None and not hist.empty and info is not None:
            # Display stock info
            col1, col2, col3, col4 = st.columns(4)
            
            currency = info.get('currency', 'USD')
            current_price = hist['Close'].iloc[-1]
            prev_close = info.get('previousClose', current_price)
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            
            with col1:
                st.metric("Current Price", format_currency(current_price, currency))
            with col2:
                st.metric("Previous Close", format_currency(prev_close, currency))
            with col3:
                st.metric("Change", format_currency(change, currency), f"{change_percent:.2f}%")
            with col4:
                st.metric("Market Cap", format_currency(info.get('marketCap', 0), currency))
            
            # Display additional info
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric("Open", format_currency(hist['Open'].iloc[-1], currency))
            with col6:
                st.metric("High", format_currency(hist['High'].iloc[-1], currency))
            with col7:
                st.metric("Low", format_currency(hist['Low'].iloc[-1], currency))
            with col8:
                st.metric("Volume", f"{hist['Volume'].iloc[-1]:,}")
            
            # Display more company info
            col9, col10, col11, col12 = st.columns(4)
            with col9:
                st.metric("52 Week High", format_currency(info.get('fiftyTwoWeekHigh', 0), currency))
            with col10:
                st.metric("52 Week Low", format_currency(info.get('fiftyTwoWeekLow', 0), currency))
            with col11:
                st.metric("PE Ratio", f"{info.get('trailingPE', 'N/A')}")
            with col12:
                st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100 if info.get('dividendYield') else 0:.2f}%")
            
            # Price chart
            st.subheader("Price Chart")
            chart_type = st.radio("Chart Type:", ["Line", "Candlestick"], horizontal=True)
            
            fig = go.Figure()
            if chart_type == "Candlestick":
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='Price'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    name='Close',
                    line=dict(color='white')
                ))
            
            fig.update_layout(
                title=f"{symbol} Price History",
                xaxis_title="Date",
                yaxis_title=f"Price ({currency})",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display company info if available
            if 'longName' in info:
                st.subheader("Company Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Name:** {info.get('longName', 'N/A')}")
                    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.write(f"**Country:** {info.get('country', 'N/A')}")
                
                with col2:
                    st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
                    st.write(f"**Website:** {info.get('website', 'N/A')}")
                    st.write(f"**CEO:** {info.get('ceo', 'N/A')}")
                    st.write(f"**IPO Year:** {info.get('ipoYear', 'N/A')}")
                
                if 'longBusinessSummary' in info:
                    with st.expander("Business Summary"):
                        st.write(info.get('longBusinessSummary', 'No summary available.'))
        else:
            st.error("Could not fetch data for the specified symbol. Please check the symbol and try again.")

# Technical Analysis Page
elif selected == "Technical Analysis":
    st.title("📈 Technical Analysis")
    
    symbol = st.text_input("Enter stock symbol:", "AAPL")
    period = st.selectbox("Select period:", ["1mo", "3mo", "6mo", "1y", "2y"])
    interval = st.selectbox("Select interval:", ["1d", "1wk", "1h"], index=0)
    
    if symbol:
        hist, info = fetch_stock_data(symbol, period, interval)
        
        if hist is not None and not hist.empty:
            currency = info.get('currency', 'USD') if info else 'USD'
            hist = get_technical_indicators(hist)
            
            # Price with SMA
            st.subheader("Moving Averages")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close', line=dict(color='white')))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name='SMA 20', line=dict(color='yellow')))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], name='SMA 200', line=dict(color='red')))
            fig.update_layout(
                title=f"{symbol} Price with Moving Averages",
                xaxis_title="Date",
                yaxis_title=f"Price ({currency})",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # MACD
            st.subheader("MACD")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD', line=dict(color='white')))
            fig2.add_trace(go.Scatter(x=hist.index, y=hist['MACD_Signal'], name='Signal', line=dict(color='yellow')))
            
            # Add histogram
            colors = ['green' if val >= 0 else 'red' for val in hist['MACD_Histogram']]
            fig2.add_trace(go.Bar(
                x=hist.index,
                y=hist['MACD_Histogram'],
                name='Histogram',
                marker_color=colors
            ))
            
            fig2.update_layout(
                title="MACD Indicator",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # RSI
            st.subheader("RSI")
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI', line=dict(color='white')))
            fig3.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig3.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig3.add_hline(y=50, line_dash="dash", line_color="gray")
            fig3.update_layout(
                title="RSI Indicator",
                xaxis_title="Date",
                yaxis_title="RSI Value",
                template="plotly_dark",
                height=400,
                yaxis_range=[0, 100]
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # Bollinger Bands
            st.subheader("Bollinger Bands")
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close', line=dict(color='white')))
            fig4.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], name='Upper Band', line=dict(color='red')))
            fig4.add_trace(go.Scatter(x=hist.index, y=hist['BB_Middle'], name='Middle Band', line=dict(color='yellow')))
            fig4.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], name='Lower Band', line=dict(color='green')))
            fig4.update_layout(
                title="Bollinger Bands",
                xaxis_title="Date",
                yaxis_title=f"Price ({currency})",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig4, use_container_width=True)
            
            # Volume
            st.subheader("Volume")
            fig5 = go.Figure()
            colors = ['green' if hist['Close'].iloc[i] >= hist['Open'].iloc[i] else 'red' for i in range(len(hist))]
            fig5.add_trace(go.Bar(
                x=hist.index,
                y=hist['Volume'],
                name='Volume',
                marker_color=colors
            ))
            fig5.update_layout(
                title="Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig5, use_container_width=True)
            
            # Technical indicators summary
            st.subheader("Technical Indicators Summary")
            
            # Calculate current values
            current_rsi = hist['RSI'].iloc[-1] if not pd.isna(hist['RSI'].iloc[-1]) else 0
            macd_line = hist['MACD'].iloc[-1] if not pd.isna(hist['MACD'].iloc[-1]) else 0
            signal_line = hist['MACD_Signal'].iloc[-1] if not pd.isna(hist['MACD_Signal'].iloc[-1]) else 0
            price = hist['Close'].iloc[-1]
            bb_upper = hist['BB_Upper'].iloc[-1] if not pd.isna(hist['BB_Upper'].iloc[-1]) else 0
            bb_lower = hist['BB_Lower'].iloc[-1] if not pd.isna(hist['BB_Lower'].iloc[-1]) else 0
            
            # Generate signals
            signals = []
            
            # RSI signals
            if current_rsi > 70:
                signals.append(("RSI", "Overbought", "bearish"))
            elif current_rsi < 30:
                signals.append(("RSI", "Oversold", "bullish"))
            else:
                signals.append(("RSI", "Neutral", "neutral"))
            
            # MACD signals
            if macd_line > signal_line:
                signals.append(("MACD", "Bullish", "bullish"))
            elif macd_line < signal_line:
                signals.append(("MACD", "Bearish", "bearish"))
            else:
                signals.append(("MACD", "Neutral", "neutral"))
            
            # Bollinger Bands signals
            if price > bb_upper:
                signals.append(("Bollinger Bands", "Overbought", "bearish"))
            elif price < bb_lower:
                signals.append(("Bollinger Bands", "Oversold", "bullish"))
            else:
                signals.append(("Bollinger Bands", "Neutral", "neutral"))
            
            # Display signals
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Indicator**")
                for signal in signals:
                    st.write(signal[0])
            
            with col2:
                st.write("**Signal**")
                for signal in signals:
                    st.write(signal[1])
            
            with col3:
                st.write("**Bias**")
                for signal in signals:
                    if signal[2] == "bullish":
                        st.success("Bullish")
                    elif signal[2] == "bearish":
                        st.error("Bearish")
                    else:
                        st.info("Neutral")

# Portfolio Manager Page
elif selected == "Portfolio Manager":
    st.title("💼 Portfolio Manager")
    
    tab1, tab2, tab3, tab4 = st.tabs(["View Portfolio", "Add Holding", "Performance Analysis", "Risk Analysis"])
    
    with tab1:
        st.subheader("Your Portfolio")
        if st.session_state.portfolio.empty:
            st.info("Your portfolio is empty. Add holdings to get started.")
        else:
            st.dataframe(st.session_state.portfolio)
            
            # Calculate portfolio value
            total_value = 0
            portfolio_data = []
            
            for _, holding in st.session_state.portfolio.iterrows():
                hist, info = fetch_stock_data(holding['Symbol'], "1d")
                if hist is not None and not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    holding_value = current_price * holding['Quantity']
                    total_value += holding_value
                    purchase_value = holding['Purchase Price'] * holding['Quantity']
                    gain_loss = holding_value - purchase_value
                    gain_loss_percent = (gain_loss / purchase_value) * 100
                    currency = info.get('currency', 'USD') if info else 'USD'
                    
                    portfolio_data.append({
                        'Symbol': holding['Symbol'],
                        'Quantity': holding['Quantity'],
                        'Current Price': current_price,
                        'Current Value': holding_value,
                        'Gain/Loss': gain_loss,
                        'Gain/Loss %': gain_loss_percent,
                        'Currency': currency
                    })
            
            portfolio_df = pd.DataFrame(portfolio_data)
            st.dataframe(portfolio_df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Portfolio Value", format_currency(total_value, "USD"))
            with col2:
                total_gain_loss = portfolio_df['Gain/Loss'].sum()
                st.metric("Total Gain/Loss", format_currency(total_gain_loss, "USD"))
            with col3:
                if total_value > 0:
                    total_gain_loss_percent = (total_gain_loss / (total_value - total_gain_loss)) * 100
                    st.metric("Total Gain/Loss %", f"{total_gain_loss_percent:.2f}%")
    
    with tab2:
        st.subheader("Add New Holding")
        with st.form("add_holding_form"):
            symbol = st.text_input("Symbol")
            quantity = st.number_input("Quantity", min_value=1, value=1)
            purchase_price = st.number_input("Purchase Price", min_value=0.0, value=0.0)
            purchase_date = st.date_input("Purchase Date", value=datetime.now())
            
            if st.form_submit_button("Add Holding"):
                # Fetch info to get sector and currency
                _, info = fetch_stock_data(symbol, "1d")
                sector = info.get('sector', 'Unknown') if info else 'Unknown'
                currency = info.get('currency', 'USD') if info else 'USD'
                
                new_holding = pd.DataFrame({
                    'Symbol': [symbol.upper()],
                    'Quantity': [quantity],
                    'Purchase Price': [purchase_price],
                    'Purchase Date': [purchase_date],
                    'Sector': [sector],
                    'Currency': [currency]
                })
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_holding], ignore_index=True)
                st.success(f"Added {quantity} shares of {symbol.upper()} to your portfolio")
    
    with tab3:
        st.subheader("Portfolio Performance")
        if not st.session_state.portfolio.empty:
            # Calculate performance for each holding
            performance_data = []
            for _, holding in st.session_state.portfolio.iterrows():
                hist, info = fetch_stock_data(holding['Symbol'], "1d")
                if hist is not None and not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    purchase_value = holding['Purchase Price'] * holding['Quantity']
                    current_value = current_price * holding['Quantity']
                    gain_loss = current_value - purchase_value
                    gain_loss_percent = (gain_loss / purchase_value) * 100
                    currency = info.get('currency', 'USD') if info else 'USD'
                    
                    performance_data.append({
                        'Symbol': holding['Symbol'],
                        'Quantity': holding['Quantity'],
                        'Purchase Price': holding['Purchase Price'],
                        'Current Price': current_price,
                        'Purchase Value': purchase_value,
                        'Current Value': current_value,
                        'Gain/Loss': gain_loss,
                        'Gain/Loss %': gain_loss_percent,
                        'Currency': currency
                    })
            
            if performance_data:
                performance_df = pd.DataFrame(performance_data)
                st.dataframe(performance_df)
                
                # Portfolio allocation chart
                fig = px.pie(performance_df, values='Current Value', names='Symbol', title='Portfolio Allocation')
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance chart
                fig2 = go.Figure()
                for _, holding in performance_df.iterrows():
                    fig2.add_trace(go.Bar(
                        x=[holding['Symbol']],
                        y=[holding['Gain/Loss %']],
                        name=holding['Symbol'],
                        text=[f"{holding['Gain/Loss %']:.2f}%"],
                        textposition='auto'
                    ))
                
                fig2.update_layout(
                    title="Gain/Loss by Holding (%)",
                    xaxis_title="Symbol",
                    yaxis_title="Gain/Loss %",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    with tab4:
        st.subheader("Portfolio Risk Analysis")
        if not st.session_state.portfolio.empty:
            risk_metrics = calculate_portfolio_risk(st.session_state.portfolio)
            
            if risk_metrics:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Portfolio Value", format_currency(risk_metrics.get('Total Value', 0), "USD"))
                    st.metric("Beta-Weighted Exposure", f"{risk_metrics.get('Beta Exposure', 0):.2f}")
                
                with col2:
                    # Sector concentration
                    st.write("**Sector Concentration:**")
                    sector_data = risk_metrics.get('Sector Concentration', {})
                    if sector_data:
                        for sector, value in sector_data.items():
                            percentage = (value / risk_metrics['Total Value']) * 100
                            st.write(f"{sector}: {percentage:.2f}%")
                    
                    # Risk assessment
                    beta_exposure = risk_metrics.get('Beta Exposure', 1.0)
                    if beta_exposure > 1.2:
                        st.error("High Risk Portfolio (Beta > 1.2)")
                    elif beta_exposure < 0.8:
                        st.success("Low Risk Portfolio (Beta < 0.8)")
                    else:
                        st.info("Moderate Risk Portfolio (Beta between 0.8 and 1.2)")
                
                # Portfolio optimization recommendations
                st.subheader("Optimization Recommendations")
                recommendations, allocation = optimize_portfolio(
                    st.session_state.portfolio, 
                    st.session_state.user_settings.get('risk_tolerance', 'medium')
                )
                
                if recommendations:
                    for recommendation in recommendations:
                        st.info(recommendation)
                else:
                    st.success("Your portfolio is well-optimized for your risk tolerance level.")

# Options Chain Page
elif selected == "Options Chain":
    st.title("📊 Options Chain Analysis")
    
    symbol = st.text_input("Enter stock symbol for options chain:", "AAPL")
    
    if symbol:
        # Get options expiration dates
        calls, puts, options_dates = get_options_chain(symbol)
        
        if options_dates:
            expiration = st.selectbox("Select expiration date:", options_dates)
            
            if expiration:
                calls, puts, _ = get_options_chain(symbol, expiration)
                
                if calls is not None and puts is not None:
                    st.subheader(f"Options Chain for {symbol} - Expiration: {expiration}")
                    
                    tab1, tab2 = st.tabs(["Call Options", "Put Options"])
                    
                    with tab1:
                        st.dataframe(calls)
                    
                    with tab2:
                        st.dataframe(puts)
                    
                    # Options strategy builder
                    st.subheader("Options Strategy Builder")
                    strategy = st.selectbox("Select a strategy:", 
                                          ["Covered Call", "Protective Put", "Collar", "Long Straddle", "Long Strangle"])
                    
                    if strategy:
                        st.info(f"**{strategy} Strategy:**")
                        
                        if strategy == "Covered Call":
                            st.write("1. Own 100 shares of the underlying stock")
                            st.write("2. Sell one call option contract for every 100 shares owned")
                            st.write("**Objective:** Generate income from premium collection while potentially selling shares at a higher price")
                        
                        elif strategy == "Protective Put":
                            st.write("1. Own 100 shares of the underlying stock")
                            st.write("2. Buy one put option contract for every 100 shares owned")
                            st.write("**Objective:** Protect against downside risk while maintaining upside potential")
                        
                        elif strategy == "Collar":
                            st.write("1. Own 100 shares of the underlying stock")
                            st.write("2. Buy one put option contract for every 100 shares owned")
                            st.write("3. Sell one call option contract for every 100 shares owned")
                            st.write("**Objective:** Protect against downside risk while generating income, with capped upside")
                        
                        elif strategy == "Long Straddle":
                            st.write("1. Buy one call option at a specific strike price")
                            st.write("2. Buy one put option at the same strike price and expiration")
                            st.write("**Objective:** Profit from significant price movement in either direction")
                        
                        elif strategy == "Long Strangle":
                            st.write("1. Buy one out-of-the-money call option")
                            st.write("2. Buy one out-of-the-money put option with the same expiration")
                            st.write("**Objective:** Profit from significant price movement in either direction with lower cost than a straddle")
                else:
                    st.error("Could not fetch options chain data. Please check the symbol and try again.")
        else:
            st.error("No options data available for this symbol. Please check the symbol and try again.")

# Market Overview Page
elif selected == "Market Overview":
    st.title("🌍 Global Market Overview")
    
    indices_df = fetch_global_indices()
    if not indices_df.empty:
        # Display metrics
        st.subheader("Key Market Indicators")
        
        # Filter for major indices
        major_indices = indices_df[indices_df['Category'] == 'Index'].head(4)
        cols = st.columns(4)
        for idx, row in major_indices.iterrows():
            with cols[idx % 4]:
                change_color = "normal" if row["Change"] >= 0 else "inverse"
                st.metric(
                    label=row["Name"],
                    value=format_currency(row["Price"], row["Currency"]),
                    delta=f"{row['Change']:.2f} ({row['Change %']:.2f}%)",
                    delta_color=change_color
                )
        
        # Display full table
        st.subheader("All Market Indicators")
        st.dataframe(indices_df)
        
        # Create performance chart
        st.subheader("Performance Comparison")
        fig = px.bar(indices_df, x='Name', y='Change %', title='Global Indices Performance (%)',
                    color='Change %', color_continuous_scale=['red', 'white', 'green'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Sector performance (simulated data)
        st.subheader("Sector Performance")
        sectors = {
            'Technology': 2.5,
            'Healthcare': 1.8,
            'Financial Services': -0.7,
            'Consumer Cyclical': 1.2,
            'Communication Services': 3.1,
            'Energy': -2.3,
            'Utilities': 0.5,
            'Real Estate': -1.2,
            'Industrials': 0.8,
            'Materials': -0.3
        }
        
        sector_df = pd.DataFrame(list(sectors.items()), columns=['Sector', 'Change %'])
        fig = px.bar(sector_df, x='Sector', y='Change %', title='Sector Performance (%)',
                    color='Change %', color_continuous_scale=['red', 'white', 'green'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Could not fetch market data. Please check your internet connection and try again.")

# Economic Calendar Page
elif selected == "Economic Calendar":
    st.title("📅 Economic Calendar")
    
    events = get_economic_calendar()
    
    if events:
        st.subheader("Upcoming Economic Events")
        
        for event in events:
            with st.expander(f"{event['date']} - {event['time']}: {event['event']} ({event['impact']} Impact)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Previous:**")
                    st.write(event['previous'] if event['previous'] else "N/A")
                
                with col2:
                    st.write("**Forecast:**")
                    st.write(event['forecast'] if event['forecast'] else "N/A")
                
                with col3:
                    st.write("**Actual:**")
                    if event['actual']:
                        if event['actual'] > event['forecast']:
                            st.success(event['actual'])
                        else:
                            st.error(event['actual'])
                    else:
                        st.info("Not yet released")
        
        # Economic indicators explanation
        st.subheader("Economic Indicators Guide")
        
        indicators = {
            'CPI': 'Consumer Price Index - Measures inflation by tracking changes in the price of a basket of consumer goods and services.',
            'GDP': 'Gross Domestic Product - The total monetary value of all finished goods and services produced within a country\'s borders in a specific time period.',
            'Unemployment Rate': 'The percentage of the total labor force that is unemployed but actively seeking employment and willing to work.',
            'Retail Sales': 'A measure of the total receipts of retail stores, tracking sales of durable and non-durable goods over a period of time.',
            'Interest Rates': 'The amount charged by lenders to borrowers for the use of money, expressed as a percentage of the principal.'
        }
        
        for indicator, description in indicators.items():
            with st.expander(indicator):
                st.write(description)
    else:
        st.info("Economic calendar data is currently unavailable. Please check back later.")

# Crypto Markets Page
elif selected == "Crypto Markets":
    st.title("₿ Crypto Markets")
    
    crypto_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'SOL-USD', 'DOT-USD', 'DOGE-USD']
    crypto_data = []
    
    for symbol in crypto_symbols:
        hist, info = fetch_stock_data(symbol, "1d")
        if hist is not None and not hist.empty:
            current_price = hist['Close'].iloc[-1]
            prev_close = info.get('previousClose', current_price) if info else current_price
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            
            crypto_data.append({
                "Symbol": symbol,
                "Name": symbol.split('-')[0],
                "Price": current_price,
                "Change": change,
                "Change %": change_percent
            })
    
    if crypto_data:
        crypto_df = pd.DataFrame(crypto_data)
        
        # Display metrics
        st.subheader("Major Cryptocurrencies")
        
        # Top 4 cryptocurrencies
        top_crypto = crypto_df.head(4)
        cols = st.columns(4)
        for idx, row in top_crypto.iterrows():
            with cols[idx % 4]:
                change_color = "normal" if row["Change"] >= 0 else "inverse"
                st.metric(
                    label=row["Name"],
                    value=format_currency(row["Price"], "USD"),
                    delta=f"{row['Change']:.2f} ({row['Change %']:.2f}%)",
                    delta_color=change_color
                )
        
        # Display full table
        st.subheader("All Cryptocurrencies")
        st.dataframe(crypto_df)
        
        # Crypto performance chart
        fig = px.bar(crypto_df, x='Name', y='Change %', title='Cryptocurrency Performance (%)',
                    color='Change %', color_continuous_scale=['red', 'white', 'green'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Crypto market overview
        st.subheader("Crypto Market Overview")
        
        # Simulated market data
        market_metrics = {
            'Total Market Cap': '1.75T',
            '24h Volume': '75.5B',
            'Bitcoin Dominance': '48.2%',
            'Fear & Greed Index': '45 (Fear)'
        }
        
        cols = st.columns(4)
        for idx, (metric, value) in enumerate(market_metrics.items()):
            with cols[idx % 4]:
                st.metric(metric, value)
        
        # Crypto news
        st.subheader("Latest Crypto News")
        crypto_news = fetch_news("cryptocurrency", 3)
        
        if crypto_news:
            for article in crypto_news:
                with st.expander(f"{article['title']} - {article['source']['name']}"):
                    if article.get('urlToImage'):
                        st.image(article['urlToImage'], width=300)
                    st.write(article['description'])
                    if article['url']:
                        st.markdown(f"[Read more]({article['url']})")
        else:
            st.info("No crypto news available at the moment.")
    else:
        st.error("Could not fetch cryptocurrency data. Please check your internet connection and try again.")

# News & Sentiment Page
elif selected == "News & Sentiment":
    st.title("📰 News & Market Sentiment")
    
    # News search
    st.subheader("Market News")
    news_query = st.text_input("Search for news:", "stock market")
    num_articles = st.slider("Number of articles:", min_value=5, max_value=20, value=10)
    
    if st.button("Fetch News"):
        news_articles = fetch_news(news_query, num_articles)
        
        if news_articles:
            for article in news_articles:
                with st.expander(f"{article['title']} - {article['source']['name']} - {article['publishedAt'][:10]}"):
                    if article.get('urlToImage'):
                        st.image(article['urlToImage'], width=300)
                    st.write(article['description'])
                    if article['url']:
                        st.markdown(f"[Read more]({article['url']})")
        else:
            st.info("No news found for your query. Try a different search term.")
    
    # Market sentiment analysis
    st.subheader("Market Sentiment Analysis")
    
    # Simulated sentiment data
    sentiment_data = {
        'Overall Market': {'Sentiment': 'Neutral', 'Score': 50},
        'Technology Sector': {'Sentiment': 'Bullish', 'Score': 65},
        'Financial Sector': {'Sentiment': 'Bearish', 'Score': 40},
        'Healthcare Sector': {'Sentiment': 'Bullish', 'Score': 70},
        'Retail Sector': {'Sentiment': 'Neutral', 'Score': 55}
    }
    
    for sector, data in sentiment_data.items():
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.write(f"**{sector}**")
        
        with col2:
            score = data['Score']
            if score >= 60:
                st.success(f"{data['Sentiment']} ({score})")
            elif score <= 40:
                st.error(f"{data['Sentiment']} ({score})")
            else:
                st.info(f"{data['Sentiment']} ({score})")
        
        with col3:
            st.progress(score / 100)
    
    # Fear & Greed Index
    st.subheader("Fear & Greed Index")
    
    fear_greed_score = 45  # Simulated score
    fear_greed_label = "Fear" if fear_greed_score < 50 else "Greed"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Current Score", f"{fear_greed_score} ({fear_greed_label})")
    
    with col2:
        if fear_greed_score < 20:
            st.error("Extreme Fear - Potential buying opportunity")
        elif fear_greed_score < 50:
            st.warning("Fear - Market may be oversold")
        elif fear_greed_score < 80:
            st.success("Greed - Market may be overbought")
        else:
            st.error("Extreme Greed - Potential market top")
    
    # Add a gauge chart for Fear & Greed Index
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = fear_greed_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fear & Greed Index"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if fear_greed_score < 50 else "green"},
            'steps': [
                {'range': [0, 20], 'color': "red"},
                {'range': [20, 50], 'color': "orange"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ]
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)

# Learning Center Page
elif selected == "Learning Center":
    st.title("📚 Learning Center")
    
    st.subheader("Educational Resources")
    
    # Investment topics
    topics = [
        {
            'title': 'Introduction to Stock Market',
            'content': '''
            The stock market is a platform where investors can buy and sell shares of publicly traded companies. 
            It serves as a barometer for the overall economy and provides opportunities for wealth creation.
            
            **Key Concepts:**
            - Stocks represent ownership in a company
            - Stock prices fluctuate based on supply and demand
            - Investors can profit through capital appreciation and dividends
            - Markets can be volatile in the short term but tend to rise over the long term
            '''
        },
        {
            'title': 'Technical Analysis Basics',
            'content': '''
            Technical analysis is the study of historical market data, primarily price and volume, to forecast future price movements.
            
            **Key Concepts:**
            - Support and resistance levels
            - Trend analysis (uptrend, downtrend, sideways)
            - Chart patterns (head and shoulders, double tops/bottoms)
            - Technical indicators (RSI, MACD, moving averages)
            - Volume analysis
            '''
        },
        {
            'title': 'Fundamental Analysis',
            'content': '''
            Fundamental analysis involves evaluating a company's financial health and business prospects to determine its intrinsic value.
            
            **Key Concepts:**
            - Financial statements (income statement, balance sheet, cash flow)
            - Valuation metrics (P/E ratio, P/B ratio, PEG ratio)
            - Economic moat and competitive advantage
            - Management quality and corporate governance
            - Industry analysis and market position
            '''
        },
        {
            'title': 'Options Trading',
            'content': '''
            Options are financial derivatives that give buyers the right, but not the obligation, to buy or sell an underlying asset at a specified price on or before a certain date.
            
            **Key Concepts:**
            - Call options (right to buy)
            - Put options (right to sell)
            - Strike price and expiration date
            - Premium (option price)
            - In-the-money, at-the-money, out-of-the-money
            - Basic strategies (covered calls, protective puts)
            '''
        },
        {
            'title': 'Portfolio Management',
            'content': '''
            Portfolio management involves selecting and overseeing a group of investments that meet the long-term financial objectives and risk tolerance of an investor.
            
            **Key Concepts:**
            - Asset allocation and diversification
            - Risk management and mitigation
            - Rebalancing strategies
            - Tax-efficient investing
            - Performance measurement and evaluation
            '''
        },
        {
            'title': 'Risk Management',
            'content': '''
            Risk management is the process of identification, analysis, and acceptance or mitigation of uncertainty in investment decisions.
            
            **Key Concepts:**
            - Types of risk (market, credit, liquidity, inflation)
            - Risk tolerance assessment
            - Position sizing and stop-loss orders
            - Hedging strategies
            - Diversification across assets and sectors
            '''
        },
        {
            'title': 'Cryptocurrency Investing',
            'content': '''
            Cryptocurrency investing involves buying and holding digital assets like Bitcoin and Ethereum with the expectation of long-term appreciation.
            
            **Key Concepts:**
            - Blockchain technology and decentralization
            - Major cryptocurrencies and their use cases
            - Wallets and storage security
            - Volatility and risk factors
            - Regulatory environment and tax implications
            '''
        },
        {
            'title': 'Market Psychology',
            'content': '''
            Market psychology refers to the overall sentiment or feeling that the market is experiencing at any particular time.
            
            **Key Concepts:**
            - Herd behavior and market bubbles
            - Fear and greed cycles
            - Confirmation bias and overconfidence
            - Loss aversion and the disposition effect
            - Contrarian investing strategies
            '''
        }
    ]
    
    for topic in topics:
        with st.expander(topic['title']):
            st.markdown(topic['content'])
    
    # Investment calculators
    st.subheader("Investment Calculators")
    
    calc_type = st.selectbox("Select Calculator:", 
                           ["Compound Interest", "Retirement Planning", "Loan Payment", "Investment Return"])
    
    if calc_type == "Compound Interest":
        st.write("**Compound Interest Calculator**")
        
        principal = st.number_input("Initial Investment ($):", min_value=0, value=10000)
        rate = st.number_input("Annual Interest Rate (%):", min_value=0.0, value=7.0)
        years = st.number_input("Time Period (Years):", min_value=1, value=10)
        contribution = st.number_input("Monthly Contribution ($):", min_value=0, value=100)
        
        if st.button("Calculate"):
            # Calculate compound interest
            rate_decimal = rate / 100
            future_value = principal * (1 + rate_decimal) ** years
            monthly_rate = rate_decimal / 12
            months = years * 12
            
            # Calculate future value of monthly contributions
            if contribution > 0:
                future_value_contributions = contribution * (((1 + monthly_rate) ** months - 1) / monthly_rate)
                future_value += future_value_contributions
            
            st.success(f"Future Value: ${future_value:,.2f}")
    
    elif calc_type == "Retirement Planning":
        st.write("**Retirement Planning Calculator**")
        
        current_age = st.number_input("Current Age:", min_value=18, max_value=100, value=30)
        retirement_age = st.number_input("Desired Retirement Age:", min_value=current_age, max_value=100, value=65)
        current_savings = st.number_input("Current Retirement Savings ($):", min_value=0, value=50000)
        monthly_contribution = st.number_input("Monthly Contribution ($):", min_value=0, value=500)
        expected_return = st.number_input("Expected Annual Return (%):", min_value=0.0, value=7.0)
        
        if st.button("Calculate Retirement Savings"):
            years_to_retirement = retirement_age - current_age
            months_to_retirement = years_to_retirement * 12
            monthly_rate = expected_return / 100 / 12
            
            # Calculate future value of current savings
            future_value_current = current_savings * (1 + monthly_rate) ** months_to_retirement
            
            # Calculate future value of monthly contributions
            future_value_contributions = monthly_contribution * (((1 + monthly_rate) ** months_to_retirement - 1) / monthly_rate)
            
            total_retirement_savings = future_value_current + future_value_contributions
            
            st.success(f"Estimated Retirement Savings: ${total_retirement_savings:,.2f}")
    
    elif calc_type == "Loan Payment":
        st.write("**Loan Payment Calculator**")
        
        loan_amount = st.number_input("Loan Amount ($):", min_value=0, value=25000)
        interest_rate = st.number_input("Annual Interest Rate (%):", min_value=0.0, value=5.0)
        loan_term = st.number_input("Loan Term (Years):", min_value=1, value=5)
        
        if st.button("Calculate Monthly Payment"):
            monthly_rate = interest_rate / 100 / 12
            months = loan_term * 12
            monthly_payment = (loan_amount * monthly_rate) / (1 - (1 + monthly_rate) ** -months)
            
            st.success(f"Monthly Payment: ${monthly_payment:,.2f}")
    
    elif calc_type == "Investment Return":
        st.write("**Investment Return Calculator**")
        
        initial_investment = st.number_input("Initial Investment ($):", min_value=0, value=10000)
        final_value = st.number_input("Final Value ($):", min_value=0, value=15000)
        years = st.number_input("Holding Period (Years):", min_value=1, value=5)
        
        if st.button("Calculate Return"):
            cagr = ((final_value / initial_investment) ** (1 / years)) - 1
            total_return = (final_value - initial_investment) / initial_investment * 100
            
            st.success(f"Total Return: {total_return:.2f}%")
            st.success(f"Compound Annual Growth Rate (CAGR): {cagr * 100:.2f}%")

# Company Info Page
elif selected == "Company Info":
    st.title("🏢 Company Information")
    
    symbol = st.text_input("Enter company symbol:", "AAPL")
    
    if symbol:
        # Fetch data
        hist, info = fetch_stock_data(symbol, "1d")
        income_stmt, balance_sheet, cash_flow = get_financial_statements(symbol)
        recommendations, latest_recommendation = get_analyst_recommendations(symbol)
        earnings_calendar = get_earnings_calendar(symbol)
        insider_transactions = get_insider_transactions(symbol)
        institutional_holders = get_institutional_holders(symbol)
        major_holders = get_major_holders(symbol)
        sustainability = get_sustainability_data(symbol)
        
        if info is not None:
            # Company profile
            st.subheader("Company Profile")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'longName' in info:
                    st.write(f"**Company Name:** {info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Country:** {info.get('country', 'N/A')}")
                st.write(f"**Exchange:** {info.get('exchange', 'N/A')}")
            
            with col2:
                st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
                st.write(f"**Website:** {info.get('website', 'N/A')}")
                st.write(f"**CEO:** {info.get('ceo', 'N/A')}")
                st.write(f"**IPO Year:** {info.get('ipoYear', 'N/A')}")
                st.write(f"**Market Cap:** {format_currency(info.get('marketCap', 0), info.get('currency', 'USD'))}")
            
            if 'longBusinessSummary' in info:
                with st.expander("Business Summary"):
                    st.write(info.get('longBusinessSummary', 'No summary available.'))
            
            # Key statistics
            st.subheader("Key Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("PE Ratio", f"{info.get('trailingPE', 'N/A')}")
                st.metric("Forward PE", f"{info.get('forwardPE', 'N/A')}")
                st.metric("PEG Ratio", f"{info.get('pegRatio', 'N/A')}")
            
            with col2:
                st.metric("EPS", f"{info.get('trailingEps', 'N/A')}")
                st.metric("Revenue Growth", f"{info.get('revenueGrowth', 'N/A')}")
                st.metric("Profit Margins", f"{info.get('profitMargins', 'N/A')}")
            
            with col3:
                st.metric("52 Week High", format_currency(info.get('fiftyTwoWeekHigh', 0), info.get('currency', 'USD')))
                st.metric("52 Week Low", format_currency(info.get('fiftyTwoWeekLow', 0), info.get('currency', 'USD')))
                st.metric("Beta", f"{info.get('beta', 'N/A')}")
            
            # Financial statements
            if income_stmt is not None and not income_stmt.empty:
                st.subheader("Financial Statements")
                
                tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
                
                with tab1:
                    st.dataframe(income_stmt)
                
                with tab2:
                    st.dataframe(balance_sheet)
                
                with tab3:
                    st.dataframe(cash_flow)
            
            # Analyst recommendations
            if recommendations is not None and not recommendations.empty:
                st.subheader("Analyst Recommendations")
                
                st.dataframe(recommendations.tail(10))
				
				                    st.write(f"**Latest Recommendation:** {latest_recommendation['Firm']} - {latest_recommendation['To Grade']} (Price Target: {latest_recommendation.get('priceTarget', 'N/A')})")
            
            # Earnings calendar
            if earnings_calendar is not None and not earnings_calendar.empty:
                st.subheader("Earnings Calendar")
                st.dataframe(earnings_calendar)
            
            # Insider transactions
            if insider_transactions is not None and not insider_transactions.empty:
                st.subheader("Insider Transactions")
                st.dataframe(insider_transactions.tail(10))
            
            # Institutional holders
            if institutional_holders is not None and not institutional_holders.empty:
                st.subheader("Institutional Holders")
                st.dataframe(institutional_holders)
            
            # Major holders
            if major_holders is not None and not major_holders.empty:
                st.subheader("Major Holders")
                st.dataframe(major_holders)
            
            # Sustainability data
            if sustainability is not None and not sustainability.empty:
                st.subheader("Sustainability Data")
                st.dataframe(sustainability)
        else:
            st.error("Could not fetch company information. Please check the symbol and try again.")

# Predictions Page
elif selected == "Predictions":
    st.title("🔮 Stock & Mutual Fund Predictions")
    
    tab1, tab2 = st.tabs(["Stock Predictions", "Mutual Fund Analysis"])
    
    with tab1:
        st.subheader("Stock Price Prediction")
        
        symbol = st.text_input("Enter symbol for prediction:", "AAPL")
        days = st.slider("Prediction days:", min_value=7, max_value=90, value=30)
        method = st.selectbox("Prediction Method:", ["linear", "polynomial", "exponential"])
        
        if st.button("Generate Prediction"):
            with st.spinner("Generating prediction..."):
                future_prices, predicted_price, confidence, method_used = predict_stock_price(symbol, days, method)
                
                if future_prices is not None:
                    hist, info = fetch_stock_data(symbol, "6mo")
                    currency = info.get('currency', 'USD') if info else 'USD'
                    current_price = hist['Close'].iloc[-1] if hist is not None and not hist.empty else 0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", format_currency(current_price, currency))
                    with col2:
                        st.metric("Predicted Price", format_currency(predicted_price, currency))
                    with col3:
                        st.metric("Confidence Level", f"{confidence:.1f}%")
                    with col4:
                        st.metric("Method Used", method_used)
                    
                    # Create prediction chart
                    dates = pd.date_range(start=datetime.now(), periods=len(future_prices), freq='D')
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=dates, y=future_prices, 
                        name='Prediction', line=dict(color='red', dash='dash')
                    ))
                    
                    if hist is not None:
                        fig.add_trace(go.Scatter(
                            x=hist.index, y=hist['Close'], 
                            name='Historical', line=dict(color='white')
                        ))
                    
                    fig.update_layout(
                        title=f"{symbol} Price Prediction ({method_used} method)",
                        xaxis_title="Date",
                        yaxis_title=f"Price ({currency})",
                        template="plotly_dark",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction analysis
                    price_change = predicted_price - current_price
                    price_change_percent = (price_change / current_price) * 100
                    
                    if price_change > 0:
                        st.success(f"Predicted increase of {format_currency(price_change, currency)} ({price_change_percent:.2f}%) over {days} days")
                    else:
                        st.error(f"Predicted decrease of {format_currency(abs(price_change), currency)} ({abs(price_change_percent):.2f}%) over {days} days")
                    
                    st.info("""
                    **Disclaimer:** Predictions are based on historical data and trend analysis. 
                    They are not guaranteed and should not be considered as financial advice. 
                    Always do your own research before making investment decisions.
                    """)
                else:
                    st.error("Could not generate prediction. Please check the symbol and try again.")
    
    with tab2:
        st.subheader("Mutual Fund Analysis")
        
        mutual_funds_df = get_mutual_funds()
        if not mutual_funds_df.empty:
            st.dataframe(mutual_funds_df)
            
            # Mutual fund comparison
            st.subheader("Mutual Fund Comparison")
            selected_funds = st.multiselect("Select funds to compare:", mutual_funds_df['Symbol'].tolist(), default=mutual_funds_df['Symbol'].tolist()[:3])
            
            if selected_funds:
                comparison_data = []
                for symbol in selected_funds:
                    fund_data = mutual_funds_df[mutual_funds_df['Symbol'] == symbol].iloc[0]
                    comparison_data.append({
                        'Symbol': symbol,
                        'Name': fund_data['Name'],
                        'Expense Ratio': fund_data['Expense Ratio'],
                        'YTD Return %': fund_data['YTD Return %'],
                        'Category': fund_data['Category']
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df)
                
                # Performance comparison chart
                fig = go.Figure()
                for symbol in selected_funds:
                    fund_data = mutual_funds_df[mutual_funds_df['Symbol'] == symbol].iloc[0]
                    fig.add_trace(go.Bar(
                        x=[fund_data['Name']],
                        y=[fund_data['YTD Return %']],
                        name=fund_data['Name'],
                        text=[f"{fund_data['YTD Return %']:.2f}%"],
                        textposition='auto'
                    ))
                
                fig.update_layout(
                    title="YTD Return Comparison (%)",
                    xaxis_title="Fund",
                    yaxis_title="YTD Return %",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Expense ratio comparison
                fig2 = go.Figure()
                for symbol in selected_funds:
                    fund_data = mutual_funds_df[mutual_funds_df['Symbol'] == symbol].iloc[0]
                    fig2.add_trace(go.Bar(
                        x=[fund_data['Name']],
                        y=[fund_data['Expense Ratio']],
                        name=fund_data['Name'],
                        text=[f"{fund_data['Expense Ratio']:.2f}%"],
                        textposition='auto'
                    ))
                
                fig2.update_layout(
                    title="Expense Ratio Comparison (%)",
                    xaxis_title="Fund",
                    yaxis_title="Expense Ratio %",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("Could not fetch mutual fund data. Please try again later.")

# Settings Page
elif selected == "Settings":
    st.title("⚙️ Settings")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Appearance", "Data Preferences", "Notifications", "Account"])
    
    with tab1:
        st.subheader("Appearance Settings")
        
        theme = st.selectbox("Color Theme", ["Dark (Red/Black)", "Light", "Blue", "Green"], 
                           index=0 if st.session_state.user_settings.get('theme') == 'dark' else 1)
        font_size = st.slider("Font Size", min_value=12, max_value=24, value=16)
        chart_style = st.selectbox("Chart Style", ["Plotly Dark", "Plotly Light", "ggplot2", "Seaborn"])
        
        if st.button("Save Appearance Settings"):
            st.session_state.user_settings['theme'] = 'dark' if theme == "Dark (Red/Black)" else 'light'
            st.session_state.user_settings['font_size'] = font_size
            st.session_state.user_settings['chart_style'] = chart_style
            st.success("Appearance settings saved successfully!")
    
    with tab2:
        st.subheader("Data Preferences")
        
        auto_refresh = st.checkbox("Auto-refresh data", value=st.session_state.user_settings.get('auto_refresh', True))
        refresh_interval = st.slider("Refresh interval (minutes)", min_value=1, max_value=60, 
                                   value=st.session_state.user_settings.get('refresh_interval', 15))
        default_currency = st.selectbox("Default Currency", ["USD", "INR", "EUR", "GBP", "JPY"], 
                                      index=0 if st.session_state.user_settings.get('default_currency') == 'USD' else 1)
        data_source = st.selectbox("Preferred Data Source", ["Yahoo Finance", "Alpha Vantage", "IEX Cloud"])
        
        if st.button("Save Data Preferences"):
            st.session_state.user_settings['auto_refresh'] = auto_refresh
            st.session_state.user_settings['refresh_interval'] = refresh_interval
            st.session_state.user_settings['default_currency'] = default_currency
            st.session_state.user_settings['data_source'] = data_source
            st.success("Data preferences saved successfully!")
    
    with tab3:
        st.subheader("Notification Settings")
        
        email_notifications = st.checkbox("Email notifications", value=st.session_state.user_settings.get('email_notifications', False))
        price_alerts = st.checkbox("Price alerts", value=st.session_state.user_settings.get('price_alerts', False))
        news_alerts = st.checkbox("News alerts", value=st.session_state.user_settings.get('news_alerts', False))
        earnings_alerts = st.checkbox("Earnings alerts", value=st.session_state.user_settings.get('earnings_alerts', False))
        
        if email_notifications:
            email_address = st.text_input("Email Address", value=st.session_state.user_settings.get('email_address', ''))
            alert_frequency = st.selectbox("Alert Frequency", ["Immediate", "Daily Digest", "Weekly Summary"])
        
        if st.button("Save Notification Settings"):
            st.session_state.user_settings['email_notifications'] = email_notifications
            st.session_state.user_settings['price_alerts'] = price_alerts
            st.session_state.user_settings['news_alerts'] = news_alerts
            st.session_state.user_settings['earnings_alerts'] = earnings_alerts
            
            if email_notifications:
                st.session_state.user_settings['email_address'] = email_address
                st.session_state.user_settings['alert_frequency'] = alert_frequency
            
            st.success("Notification settings saved successfully!")
    
    with tab4:
        st.subheader("Account Settings")
        
        risk_tolerance = st.selectbox("Risk Tolerance", 
                                    ["Low (Conservative)", "Medium (Moderate)", "High (Aggressive)"],
                                    index=1 if st.session_state.user_settings.get('risk_tolerance') == 'medium' else 0)
        
        investment_goal = st.selectbox("Primary Investment Goal", 
                                     ["Capital Preservation", "Income", "Growth", "Speculation"])
        
        investment_horizon = st.selectbox("Investment Horizon", 
                                        ["Short-term (<1 year)", "Medium-term (1-5 years)", "Long-term (>5 years)"])
        
        if st.button("Save Account Settings"):
            st.session_state.user_settings['risk_tolerance'] = risk_tolerance.split(' ')[0].lower()
            st.session_state.user_settings['investment_goal'] = investment_goal
            st.session_state.user_settings['investment_horizon'] = investment_horizon
            st.success("Account settings saved successfully!")
        
        st.subheader("Data Management")
        if st.button("Clear All Cached Data"):
            st.cache_data.clear()
            st.success("All cached data has been cleared!")
        
        if st.button("Export Portfolio Data"):
            # Create a CSV file for download
            csv = st.session_state.portfolio.to_csv(index=False)
            st.download_button(
                label="Download Portfolio CSV",
                data=csv,
                file_name="portfolio_export.csv",
                mime="text/csv"
            )
        
        if st.button("Reset Application"):
            st.session_state.cached_data = {}
            st.session_state.watchlist = []
            st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Quantity', 'Purchase Price', 'Purchase Date', 'Sector', 'Currency'])
            st.session_state.user_settings = {
                'theme': 'dark',
                'auto_refresh': True,
                'refresh_interval': 15,
                'default_currency': 'USD',
                'risk_tolerance': 'medium'
            }
            st.success("Application has been reset to default settings!")

# Add a footer to all pages
st.write("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("<div style='text-align: center; color: #FF0000;'><b>MarketMentor Pro</b><br>Advanced Financial Analytics Platform</div>", 
                unsafe_allow_html=True)
with footer_col2:
    st.markdown("<div style='text-align: center;'><b>Disclaimer:</b><br>Not investment advice. Data may be delayed.</div>", 
                unsafe_allow_html=True)
with footer_col3:
    st.markdown("<div style='text-align: center;'>© 2023 MarketMentor<br>Version 2.1.0</div>", 
                unsafe_allow_html=True)

# LinkedIn profile footer
linkedin_url = "https://www.linkedin.com/in/ashwik-bire-b2a000186"
st.markdown(f"""
<div style="width:100%; background-color:#0D0D0D; border-top:2px solid #FF0000; padding:10px 0; text-align:center; font-family:sans-serif;">
    <a href="{linkedin_url}" target="_blank" style="text-decoration:none; display:inline-flex; align-items:center;">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="30" style="border-radius:50%; margin-right:8px;">
        <span style="color:#0A66C2; font-size:16px; font-weight:600;">Connect on LinkedIn</span>
    </a>
</div>
""", unsafe_allow_html=True)

# Add custom JavaScript for performance
st.markdown("""
<script>
// Performance optimization: Lazy load images
document.addEventListener('DOMContentLoaded', function() {
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.loading = 'lazy';
    });
});

// Auto-refresh functionality
function autoRefresh() {
    setTimeout(function() {
        window.location.reload();
    }, %REFRESH_INTERVAL% * 60 * 1000);
}

if (%AUTO_REFRESH%) {
    autoRefresh();
}
</script>
""".replace("%AUTO_REFRESH%", str(st.session_state.user_settings.get('auto_refresh', True)).lower())
 .replace("%REFRESH_INTERVAL%", str(st.session_state.user_settings.get('refresh_interval', 15))), 
 unsafe_allow_html=True)
                
                
