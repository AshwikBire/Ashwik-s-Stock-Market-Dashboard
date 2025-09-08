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
import ta
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import time
from newsapi import NewsApiClient
import pandas_ta as pta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
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
    
    /* Custom card style */
    .custom-card {
        background-color: #0A0A0A;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FF0000;
        box-shadow: 0 4px 6px rgba(255, 0, 0, 0.1);
    }
    
    /* Custom table style */
    .custom-table {
        background-color: #0A0A0A;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #2A0A0A;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for caching and user data
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = {}
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Quantity', 'Purchase Price', 'Purchase Date', 'Exchange'])
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'default_currency': 'INR',
        'theme': 'dark',
        'notifications': False,
        'auto_refresh': True
    }
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Indian stock symbols mapping
INDIAN_STOCKS = {
    'RELIANCE': 'RELIANCE.NS',
    'TATASTEEL': 'TATASTEEL.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'INFY': 'INFY.NS',
    'TCS': 'TCS.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'SBIN': 'SBIN.NS',
    'HINDUNILVR': 'HINDUNILVR.NS',
    'ITC': 'ITC.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'AXISBANK': 'AXISBANK.NS',
    'ASIANPAINT': 'ASIANPAINT.NS',
    'MARUTI': 'MARUTI.NS',
    'HCLTECH': 'HCLTECH.NS',
    'LT': 'LT.NS',
    'WIPRO': 'WIPRO.NS',
    'ONGC': 'ONGC.NS',
    'SUNPHARMA': 'SUNPHARMA.NS'
}

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
    elif currency == "HKD":
        return f"HK${value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"

# Format large numbers for better readability
def format_large_number(value):
    if pd.isna(value):
        return "N/A"
    
    if value >= 1e12:
        return f"{value/1e12:.2f}T"
    elif value >= 1e9:
        return f"{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{value/1e3:.2f}K"
    else:
        return f"{value:,.2f}"

# Get full symbol with exchange suffix
def get_full_symbol(symbol, exchange="NSE"):
    symbol = symbol.upper().strip()
    
    # Check if symbol already has an exchange suffix
    if '.' in symbol:
        return symbol
    
    # Add exchange suffix based on the exchange
    if exchange == "NSE":
        return symbol + ".NS"
    elif exchange == "BSE":
        return symbol + ".BO"
    else:
        return symbol

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
        "^GSPC": {"name": "S&P 500", "currency": "USD"},
        "^DJI": {"name": "Dow Jones", "currency": "USD"},
        "^IXIC": {"name": "NASDAQ", "currency": "USD"},
        "^NSEI": {"name": "Nifty 50", "currency": "INR"},
        "^BSESN": {"name": "Sensex", "currency": "INR"},
        "^FTSE": {"name": "FTSE 100", "currency": "GBP"},
        "^GDAXI": {"name": "DAX", "currency": "EUR"},
        "^FCHI": {"name": "CAC 40", "currency": "EUR"},
        "^N225": {"name": "Nikkei 225", "currency": "JPY"},
        "^HSI": {"name": "Hang Seng", "currency": "HKD"},
        "GC=F": {"name": "Gold", "currency": "INR"},
        "SI=F": {"name": "Silver", "currency": "INR"},
        "PL=F": {"name": "Platinum", "currency": "INR"},
        "CL=F": {"name": "Crude Oil", "currency": "USD"},
        "NG=F": {"name": "Natural Gas", "currency": "USD"},
        "BTC-USD": {"name": "Bitcoin", "currency": "USD"},
        "ETH-USD": {"name": "Ethereum", "currency": "USD"},
        "^VIX": {"name": "VIX Volatility", "currency": "USD"},
        "^TNX": {"name": "10-Year Treasury", "currency": "USD"}
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
                    "Currency": info["currency"]
                })
        except:
            continue
    
    return pd.DataFrame(data)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(query="stock market", country="in", category="business"):
    try:
        # Using NewsAPI (you need to get your own API key)
        NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"  # This is a demo key, replace with your own
        url = f"https://newsapi.org/v2/top-headlines?country={country}&category={category}&apiKey={NEWS_API_KEY}&pageSize=10"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get("articles", [])
        return []
    except:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_technical_indicators(df):
    if df.empty:
        return df
    
    # Calculate moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate exponential moving averages
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # Calculate MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Calculate Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stochastic_%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stochastic_%D'] = df['Stochastic_%K'].rolling(window=3).mean()
    
    # Calculate Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    return df

# Advanced technical analysis with ta library
def get_advanced_technical_indicators(df):
    if df.empty:
        return df
    
    try:
        # Use ta library for additional indicators
        df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )
    except:
        # Fallback to manual calculation if ta library fails
        pass
    
    return df

# Calculate support and resistance levels
def calculate_support_resistance(df, window=20):
    if df.empty:
        return [], []
    
    supports = []
    resistances = []
    
    for i in range(window, len(df)-window):
        window_high = df['High'].iloc[i-window:i+window]
        window_low = df['Low'].iloc[i-window:i+window]
        
        # Check if current point is a local maximum (resistance)
        if df['High'].iloc[i] == window_high.max():
            resistances.append((df.index[i], df['High'].iloc[i]))
        
        # Check if current point is a local minimum (support)
        if df['Low'].iloc[i] == window_low.min():
            supports.append((df.index[i], df['Low'].iloc[i]))
    
    return supports, resistances

# Prediction functions
@st.cache_data(ttl=3600, show_spinner=False)
def predict_stock_price(ticker, days=30):
    """ Advanced stock price prediction using multiple models """
    try:
        # Get historical data
        hist, _ = fetch_stock_data(ticker, "2y")
        if hist is None or hist.empty:
            return None, None, None, None, None
        
        # Prepare data for prediction
        prices = hist['Close'].values
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train linear regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_mae = mean_absolute_error(y_test, lr_pred)
        
        # Train random forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        
        # Choose the better model
        if lr_mae < rf_mae:
            model = lr_model
            model_type = "Linear Regression"
            mae = lr_mae
        else:
            model = rf_model
            model_type = "Random Forest"
            mae = rf_mae
        
        # Predict future prices
        future_x = np.arange(len(prices), len(prices) + days).reshape(-1, 1)
        future_prices = model.predict(future_x)
        
        # Calculate confidence intervals
        current_price = prices[-1]
        predicted_price = future_prices[-1]
        price_change = predicted_price - current_price
        price_change_percent = (price_change / current_price) * 100
        
        # Calculate confidence based on model performance
        confidence = max(0, min(100, 100 - (mae / current_price * 100)))
        
        return future_prices, predicted_price, confidence, price_change_percent, model_type
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, None, None

# Financial metrics calculation
def calculate_financial_metrics(hist, info):
    if hist.empty or not info:
        return {}
    
    metrics = {}
    
    # Basic metrics
    current_price = hist['Close'].iloc[-1]
    prev_close = info.get('previousClose', current_price)
    change = current_price - prev_close
    change_percent = (change / prev_close) * 100
    
    metrics['current_price'] = current_price
    metrics['prev_close'] = prev_close
    metrics['change'] = change
    metrics['change_percent'] = change_percent
    
    # Volume metrics
    metrics['volume'] = hist['Volume'].iloc[-1]
    metrics['avg_volume'] = hist['Volume'].mean()
    
    # High/low metrics
    metrics['day_high'] = hist['High'].iloc[-1]
    metrics['day_low'] = hist['Low'].iloc[-1]
    metrics['52_week_high'] = info.get('fiftyTwoWeekHigh', hist['High'].max())
    metrics['52_week_low'] = info.get('fiftyTwoWeekLow', hist['Low'].min())
    
    # Valuation metrics
    metrics['pe_ratio'] = info.get('trailingPE', None)
    metrics['forward_pe'] = info.get('forwardPE', None)
    metrics['peg_ratio'] = info.get('pegRatio', None)
    metrics['eps'] = info.get('trailingEps', None)
    metrics['book_value'] = info.get('bookValue', None)
    metrics['price_to_book'] = info.get('priceToBook', None)
    metrics['dividend_yield'] = info.get('dividendYield', None) * 100 if info.get('dividendYield') else None
    metrics['market_cap'] = info.get('marketCap', None)
    metrics['enterprise_value'] = info.get('enterpriseValue', None)
    metrics['ebitda'] = info.get('ebitda', None)
    metrics['profit_margins'] = info.get('profitMargins', None) * 100 if info.get('profitMargins') else None
    
    # Growth metrics
    metrics['revenue_growth'] = info.get('revenueGrowth', None) * 100 if info.get('revenueGrowth') else None
    metrics['earnings_growth'] = info.get('earningsGrowth', None) * 100 if info.get('earningsGrowth') else None
    
    return metrics

# Portfolio analysis functions
def calculate_portfolio_performance():
    if st.session_state.portfolio.empty:
        return None, None, None, None
    
    performance_data = []
    total_investment = 0
    total_current_value = 0
    
    for _, holding in st.session_state.portfolio.iterrows():
        symbol = holding['Symbol']
        quantity = holding['Quantity']
        purchase_price = holding['Purchase Price']
        exchange = holding.get('Exchange', 'NSE')
        
        full_symbol = get_full_symbol(symbol, exchange)
        hist, info = fetch_stock_data(full_symbol, "1d")
        
        if hist is not None and not hist.empty:
            current_price = hist['Close'].iloc[-1]
            currency = info.get('currency', 'USD') if info else 'USD'
            
            investment = purchase_price * quantity
            current_value = current_price * quantity
            gain_loss = current_value - investment
            gain_loss_percent = (gain_loss / investment) * 100
            
            performance_data.append({
                'Symbol': symbol,
                'Quantity': quantity,
                'Purchase Price': purchase_price,
                'Current Price': current_price,
                'Investment': investment,
                'Current Value': current_value,
                'Gain/Loss': gain_loss,
                'Gain/Loss %': gain_loss_percent,
                'Currency': currency
            })
            
            total_investment += investment
            total_current_value += current_value
    
    if not performance_data:
        return None, None, None, None
    
    total_gain_loss = total_current_value - total_investment
    total_gain_loss_percent = (total_gain_loss / total_investment) * 100 if total_investment > 0 else 0
    
    performance_df = pd.DataFrame(performance_data)
    return performance_df, total_investment, total_current_value, total_gain_loss_percent

# Options chain data (simulated as yfinance doesn't provide full options chain)
@st.cache_data(ttl=3600, show_spinner=False)
def get_options_chain(ticker, expiration=None):
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
    except:
        return None, None, None

# Economic calendar data (simulated)
@st.cache_data(ttl=3600, show_spinner=False)
def get_economic_calendar():
    # Simulated economic calendar data
    events = [
        {'date': '2023-12-15', 'country': 'US', 'event': 'Federal Reserve Interest Rate Decision', 'impact': 'High'},
        {'date': '2023-12-20', 'country': 'US', 'event': 'GDP Quarterly Report', 'impact': 'Medium'},
        {'date': '2023-12-22', 'country': 'EU', 'event': 'ECB Monetary Policy Statement', 'impact': 'High'},
        {'date': '2023-12-25', 'country': 'Global', 'event': 'Christmas Holiday', 'impact': 'Low'},
        {'date': '2023-12-28', 'country': 'US', 'event': 'Initial Jobless Claims', 'impact': 'Medium'},
        {'date': '2024-01-05', 'country': 'US', 'event': 'Non-Farm Payrolls', 'impact': 'High'},
        {'date': '2024-01-10', 'country': 'IN', 'event': 'RBI Monetary Policy Meeting', 'impact': 'High'},
        {'date': '2024-01-15', 'country': 'US', 'event': 'Retail Sales Data', 'impact': 'Medium'},
        {'date': '2024-01-20', 'country': 'CN', 'event': 'GDP Growth Rate', 'impact': 'High'},
        {'date': '2024-01-25', 'country': 'JP', 'event': 'Bank of Japan Policy Decision', 'impact': 'High'},
    ]
    
    return pd.DataFrame(events)

# Mutual funds data
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
        'MIRAX': {'name': 'Mirae Asset Emerging Bluechip Fund', 'category': 'Large-Cap', 'expense_ratio': 0.58},
        'AXISBLU': {'name': 'Axis Bluechip Fund', 'category': 'Large-Cap', 'expense_ratio': 0.52},
        'PARAGP': {'name': 'Parag Parikh Flexi Cap Fund', 'category': 'Flexi-Cap', 'expense_ratio': 0.77},
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
                
                data.append({
                    "Symbol": symbol,
                    "Name": info["name"],
                    "Category": info["category"],
                    "Expense Ratio": info["expense_ratio"],
                    "Price": current_price,
                    "Change": change,
                    "Change %": change_percent
                })
        except:
            continue
    
    return pd.DataFrame(data)

# Sidebar navigation
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #FF0000;'>MarketMentor Pro</h1>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Dashboard", "Stock Analysis", "Technical Analysis", "Portfolio Manager", 
                 "Options Chain", "Market Overview", "Economic Calendar", "Crypto Markets", 
                 "News & Sentiment", "Learning Center", "Company Info", "Predictions", "Mutual Funds", "Settings"],
        icons=["house", "graph-up", "bar-chart", "wallet", 
               "diagram-3", "globe", "calendar", "currency-bitcoin",
               "newspaper", "book", "building", "lightbulb", "piggy-bank", "gear"],
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#0A0A0A"},
            "icon": {"color": "#FF0000", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#FFFFFF"},
            "nav-link-selected": {"background-color": "#FF0000", "color": "#000000", "font-weight": "bold"},
        }
    )
    
    # Watchlist section in sidebar
    st.subheader("My Watchlist")
    watchlist_col1, watchlist_col2 = st.columns([3, 1])
    with watchlist_col1:
        watchlist_symbol = st.text_input("Add symbol:", key="watchlist_input")
    with watchlist_col2:
        watchlist_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NASDAQ", "NYSE"], key="watchlist_exchange")
    
    if st.button("Add to Watchlist", key="add_watchlist"):
        if watchlist_symbol:
            full_symbol = get_full_symbol(watchlist_symbol, watchlist_exchange)
            if full_symbol not in st.session_state.watchlist:
                st.session_state.watchlist.append(full_symbol)
                st.success(f"Added {full_symbol} to watchlist")
    
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
    
    # Quick access to popular Indian stocks
    st.subheader("Popular Indian Stocks")
    indian_stocks_cols = st.columns(2)
    for i, (name, symbol) in enumerate(INDIAN_STOCKS.items()):
        with indian_stocks_cols[i % 2]:
            if st.button(name, key=f"indian_{name}"):
                st.session_state.analyze_symbol = symbol
                if selected != "Stock Analysis":
                    st.switch_page("Stock Analysis")

# Dashboard Page
if selected == "Dashboard":
    st.title("📈 Market Dashboard")
    
    # Market overview with global indices
    st.subheader("Global Market Overview")
    indices_df = fetch_global_indices()
    
    if not indices_df.empty:
        cols = st.columns(4)
        for idx, row in indices_df.iterrows():
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
    
    # Latest news
    st.subheader("Latest Market News")
    news_articles = fetch_news()
    
    if news_articles:
        for article in news_articles[:5]:
            with st.expander(f"{article.get('title', 'No title')} - {article.get('source', {}).get('name', 'Unknown')}"):
                if article.get('urlToImage'):
                    st.image(article['urlToImage'], width=300)
                st.write(article.get('description', 'No description available'))
                if article.get('url'):
                    st.markdown(f"[Read more]({article['url']})")
    else:
        st.info("No news available at the moment. Check your internet connection or try again later.")

# Stock Analysis Page
elif selected == "Stock Analysis":
    st.title("📊 Stock Analysis")
    
    # Symbol input with exchange selection
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Enter stock symbol:", value=getattr(st.session_state, 'analyze_symbol', 'RELIANCE.NS'))
    with col2:
        exchange = st.selectbox("Exchange", ["NSE", "BSE", "NASDAQ", "NYSE"], index=0)
    
    full_symbol = get_full_symbol(symbol, exchange)
    
    # Period selection
    period = st.selectbox("Select period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"], index=2)
    
    # Fetch data
    hist, info = fetch_stock_data(full_symbol, period)
    
    if hist is not None and not hist.empty and info is not None:
        # Display stock info
        st.subheader(f"{info.get('longName', full_symbol)} ({full_symbol})")
        
        # Calculate financial metrics
        metrics = calculate_financial_metrics(hist, info)
        currency = info.get('currency', 'USD')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", format_currency(metrics['current_price'], currency), 
                     f"{metrics['change']:.2f} ({metrics['change_percent']:.2f}%)")
        with col2:
            st.metric("Market Cap", format_currency(metrics['market_cap'], currency) if metrics['market_cap'] else "N/A")
        with col3:
            st.metric("PE Ratio", f"{metrics['pe_ratio']:.2f}" if metrics['pe_ratio'] else "N/A")
        with col4:
            st.metric("Volume", f"{metrics['volume']:,.0f}")
        
        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("52W High", format_currency(metrics['52_week_high'], currency))
        with col6:
            st.metric("52W Low", format_currency(metrics['52_week_low'], currency))
        with col7:
            st.metric("Dividend Yield", f"{metrics['dividend_yield']:.2f}%" if metrics['dividend_yield'] else "N/A")
        with col8:
            st.metric("Avg Volume", f"{metrics['avg_volume']:,.0f}")
        
        # Price chart
        st.subheader("Price Chart")
        chart_type = st.radio("Chart Type", ["Line", "Candlestick", "OHLC"], horizontal=True)
        
        fig = go.Figure()
        
        if chart_type == "Line":
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close', line=dict(color='white')))
        elif chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Price'
            ))
        else:  # OHLC
            fig.add_trace(go.Ohlc(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='OHLC'
            ))
        
        fig.update_layout(
            title=f"{full_symbol} Price History",
            xaxis_title="Date",
            yaxis_title=f"Price ({currency})",
            template="plotly_dark",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        st.subheader("Volume")
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume', marker_color='red'))
        fig_volume.update_layout(
            title="Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Display company info if available
        if 'longName' in info:
            st.subheader("Company Information")
            col9, col10 = st.columns(2)
            with col9:
                st.write(f"**Name:** {info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Country:** {info.get('country', 'N/A')}")
            with col10:
                st.write(f"**Website:** {info.get('website', 'N/A')}")
                st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
                st.write(f"**CEO:** {info.get('ceo', 'N/A')}")
            
            if 'longBusinessSummary' in info:
                with st.expander("Business Summary"):
                    st.write(info.get('longBusinessSummary', 'No summary available.'))
        
        # Financial ratios
        st.subheader("Financial Ratios")
        ratios_col1, ratios_col2, ratios_col3 = st.columns(3)
        
        with ratios_col1:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.write("**Valuation Ratios**")
            st.write(f"P/E Ratio: {metrics['pe_ratio']:.2f}" if metrics['pe_ratio'] else "P/E Ratio: N/A")
            st.write(f"Forward P/E: {metrics['forward_pe']:.2f}" if metrics['forward_pe'] else "Forward P/E: N/A")
            st.write(f"PEG Ratio: {metrics['peg_ratio']:.2f}" if metrics['peg_ratio'] else "PEG Ratio: N/A")
            st.write(f"Price to Book: {metrics['price_to_book']:.2f}" if metrics['price_to_book'] else "Price to Book: N/A")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with ratios_col2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.write("**Profitability Ratios**")
            st.write(f"Profit Margin: {metrics['profit_margins']:.2f}%" if metrics['profit_margins'] else "Profit Margin: N/A")
            st.write(f"EPS: {metrics['eps']:.2f}" if metrics['eps'] else "EPS: N/A")
            st.write(f"ROE: {info.get('returnOnEquity', 'N/A')}")
            st.write(f"ROA: {info.get('returnOnAssets', 'N/A')}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with ratios_col3:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.write("**Growth Ratios**")
            st.write(f"Revenue Growth: {metrics['revenue_growth']:.2f}%" if metrics['revenue_growth'] else "Revenue Growth: N/A")
            st.write(f"Earnings Growth: {metrics['earnings_growth']:.2f}%" if metrics['earnings_growth'] else "Earnings Growth: N/A")
            st.write(f"EBITDA: {format_currency(metrics['ebitda'], currency) if metrics['ebitda'] else 'N/A'}")
            st.write(f"Enterprise Value: {format_currency(metrics['enterprise_value'], currency) if metrics['enterprise_value'] else 'N/A'}")
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.error("Could not fetch data for the specified symbol. Please check the symbol and try again.")

# Technical Analysis Page
elif selected == "Technical Analysis":
    st.title("📈 Technical Analysis")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Enter stock symbol:", "RELIANCE.NS")
    with col2:
        exchange = st.selectbox("Exchange", ["NSE", "BSE", "NASDAQ", "NYSE"], index=0)
    
    full_symbol = get_full_symbol(symbol, exchange)
    
    period = st.selectbox("Select period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("Select interval:", ["1d", "1wk", "1h", "30m", "15m", "5m", "1m"], index=0)
    
    if full_symbol:
        hist, info = fetch_stock_data(full_symbol, period, interval)
        
        if hist is not None and not hist.empty:
            currency = info.get('currency', 'USD') if info else 'USD'
            hist = get_technical_indicators(hist)
            
            # Price with moving averages
            st.subheader("Moving Averages")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close', line=dict(color='white')))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name='SMA 20', line=dict(color='yellow')))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], name='SMA 200', line=dict(color='red')))
            fig.update_layout(
                title=f"{full_symbol} Price with Moving Averages",
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
            fig2.add_trace(go.Bar(x=hist.index, y=hist['MACD_Histogram'], name='Histogram', marker_color='red'))
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
                height=400
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
            
            # Stochastic Oscillator
            st.subheader("Stochastic Oscillator")
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=hist.index, y=hist['Stochastic_%K'], name='%K', line=dict(color='white')))
            fig5.add_trace(go.Scatter(x=hist.index, y=hist['Stochastic_%D'], name='%D', line=dict(color='yellow')))
            fig5.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig5.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig5.update_layout(
                title="Stochastic Oscillator",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig5, use_container_width=True)
            
            # Volume indicators
            st.subheader("Volume Indicators")
            fig6 = make_subplots(rows=2, cols=1, subplot_titles=("On-Balance Volume (OBV)", "Volume Weighted Average Price (VWAP)"))
            
            fig6.add_trace(go.Scatter(x=hist.index, y=hist['OBV'], name='OBV', line=dict(color='white')), row=1, col=1)
            fig6.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='yellow')), row=2, col=1)
            fig6.add_trace(go.Scatter(x=hist.index, y=hist['VWAP'], name='VWAP', line=dict(color='red')), row=2, col=1)
            
            fig6.update_layout(
                title="Volume Indicators",
                template="plotly_dark",
                height=600
            )
            st.plotly_chart(fig6, use_container_width=True)
            
            # Technical indicators summary
            st.subheader("Technical Indicators Summary")
            
            # Calculate current values
            current_rsi = hist['RSI'].iloc[-1] if not hist['RSI'].isna().iloc[-1] else 0
            current_macd = hist['MACD'].iloc[-1] if not hist['MACD'].isna().iloc[-1] else 0
            current_macd_signal = hist['MACD_Signal'].iloc[-1] if not hist['MACD_Signal'].isna().iloc[-1] else 0
            current_stochastic_k = hist['Stochastic_%K'].iloc[-1] if not hist['Stochastic_%K'].isna().iloc[-1] else 0
            current_stochastic_d = hist['Stochastic_%D'].iloc[-1] if not hist['Stochastic_%D'].isna().iloc[-1] else 0
            
            # Generate signals
            signals = []
            
            # RSI signal
            if current_rsi > 70:
                signals.append(("RSI", "Bearish", "Overbought"))
            elif current_rsi < 30:
                signals.append(("RSI", "Bullish", "Oversold"))
            else:
                signals.append(("RSI", "Neutral", "In range"))
            
            # MACD signal
            if current_macd > current_macd_signal:
                signals.append(("MACD", "Bullish", "Above signal line"))
            else:
                signals.append(("MACD", "Bearish", "Below signal line"))
            
            # Stochastic signal
            if current_stochastic_k > 80 and current_stochastic_d > 80:
                signals.append(("Stochastic", "Bearish", "Overbought"))
            elif current_stochastic_k < 20 and current_stochastic_d < 20:
                signals.append(("Stochastic", "Bullish", "Oversold"))
            else:
                signals.append(("Stochastic", "Neutral", "In range"))
            
            # Moving averages signal
            if hist['SMA_20'].iloc[-1] > hist['SMA_50'].iloc[-1] > hist['SMA_200'].iloc[-1]:
                signals.append(("Moving Averages", "Bullish", "Uptrend"))
            elif hist['SMA_20'].iloc[-1] < hist['SMA_50'].iloc[-1] < hist['SMA_200'].iloc[-1]:
                signals.append(("Moving Averages", "Bearish", "Downtrend"))
            else:
                signals.append(("Moving Averages", "Neutral", "Sideways"))
            
            # Display signals
            signals_df = pd.DataFrame(signals, columns=["Indicator", "Signal", "Reason"])
            st.dataframe(signals_df)
            
        else:
            st.error("Could not fetch data for the specified symbol. Please check the symbol and try again.")

# Portfolio Manager Page
elif selected == "Portfolio Manager":
    st.title("💼 Portfolio Manager")
    
    tab1, tab2, tab3, tab4 = st.tabs(["View Portfolio", "Add Holding", "Performance Analysis", "Portfolio Optimization"])
    
    with tab1:
        st.subheader("Your Portfolio")
        if st.session_state.portfolio.empty:
            st.info("Your portfolio is empty. Add holdings to get started.")
        else:
            st.dataframe(st.session_state.portfolio)
            
            # Calculate portfolio value
            performance_df, total_investment, total_current_value, total_gain_loss_percent = calculate_portfolio_performance()
            
            if performance_df is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Investment", format_currency(total_investment, "USD"))
                with col2:
                    st.metric("Current Value", format_currency(total_current_value, "USD"))
                with col3:
                    gain_color = "normal" if total_gain_loss_percent >= 0 else "inverse"
                    st.metric("Total Gain/Loss", f"{total_gain_loss_percent:.2f}%", delta_color=gain_color)
    
    with tab2:
        st.subheader("Add New Holding")
        with st.form("add_holding_form"):
            col1, col2 = st.columns(2)
            with col1:
                symbol = st.text_input("Symbol")
            with col2:
                exchange = st.selectbox("Exchange", ["NSE", "BSE", "NASDAQ", "NYSE"], index=0)
            
            quantity = st.number_input("Quantity", min_value=1, value=1)
            purchase_price = st.number_input("Purchase Price", min_value=0.0, value=0.0, step=0.01)
            purchase_date = st.date_input("Purchase Date", value=datetime.now())
            
            if st.form_submit_button("Add Holding"):
                new_holding = pd.DataFrame({
                    'Symbol': [symbol.upper()],
                    'Quantity': [quantity],
                    'Purchase Price': [purchase_price],
                    'Purchase Date': [purchase_date],
                    'Exchange': [exchange]
                })
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_holding], ignore_index=True)
                st.success(f"Added {quantity} shares of {symbol.upper()} to your portfolio")
    
    with tab3:
        st.subheader("Portfolio Performance")
        if not st.session_state.portfolio.empty:
            performance_df, total_investment, total_current_value, total_gain_loss_percent = calculate_portfolio_performance()
            
            if performance_df is not None:
                st.dataframe(performance_df)
                
                # Portfolio allocation chart
                fig = px.pie(performance_df, values='Current Value', names='Symbol', title='Portfolio Allocation')
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance by holding
                fig2 = px.bar(performance_df, x='Symbol', y='Gain/Loss %', title='Performance by Holding (%)',
                             color='Gain/Loss %', color_continuous_scale=['red', 'white', 'green'])
                st.plotly_chart(fig2, use_container_width=True)
    
    with tab4:
        st.subheader("Portfolio Optimization")
        st.info("Portfolio optimization feature is coming soon. Check back in the next update!")

# Options Chain Page
elif selected == "Options Chain":
    st.title("📊 Options Chain Analysis")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Enter stock symbol:", "RELIANCE.NS")
    with col2:
        exchange = st.selectbox("Exchange", ["NSE", "BSE", "NASDAQ", "NYSE"], index=0)
    
    full_symbol = get_full_symbol(symbol, exchange)
    
    if full_symbol:
        calls, puts, options_dates = get_options_chain(full_symbol)
        
        if options_dates:
            expiration = st.selectbox("Select expiration date:", options_dates)
            
            if expiration:
                calls, puts, _ = get_options_chain(full_symbol, expiration)
                
                if calls is not None and puts is not None:
                    st.subheader(f"Options Chain for {full_symbol} - {expiration}")
                    
                    tab1, tab2 = st.tabs(["Call Options", "Put Options"])
                    
                    with tab1:
                        st.dataframe(calls)
                    
                    with tab2:
                        st.dataframe(puts)
                    
                    # Options strategy analyzer
                    st.subheader("Options Strategy Analyzer")
                    strategy = st.selectbox("Select strategy:", ["Long Call", "Short Call", "Long Put", "Short Put", "Straddle", "Strangle", "Bull Call Spread", "Bear Put Spread"])
                    
                    if strategy:
                        st.info(f"{strategy} strategy analysis is coming soon. Check back in the next update!")
                else:
                    st.error("Could not fetch options data. Please check the symbol and try again.")
        else:
            st.error("No options data available for this symbol.")

# Market Overview Page
elif selected == "Market Overview":
    st.title("🌍 Global Market Overview")
    
    indices_df = fetch_global_indices()
    if not indices_df.empty:
        # Display indices as metrics
        st.subheader("Global Indices")
        cols = st.columns(4)
        for idx, row in indices_df.iterrows():
            col_idx = idx % 4
            with cols[col_idx]:
                change_color = "normal" if row["Change"] >= 0 else "inverse"
                st.metric(
                    label=f"{row['Name']} ({row['Symbol']})",
                    value=format_currency(row["Price"], row["Currency"]),
                    delta=f"{row['Change']:.2f} ({row['Change %']:.2f}%)",
                    delta_color=change_color
                )
        
        # Create a bar chart of performance
        st.subheader("Global Indices Performance")
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
            'Energy': -2.3,
            'Utilities': 0.5,
            'Real Estate': -1.1,
            'Communication Services': 3.2,
            'Industrials': 0.8,
            'Materials': -0.3
        }
        
        sector_df = pd.DataFrame(list(sectors.items()), columns=['Sector', 'Change %'])
        fig2 = px.bar(sector_df, x='Sector', y='Change %', title='Sector Performance (%)',
                     color='Change %', color_continuous_scale=['red', 'white', 'green'])
        st.plotly_chart(fig2, use_container_width=True)
        
        # Market sentiment
        st.subheader("Market Sentiment")
        sentiment_col1, sentiment_col2, sentiment_col3 = st.columns(3)
        
        with sentiment_col1:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.write("**Overall Sentiment**")
            st.metric("Bullish", "62%", "5%")
            st.metric("Bearish", "38%", "-5%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with sentiment_col2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.write("**Volatility**")
            st.metric("VIX Index", "18.5", "-1.2")
            st.metric("Fear & Greed", "64 (Greed)", "4")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with sentiment_col3:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.write("**Market Breadth**")
            st.metric("Advancers", "1,245", "125")
            st.metric("Decliners", "856", "-85")
            st.markdown("</div>", unsafe_allow_html=True)

# Economic Calendar Page
elif selected == "Economic Calendar":
    st.title("📅 Economic Calendar")
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now() + timedelta(days=30))
    
    # Country filter
    countries = ['US', 'EU', 'UK', 'JP', 'CN', 'IN', 'AU', 'CA', 'Global']
    selected_countries = st.multiselect("Select Countries", countries, default=['US', 'EU', 'IN'])
    
    # Impact filter
    impact_levels = ['High', 'Medium', 'Low']
    selected_impact = st.multiselect("Select Impact Levels", impact_levels, default=['High', 'Medium'])
    
    # Get economic calendar data
    econ_calendar = get_economic_calendar()
    
    if not econ_calendar.empty:
        # Filter data based on selections
        filtered_calendar = econ_calendar[
            (econ_calendar['date'] >= start_date.strftime('%Y-%m-%d')) &
            (econ_calendar['date'] <= end_date.strftime('%Y-%m-%d')) &
            (econ_calendar['country'].isin(selected_countries)) &
            (econ_calendar['impact'].isin(selected_impact))
        ]
        
        st.dataframe(filtered_calendar)
        
        # Count events by country
        st.subheader("Events by Country")
        country_counts = filtered_calendar['country'].value_counts()
        fig = px.pie(values=country_counts.values, names=country_counts.index, title='Events by Country')
        st.plotly_chart(fig, use_container_width=True)
        
        # Count events by impact
        st.subheader("Events by Impact")
        impact_counts = filtered_calendar['impact'].value_counts()
        fig2 = px.bar(x=impact_counts.index, y=impact_counts.values, title='Events by Impact Level',
                     color=impact_counts.index, color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Economic calendar data is not available at the moment.")

# Crypto Markets Page
elif selected == "Crypto Markets":
    st.title("₿ Crypto Markets")
    
    crypto_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'SOL-USD', 'DOT-USD', 'DOGE-USD', 'SHIB-USD', 'MATIC-USD']
    crypto_data = []
    
    for symbol in crypto_symbols:
        hist, info = fetch_stock_data(symbol, "1d")
        if hist is not None and not hist.empty:
            current_price = hist['Close'].iloc[-1]
            prev_close = info.get('previousClose', current_price) if info else current_price
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            
            market_cap = info.get('marketCap', 0) if info else 0
            volume = info.get('volume', 0) if info else 0
            
            crypto_data.append({
                "Symbol": symbol,
                "Name": symbol.split('-')[0],
                "Price": current_price,
                "Change": change,
                "Change %": change_percent,
                "Market Cap": market_cap,
                "Volume": volume
            })
    
    if crypto_data:
        crypto_df = pd.DataFrame(crypto_data)
        
        # Display crypto metrics
        st.subheader("Cryptocurrency Prices")
        cols = st.columns(5)
        for idx, row in crypto_df.iterrows():
            col_idx = idx % 5
            with cols[col_idx]:
                change_color = "normal" if row["Change"] >= 0 else "inverse"
                st.metric(
                    label=row["Name"],
                    value=f"${row['Price']:,.2f}",
                    delta=f"{row['Change']:.2f} ({row['Change %']:.2f}%)",
                    delta_color=change_color
                )
        
        # Crypto performance chart
        st.subheader("Cryptocurrency Performance")
        fig = px.bar(crypto_df, x='Name', y='Change %', title='Cryptocurrency Performance (%)',
                    color='Change %', color_continuous_scale=['red', 'white', 'green'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Market cap comparison
        st.subheader("Market Capitalization")
        fig2 = px.pie(crypto_df, values='Market Cap', names='Name', title='Cryptocurrency Market Cap Distribution')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Individual crypto analysis
        selected_crypto = st.selectbox("Select cryptocurrency for detailed analysis:", crypto_df['Symbol'].tolist())
        
        if selected_crypto:
            crypto_hist, crypto_info = fetch_stock_data(selected_crypto, "1mo")
            
            if crypto_hist is not None and not crypto_hist.empty:
                st.subheader(f"{selected_crypto} Price Chart")
                fig3 = go.Figure()
                fig3.add_trace(go.Candlestick(
                    x=crypto_hist.index,
                    open=crypto_hist['Open'],
                    high=crypto_hist['High'],
                    low=crypto_hist['Low'],
                    close=crypto_hist['Close'],
                    name='Price'
                ))
                fig3.update_layout(
                    title=f"{selected_crypto} Price History",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig3, use_container_width=True)

# News & Sentiment Page
elif selected == "News & Sentiment":
    st.title("📰 News & Market Sentiment")
    
    # News category selection
    news_category = st.selectbox("Select News Category", ["business", "technology", "general", "health", "science", "sports", "entertainment"])
    
    # Country selection
    news_country = st.selectbox("Select Country", ["in", "us", "gb", "ca", "au", "de", "fr", "jp", "cn"])
    
    # Fetch news
    news_articles = fetch_news(query="stock market", country=news_country, category=news_category)
    
    if news_articles:
        st.subheader("Latest News")
        
        for article in news_articles:
            with st.expander(f"{article.get('title', 'No title')} - {article.get('source', {}).get('name', 'Unknown')}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if article.get('urlToImage'):
                        st.image(article['urlToImage'], width=200)
                
                with col2:
                    st.write(f"**Published at:** {article.get('publishedAt', 'Unknown')}")
                    st.write(article.get('description', 'No description available'))
                    
                    if article.get('url'):
                        st.markdown(f"[Read full article]({article['url']})")
        
        # Sentiment analysis (simulated)
        st.subheader("Market Sentiment Analysis")
        
        sentiment_col1, sentiment_col2, sentiment_col3 = st.columns(3)
        
        with sentiment_col1:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.write("**Overall Sentiment**")
            st.metric("Bullish", "65%", "3%")
            st.metric("Bearish", "35%", "-3%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with sentiment_col2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.write("**Sector Sentiment**")
            st.write("Technology: 72% Bullish")
            st.write("Healthcare: 58% Bullish")
            st.write("Financials: 45% Bullish")
            st.write("Energy: 38% Bullish")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with sentiment_col3:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.write("**Social Media Sentiment**")
            st.metric("Twitter", "58% Positive", "5%")
            st.metric("Reddit", "62% Positive", "3%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Word cloud (simulated)
        st.subheader("Trending Topics")
        topics = ["AI Stocks", "Interest Rates", "Earnings Season", "Market Rally", "IPO News", "Crypto Regulation", "Global Economy", "Inflation Data"]
        
        for topic in topics:
            st.write(f"- {topic}")
    
    else:
        st.info("No news available at the moment. Check your internet connection or try again later.")

# Learning Center Page
elif selected == "Learning Center":
    st.title("📚 Learning Center")
    
    st.subheader("Educational Resources")
    
    topics = [
        {
            "title": "Introduction to Stock Market",
            "content": """
            The stock market is a platform where investors buy and sell shares of publicly traded companies. 
            It serves as a barometer for the overall economy and provides opportunities for wealth creation.
            
            **Key Concepts:**
            - Stocks represent ownership in a company
            - Stock exchanges facilitate trading (NSE, BSE, NYSE, NASDAQ)
            - Indices track market performance (Nifty 50, Sensex, S&P 500)
            - Bull and bear markets indicate rising and falling trends
            """,
            "level": "Beginner"
        },
        {
            "title": "Technical Analysis Basics",
            "content": """
            Technical analysis involves studying historical price and volume data to predict future market movements.
            
            **Key Concepts:**
            - Support and resistance levels
            - Trend lines and chart patterns
            - Technical indicators (RSI, MACD, Moving Averages)
            - Volume analysis
            - Candlestick patterns
            """,
            "level": "Intermediate"
        },
        {
            "title": "Fundamental Analysis",
            "content": """
            Fundamental analysis evaluates a company's financial health and intrinsic value by examining economic, financial, and qualitative factors.
            
            **Key Concepts:**
            - Financial statements (Income Statement, Balance Sheet, Cash Flow)
            - Valuation ratios (P/E, P/B, PEG)
            - Profitability ratios (ROE, ROA, Profit Margin)
            - Economic moat and competitive advantage
            - Industry analysis
            """,
            "level": "Intermediate"
        },
        {
            "title": "Options Trading",
            "content": """
            Options are financial derivatives that give buyers the right, but not the obligation, to buy or sell an asset at a specified price on or before a certain date.
            
            **Key Concepts:**
            - Call and put options
            - Strike price and expiration date
            - In-the-money, at-the-money, out-of-the-money
            - Options strategies (covered calls, protective puts, spreads)
            - Implied volatility and Greek letters
            """,
            "level": "Advanced"
        },
        {
            "title": "Portfolio Management",
            "content": """
            Portfolio management involves selecting and managing investments to meet specific financial goals while balancing risk and return.
            
            **Key Concepts:**
            - Asset allocation and diversification
            - Risk management and hedging
            - Modern Portfolio Theory
            - Active vs. passive management
            - Rebalancing strategies
            """,
            "level": "Intermediate"
        },
        {
            "title": "Risk Management",
            "content": """
            Risk management is the process of identifying, assessing, and controlling threats to an investment portfolio.
            
            **Key Concepts:**
            - Types of risk (market, credit, liquidity, inflation)
            - Risk tolerance assessment
            - Position sizing
            - Stop-loss orders
            - Hedging strategies
            """,
            "level": "Intermediate"
        },
        {
            "title": "Cryptocurrency Investing",
            "content": """
            Cryptocurrency investing involves buying and holding digital assets like Bitcoin and Ethereum as investments.
            
            **Key Concepts:**
            - Blockchain technology
            - Different types of cryptocurrencies
            - Wallets and exchanges
            - Volatility and risk factors
            - Regulatory environment
            """,
            "level": "Intermediate"
        },
        {
            "title": "Market Psychology",
            "content": """
            Market psychology studies how collective investor behavior influences financial markets.
            
            **Key Concepts:**
            - Herd mentality and market bubbles
            - Fear and greed indicators
            - Behavioral finance biases
            - Contrarian investing
            - Emotional discipline in trading
            """,
            "level": "Advanced"
        }
    ]
    
    for topic in topics:
        with st.expander(f"{topic['title']} ({topic['level']})"):
            st.write(topic['content'])
            if st.button("Take Quiz", key=f"quiz_{topic['title']}"):
                st.info("Quiz feature is coming soon. Check back in the next update!")

# Company Info Page
elif selected == "Company Info":
    st.title("🏢 Company Information")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Enter company symbol:", "RELIANCE.NS")
    with col2:
        exchange = st.selectbox("Exchange", ["NSE", "BSE", "NASDAQ", "NYSE"], index=0)
    
    full_symbol = get_full_symbol(symbol, exchange)
    
    if full_symbol:
        hist, info = fetch_stock_data(full_symbol, "1d")
        
        if info is not None:
            st.subheader("Company Profile")
            col3, col4 = st.columns(2)
            
            with col3:
                st.write(f"**Company Name:** {info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Country:** {info.get('country', 'N/A')}")
                st.write(f"**Exchange:** {info.get('exchange', 'N/A')}")
            
            with col4:
                st.write(f"**Website:** {info.get('website', 'N/A')}")
                st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
                st.write(f"**CEO:** {info.get('ceo', 'N/A')}")
                st.write(f"**Founded:** {info.get('founded', 'N/A')}")
            
            if 'longBusinessSummary' in info:
                with st.expander("Business Summary"):
                    st.write(info.get('longBusinessSummary', 'No summary available.'))
            
            # Key executives
            if 'companyOfficers' in info and info['companyOfficers']:
                st.subheader("Key Executives")
                executives = info['companyOfficers'][:5]  # Show top 5 executives
                exec_data = []
                for exec in executives:
                    exec_data.append({
                        'Name': exec.get('name', 'N/A'),
                        'Title': exec.get('title', 'N/A'),
                        'Age': exec.get('age', 'N/A'),
                        'Total Pay': format_currency(exec.get('totalPay', 'N/A'), 'USD') if exec.get('totalPay') else 'N/A'
                    })
                st.table(pd.DataFrame(exec_data))
            
            # Financial highlights
            st.subheader("Financial Highlights")
            financials_col1, financials_col2, financials_col3 = st.columns(3)
            
            with financials_col1:
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.write("**Valuation**")
                st.write(f"Market Cap: {format_currency(info.get('marketCap', 'N/A'), info.get('currency', 'USD'))}")
                st.write(f"Enterprise Value: {format_currency(info.get('enterpriseValue', 'N/A'), info.get('currency', 'USD'))}")
                st.write(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
                st.write(f"P/B Ratio: {info.get('priceToBook', 'N/A')}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with financials_col2:
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.write("**Profitability**")
                st.write(f"ROE: {info.get('returnOnEquity', 'N/A')}")
                st.write(f"ROA: {info.get('returnOnAssets', 'N/A')}")
                st.write(f"Profit Margin: {info.get('profitMargins', 'N/A')}")
                st.write(f"EBITDA: {format_currency(info.get('ebitda', 'N/A'), info.get('currency', 'USD'))}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with financials_col3:
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.write("**Growth**")
                st.write(f"Revenue Growth: {info.get('revenueGrowth', 'N/A')}")
                st.write(f"Earnings Growth: {info.get('earningsGrowth', 'N/A')}")
                st.write(f"Quarterly Revenue Growth: {info.get('quarterlyRevenueGrowth', 'N/A')}")
                st.write(f"Quarterly Earnings Growth: {info.get('quarterlyEarningsGrowth', 'N/A')}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # ESG scores (if available)
            if 'esgScores' in info:
                st.subheader("ESG Scores")
                esg = info['esgScores']
                esg_col1, esg_col2, esg_col3 = st.columns(3)
                
                with esg_col1:
                    st.metric("Environment Score", f"{esg.get('environmentScore', 'N/A')}")
                with esg_col2:
                    st.metric("Social Score", f"{esg.get('socialScore', 'N/A')}")
                with esg_col3:
                    st.metric("Governance Score", f"{esg.get('governanceScore', 'N/A')}")
                
                st.metric("Total ESG Score", f"{esg.get('totalEsg', 'N/A')}")
        else:
            st.error("Could not fetch company information. Please check the symbol and try again.")

# Predictions Page
elif selected == "Predictions":
    st.title("🔮 Stock & Mutual Fund Predictions")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Enter symbol for prediction:", "RELIANCE.NS")
    with col2:
        exchange = st.selectbox("Exchange", ["NSE", "BSE", "NASDAQ", "NYSE"], index=0)
    
    full_symbol = get_full_symbol(symbol, exchange)
    
    days = st.slider("Prediction days:", min_value=7, max_value=90, value=30)
    model_type = st.selectbox("Select prediction model:", ["Linear Regression", "Random Forest", "Auto Select"])
    
    if st.button("Generate Prediction"):
        with st.spinner("Generating prediction..."):
            future_prices, predicted_price, confidence, price_change_percent, selected_model = predict_stock_price(full_symbol, days)
            
            if future_prices is not None:
                hist, info = fetch_stock_data(full_symbol, "6mo")
                currency = info.get('currency', 'USD') if info else 'USD'
                current_price = hist['Close'].iloc[-1] if hist is not None and not hist.empty else 0
                
                col3, col4, col5, col6 = st.columns(4)
                with col3:
                    st.metric("Current Price", format_currency(current_price, currency))
                with col4:
                    st.metric("Predicted Price", format_currency(predicted_price, currency))
                with col5:
                    st.metric("Expected Change", f"{price_change_percent:.2f}%")
                with col6:
                    st.metric("Confidence Level", f"{confidence:.1f}%")
                
                st.write(f"**Model used:** {selected_model}")
                
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
                    title=f"{full_symbol} Price Prediction",
                    xaxis_title="Date",
                    yaxis_title=f"Price ({currency})",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction analysis
                st.subheader("Prediction Analysis")
                
                if price_change_percent > 5:
                    st.success(f"Strong bullish signal predicted. The model expects {full_symbol} to increase by {price_change_percent:.2f}% in the next {days} days.")
                elif price_change_percent > 0:
                    st.info(f"Mild bullish signal predicted. The model expects {full_symbol} to increase by {price_change_percent:.2f}% in the next {days} days.")
                elif price_change_percent > -5:
                    st.warning(f"Mild bearish signal predicted. The model expects {full_symbol} to decrease by {abs(price_change_percent):.2f}% in the next {days} days.")
                else:
                    st.error(f"Strong bearish signal predicted. The model expects {full_symbol} to decrease by {abs(price_change_percent):.2f}% in the next {days} days.")
                
                st.info("""
                **Disclaimer:** Predictions are based on historical data and trend analysis. 
                They are not guaranteed and should not be considered as financial advice. 
                Always do your own research before making investment decisions.
                """)
            else:
                st.error("Could not generate prediction. Please check the symbol and try again.")

# Mutual Funds Page
elif selected == "Mutual Funds":
    st.title("📊 Mutual Funds Analysis")
    
    mutual_funds_df = get_mutual_funds()
    
    if not mutual_funds_df.empty:
        st.subheader("Popular Mutual Funds")
        
        # Display mutual funds as metrics
        cols = st.columns(4)
        for idx, row in mutual_funds_df.iterrows():
            col_idx = idx % 4
            with cols[col_idx]:
                change_color = "normal" if row["Change"] >= 0 else "inverse"
                st.metric(
                    label=row["Name"],
                    value=f"${row['Price']:.2f}",
                    delta=f"{row['Change']:.2f} ({row['Change %']:.2f}%)",
                    delta_color=change_color
                )
        
        # Mutual funds performance chart
        st.subheader("Mutual Funds Performance")
        fig = px.bar(mutual_funds_df, x='Name', y='Change %', title='Mutual Funds Performance (%)',
                    color='Change %', color_continuous_scale=['red', 'white', 'green'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Expense ratio comparison
        st.subheader("Expense Ratio Comparison")
        fig2 = px.bar(mutual_funds_df, x='Name', y='Expense Ratio', title='Expense Ratios (%)',
                     color='Expense Ratio', color_continuous_scale=['green', 'yellow', 'red'])
        st.plotly_chart(fig2, use_container_width=True)
        
        # Category analysis
        st.subheader("Performance by Category")
        category_avg = mutual_funds_df.groupby('Category')['Change %'].mean().reset_index()
        fig3 = px.bar(category_avg, x='Category', y='Change %', title='Average Performance by Category (%)')
        st.plotly_chart(fig3, use_container_width=True)
        
        # Individual fund analysis
        selected_fund = st.selectbox("Select mutual fund for detailed analysis:", mutual_funds_df['Symbol'].tolist())
        
        if selected_fund:
            fund_hist, fund_info = fetch_stock_data(selected_fund, "1y")
            
            if fund_hist is not None and not fund_hist.empty:
                st.subheader(f"{selected_fund} Performance")
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=fund_hist.index, y=fund_hist['Close'], 
                    name='NAV', line=dict(color='white')
                ))
                fig4.update_layout(
                    title=f"{selected_fund} NAV History",
                    xaxis_title="Date",
                    yaxis_title="Net Asset Value (USD)",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig4, use_container_width=True)
                
                # Fund details
                st.subheader("Fund Details")
                fund_details = mutual_funds_df[mutual_funds_df['Symbol'] == selected_fund].iloc[0]
                
                col7, col8 = st.columns(2)
                with col7:
                    st.write(f"**Name:** {fund_details['Name']}")
                    st.write(f"**Category:** {fund_details['Category']}")
                    st.write(f"**Expense Ratio:** {fund_details['Expense Ratio']}%")
                with col8:
                    st.write(f"**Current NAV:** ${fund_details['Price']:.2f}")
                    st.write(f"**Daily Change:** {fund_details['Change']:.2f} ({fund_details['Change %']:.2f}%)")
    else:
        st.error("Could not fetch mutual funds data. Please try again later.")

# Settings Page
elif selected == "Settings":
    st.title("⚙️ Settings")
    
    st.subheader("Appearance")
    theme = st.selectbox("Color Theme", ["Dark (Red/Black)", "Light", "Blue", "Green"])
    st.info(f"Selected theme: {theme}. Note: Theme changes require app restart to take effect.")
    
    st.subheader("Data Preferences")
    auto_refresh = st.checkbox("Auto-refresh data", value=st.session_state.user_preferences['auto_refresh'])
    refresh_interval = st.slider("Refresh interval (minutes)", min_value=1, max_value=60, value=15)
    default_currency = st.selectbox("Default Currency", ["INR", "USD", "EUR", "GBP", "JPY"], 
                                   index=["INR", "USD", "EUR", "GBP", "JPY"].index(st.session_state.user_preferences['default_currency']))
    
    st.subheader("Notifications")
    email_notifications = st.checkbox("Email notifications", value=st.session_state.user_preferences['notifications'])
    price_alerts = st.checkbox("Price alerts", value=False)
    
    st.subheader("Data Management")
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared successfully!")
    
    if st.button("Reset to Defaults"):
        st.session_state.user_preferences = {
            'default_currency': 'INR',
            'theme': 'dark',
            'notifications': False,
            'auto_refresh': True
        }
        st.success("Settings reset to defaults!")
    
    if st.button("Save Settings"):
        st.session_state.user_preferences = {
            'default_currency': default_currency,
            'theme': theme,
            'notifications': email_notifications,
            'auto_refresh': auto_refresh
        }
        st.success("Settings saved successfully!")

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
    st.markdown("<div style='text-align: center;'>© 2023 MarketMentor<br>Version 3.0.0</div>", 
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
</script>
""", unsafe_allow_html=True)
