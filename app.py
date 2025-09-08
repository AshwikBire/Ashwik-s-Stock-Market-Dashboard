import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from plotly import graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
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
</style>
""", unsafe_allow_html=True)

# Initialize session state for caching and user data
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = {}
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Quantity', 'Purchase Price', 'Purchase Date', 'Currency'])
if 'selected_currency' not in st.session_state:
    st.session_state.selected_currency = 'INR'
if 'selected_theme' not in st.session_state:
    st.session_state.selected_theme = 'Dark'

# Currency conversion rates (simplified - in a real app, use an API)
CURRENCY_RATES = {
    'USD': 1.0,
    'INR': 83.0,
    'EUR': 0.93,
    'GBP': 0.80,
    'JPY': 150.0
}

# Optimized data fetching with caching
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker, period="1mo"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except:
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
        "GC=F": {"name": "Gold", "currency": "USD"},
        "SI=F": {"name": "Silver", "currency": "USD"},
        "PL=F": {"name": "Platinum", "currency": "USD"},
        "CL=F": {"name": "Crude Oil", "currency": "USD"},
        "NG=F": {"name": "Natural Gas", "currency": "USD"},
        "BTC-USD": {"name": "Bitcoin", "currency": "USD"},
        "ETH-USD": {"name": "Ethereum", "currency": "USD"}
    }
    
    data = {}
    for symbol, info in indices.items():
        try:
            stock_data = yf.Ticker(symbol)
            hist = stock_data.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                previous_close = stock_data.info.get('previousClose', current_price)
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100
                
                data[symbol] = {
                    "name": info["name"],
                    "price": current_price,
                    "change": change,
                    "change_percent": change_percent,
                    "currency": info["currency"]
                }
        except:
            continue
    
    return data

@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(query="stock market"):
    try:
        NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"  # Replace with your actual API key
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&pageSize=5"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json().get("articles", [])
        return []
    except:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_technical_indicators(df):
    # Calculate simple technical indicators without external dependencies
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
    df['RSI'] = compute_rsi(df['Close'])
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def compute_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            up_val = delta
            down_val = 0.
        else:
            up_val = 0.
            down_val = -delta

        up = (up*(period-1) + up_val)/period
        down = (down*(period-1) + down_val)/period
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi

# Prediction functions
@st.cache_data(ttl=3600, show_spinner=False)
def predict_stock_price(ticker, days=30):
    """Simple stock price prediction using historical data and trend analysis"""
    try:
        # Get historical data
        hist, _ = fetch_stock_data(ticker, "1y")
        if hist is None or hist.empty:
            return None, None, None
        
        # Simple linear regression for prediction
        prices = hist['Close'].values
        x = np.arange(len(prices))
        
        # Fit a polynomial (degree 2) to the data
        coefficients = np.polyfit(x, prices, 2)
        polynomial = np.poly1d(coefficients)
        
        # Predict future values
        future_x = np.arange(len(prices) + days)
        future_prices = polynomial(future_x)
        
        current_price = prices[-1]
        predicted_price = future_prices[-1]
        price_change = predicted_price - current_price
        percent_change = (price_change / current_price) * 100
        
        return current_price, predicted_price, percent_change
    except:
        return None, None, None

@st.cache_data(ttl=3600, show_spinner=False)
def get_mutual_funds():
    """Get a list of popular mutual funds with their performance data"""
    mutual_funds = {
        'VFIAX': {'name': 'Vanguard 500 Index Fund', 'category': 'Large Blend', 'expense_ratio': 0.04},
        'VTSAX': {'name': 'Vanguard Total Stock Market Index Fund', 'category': 'Large Blend', 'expense_ratio': 0.04},
        'VGSLX': {'name': 'Vanguard Real Estate Index Fund', 'category': 'Real Estate', 'expense_ratio': 0.12},
        'VIMAX': {'name': 'Vanguard Mid-Cap Index Fund', 'category': 'Mid-Cap Blend', 'expense_ratio': 0.05},
        'VSMAX': {'name': 'Vanguard Small-Cap Index Fund', 'category': 'Small-Cap Blend', 'expense_ratio': 0.05},
        'VTIAX': {'name': 'Vanguard Total International Stock Index Fund', 'category': 'International', 'expense_ratio': 0.11},
        'VBTLX': {'name': 'Vanguard Total Bond Market Index Fund', 'category': 'Intermediate-Term Bond', 'expense_ratio': 0.05},
    }
    
    # Add performance data
    for symbol in mutual_funds:
        try:
            fund_data = yf.Ticker(symbol)
            hist = fund_data.history(period="1y")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                year_high = hist['High'].max()
                year_low = hist['Low'].min()
                
                mutual_funds[symbol]['current_price'] = current_price
                mutual_funds[symbol]['year_high'] = year_high
                mutual_funds[symbol]['year_low'] = year_low
                mutual_funds[symbol]['ytd_return'] = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
        except:
            continue
    
    return mutual_funds

@st.cache_data(ttl=3600, show_spinner=False)
def get_indian_stocks():
    """Get a list of popular Indian stocks"""
    indian_stocks = {
        'RELIANCE.NS': {'name': 'Reliance Industries', 'sector': 'Conglomerate'},
        'TCS.NS': {'name': 'Tata Consultancy Services', 'sector': 'IT Services'},
        'HDFCBANK.NS': {'name': 'HDFC Bank', 'sector': 'Banking'},
        'INFY.NS': {'name': 'Infosys', 'sector': 'IT Services'},
        'ICICIBANK.NS': {'name': 'ICICI Bank', 'sector': 'Banking'},
        'HINDUNILVR.NS': {'name': 'Hindustan Unilever', 'sector': 'FMCG'},
        'SBIN.NS': {'name': 'State Bank of India', 'sector': 'Banking'},
        'BHARTIARTL.NS': {'name': 'Bharti Airtel', 'sector': 'Telecom'},
        'ITC.NS': {'name': 'ITC Limited', 'sector': 'Conglomerate'},
        'KOTAKBANK.NS': {'name': 'Kotak Mahindra Bank', 'sector': 'Banking'},
        'BAJFINANCE.NS': {'name': 'Bajaj Finance', 'sector': 'Financial Services'},
        'HCLTECH.NS': {'name': 'HCL Technologies', 'sector': 'IT Services'},
        'AXISBANK.NS': {'name': 'Axis Bank', 'sector': 'Banking'},
        'ASIANPAINT.NS': {'name': 'Asian Paints', 'sector': 'Paints'},
        'MARUTI.NS': {'name': 'Maruti Suzuki', 'sector': 'Automobile'},
        'TITAN.NS': {'name': 'Titan Company', 'sector': 'Retail'},
        'NTPC.NS': {'name': 'NTPC Limited', 'sector': 'Power'},
        'ONGC.NS': {'name': 'Oil and Natural Gas Corporation', 'sector': 'Oil & Gas'},
        'SUNPHARMA.NS': {'name': 'Sun Pharmaceutical', 'sector': 'Pharmaceuticals'},
        'WIPRO.NS': {'name': 'Wipro', 'sector': 'IT Services'}
    }
    return indian_stocks

@st.cache_data(ttl=3600, show_spinner=False)
def get_sector_performance():
    """Get sector performance data"""
    sectors = {
        'Technology': {'change': 2.5, 'top_stocks': ['AAPL', 'MSFT', 'NVDA']},
        'Healthcare': {'change': 1.8, 'top_stocks': ['JNJ', 'PFE', 'UNH']},
        'Financial Services': {'change': 1.2, 'top_stocks': ['JPM', 'V', 'MA']},
        'Consumer Cyclical': {'change': 0.9, 'top_stocks': ['AMZN', 'TSLA', 'HD']},
        'Energy': {'change': -0.5, 'top_stocks': ['XOM', 'CVX', 'COP']},
        'Real Estate': {'change': 1.5, 'top_stocks': ['AMT', 'PLD', 'EQIX']},
        'Utilities': {'change': 0.3, 'top_stocks': ['NEE', 'DUK', 'SO']},
        'Industrials': {'change': 1.1, 'top_stocks': ['HON', 'UPS', 'CAT']}
    }
    return sectors

def format_currency(value, currency='INR'):
    """Format currency based on selected currency"""
    if currency == 'INR':
        return f"₹{value:,.2f}"
    elif currency == 'USD':
        return f"${value:,.2f}"
    elif currency == 'EUR':
        return f"€{value:,.2f}"
    elif currency == 'GBP':
        return f"£{value:,.2f}"
    elif currency == 'JPY':
        return f"¥{value:,.0f}"
    else:
        return f"{value:,.2f}"

def convert_currency(amount, from_currency, to_currency):
    """Convert currency based on predefined rates"""
    if from_currency == to_currency:
        return amount
    return amount * (CURRENCY_RATES[to_currency] / CURRENCY_RATES[from_currency])

# Sidebar navigation
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #FF0000;'>MarketMentor Pro</h1>", unsafe_allow_html=True)
    
    # Currency selector
    st.session_state.selected_currency = st.selectbox(
        "Select Currency",
        ["INR", "USD", "EUR", "GBP", "JPY"],
        index=0
    )
    
    # Theme selector
    st.session_state.selected_theme = st.selectbox(
        "Select Theme",
        ["Dark", "Light", "System"],
        index=0
    )
    
    # Navigation menu
    selected = option_menu(
        menu_title="Navigation",
        options=["Dashboard", "Stock Analysis", "Technical Analysis", "Portfolio Manager", 
                "Options Chain", "Market Overview", "Economic Calendar", "Crypto Markets", 
                "News & Sentiment", "Learning Center", "Company Info", "Predictions", "Settings"],
        icons=["house", "graph-up", "bar-chart", "wallet", 
               "diagram-3", "globe", "calendar", "currency-bitcoin",
               "newspaper", "book", "building", "lightbulb", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#0A0A0A"},
            "icon": {"color": "#FF0000", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#FFFFFF"},
            "nav-link-selected": {"background-color": "#FF0000", "color": "#000000"},
        }
    )
    
    # Watchlist section in sidebar
    st.subheader("📋 Watchlist")
    watchlist_symbol = st.text_input("Add symbol to watchlist", "AAPL")
    if st.button("Add to Watchlist"):
        if watchlist_symbol not in st.session_state.watchlist:
            st.session_state.watchlist.append(watchlist_symbol)
            st.success(f"Added {watchlist_symbol} to watchlist")
    
    if st.session_state.watchlist:
        st.write("Your Watchlist:")
        for symbol in st.session_state.watchlist:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(symbol)
            with col2:
                if st.button("X", key=f"remove_{symbol}"):
                    st.session_state.watchlist.remove(symbol)
                    st.rerun()
    else:
        st.info("Your watchlist is empty. Add symbols to track.")

# Dashboard Page
if selected == "Dashboard":
    st.title("Market Dashboard")
    
    # Display market indices
    st.subheader("🌍 Global Market Indices")
    indices_data = fetch_global_indices()
    
    if indices_data:
        cols = st.columns(4)
        index_count = 0
        for symbol, data in indices_data.items():
            with cols[index_count % 4]:
                change_color = "#00FF00" if data['change'] >= 0 else "#FF0000"
                change_icon = "📈" if data['change'] >= 0 else "📉"
                
                # Convert to selected currency if needed
                display_price = data['price']
                if data['currency'] != st.session_state.selected_currency:
                    display_price = convert_currency(data['price'], data['currency'], st.session_state.selected_currency)
                
                st.metric(
                    label=f"{data['name']} ({symbol})",
                    value=format_currency(display_price, st.session_state.selected_currency),
                    delta=f"{data['change_percent']:.2f}%",
                    delta_color="normal" if data['change'] >= 0 else "inverse"
                )
            index_count += 1
    
    # Display watchlist if available
    if st.session_state.watchlist:
        st.subheader("⭐ Your Watchlist")
        watchlist_cols = st.columns(4)
        for i, symbol in enumerate(st.session_state.watchlist):
            with watchlist_cols[i % 4]:
                try:
                    stock_data = yf.Ticker(symbol)
                    hist = stock_data.history(period="1d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        info = stock_data.info
                        previous_close = info.get('previousClose', current_price)
                        change = current_price - previous_close
                        change_percent = (change / previous_close) * 100
                        
                        # Get currency from info or default to USD
                        currency = info.get('currency', 'USD')
                        if currency == 'INR':
                            currency = 'INR'
                        
                        # Convert to selected currency if needed
                        display_price = current_price
                        if currency != st.session_state.selected_currency:
                            display_price = convert_currency(current_price, currency, st.session_state.selected_currency)
                        
                        st.metric(
                            label=symbol,
                            value=format_currency(display_price, st.session_state.selected_currency),
                            delta=f"{change_percent:.2f}%",
                            delta_color="normal" if change >= 0 else "inverse"
                        )
                except:
                    st.error(f"Error fetching data for {symbol}")
    
    # Display recent news
    st.subheader("📰 Latest Financial News")
    news_data = fetch_news()
    
    if news_data:
        for i, article in enumerate(news_data[:3]):
            with st.expander(f"{article['title']}"):
                st.write(f"**Source:** {article['source']['name']}")
                st.write(f"**Published:** {article['publishedAt'][:10]}")
                st.write(article['description'])
                if article['url']:
                    st.markdown(f"[Read more]({article['url']})")
    else:
        st.info("No news available at the moment.")

# Stock Analysis Page
elif selected == "Stock Analysis":
    st.title("Stock Analysis")
    
    # Stock selection
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_symbol = st.text_input("Enter stock symbol (e.g., AAPL, RELIANCE.NS):", "RELIANCE.NS")
    with col2:
        period = st.selectbox("Select period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    
    if st.button("Analyze Stock"):
        with st.spinner("Fetching stock data..."):
            hist, info = fetch_stock_data(stock_symbol, period)
            
            if hist is not None and not hist.empty:
                # Display basic info
                st.subheader(f"📊 {info.get('longName', stock_symbol)} Analysis")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    current_price = hist['Close'].iloc[-1]
                    previous_close = info.get('previousClose', current_price)
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100
                    
                    # Get currency
                    currency = info.get('currency', 'USD')
                    if currency == 'INR':
                        currency = 'INR'
                    
                    # Convert to selected currency if needed
                    display_price = current_price
                    if currency != st.session_state.selected_currency:
                        display_price = convert_currency(current_price, currency, st.session_state.selected_currency)
                    
                    st.metric(
                        label="Current Price",
                        value=format_currency(display_price, st.session_state.selected_currency),
                        delta=f"{change_percent:.2f}%",
                        delta_color="normal" if change >= 0 else "inverse"
                    )
                
                with col2:
                    day_high = hist['High'].iloc[-1]
                    day_low = hist['Low'].iloc[-1]
                    
                    # Convert to selected currency if needed
                    display_high = day_high
                    display_low = day_low
                    if currency != st.session_state.selected_currency:
                        display_high = convert_currency(day_high, currency, st.session_state.selected_currency)
                        display_low = convert_currency(day_low, currency, st.session_state.selected_currency)
                    
                    st.metric("Day High", format_currency(display_high, st.session_state.selected_currency))
                
                with col3:
                    st.metric("Day Low", format_currency(display_low, st.session_state.selected_currency))
                
                # Display additional info
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**52 Week High:** {format_currency(info.get('fiftyTwoWeekHigh', 0), currency)}")
                    st.write(f"**52 Week Low:** {format_currency(info.get('fiftyTwoWeekLow', 0), currency)}")
                    st.write(f"**Volume:** {info.get('volume', 0):,}")
                
                with col2:
                    st.write(f"**Market Cap:** {format_currency(info.get('marketCap', 0), currency)}")
                    st.write(f"**PE Ratio:** {info.get('trailingPE', 'N/A')}")
                    st.write(f"**Dividend Yield:** {info.get('dividendYield', 0)*100 if info.get('dividendYield') else 'N/A'}%")
                
                # Display chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='Price'
                ))
                
                fig.update_layout(
                    title=f"{stock_symbol} Price History",
                    xaxis_title="Date",
                    yaxis_title=f"Price ({st.session_state.selected_currency})",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display historical data
                st.subheader("Historical Data")
                st.dataframe(hist.tail(10), use_container_width=True)
                
            else:
                st.error("Could not fetch data for the specified symbol. Please check the symbol and try again.")

# Technical Analysis Page
elif selected == "Technical Analysis":
    st.title("Technical Analysis")
    
    # Stock selection
    stock_symbol = st.text_input("Enter stock symbol for technical analysis:", "RELIANCE.NS")
    period = st.selectbox("Select period for analysis:", ["1mo", "3mo", "6mo", "1y"], index=2)
    
    if st.button("Perform Technical Analysis"):
        with st.spinner("Calculating technical indicators..."):
            hist, info = fetch_stock_data(stock_symbol, period)
            
            if hist is not None and not hist.empty:
                # Calculate technical indicators
                hist = get_technical_indicators(hist)
                
                # Display current price and RSI
                current_price = hist['Close'].iloc[-1]
                current_rsi = hist['RSI'].iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Get currency
                    currency = info.get('currency', 'USD') if info else 'USD'
                    if currency == 'INR':
                        currency = 'INR'
                    
                    # Convert to selected currency if needed
                    display_price = current_price
                    if currency != st.session_state.selected_currency:
                        display_price = convert_currency(current_price, currency, st.session_state.selected_currency)
                    
                    st.metric("Current Price", format_currency(display_price, st.session_state.selected_currency))
                
                with col2:
                    st.metric("RSI", f"{current_rsi:.2f}")
                
                with col3:
                    rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    st.metric("RSI Signal", rsi_signal)
                
                # RSI Chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], mode='lines', name='RSI', line=dict(color='#FF0000')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="green")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="red")
                fig_rsi.update_layout(title="RSI Indicator", height=300, template="plotly_dark")
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # MACD Chart
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], mode='lines', name='MACD', line=dict(color='#FF9900')))
                fig_macd.add_trace(go.Scatter(x=hist.index, y=hist['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='#00FF00')))
                fig_macd.update_layout(title="MACD Indicator", height=300, template="plotly_dark")
                st.plotly_chart(fig_macd, use_container_width=True)
                
                # Moving Averages Chart
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Price', line=dict(color='#FFFFFF')))
                fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], mode='lines', name='SMA 20', line=dict(color='#FF0000')))
                fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], mode='lines', name='SMA 50', line=dict(color='#00FF00')))
                fig_ma.update_layout(title="Moving Averages", height=300, template="plotly_dark")
                st.plotly_chart(fig_ma, use_container_width=True)
                
                # Bollinger Bands Chart
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], mode='lines', name='Upper Band', line=dict(color='#FF0000')))
                fig_bb.add_trace(go.Scatter(x=hist.index, y=hist['BB_Middle'], mode='lines', name='Middle Band', line=dict(color='#FFFFFF')))
                fig_bb.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], mode='lines', name='Lower Band', line=dict(color='#00FF00')))
                fig_bb.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Price', line=dict(color='#FF9900')))
                fig_bb.update_layout(title="Bollinger Bands", height=300, template="plotly_dark")
                st.plotly_chart(fig_bb, use_container_width=True)
                
                # Technical signals summary
                st.subheader("Technical Signals Summary")
                
                # Generate signals
                signals = []
                
                # RSI signal
                if current_rsi > 70:
                    signals.append(("RSI", "Overbought", "Bearish"))
                elif current_rsi < 30:
                    signals.append(("RSI", "Oversold", "Bullish"))
                else:
                    signals.append(("RSI", "Neutral", "Neutral"))
                
                # MACD signal
                if hist['MACD'].iloc[-1] > hist['Signal_Line'].iloc[-1]:
                    signals.append(("MACD", "Above Signal Line", "Bullish"))
                else:
                    signals.append(("MACD", "Below Signal Line", "Bearish"))
                
                # Moving Average signal
                if hist['SMA_20'].iloc[-1] > hist['SMA_50'].iloc[-1]:
                    signals.append(("Moving Averages", "Short-term above Long-term", "Bullish"))
                else:
                    signals.append(("Moving Averages", "Short-term below Long-term", "Bearish"))
                
                # Bollinger Bands signal
                if hist['Close'].iloc[-1] > hist['BB_Upper'].iloc[-1]:
                    signals.append(("Bollinger Bands", "Price above Upper Band", "Overbought"))
                elif hist['Close'].iloc[-1] < hist['BB_Lower'].iloc[-1]:
                    signals.append(("Bollinger Bands", "Price below Lower Band", "Oversold"))
                else:
                    signals.append(("Bollinger Bands", "Price within Bands", "Neutral"))
                
                # Display signals in a table
                signals_df = pd.DataFrame(signals, columns=["Indicator", "Signal", "Interpretation"])
                st.dataframe(signals_df, use_container_width=True)
                
            else:
                st.error("Could not fetch data for the specified symbol. Please check the symbol and try again.")

# Portfolio Manager Page
elif selected == "Portfolio Manager":
    st.title("Portfolio Manager")
    
    tab1, tab2, tab3 = st.tabs(["View Portfolio", "Add Holding", "Performance Analysis"])
    
    with tab1:
        st.subheader("Your Investment Portfolio")
        
        if not st.session_state.portfolio.empty:
            # Calculate current values
            portfolio_df = st.session_state.portfolio.copy()
            current_values = []
            total_investment = 0
            total_current = 0
            
            for _, row in portfolio_df.iterrows():
                try:
                    stock_data = yf.Ticker(row['Symbol'])
                    hist = stock_data.history(period="1d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        current_value = current_price * row['Quantity']
                        investment_value = row['Purchase Price'] * row['Quantity']
                        
                        # Convert to selected currency if needed
                        if row['Currency'] != st.session_state.selected_currency:
                            current_value = convert_currency(current_value, row['Currency'], st.session_state.selected_currency)
                            investment_value = convert_currency(investment_value, row['Currency'], st.session_state.selected_currency)
                        
                        current_values.append(current_value)
                        total_investment += investment_value
                        total_current += current_value
                    else:
                        current_values.append(0)
                except:
                    current_values.append(0)
            
            portfolio_df['Current Value'] = current_values
            portfolio_df['Investment Value'] = portfolio_df['Purchase Price'] * portfolio_df['Quantity']
            portfolio_df['Gain/Loss'] = portfolio_df['Current Value'] - portfolio_df['Investment Value']
            portfolio_df['Gain/Loss %'] = (portfolio_df['Gain/Loss'] / portfolio_df['Investment Value']) * 100
            
            # Format currency
            for col in ['Purchase Price', 'Current Value', 'Investment Value', 'Gain/Loss']:
                portfolio_df[col] = portfolio_df[col].apply(lambda x: format_currency(x, st.session_state.selected_currency))
            
            st.dataframe(portfolio_df, use_container_width=True)
            
            # Display portfolio summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Investment", format_currency(total_investment, st.session_state.selected_currency))
            with col2:
                st.metric("Current Value", format_currency(total_current, st.session_state.selected_currency))
            with col3:
                gain_loss = total_current - total_investment
                gain_loss_percent = (gain_loss / total_investment) * 100 if total_investment > 0 else 0
                st.metric(
                    "Total Gain/Loss", 
                    format_currency(gain_loss, st.session_state.selected_currency),
                    f"{gain_loss_percent:.2f}%"
                )
        else:
            st.info("Your portfolio is empty. Add holdings to get started.")
    
    with tab2:
        st.subheader("Add New Holding")
        
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Symbol", "RELIANCE.NS")
            quantity = st.number_input("Quantity", min_value=1, value=10)
        with col2:
            purchase_price = st.number_input("Purchase Price", min_value=0.0, value=2500.0)
            purchase_date = st.date_input("Purchase Date", value=datetime.now().date())
        
        # Get currency for the symbol
        currency = "USD"  # default
        try:
            stock_data = yf.Ticker(symbol)
            info = stock_data.info
            currency = info.get('currency', 'USD')
            if currency == 'INR':
                currency = 'INR'
        except:
            pass
        
        if st.button("Add to Portfolio"):
            new_holding = {
                'Symbol': symbol,
                'Quantity': quantity,
                'Purchase Price': purchase_price,
                'Purchase Date': purchase_date,
                'Currency': currency
            }
            
            st.session_state.portfolio = pd.concat([
                st.session_state.portfolio, 
                pd.DataFrame([new_holding])
            ], ignore_index=True)
            
            st.success(f"Added {quantity} shares of {symbol} to your portfolio")
    
    with tab3:
        st.subheader("Portfolio Performance Analysis")
        
        if not st.session_state.portfolio.empty:
            # Create a pie chart of portfolio allocation
            symbols = st.session_state.portfolio['Symbol'].unique()
            allocations = []
            
            for symbol in symbols:
                symbol_holdings = st.session_state.portfolio[st.session_state.portfolio['Symbol'] == symbol]
                total_investment = (symbol_holdings['Purchase Price'] * symbol_holdings['Quantity']).sum()
                allocations.append(total_investment)
            
            fig = px.pie(
                values=allocations, 
                names=symbols, 
                title="Portfolio Allocation by Symbol",
                color_discrete_sequence=px.colors.sequential.Reds_r
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a line chart of portfolio performance over time
            st.info("Portfolio performance chart would be implemented with historical data in a production version.")
        else:
            st.info("Your portfolio is empty. Add holdings to see performance analysis.")

# Options Chain Page
elif selected == "Options Chain":
    st.title("📊 Options Chain Analysis")
    
    st.info("""
    Options chain analysis provides detailed information about available options contracts for a particular stock.
    This feature requires a premium data subscription for real-time options data.
    """)
    
    symbol = st.text_input("Enter stock symbol for options analysis:", "RELIANCE.NS")
    expiry_date = st.date_input("Select expiration date:", value=datetime.now().date() + timedelta(days=30))
    
    if st.button("Fetch Options Chain"):
        st.warning("Options chain data is not available in the demo version. Please upgrade to a premium subscription.")
        
        # Mock options data for demonstration
        st.subheader(f"Options Chain for {symbol} (Mock Data)")
        
        # Generate mock options data
        strike_prices = np.arange(2400, 2600, 25)
        calls = []
        puts = []
        
        for strike in strike_prices:
            calls.append({
                "Strike": strike,
                "Last Price": round(np.random.uniform(5, 50), 2),
                "Bid": round(np.random.uniform(4, 49), 2),
                "Ask": round(np.random.uniform(6, 51), 2),
                "Volume": np.random.randint(100, 1000),
                "Open Interest": np.random.randint(500, 5000),
                "Implied Volatility": round(np.random.uniform(0.2, 0.5), 3)
            })
            
            puts.append({
                "Strike": strike,
                "Last Price": round(np.random.uniform(5, 50), 2),
                "Bid": round(np.random.uniform(4, 49), 2),
                "Ask": round(np.random.uniform(6, 51), 2),
                "Volume": np.random.randint(100, 1000),
                "Open Interest": np.random.randint(500, 5000),
                "Implied Volatility": round(np.random.uniform(0.2, 0.5), 3)
            })
        
        calls_df = pd.DataFrame(calls)
        puts_df = pd.DataFrame(puts)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Call Options")
            st.dataframe(calls_df, use_container_width=True)
        with col2:
            st.subheader("Put Options")
            st.dataframe(puts_df, use_container_width=True)
        
        # Options strategy suggestions
        st.subheader("Options Strategy Suggestions")
        
        strategies = [
            ("Covered Call", "Sell call options against shares you own to generate income"),
            ("Cash-Secured Put", "Sell put options with cash reserved to buy the stock if assigned"),
            ("Protective Put", "Buy put options to protect against downside risk in a stock you own"),
            ("Collar", "Combine covered call and protective put to limit both upside and downside"),
            ("Long Straddle", "Buy both call and put options with same strike and expiration to profit from volatility")
        ]
        
        for strategy, description in strategies:
            with st.expander(strategy):
                st.write(description)

# Market Overview Page
elif selected == "Market Overview":
    st.title("Global Market Overview")
    
    # Sector performance
    st.subheader("📈 Sector Performance")
    sectors = get_sector_performance()
    
    sector_cols = st.columns(4)
    for i, (sector, data) in enumerate(sectors.items()):
        with sector_cols[i % 4]:
            change_color = "#00FF00" if data['change'] >= 0 else "#FF0000"
            st.metric(
                label=sector,
                value=f"{data['change']:.1f}%",
                delta_color="normal" if data['change'] >= 0 else "inverse"
            )
    
    # Indian stocks
    st.subheader("🇮🇳 Indian Stocks Overview")
    indian_stocks = get_indian_stocks()
    
    selected_sector = st.selectbox("Filter by sector:", ["All"] + list(set([stock['sector'] for stock in indian_stocks.values()])))
    
    filtered_stocks = indian_stocks
    if selected_sector != "All":
        filtered_stocks = {k: v for k, v in indian_stocks.items() if v['sector'] == selected_sector}
    
    stock_cols = st.columns(4)
    for i, (symbol, info) in enumerate(filtered_stocks.items()):
        if i >= 16:  # Limit display to 16 stocks
            break
            
        with stock_cols[i % 4]:
            try:
                stock_data = yf.Ticker(symbol)
                hist = stock_data.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    info_data = stock_data.info
                    previous_close = info_data.get('previousClose', current_price)
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100
                    
                    # Get currency
                    currency = info_data.get('currency', 'USD')
                    if currency == 'INR':
                        currency = 'INR'
                    
                    # Convert to selected currency if needed
                    display_price = current_price
                    if currency != st.session_state.selected_currency:
                        display_price = convert_currency(current_price, currency, st.session_state.selected_currency)
                    
                    st.metric(
                        label=f"{info['name']} ({symbol})",
                        value=format_currency(display_price, st.session_state.selected_currency),
                        delta=f"{change_percent:.2f}%",
                        delta_color="normal" if change >= 0 else "inverse"
                    )
            except:
                st.error(f"Error fetching data for {symbol}")

# Economic Calendar Page
elif selected == "Economic Calendar":
    st.title("📅 Economic Calendar")
    
    st.info("""
    The economic calendar shows upcoming economic events and indicators that may impact financial markets.
    This feature requires a premium data subscription for real-time economic data.
    """)
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now().date())
    with col2:
        end_date = st.date_input("End Date", value=datetime.now().date() + timedelta(days=7))
    
    # Country filter
    countries = ["All", "US", "UK", "EU", "Japan", "China", "India"]
    selected_country = st.selectbox("Filter by country:", countries)
    
    if st.button("Load Economic Events"):
        st.warning("Economic calendar data is not available in the demo version. Please upgrade to a premium subscription.")
        
        # Mock economic events for demonstration
        events = [
            {"Date": "2023-11-15", "Time": "14:30", "Country": "US", "Event": "CPI Data", "Impact": "High"},
            {"Date": "2023-11-16", "Time": "10:00", "Country": "EU", "Event": "ECB Interest Rate Decision", "Impact": "High"},
            {"Date": "2023-11-17", "Time": "09:30", "Country": "UK", "Event": "Retail Sales", "Impact": "Medium"},
            {"Date": "2023-11-18", "Time": "08:00", "Country": "Japan", "Event": "BoJ Monetary Policy Statement", "Impact": "High"},
            {"Date": "2023-11-19", "Time": "13:00", "Country": "US", "Event": "FOMC Meeting Minutes", "Impact": "High"},
            {"Date": "2023-11-20", "Time": "11:30", "Country": "India", "Event": "RBI Repo Rate Decision", "Impact": "High"},
        ]
        
        events_df = pd.DataFrame(events)
        
        # Filter by country if selected
        if selected_country != "All":
            events_df = events_df[events_df["Country"] == selected_country]
        
        # Color code by impact
        def color_impact(val):
            if val == "High":
                color = "#FF0000"
            elif val == "Medium":
                color = "#FF9900"
            else:
                color = "#00FF00"
            return f"color: {color}; font-weight: bold;"
        
        styled_df = events_df.style.applymap(color_impact, subset=["Impact"])
        st.dataframe(styled_df, use_container_width=True)

# Crypto Markets Page
elif selected == "Crypto Markets":
    st.title("₿ Crypto Markets")
    
    # Major cryptocurrencies
    cryptocurrencies = {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "BNB-USD": "Binance Coin",
        "ADA-USD": "Cardano",
        "XRP-USD": "Ripple",
        "SOL-USD": "Solana",
        "DOT-USD": "Polkadot",
        "DOGE-USD": "Dogecoin",
    }
    
    st.subheader("Major Cryptocurrencies")
    
    crypto_cols = st.columns(4)
    for i, (symbol, name) in enumerate(cryptocurrencies.items()):
        with crypto_cols[i % 4]:
            try:
                crypto_data = yf.Ticker(symbol)
                hist = crypto_data.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    info = crypto_data.info
                    previous_close = info.get('previousClose', current_price)
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100
                    
                    # Convert to selected currency if needed
                    display_price = current_price
                    if st.session_state.selected_currency != 'USD':
                        display_price = convert_currency(current_price, 'USD', st.session_state.selected_currency)
                    
                    st.metric(
                        label=name,
                        value=format_currency(display_price, st.session_state.selected_currency),
                        delta=f"{change_percent:.2f}%",
                        delta_color="normal" if change >= 0 else "inverse"
                    )
            except:
                st.error(f"Error fetching data for {name}")
    
    # Crypto news
    st.subheader("Cryptocurrency News")
    crypto_news = fetch_news("cryptocurrency")
    
    if crypto_news:
        for article in crypto_news[:3]:
            with st.expander(f"{article['title']}"):
                st.write(f"**Source:** {article['source']['name']}")
                st.write(f"**Published:** {article['publishedAt'][:10]}")
                st.write(article['description'])
                if article['url']:
                    st.markdown(f"[Read more]({article['url']})")
    else:
        st.info("No cryptocurrency news available at the moment.")

# News & Sentiment Page
elif selected == "News & Sentiment":
    st.title("News & Market Sentiment")
    
    # Search for news
    news_query = st.text_input("Search for news:", "stock market")
    
    if st.button("Search News"):
        with st.spinner("Fetching news..."):
            news_data = fetch_news(news_query)
            
            if news_data:
                st.subheader(f"News about '{news_query}'")
                
                for article in news_data:
                    with st.expander(f"{article['title']}"):
                        st.write(f"**Source:** {article['source']['name']}")
                        st.write(f"**Published:** {article['publishedAt'][:10]}")
                        st.write(article['description'])
                        if article['url']:
                            st.markdown(f"[Read more]({article['url']})")
            else:
                st.info("No news found for your search query.")
    
    # Market sentiment analysis
    st.subheader("Market Sentiment Analysis")
    
    st.info("""
    Market sentiment analysis evaluates the overall attitude of investors toward a particular security or financial market.
    This feature requires a premium subscription for advanced sentiment analysis.
    """)
    
    sentiment_symbol = st.text_input("Enter symbol for sentiment analysis:", "AAPL")
    
    if st.button("Analyze Sentiment"):
        st.warning("Advanced sentiment analysis is not available in the demo version. Please upgrade to a premium subscription.")
        
        # Mock sentiment analysis
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Sentiment", "Bullish", "Positive")
        with col2:
            st.metric("News Sentiment", "72% Positive", "5%")
        with col3:
            st.metric("Social Media Sentiment", "65% Positive", "3%")
        
        # Sentiment over time chart
        dates = pd.date_range(end=datetime.now(), periods=30).tolist()
        positive = np.random.normal(65, 5, 30)
        negative = 100 - positive
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=positive, mode='lines', name='Positive', line=dict(color='#00FF00')))
        fig.add_trace(go.Scatter(x=dates, y=negative, mode='lines', name='Negative', line=dict(color='#FF0000')))
        fig.update_layout(
            title="Sentiment Over Time",
            xaxis_title="Date",
            yaxis_title="Sentiment (%)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Learning Center Page
elif selected == "Learning Center":
    st.title("Learning Center")
    
    st.subheader("Educational Resources")
    
    topics = [
        {
            "title": "Stock Market Basics",
            "description": "Learn the fundamentals of stock market investing",
            "content": """
            - What are stocks and how do they work?
            - Different types of orders (market, limit, stop)
            - Understanding stock indices
            - Fundamental vs. technical analysis
            """
        },
        {
            "title": "Technical Analysis",
            "description": "Learn how to analyze price charts and indicators",
            "content": """
            - Support and resistance levels
            - Moving averages and trends
            - RSI, MACD, and other indicators
            - Chart patterns (head and shoulders, triangles, etc.)
            """
        },
        {
            "title": "Options Trading",
            "description": "Understand options contracts and strategies",
            "content": """
            - Calls and puts explained
            - In-the-money, at-the-money, out-of-the-money
            - Basic options strategies (covered calls, protective puts)
            - Understanding implied volatility
            """
        },
        {
            "title": "Portfolio Management",
            "description": "Learn how to build and manage a diversified portfolio",
            "content": """
            - Asset allocation strategies
            - Risk management techniques
            - Rebalancing your portfolio
            - Tax-efficient investing
            """
        }
    ]
    
    for topic in topics:
        with st.expander(f"{topic['title']}: {topic['description']}"):
            st.markdown(topic['content'])
    
    st.subheader("Recommended Books")
    books = [
        {"title": "The Intelligent Investor", "author": "Benjamin Graham", "year": 1949},
        {"title": "A Random Walk Down Wall Street", "author": "Burton Malkiel", "year": 1973},
        {"title": "One Up On Wall Street", "author": "Peter Lynch", "year": 1989},
        {"title": "The Little Book of Common Sense Investing", "author": "John C. Bogle", "year": 2007},
        {"title": "Rich Dad Poor Dad", "author": "Robert Kiyosaki", "year": 1997},
    ]
    
    for book in books:
        st.write(f"- **{book['title']}** by {book['author']} ({book['year']})")

# Company Info Page
elif selected == "Company Info":
    st.title("Company Information")
    
    symbol = st.text_input("Enter company symbol:", "RELIANCE.NS")
    
    if st.button("Get Company Info"):
        with st.spinner("Fetching company information..."):
            stock_data = yf.Ticker(symbol)
            info = stock_data.info
            
            if info:
                st.subheader(f"{info.get('longName', 'N/A')} ({symbol})")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.write(f"**Country:** {info.get('country', 'N/A')}")
                    st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A'):,}")
                
                with col2:
                    st.write(f"**Website:** {info.get('website', 'N/A')}")
                    st.write(f"**CEO:** {info.get('ceo', 'N/A')}")
                    st.write(f"**Founded:** {info.get('founded', 'N/A')}")
                
                # Financial metrics
                st.subheader("Financial Metrics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Market Cap", format_currency(info.get('marketCap', 0), info.get('currency', 'USD')))
                    st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
                    st.metric("EPS", f"{info.get('trailingEps', 'N/A')}")
                
                with col2:
                    st.metric("ROE", f"{info.get('returnOnEquity', 'N/A')}")
                    st.metric("Profit Margins", f"{info.get('profitMargins', 'N/A')}")
                    st.metric("Revenue Growth", f"{info.get('revenueGrowth', 'N/A')}")
                
                with col3:
                    st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100 if info.get('dividendYield') else 'N/A'}%")
                    st.metric("Beta", f"{info.get('beta', 'N/A')}")
                    st.metric("52 Week High", format_currency(info.get('fiftyTwoWeekHigh', 0), info.get('currency', 'USD')))
                
                # Company description
                if info.get('longBusinessSummary'):
                    st.subheader("Business Summary")
                    st.write(info.get('longBusinessSummary'))
            else:
                st.error("Could not fetch company information. Please check the symbol and try again.")

# Predictions Page
elif selected == "Predictions":
    st.title("Stock & Mutual Fund Predictions")
    
    tab1, tab2 = st.tabs(["Stock Predictions", "Mutual Fund Analysis"])
    
    with tab1:
        st.subheader("Stock Price Prediction")
        
        symbol = st.text_input("Enter stock symbol for prediction:", "RELIANCE.NS")
        prediction_days = st.slider("Prediction period (days):", 7, 90, 30)
        
        if st.button("Generate Prediction"):
            with st.spinner("Analyzing historical data and generating prediction..."):
                current_price, predicted_price, percent_change = predict_stock_price(symbol, prediction_days)
                
                if current_price is not None:
                    # Get currency
                    stock_data = yf.Ticker(symbol)
                    info = stock_data.info
                    currency = info.get('currency', 'USD')
                    if currency == 'INR':
                        currency = 'INR'
                    
                    # Convert to selected currency if needed
                    display_current = current_price
                    display_predicted = predicted_price
                    if currency != st.session_state.selected_currency:
                        display_current = convert_currency(current_price, currency, st.session_state.selected_currency)
                        display_predicted = convert_currency(predicted_price, currency, st.session_state.selected_currency)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", format_currency(display_current, st.session_state.selected_currency))
                    with col2:
                        st.metric(
                            f"Predicted Price ({prediction_days} days)", 
                            format_currency(display_predicted, st.session_state.selected_currency)
                        )
                    with col3:
                        st.metric(
                            "Expected Change", 
                            f"{percent_change:.2f}%",
                            delta_color="normal" if percent_change >= 0 else "inverse"
                        )
                    
                    # Disclaimer
                    st.warning("""
                    **Disclaimer:** This prediction is based on historical data and trend analysis using simple algorithms. 
                    It should not be considered as financial advice. Past performance is not indicative of future results. 
                    Always do your own research and consider consulting with a qualified financial advisor before making investment decisions.
                    """)
                else:
                    st.error("Could not generate prediction for the specified symbol. Please check the symbol and try again.")
    
    with tab2:
        st.subheader("Mutual Fund Analysis")
        
        mutual_funds = get_mutual_funds()
        
        selected_fund = st.selectbox(
            "Select a mutual fund:",
            options=list(mutual_funds.keys()),
            format_func=lambda x: f"{x} - {mutual_funds[x]['name']}"
        )
        
        if selected_fund:
            fund_data = mutual_funds[selected_fund]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'current_price' in fund_data:
                    st.metric("Current NAV", format_currency(fund_data['current_price'], 'USD'))
                else:
                    st.metric("Current NAV", "N/A")
            with col2:
                st.metric("Category", fund_data['category'])
            with col3:
                st.metric("Expense Ratio", f"{fund_data['expense_ratio']}%")
            
            if 'ytd_return' in fund_data:
                st.metric("YTD Return", f"{fund_data['ytd_return']:.2f}%")
            
            # Fund comparison
            st.subheader("Compare Mutual Funds")
            
            compare_funds = st.multiselect(
                "Select funds to compare:",
                options=list(mutual_funds.keys()),
                format_func=lambda x: f"{x} - {mutual_funds[x]['name']}",
                default=[selected_fund]
            )
            
            if compare_funds:
                comparison_data = []
                for fund in compare_funds:
                    if fund in mutual_funds and 'ytd_return' in mutual_funds[fund]:
                        comparison_data.append({
                            'Fund': fund,
                            'Name': mutual_funds[fund]['name'],
                            'YTD Return': mutual_funds[fund]['ytd_return'],
                            'Expense Ratio': mutual_funds[fund]['expense_ratio'],
                            'Category': mutual_funds[fund]['category']
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # YTD Return comparison chart
                    fig = px.bar(
                        comparison_df, 
                        x='Fund', 
                        y='YTD Return',
                        title="YTD Return Comparison",
                        color='YTD Return',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)

# Settings Page
elif selected == "Settings":
    st.title("Settings")
    
    st.subheader("Appearance Settings")
    
    # Theme customization
    theme_options = ["Dark", "Light", "System"]
    selected_theme = st.selectbox("Theme", theme_options, index=theme_options.index(st.session_state.selected_theme))
    
    # Currency preferences
    currency_options = ["INR", "USD", "EUR", "GBP", "JPY"]
    selected_currency = st.selectbox("Default Currency", currency_options, index=currency_options.index(st.session_state.selected_currency))
    
    # Data refresh interval
    refresh_options = ["15 minutes", "30 minutes", "1 hour", "2 hours"]
    selected_refresh = st.selectbox("Data Refresh Interval", refresh_options, index=2)
    
    if st.button("Save Settings"):
        st.session_state.selected_theme = selected_theme
        st.session_state.selected_currency = selected_currency
        st.success("Settings saved successfully!")
    
    st.subheader("Data Management")
    
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared successfully!")
    
    if st.button("Reset Portfolio"):
        st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Quantity', 'Purchase Price', 'Purchase Date', 'Currency'])
        st.success("Portfolio reset successfully!")
    
    st.subheader("About MarketMentor Pro")
    st.write("""
    MarketMentor Pro is an advanced financial analytics platform designed to help investors make informed decisions.
    
    **Version:** 2.1.0
    **Last Updated:** November 2023
    **Data Provider:** Yahoo Finance
    
    For support and feedback, please contact us at support@marketmentor.com
    """)

# Add a footer to all pages
st.write("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("<div style='text-align: center; color: #FF0000;'><b>MarketMentor Pro</b><br>Advanced Financial Analytics Platform</div>", unsafe_allow_html=True)
with footer_col2:
    st.markdown("<div style='text-align: center;'><b>Disclaimer:</b><br>Not investment advice. Data may be delayed.</div>", unsafe_allow_html=True)
with footer_col3:
    st.markdown("<div style='text-align: center;'>© 2023 MarketMentor<br>Version 2.1.0</div>", unsafe_allow_html=True)

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
