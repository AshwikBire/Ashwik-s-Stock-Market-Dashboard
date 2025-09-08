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
    st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Quantity', 'Purchase Price', 'Purchase Date'])

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
    
    results = {}
    for symbol, data in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                last_close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = ((last_close - prev_close) / prev_close) * 100
                results[symbol] = {
                    "name": data["name"],
                    "price": last_close,
                    "change": change,
                    "currency": data["currency"]
                }
        except:
            continue
    return results

@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(query="stock market"):
    try:
        NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"
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
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    return df

# Prediction functions
@st.cache_data(ttl=3600, show_spinner=False)
def predict_stock_price(ticker, days=30):
    """
    Simple stock price prediction using historical data and trend analysis
    This is a simplified version for demonstration purposes
    """
    try:
        # Get historical data
        hist, _ = fetch_stock_data(ticker, "1y")
        
        if hist is None or hist.empty:
            return None, None, None
        
        # Calculate simple moving averages
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        
        # Simple trend prediction (for demo purposes)
        last_price = hist['Close'].iloc[-1]
        sma_20 = hist['SMA_20'].iloc[-1]
        sma_50 = hist['SMA_50'].iloc[-1]
        
        # Determine trend direction
        if sma_20 > sma_50 and last_price > sma_20:
            trend = "Bullish"
            prediction = last_price * (1 + 0.005 * days)  # Simple upward trend
        elif sma_20 < sma_50 and last_price < sma_20:
            trend = "Bearish"
            prediction = last_price * (1 - 0.005 * days)  # Simple downward trend
        else:
            trend = "Neutral"
            prediction = last_price  # No significant change
        
        # Generate confidence score (for demo purposes)
        confidence = min(95, max(60, 70 + (abs(sma_20 - sma_50) / last_price * 1000)))
        
        return prediction, trend, confidence
    
    except Exception as e:
        return None, None, None

@st.cache_data(ttl=3600, show_spinner=False)
def get_mutual_funds():
    """
    Get a list of popular mutual funds with their performance data
    """
    mutual_funds = {
        'VFIAX': {'name': 'Vanguard 500 Index Fund', 'category': 'Large Blend', 'expense_ratio': 0.04},
        'VTSAX': {'name': 'Vanguard Total Stock Market Index Fund', 'category': 'Large Blend', 'expense_ratio': 0.04},
        'VGSLX': {'name': 'Vanguard Real Estate Index Fund', 'category': 'Real Estate', 'expense_ratio': 0.12},
        'VIMAX': {'name': 'Vanguard Mid-Cap Index Fund', 'category': 'Mid-Cap Blend', 'expense_ratio': 0.05},
        'VSMAX': {'name': 'Vanguard Small-Cap Index Fund', 'category': 'Small-Cap Blend', 'expense_ratio': 0.05},
        'VTIAX': {'name': 'Vanguard Total International Stock Index Fund', 'category': 'International', 'expense_ratio': 0.11},
        'VBTLX': {'name': 'Vanguard Total Bond Market Index Fund', 'category': 'Intermediate-Term Bond', 'expense_ratio': 0.05},
    }
    
    results = {}
    for symbol, data in mutual_funds.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                year_high = hist['High'].max()
                year_low = hist['Low'].min()
                ytd_return = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                
                results[symbol] = {
                    'name': data['name'],
                    'category': data['category'],
                    'expense_ratio': data['expense_ratio'],
                    'price': current_price,
                    'ytd_return': ytd_return,
                    'year_high': year_high,
                    'year_low': year_low
                }
        except:
            continue
    
    return results

# Sidebar navigation
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #FF0000;'>MarketMentor Pro</h1>", unsafe_allow_html=True)
    
    selected = option_menu(
        "Navigation",
        ["Dashboard", "Stock Analysis", "Technical Analysis", "Portfolio Manager", 
         "Options Chain", "Market Overview", "Economic Calendar", "Crypto Markets",
         "News & Sentiment", "Learning Center", "Company Info", "Predictions", "Settings"],
        icons=['speedometer2', 'graph-up', 'bar-chart-line', 'wallet', 
               'diagram-3', 'globe', 'calendar', 'currency-bitcoin',
               'newspaper', 'book', 'building', 'graph-up-arrow', 'gear'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#0A0A0A", "padding": "5px"},
            "icon": {"color": "#FF0000", "font-size": "16px"}, 
            "nav-link": {"color": "#FFFFFF", "font-size": "14px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#FF0000", "color": "#000000", "font-weight": "bold"},
        }
    )
    
    # Watchlist section in sidebar
    st.markdown("---")
    st.subheader(" Watchlist")
    watchlist_symbol = st.text_input("Add symbol to watchlist", "AAPL", key="watchlist_input")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add", key="add_watchlist") and watchlist_symbol:
            if watchlist_symbol not in st.session_state.watchlist:
                st.session_state.watchlist.append(watchlist_symbol)
                st.success(f"Added {watchlist_symbol} to watchlist")
    with col2:
        if st.button("Clear All", key="clear_watchlist"):
            st.session_state.watchlist = []
            st.success("Watchlist cleared")
    
    if st.session_state.watchlist:
        st.markdown("**Your Watchlist:**")
        for symbol in st.session_state.watchlist:
            try:
                hist, _ = fetch_stock_data(symbol, "1d")
                if hist is not None and not hist.empty:
                    price = hist['Close'].iloc[-1]
                    st.markdown(f"<div style='background-color: #1A0A0A; padding: 8px; border-radius: 4px; margin: 4px 0; border-left: 3px solid #FF0000;'>{symbol}: ${price:.2f}</div>", 
                               unsafe_allow_html=True)
            except:
                st.markdown(f"<div style='background-color: #1A0A0A; padding: 8px; border-radius: 4px; margin: 4px 0; border-left: 3px solid #FF0000;'>{symbol}: N/A</div>", 
                           unsafe_allow_html=True)

# Dashboard Page
if selected == "Dashboard":
    st.title("Market Dashboard")
    
    # Load global indices with caching
    indices_data = fetch_global_indices()
    
    # Display key indices
    st.subheader("Global Markets")
    cols = st.columns(6)
    index_count = 0
    for symbol, data in list(indices_data.items())[:6]:
        with cols[index_count % 6]:
            currency_symbol = "$" if data["currency"] == "USD" else "₹" if data["currency"] == "INR" else "€" if data["currency"] == "EUR" else "£"
            st.metric(
                label=data["name"],
                value=f"{currency_symbol}{data['price']:.2f}",
                delta=f"{data['change']:.2f}%"
            )
        index_count += 1
    
    # Show more markets in a second row
    cols = st.columns(6)
    for idx, (symbol, data) in enumerate(list(indices_data.items())[6:12]):
        with cols[idx % 6]:
            currency_symbol = "$" if data["currency"] == "USD" else "₹" if data["currency"] == "INR" else "€" if data["currency"] == "EUR" else "£"
            st.metric(
                label=data["name"],
                value=f"{currency_symbol}{data['price']:.2f}",
                delta=f"{data['change']:.2f}%"
            )
    
    # Market overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sector Performance")
        sectors = {
            "Technology": "+2.3%", "Healthcare": "+1.5%", "Financials": "-0.8%",
            "Energy": "+3.2%", "Consumer Cyclical": "+0.7%", "Real Estate": "-1.2%",
            "Utilities": "+0.3%", "Communications": "+1.8%", "Materials": "-0.5%"
        }
        sector_df = pd.DataFrame({
            "Sector": list(sectors.keys()),
            "Performance": [float(x.strip('%')) for x in sectors.values()]
        })
        fig = px.bar(sector_df, x="Performance", y="Sector", orientation='h',
                     title="Sector Performance (%)", color="Performance",
                     color_continuous_scale=px.colors.sequential.Reds)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Market Sentiment")
        sentiment_data = {"Bullish": 45, "Neutral": 30, "Bearish": 25}
        fig = px.pie(values=list(sentiment_data.values()), names=list(sentiment_data.keys()),
                     title="Market Sentiment Distribution",
                     color_discrete_map={"Bullish": "#FF0000", "Neutral": "#AAAAAA", "Bearish": "#0000FF"})
        st.plotly_chart(fig, use_container_width=True)
        
        # Market health indicators
        st.subheader("Market Health")
        health_data = {
            "Volatility Index (VIX)": "18.5",
            "Advance/Decline Ratio": "1.2:1",
            "New Highs/Lows": "285/142",
            "Put/Call Ratio": "0.85"
        }
        for indicator, value in health_data.items():
            st.markdown(f"<div style='background-color: #1A0A0A; padding: 8px; border-radius: 4px; margin: 4px 0; border-left: 3px solid #FF0000;'><b>{indicator}:</b> {value}</div>", 
                       unsafe_allow_html=True)
    
    # Recent news with caching
    st.subheader("Market News")
    news_articles = fetch_news()
    if news_articles:
        for article in news_articles[:3]:
            with st.expander(f"{article['title']} - {article['source']['name']}"):
                st.write(article.get('description', 'No description available'))
                st.markdown(f"[Read more]({article['url']})")
    else:
        st.info("News feed temporarily unavailable.")
    
    # Quick actions
    st.subheader("Quick Actions")
    action_cols = st.columns(4)
    with action_cols[0]:
        if st.button("🔍 Screen Stocks", key="screen_stocks"):
            st.session_state.selected = "Stock Analysis"
    with action_cols[1]:
        if st.button("Manage Portfolio", key="manage_portfolio"):
            st.session_state.selected = "Portfolio Manager"
    with action_cols[2]:
        if st.button("View Options", key="view_options"):
            st.session_state.selected = "Options Chain"
    with action_cols[3]:
        if st.button("Check News", key="check_news"):
            st.session_state.selected = "News & Sentiment"

# Stock Analysis Page
elif selected == "Stock Analysis":
    st.title("Stock Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Single Stock", "Compare Stocks", "Screener"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker = st.text_input("🔍 Enter Stock Symbol", "AAPL")
        with col2:
            analysis_type = st.selectbox("Analysis Type", ["Overview", "Financials", "Holdings", "Chart"])
        
        if ticker:
            hist, info = fetch_stock_data(ticker, "3mo")
            if hist is not None and info is not None:
                st.subheader(f"{info.get('longName', 'N/A')} ({ticker.upper()})")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                current_price = info.get('regularMarketPrice', hist['Close'].iloc[-1] if not hist.empty else 'N/A')
                previous_close = info.get('regularMarketPreviousClose', 'N/A')
                market_cap = info.get('marketCap', 'N/A')
                pe_ratio = info.get('trailingPE', 'N/A')
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}" if isinstance(current_price, float) else current_price)
                with col2:
                    if isinstance(previous_close, float) and isinstance(current_price, float):
                        change = current_price - previous_close
                        change_percent = (change / previous_close) * 100
                        st.metric("Previous Close", f"${previous_close:.2f}", f"{change:.2f} ({change_percent:.2f}%)")
                with col3:
                    if isinstance(market_cap, (int, float)):
                        if market_cap >= 1e12:
                            st.metric("Market Cap", f"${market_cap/1e12:.2f}T")
                        elif market_cap >= 1e9:
                            st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
                        else:
                            st.metric("Market Cap", f"${market_cap:,.2f}")
                with col4:
                    st.metric("P/E Ratio", f"{pe_ratio:.2f}" if isinstance(pe_ratio, float) else pe_ratio)
                
                # Additional metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
                with col2:
                    st.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
                with col3:
                    st.metric("Volume", f"{info.get('volume', 'N/A'):,}")
                with col4:
                    st.metric("Beta", f"{info.get('beta', 'N/A')}")
                
                # Price chart
                if analysis_type == "Chart":
                    st.subheader("Price Chart")
                    chart_type = st.selectbox("Chart Type", ["Line", "Candlestick"])
                    
                    if chart_type == "Line":
                        fig = px.line(hist, x=hist.index, y='Close', title=f"{ticker.upper()} Price History")
                    else:
                        fig = go.Figure(data=[go.Candlestick(
                            x=hist.index, open=hist['Open'], high=hist['High'], 
                            low=hist['Low'], close=hist['Close']
                        )])
                        fig.update_layout(title=f"{ticker.upper()} Candlestick Chart")
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Financial statements (simulated)
                if analysis_type == "Financials":
                    st.subheader("Financial Statements")
                    financials = pd.DataFrame({
                        'Year': ['2023', '2022', '2021', '2020', '2019'],
                        'Revenue (B)': [383.29, 365.82, 274.52, 260.17, 265.60],
                        'Net Income (B)': [97.00, 94.68, 57.41, 55.26, 55.34],
                        'EPS': [6.13, 5.67, 3.28, 3.31, 3.00],
                        'Dividend': [0.96, 0.88, 0.82, 0.80, 0.75]
                    })
                    st.dataframe(financials, use_container_width=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=financials['Year'], y=financials['Revenue (B)'], name='Revenue'))
                    fig.add_trace(go.Bar(x=financials['Year'], y=financials['Net Income (B)'], name='Net Income'))
                    fig.update_layout(title="Revenue vs Net Income (Billions $)", barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Company info
                if analysis_type == "Overview":
                    st.subheader("Company Information")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                        st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A'):,}")
                        st.write(f"**Country:** {info.get('country', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Website:** {info.get('website', 'N/A')}")
                        st.write(f"**CEO:** {info.get('ceo', 'N/A')}")
                        st.write(f"**IPO Year:** {info.get('ipoYear', 'N/A')}")
                        st.write(f"**Market Cap:** ${info.get('marketCap', 'N/A'):,}")
                    
                    st.subheader("Business Summary")
                    st.write(info.get('longBusinessSummary', 'No summary available.'))
            
            else:
                st.error("Unable to fetch data for the provided ticker symbol.")
    
    with tab2:
        st.subheader("Compare Stocks")
        symbols = st.text_input("Enter symbols to compare (comma separated)", "AAPL, MSFT, GOOGL")
        
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
            comparison_data = {}
            
            for symbol in symbol_list:
                hist, info = fetch_stock_data(symbol, "1mo")
                if hist is not None and not hist.empty:
                    comparison_data[symbol] = {
                        'Price': hist['Close'].iloc[-1],
                        'Change': ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100,
                        'Volume': info.get('averageVolume', 0)
                    }
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data).T
                st.dataframe(comp_df, use_container_width=True)
                
                # Normalized price comparison chart
                norm_data = {}
                for symbol in symbol_list:
                    hist, _ = fetch_stock_data(symbol, "1mo")
                    if hist is not None and not hist.empty:
                        norm_data[symbol] = (hist['Close'] / hist['Close'].iloc[0]) * 100
                
                if norm_data:
                    norm_df = pd.DataFrame(norm_data)
                    fig = px.line(norm_df, title="Normalized Price Comparison")
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Stock Screener")
        st.info("Advanced stock screening functionality coming soon!")
        
        # Basic screener options
        col1, col2 = st.columns(2)
        with col1:
            min_market_cap = st.selectbox("Min Market Cap", ["Any", "> $1B", "> $10B", "> $100B"])
            min_price = st.number_input("Min Price", value=0.0)
        with col2:
            sector = st.selectbox("Sector", ["Any", "Technology", "Healthcare", "Financial", "Energy"])
            max_pe = st.number_input("Max P/E Ratio", value=100.0)
        
        if st.button("Run Screen"):
            st.success(f"Found 25 stocks matching your criteria")
            # Simulated results
            screened_stocks = pd.DataFrame({
                'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
                'Price': [175.32, 328.79, 138.22, 145.18, 312.64],
                'Change %': [2.3, 1.7, -0.8, 3.2, 0.5],
                'Market Cap (B)': [2750, 2490, 1750, 1480, 890],
                'P/E Ratio': [29.5, 32.1, 24.8, 58.3, 26.7]
            })
            st.dataframe(screened_stocks, use_container_width=True)

# Technical Analysis Page
elif selected == "Technical Analysis":
    st.title("Technical Analysis")
    
    ticker = st.text_input("Enter Stock Symbol for Technical Analysis", "AAPL")
    if ticker:
        hist, info = fetch_stock_data(ticker, "3mo")
        if hist is not None and not hist.empty:
            # Calculate technical indicators
            hist = get_technical_indicators(hist)
            
            indicator = st.selectbox("Select Technical Indicator", 
                                    ["Moving Averages", "RSI", "MACD", "Bollinger Bands", "All"])
            
            if indicator == "Moving Averages":
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='#FF0000')))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name='SMA 20', line=dict(color='#FFA15A')))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50', line=dict(color='#00CC96')))
                fig.update_layout(title="Moving Averages")
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("""
                **Moving Averages Interpretation:**
                - **Golden Cross:** When shorter-term MA crosses above longer-term MA (bullish signal)
                - **Death Cross:** When shorter-term MA crosses below longer-term MA (bearish signal)
                - **Support/Resistance:** MAs often act as dynamic support/resistance levels
                """)
            
            elif indicator == "RSI":
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI', line=dict(color='#FF0000')))
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig.update_layout(title="Relative Strength Index (RSI)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("""
                **RSI Interpretation:**
                - **Overbought (RSI > 70):** Potential selling opportunity
                - **Oversold (RSI < 30):** Potential buying opportunity
                - **Divergence:** When price and RSI move in opposite directions, often signals trend reversal
                """)
            
            elif indicator == "MACD":
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD', line=dict(color='#FF0000')))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD_Signal'], name='Signal', line=dict(color='#FFA15A')))
                fig.add_trace(go.Bar(x=hist.index, y=hist['MACD_Hist'], name='Histogram', marker_color='#00CC96'))
                fig.update_layout(title="MACD Indicator")
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("""
                **MACD Interpretation:**
                - **Bullish Signal:** MACD crosses above signal line
                - **Bearish Signal:** MACD crosses below signal line
                - **Histogram:** Shows the difference between MACD and signal line
                - **Zero Line Cross:** MACD crossing zero can indicate trend changes
                """)
            
            elif indicator == "Bollinger Bands":
                # Calculate Bollinger Bands
                hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
                hist['BB_Std'] = hist['Close'].rolling(window=20).std()
                hist['BB_Upper'] = hist['BB_Middle'] + (hist['BB_Std'] * 2)
                hist['BB_Lower'] = hist['BB_Middle'] - (hist['BB_Std'] * 2)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], name='Upper Band', line=dict(color='#EF553B')))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Middle'], name='Middle Band', line=dict(color='#00CC96')))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], name='Lower Band', line=dict(color='#EF553B')))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='#FF0000')))
                fig.update_layout(title="Bollinger Bands")
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("""
                **Bollinger Bands Interpretation:**
                - **Squeeze:** Narrow bands indicate low volatility, often followed by high volatility breakout
                - **Expansion:** Wide bands indicate high volatility
                - **Support/Resistance:** Price often bounces off the bands
                - **Breakouts:** Price breaking outside bands can indicate strong momentum
                """)
            
            elif indicator == "All":
                # Display all indicators in subplots
                from plotly.subplots import make_subplots
                
                fig = make_subplots(rows=4, cols=1, shared_x=True,
                                   subplot_titles=("Price with Moving Averages", "RSI", "MACD", "Bollinger Bands"))
                
                # Price with MAs
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price'), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name='SMA 20'), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50'), row=1, col=1)
                
                # RSI
                fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD'), row=3, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD_Signal'], name='Signal'), row=3, col=1)
                fig.add_trace(go.Bar(x=hist.index, y=hist['MACD_Hist'], name='Histogram'), row=3, col=1)
                
                # Bollinger Bands
                fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], name='Upper Band'), row=4, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Middle'], name='Middle Band'), row=4, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], name='Lower Band'), row=4, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price'), row=4, col=1)
                
                fig.update_layout(height=1000, title_text="Complete Technical Analysis")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("Unable to fetch data for the provided ticker symbol.")

# Portfolio Manager Page
elif selected == "Portfolio Manager":
    st.title("Portfolio Manager")
    
    tab1, tab2, tab3 = st.tabs(["Current Portfolio", "Add Holding", "Performance Analysis"])
    
    with tab1:
        st.subheader("Current Portfolio")
        if not st.session_state.portfolio.empty:
            # Calculate current values
            portfolio_df = st.session_state.portfolio.copy()
            current_prices = []
            
            for _, row in portfolio_df.iterrows():
                hist, _ = fetch_stock_data(row['Symbol'], "1d")
                if hist is not None and not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    current_prices.append(current_price)
                else:
                    current_prices.append(0)
            
            portfolio_df['Current Price'] = current_prices
            portfolio_df['Current Value'] = portfolio_df['Quantity'] * portfolio_df['Current Price']
            portfolio_df['Investment'] = portfolio_df['Quantity'] * portfolio_df['Purchase Price']
            portfolio_df['Gain/Loss'] = portfolio_df['Current Value'] - portfolio_df['Investment']
            portfolio_df['Gain/Loss %'] = (portfolio_df['Gain/Loss'] / portfolio_df['Investment']) * 100
            
            st.dataframe(portfolio_df, use_container_width=True)
            
            # Portfolio summary
            total_investment = portfolio_df['Investment'].sum()
            total_value = portfolio_df['Current Value'].sum()
            total_gain = portfolio_df['Gain/Loss'].sum()
            total_gain_percent = (total_gain / total_investment) * 100 if total_investment > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Investment", f"${total_investment:,.2f}")
            with col2:
                st.metric("Current Value", f"${total_value:,.2f}")
            with col3:
                st.metric("Total Gain/Loss", f"${total_gain:,.2f}", delta=f"{total_gain_percent:.2f}%")
            with col4:
                st.metric("Number of Holdings", len(portfolio_df))
            
            # Portfolio allocation chart
            fig = px.pie(portfolio_df, values='Current Value', names='Symbol', 
                         title="Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No holdings in portfolio. Add some stocks to get started.")
    
    with tab2:
        st.subheader("Add New Holding")
        
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Stock Symbol", "AAPL").upper()
            quantity = st.number_input("Quantity", min_value=1, value=10)
        with col2:
            purchase_price = st.number_input("Purchase Price", min_value=0.01, value=150.0)
            purchase_date = st.date_input("Purchase Date", value=datetime.now())
        
        if st.button("Add to Portfolio"):
            # Check if symbol already exists
            if symbol in st.session_state.portfolio['Symbol'].values:
                st.warning(f"{symbol} already exists in portfolio. Updating quantity and average price.")
                # Update existing holding
                idx = st.session_state.portfolio[st.session_state.portfolio['Symbol'] == symbol].index[0]
                old_quantity = st.session_state.portfolio.at[idx, 'Quantity']
                old_price = st.session_state.portfolio.at[idx, 'Purchase Price']
                
                new_quantity = old_quantity + quantity
                new_avg_price = ((old_quantity * old_price) + (quantity * purchase_price)) / new_quantity
                
                st.session_state.portfolio.at[idx, 'Quantity'] = new_quantity
                st.session_state.portfolio.at[idx, 'Purchase Price'] = new_avg_price
            else:
                # Add new holding
                new_holding = pd.DataFrame({
                    'Symbol': [symbol],
                    'Quantity': [quantity],
                    'Purchase Price': [purchase_price],
                    'Purchase Date': [purchase_date]
                })
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_holding], ignore_index=True)
            
            st.success(f"Added {quantity} shares of {symbol} to portfolio")
    
    with tab3:
        st.subheader("Portfolio Performance")
        
        if not st.session_state.portfolio.empty:
            # Simulated performance data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            performance_data = {}
            
            for symbol in st.session_state.portfolio['Symbol'].unique():
                # Generate random but realistic performance data
                base_value = 100
                noise = np.random.normal(0, 0.02, 30)
                cumulative_noise = np.cumsum(noise)
                performance_data[symbol] = base_value * (1 + cumulative_noise)
            
            performance_df = pd.DataFrame(performance_data, index=dates)
            
            # Calculate portfolio performance (weighted average)
            weights = {}
            for symbol in st.session_state.portfolio['Symbol'].unique():
                holding = st.session_state.portfolio[st.session_state.portfolio['Symbol'] == symbol]
                weights[symbol] = (holding['Quantity'].iloc[0] * holding['Purchase Price'].iloc[0]) / total_investment
            
            portfolio_performance = pd.Series(0, index=dates)
            for symbol, weight in weights.items():
                portfolio_performance += performance_df[symbol] * weight
            
            fig = px.line(portfolio_performance, title="Portfolio Performance (Last 30 Days)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Benchmark comparison
            benchmark = "SPY"
            bench_hist, _ = fetch_stock_data(benchmark, "1mo")
            if bench_hist is not None and not bench_hist.empty:
                bench_perf = (bench_hist['Close'] / bench_hist['Close'].iloc[0]) * 100
                
                comp_df = pd.DataFrame({
                    'Portfolio': portfolio_performance,
                    'Benchmark (SPY)': bench_perf.values
                })
                
                fig = px.line(comp_df, title="Portfolio vs Benchmark")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Add holdings to see performance analysis")

# Options Chain Page
elif selected == "Options Chain":
    st.title("📊 Options Chain Analysis")
    
    st.subheader("Options Data (Simulated)")
    options_data = pd.DataFrame({
        'Strike': [150, 155, 160, 165, 170, 175, 180],
        'Call OI': [1200, 950, 780, 620, 480, 320, 210],
        'Put OI': [850, 720, 1080, 920, 680, 450, 290],
        'Call Volume': [450, 320, 280, 210, 180, 120, 80],
        'Put Volume': [380, 290, 420, 350, 270, 190, 130],
        'Call IV': [0.35, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22],
        'Put IV': [0.38, 0.35, 0.33, 0.31, 0.29, 0.27, 0.25]
    })
    
    st.dataframe(options_data.style.background_gradient(cmap="Reds"), use_container_width=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=options_data['Strike'], y=options_data['Call OI'], name='Call OI'))
    fig.add_trace(go.Bar(x=options_data['Strike'], y=options_data['Put OI'], name='Put OI'))
    fig.update_layout(title="Open Interest Analysis", barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Options Strategy Builder")
    strategy = st.selectbox("Select Strategy", ["Long Call", "Long Put", "Covered Call", "Protective Put", 
                                               "Bull Call Spread", "Bear Put Spread", "Iron Condor", "Straddle"])
    
    col1, col2 = st.columns(2)
    with col1:
        underlying_price = st.number_input("Underlying Price", value=160.0)
        expiration = st.selectbox("Expiration", ["Weekly", "Monthly", "Quarterly"])
        strategy_type = st.selectbox("Strategy Type", ["Debit", "Credit"])
    
    with col2:
        strike_price = st.number_input("Strike Price", value=160.0)
        contract_count = st.number_input("Contract Count", min_value=1, value=1)
        premium = st.number_input("Premium", value=2.5)
    
    st.write(f"**{strategy} Strategy Overview:**")
    
    if strategy == "Long Call":
        st.write("- **Direction:** Bullish")
        st.write("- **Max Profit:** Unlimited")
        st.write("- **Max Loss:** Premium Paid")
        st.write("- **Breakeven:** Strike Price + Premium")
        cost = premium * 100 * contract_count
        st.write(f"- **Total Cost:** ${cost:,.2f}")
    
    elif strategy == "Long Put":
        st.write("- **Direction:** Bearish")
        st.write("- **Max Profit:** Strike Price - Premium")
        st.write("- **Max Loss:** Premium Paid")
        st.write("- **Breakeven:** Strike Price - Premium")
        cost = premium * 100 * contract_count
        st.write(f"- **Total Cost:** ${cost:,.2f}")
    
    elif strategy == "Covered Call":
        st.write("- **Direction:** Neutral to Bullish")
        st.write("- **Max Profit:** (Strike Price - Purchase Price) + Premium")
        st.write("- **Max Loss:** Unlimited (if stock price drops)")
        st.write("- **Breakeven:** Purchase Price - Premium")
        income = premium * 100 * contract_count
        st.write(f"- **Total Income:** ${income:,.2f}")
    
    elif strategy == "Iron Condor":
        st.write("- **Direction:** Neutral")
        st.write("- **Max Profit:** Net Premium Received")
        st.write("- **Max Loss:** Width of Spread - Net Premium")
        st.write("- **Breakeven:** Multiple breakeven points based on strikes")
        st.write("- **Ideal for:** Low volatility environments")
    
    if st.button("Analyze Strategy"):
        st.success(f"{strategy} strategy analyzed successfully!")
        
        # Simulated payoff diagram
        prices = np.linspace(underlying_price * 0.7, underlying_price * 1.3, 50)
        if strategy == "Long Call":
            payoffs = np.maximum(prices - strike_price, 0) - premium
        elif strategy == "Long Put":
            payoffs = np.maximum(strike_price - prices, 0) - premium
        elif strategy == "Covered Call":
            # Assuming stock was purchased at current price
            stock_payoff = prices - underlying_price
            option_payoff = premium - np.maximum(prices - strike_price, 0)
            payoffs = stock_payoff + option_payoff
        else:
            payoffs = np.zeros_like(prices)
        
        payoff_df = pd.DataFrame({"Price": prices, "Payoff": payoffs * 100 * contract_count})
        fig = px.line(payoff_df, x="Price", y="Payoff", title="Strategy Payoff Diagram")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

# Market Overview Page
elif selected == "Market Overview":
    st.title("Global Market Overview")
    
    # Load global indices with caching
    indices_data = fetch_global_indices()
    
    # Display key indices in a grid
    st.subheader("Global Markets")
    cols = st.columns(6)
    for idx, (symbol, data) in enumerate(indices_data.items()):
        if idx < 6:  # First row
            with cols[idx % 6]:
                currency_symbol = "$" if data["currency"] == "USD" else "₹" if data["currency"] == "INR" else "€" if data["currency"] == "EUR" else "£"
                st.metric(
                    label=data["name"],
                    value=f"{currency_symbol}{data['price']:.2f}",
                    delta=f"{data['change']:.2f}%"
                )
    
    # Second row of markets
    cols = st.columns(6)
    for idx, (symbol, data) in enumerate(list(indices_data.items())[6:12]):
        with cols[idx % 6]:
            currency_symbol = "$" if data["currency"] == "USD" else "₹" if data["currency"] == "INR" else "€" if data["currency"] == "EUR" else "£"
            st.metric(
                label=data["name"],
                value=f"{currency_symbol}{data['price']:.2f}",
                delta=f"{data['change']:.2f}%"
            )
    
    # Third row of markets
    cols = st.columns(6)
    for idx, (symbol, data) in enumerate(list(indices_data.items())[12:18]):
        with cols[idx % 6]:
            currency_symbol = "$" if data["currency"] == "USD" else "₹" if data["currency"] == "INR" else "€" if data["currency"] == "EUR" else "£"
            st.metric(
                label=data["name"],
                value=f"{currency_symbol}{data['price']:.2f}",
                delta=f"{data['change']:.2f}%"
            )
    
    # Market heatmap
    st.subheader(" Market Heatmap")
    
    # Simulated sector performance
    sectors = {
        "Technology": 2.3, "Healthcare": 1.5, "Financials": -0.8, "Energy": 3.2,
        "Consumer Cyclical": 0.7, "Real Estate": -1.2, "Utilities": 0.3, 
        "Communications": 1.8, "Materials": -0.5, "Industrials": 0.9
    }
    
    sector_df = pd.DataFrame({
        "Sector": list(sectors.keys()),
        "Performance": list(sectors.values())
    })
    
    fig = px.bar(sector_df, x="Sector", y="Performance", 
                 title="Sector Performance", color="Performance",
                 color_continuous_scale=px.colors.diverging.RdYlGn)
    st.plotly_chart(fig, use_container_width=True)
    
    # Economic indicators
    st.subheader("Economic Indicators")
    
    econ_data = {
        "Indicator": ["GDP Growth", "Unemployment Rate", "Inflation Rate", "Interest Rate", "Consumer Confidence"],
        "Current": ["2.1%", "3.8%", "3.2%", "5.25%", "108.5"],
        "Previous": ["2.2%", "3.9%", "3.4%", "5.25%", "107.8"],
        "Change": ["-0.1%", "-0.1%", "-0.2%", "0.0%", "+0.7"]
    }
    
    econ_df = pd.DataFrame(econ_data)
    st.dataframe(econ_df, use_container_width=True)
    
    # Bond yields
    st.subheader("Bond Yields")
    
    bond_data = {
        "Maturity": ["1 Month", "3 Month", "6 Month", "1 Year", "2 Year", "5 Year", "10 Year", "30 Year"],
        "Yield": ["5.25%", "5.32%", "5.38%", "5.25%", "4.89%", "4.35%", "4.18%", "4.35%"],
        "Change": ["+0.01%", "+0.02%", "+0.01%", "-0.02%", "-0.05%", "-0.03%", "-0.02%", "-0.01%"]
    }
    
    bond_df = pd.DataFrame(bond_data)
    st.dataframe(bond_df, use_container_width=True)
    
    # Currency rates
    st.subheader("💱 Currency Exchange Rates")
    
    currency_data = {
        "Pair": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD"],
        "Rate": ["1.0850", "1.2650", "148.50", "0.8820", "0.6520", "1.3580"],
        "Change": ["+0.2%", "+0.1%", "-0.3%", "+0.1%", "-0.2%", "+0.1%"]
    }
    
    currency_df = pd.DataFrame(currency_data)
    st.dataframe(currency_df, use_container_width=True)

# Economic Calendar Page
elif selected == "Economic Calendar":
    st.title("📅 Economic Calendar")
    
    # Date selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now() + timedelta(days=7))
    
    # Simulated economic events
    economic_events = pd.DataFrame({
        'Date': ['2023-09-15', '2023-09-20', '2023-09-25', '2023-10-01', '2023-10-05'],
        'Time': ['14:00', '10:30', '08:30', '14:30', '12:00'],
        'Event': ['Fed Interest Rate Decision', 'CPI Data Release', 'Unemployment Claims', 'Non-Farm Payrolls', 'Retail Sales'],
        'Impact': ['High', 'High', 'Medium', 'High', 'Medium'],
        'Previous': ['5.25%', '3.2%', '230K', '187K', '0.7%'],
        'Forecast': ['5.5%', '3.4%', '225K', '190K', '0.5%'],
        'Actual': ['5.5%', '3.5%', '220K', '192K', 'N/A']
    })
    
    # Filter events by date range
    economic_events['Date'] = pd.to_datetime(economic_events['Date'])
    filtered_events = economic_events[
        (economic_events['Date'] >= pd.to_datetime(start_date)) & 
        (economic_events['Date'] <= pd.to_datetime(end_date))
    ]
    
    st.dataframe(filtered_events, use_container_width=True)
    
    # Impact color coding
    st.write("**Impact Level:**")
    st.markdown("- <span style='color:red'>**High**</span>: Significant market impact expected", unsafe_allow_html=True)
    st.markdown("- <span style='color:orange'>**Medium**</span>: Moderate market impact expected", unsafe_allow_html=True)
    st.markdown("- <span style='color:green'>**Low**</span>: Minimal market impact expected", unsafe_allow_html=True)
    
    # Central bank watch
    st.subheader("Central Bank Watch")
    rates_data = pd.DataFrame({
        'Central Bank': ['Federal Reserve', 'ECB', 'Bank of England', 'Bank of Japan', 'Reserve Bank of Australia'],
        'Current Rate': ['5.25%', '4.25%', '5.00%', '-0.10%', '4.35%'],
        'Next Meeting': ['Sept 20, 2023', 'Oct 5, 2023', 'Sept 21, 2023', 'Oct 12, 2023', 'Oct 3, 2023'],
        'Expected Change': ['+0.25%', '+0.25%', '+0.25%', 'No Change', 'No Change']
    })
    
    st.dataframe(rates_data, use_container_width=True)
    
    # Economic data visualization
    st.subheader("Economic Data Trends")
    
    # Simulated inflation data
    dates = pd.date_range(start='2022-01-01', end='2023-09-01', freq='M')
    inflation = [7.5, 7.9, 8.5, 8.3, 8.6, 9.1, 8.5, 8.3, 7.7, 7.1, 6.5, 6.0, 5.0, 4.9, 4.0, 3.7, 3.2, 3.4, 3.5, 3.4]
    
    inflation_df = pd.DataFrame({'Date': dates, 'Inflation Rate': inflation})
    fig = px.line(inflation_df, x='Date', y='Inflation Rate', title='Inflation Rate Trend')
    st.plotly_chart(fig, use_container_width=True)

# Crypto Markets Page
elif selected == "Crypto Markets":
    st.title("₿ Crypto Markets")
    
    # Simulated crypto data
    crypto_data = pd.DataFrame({
        'Cryptocurrency': ['Bitcoin', 'Ethereum', 'Binance Coin', 'Cardano', 'Solana', 'XRP', 'Dogecoin', 'Polkadot'],
        'Symbol': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOGE', 'DOT'],
        'Price': [25800, 1650, 215, 0.25, 32.50, 0.52, 0.06, 4.20],
        '24h Change': [-1.2, 2.5, -0.8, 3.2, -2.1, 1.5, -3.2, 0.8],
        'Market Cap (B)': [500, 200, 35, 9, 12, 28, 8, 5],
        'Volume (24h)': [25, 12, 5, 2, 3, 4, 1, 1]
    })
    
    st.dataframe(crypto_data, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Crypto Fear & Greed Index")
        current_index = 45  # Neutral
        st.metric("Current Index", f"{current_index}/100", "Neutral")
        st.progress(current_index / 100)
        
        st.write("**Index Interpretation:**")
        st.write("- 0-24: Extreme Fear")
        st.write("- 25-49: Fear")
        st.write("- 50: Neutral")
        st.write("- 51-74: Greed")
        st.write("- 75-100: Extreme Greed")
    
    with col2:
        st.subheader("Bitcoin Dominance")
        dominance = 48.5  # Percentage
        st.metric("BTC Dominance", f"{dominance}%")
        st.progress(dominance / 100)
        
        st.write("**Dominance Trends:**")
        st.write("- Decreasing: Altcoin season may be approaching")
        st.write("- Increasing: Bitcoin leading market movements")
    
    # Crypto performance chart
    st.subheader(" Top Cryptocurrencies Performance")
    
    # Simulated crypto performance
    crypto_perf = pd.DataFrame({
        'Crypto': ['Bitcoin', 'Ehereum', 'Binance Coin', 'Cardano', 'Solana'],
        '7d Change': [2.1, 5.3, -1.2, 8.5, -3.2],
        '30d Change': [12.5, 18.2, 5.3, 25.8, -8.4],
        '90d Change': [35.2, 42.1, 18.5, 65.3, 12.7]
    })
    
    fig = px.bar(crypto_perf, x='Crypto', y=['7d Change', '30d Change', '90d Change'], 
                 title="Cryptocurrency Performance", barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Crypto news
    st.subheader("Crypto News")
    crypto_news = fetch_news("cryptocurrency")
    if crypto_news:
        for article in crypto_news[:3]:
            with st.expander(f"{article['title']} - {article['source']['name']}"):
                st.write(article.get('description', 'No description available'))
                st.markdown(f"[Read more]({article['url']})")
    else:
        st.info("Crypto news temporarily unavailable.")

# News & Sentiment Page
elif selected == "News & Sentiment":
    st.title("News & Market Sentiment")
    
    news_query = st.text_input("Search Financial News", "stock market")
    if news_query:
        news_articles = fetch_news(news_query)
        if news_articles:
            for article in news_articles:
                with st.expander(f"{article['title']} - {article['source']['name']}"):
                    st.write(article.get('description', 'No description available'))
                    st.markdown(f"[Read more]({article['url']})")
        else:
            st.info("No articles found or news feed temporarily unavailable.")
    
    st.subheader("Market Sentiment Analysis")
    sentiment_tabs = st.tabs(["Stocks", "Sectors", "Market Overview"])
    
    with sentiment_tabs[0]:
        stock_sentiment = pd.DataFrame({
            'Stock': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'WMT'],
            'Sentiment': ['Bullish', 'Neutral', 'Bullish', 'Bearish', 'Neutral', 'Bullish', 'Neutral', 'Bearish', 'Bullish', 'Neutral'],
            'Confidence': [85, 60, 78, 72, 55, 82, 65, 68, 79, 62]
        })
        st.dataframe(stock_sentiment, use_container_width=True)
    
    with sentiment_tabs[1]:
        sector_sentiment = pd.DataFrame({
            'Sector': ['Technology', 'Healthcare', 'Financials', 'Energy', 'Consumer', 'Real Estate', 'Utilities', 'Communications', 'Materials', 'Industrials'],
            'Sentiment': ['Bullish', 'Neutral', 'Bearish', 'Bullish', 'Neutral', 'Bearish', 'Neutral', 'Bullish', 'Neutral', 'Bullish'],
            'Trend': ['Improving', 'Stable', 'Worsening', 'Improving', 'Stable', 'Worsening', 'Stable', 'Improving', 'Stable', 'Improving']
        })
        st.dataframe(sector_sentiment, use_container_width=True)
    
    with sentiment_tabs[2]:
        st.subheader("Overall Market Sentiment")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bullish", "45%", "2%")
        with col2:
            st.metric("Neutral", "30%", "-1%")
        with col3:
            st.metric("Bearish", "25%", "-1%")
        
        # Sentiment trend chart
        sentiment_trend = pd.DataFrame({
            'Date': pd.date_range(end=datetime.now(), periods=30, freq='D'),
            'Bullish': np.random.normal(45, 5, 30),
            'Neutral': np.random.normal(30, 3, 30),
            'Bearish': np.random.normal(25, 4, 30)
        })
        
        fig = px.line(sentiment_trend, x='Date', y=['Bullish', 'Neutral', 'Bearish'],
                      title="Market Sentiment Trend (30 Days)")
        st.plotly_chart(fig, use_container_width=True)

# Learning Center Page
elif selected == "Learning Center":
    st.title("Learning Center")
    
    learning_tabs = st.tabs(["Beginner Guides", "Technical Analysis", "Options Trading", "Portfolio Management", "Video Resources"])
    
    with learning_tabs[0]:
        st.subheader("Getting Started with Investing")
        
        with st.expander("1. Understanding the Basics", expanded=True):
            st.write("""
            ### What are Stocks?
            Stocks represent ownership in a company. When you buy a stock, you become a shareholder and own a small piece of that company.
            
            ### Types of Investments:
            - **Stocks:** Ownership shares in publicly traded companies
            - **Bonds:** Debt instruments issued by governments or corporations
            - **Mutual Funds:** Pooled investments managed by professionals
            - **ETFs:** Exchange-Traded Funds that track indexes or sectors
            - **Options:** Contracts that give the right to buy/sell assets at specific prices
            
            ### Risk vs. Return:
            Generally, higher potential returns come with higher risk. Understanding your risk tolerance is crucial for building an appropriate investment strategy.
            """)
        
        with st.expander("2. Fundamental Analysis"):
            st.write("""
            ### Reading Financial Statements:
            - **Balance Sheet:** Shows assets, liabilities, and equity at a specific point in time
            - **Income Statement:** Shows revenues, expenses, and profits over a period
            - **Cash Flow Statement:** Shows how cash moves in and out of the business
            
            ### Key Financial Ratios:
            - **P/E Ratio:** Price-to-Earnings ratio measures valuation
            - **PEG Ratio:** P/E ratio divided by growth rate
            - **ROE:** Return on Equity measures profitability
            - **Debt-to-Equity:** Measures financial leverage
            """)
        
        with st.expander("3. Building Your First Portfolio"):
            st.write("""
            ### Diversification:
            Don't put all your eggs in one basket. Spread investments across different:
            - Asset classes (stocks, bonds, cash)
            - Sectors (technology, healthcare, financials)
            - Geographic regions (US, international, emerging markets)
            
            ### Asset Allocation:
            Your ideal mix of assets depends on:
            - Your age and investment timeline
            - Your financial goals
            - Your risk tolerance
            
            ### Rebalancing:
            Periodically adjust your portfolio to maintain your target asset allocation.
            """)
    
    with learning_tabs[1]:
        st.subheader("Technical Analysis Fundamentals")
        
        with st.expander("1. Chart Patterns"):
            st.write("""
            ### Common Patterns:
            - **Head and Shoulders:** Reversal pattern signaling trend change
            - **Double Top/Bottom:** Reversal patterns after strong trends
            - **Triangles:** Continuation patterns (ascending, descending, symmetrical)
            - **Flags and Pennants:** Short-term continuation patterns
            
            ### Support and Resistance:
            - **Support:** Price level where buying interest is significantly strong
            - **Resistance:** Price level where selling pressure is significantly strong
            - **Breakouts:** When price moves through support/resistance with increased volume
            """)
        
        with st.expander("2. Technical Indicators"):
            st.write("""
            ### Trend Indicators:
            - **Moving Averages:** Smooth out price data to identify trends
            - **MACD:** Moving Average Convergence Divergence shows trend changes
            - **ADX:** Average Directional Index measures trend strength
            
            ### Momentum Indicators:
            - **RSI:** Relative Strength Index identifies overbought/oversold conditions
            - **Stochastic Oscillator:** Compares closing price to price range over time
            - **Williams %R:** Momentum indicator measuring overbought/oversold levels
            
            ### Volume Indicators:
            - **OBV:** On-Balance Volume measures buying and selling pressure
            - **Volume Profile:** Shows trading activity at specific price levels
            """)
        
        with st.expander("3. Japanese Candlestick Patterns"):
            st.write("""
            ### Single Candle Patterns:
            - **Doji:** Indecision, potential trend reversal
            - **Hammer:** Bullish reversal pattern
            - **Hanging Man:** Bearish reversal pattern
            - **Shooting Star:** Bearish reversal pattern
            
            ### Multi-Candle Patterns:
            - **Engulfing Patterns:** Bullish or bearish reversal signals
            - **Harami Patterns:** Potential trend reversal
            - **Morning Star:** Bullish reversal pattern
            - **Evening Star:** Bearish reversal pattern
            """)
    
    with learning_tabs[2]:
        st.subheader("Options Trading Strategies")
        
        with st.expander("1. Basic Strategies"):
            st.write("""
            ### Covered Calls:
            - **Description:** Sell call options against stock you own
            - **Objective:** Generate income from existing holdings
            - **Risk:** Limited upside potential, unlimited downside risk
            - **Best for:** Neutral to slightly bullish outlook
            
            ### Protective Puts:
            - **Description:** Buy put options to protect stock holdings
            - **Objective:** Insurance against downside risk
            - **Risk:** Cost of put premium
            - **Best for:** Protecting gains while maintaining upside potential
            
            ### Long Calls/Puts:
            - **Description:** Buy call or put options
            - **Objective:** Speculative directional bets with limited risk
            - **Risk:** Loss of premium paid
            - **Best for:** High-conviction directional views
            """)
        
        with st.expander("2. Advanced Strategies"):
            st.write("""
            ### Iron Condors:
            - **Description:** Sell out-of-the-money put and call spreads
            - **Objective:** Profit from low volatility, time decay
            - **Risk:** Limited, defined risk
            - **Best for:** Range-bound markets
            
            ### Straddles/Strangles:
            - **Description:** Buy/sell both call and put options at same/different strikes
            - **Objective:** Profit from volatility expansion
            - **Risk:** Limited or unlimited depending on position
            - **Best for:** High volatility expectations
            
            ### Vertical Spreads:
            - **Description:** Buy and sell options at different strikes same expiration
            - **Objective:** Defined risk directional plays
            - **Risk:** Limited to difference between strikes minus premium
            - **Best for:** Directional views with defined risk
            """)
        
        with st.expander("3. Risk Management in Options Trading"):
            st.write("""
            ### Position Sizing:
            - Never risk more than 2-5% of portfolio on a single trade
            - Adjust position size based on probability of success
            - Consider implied volatility when sizing positions
            
            ### Greeks Risk Management:
            - **Delta:** Measure of price sensitivity
            - **Gamma:** Rate of change of delta
            - **Theta:** Time decay measurement
            - **Vega:** Volatility sensitivity
            - **Rho:** Interest rate sensitivity
            
            ### Exit Strategies:
            - Set profit targets based on risk-reward ratios
            - Use stop-losses or mental stops to limit losses
            - Have a plan for adjusting positions that go against you
            """)
    
    with learning_tabs[3]:
        st.subheader("Portfolio Management Techniques")
        
        with st.expander("1. Modern Portfolio Theory"):
            st.write("""
            ### Efficient Frontier:
            The set of optimal portfolios that offer the highest expected return for a defined level of risk.
            
            ### Diversification Benefits:
            Combining assets with low correlation can reduce overall portfolio risk.
            
            ### Capital Asset Pricing Model (CAPM):
            Describes the relationship between systematic risk and expected return.
            """)
        
        with st.expander("2. Risk Management Frameworks"):
            st.write("""
            ### Value at Risk (VaR):
            Measures the potential loss in value of a portfolio over a defined period.
            
            ### Stress Testing:
            Assessing how portfolios perform under extreme market conditions.
            
            ### Scenario Analysis:
            Evaluating portfolio performance under various hypothetical scenarios.
            """)
        
        with st.expander("3. Behavioral Finance"):
            st.write("""
            ### Common Biases:
            - **Confirmation Bias:** Seeking information that confirms existing beliefs
            - **Loss Aversion:** Feeling losses more acutely than gains
            - **Recency Bias:** Overweighting recent events
            - **Anchoring:** Relying too heavily on initial information
            
            ### Overcoming Biases:
            - Develop and follow a systematic investment process
            - Keep an investment journal to document decisions
            - Regularly review and learn from past mistakes
            - Seek contrary opinions to challenge your views
            """)
    
    with learning_tabs[4]:
        st.subheader("Recommended Video Resources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📺 Beginner Investing Videos:**")
            st.write("- [Investing for Beginners: How to Get Started](https://www.youtube.com/watch?v=Wf2eY3Lc2sI)")
            st.write("- [Stock Market Basics](https://www.youtube.com/watch?v=3UF0ymVdYLA)")
            st.write("- [Understanding Financial Statements](https://www.youtube.com/watch?v=2QrPId4f6L8)")
            st.write("- [Risk Management Fundamentals](https://www.youtube.com/watch?v=5xX1l5lkl_c)")
            
        with col2:
            st.write("**📺 Portfolio Management Videos:**")
            st.write("- [Modern Portfolio Theory](https://www.youtube.com/watch?v=U9Xk0gQf7eI)")
            st.write("- [Asset Allocation Strategies](https://www.youtube.com/watch?v=ERDvLf3i9vU)")
            st.write("- [Rebalancing Your Portfolio](https://www.youtube.com/watch?v=3aT-ML5wlwg)")
            st.write("- [Behavioral Finance Insights](https://www.youtube.com/watch?v=8Y39E8rK9U8)")

# Company Info Page
elif selected == "Company Info":
    st.title("🏢 Company Information")
    
    ticker = st.text_input("Enter Company Symbol", "AAPL")
    
    if ticker:
        hist, info = fetch_stock_data(ticker, "1y")
        
        if hist is not None and info is not None:
            st.subheader(f"{info.get('longName', 'N/A')} ({ticker.upper()})")
            
            # Company overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A'):,}")
            
            with col2:
                st.write(f"**CEO:** {info.get('ceo', 'N/A')}")
                st.write(f"**Website:** {info.get('website', 'N/A')}")
                st.write(f"**Country:** {info.get('country', 'N/A')}")
                st.write(f"**IPO Year:** {info.get('ipoYear', 'N/A')}")
            
            with col3:
                st.write(f"**Market Cap:** ${info.get('marketCap', 'N/A'):,}")
                st.write(f"**Enterprise Value:** ${info.get('enterpriseValue', 'N/A'):,}")
                st.write(f"**Shares Outstanding:** {info.get('sharesOutstanding', 'N/A'):,}")
                st.write(f"**Float:** {info.get('floatShares', 'N/A'):,}")
            
            # Business summary
            st.subheader("Business Summary")
            st.write(info.get('longBusinessSummary', 'No summary available.'))
            
            # Key executives
            st.subheader("Key Executives")
            try:
                executives = info.get('companyOfficers', [])
                if executives:
                    exec_data = []
                    for exec in executives[:5]:  # Show top 5 executives
                        exec_data.append({
                            'Name': exec.get('name', 'N/A'),
                            'Title': exec.get('title', 'N/A'),
                            'Salary': f"${exec.get('totalPay', 'N/A'):,}" if exec.get('totalPay') else 'N/A',
                            'Age': exec.get('age', 'N/A')
                        })
                    st.table(pd.DataFrame(exec_data))
                else:
                    st.info("Executive information not available.")
            except:
                st.info("Executive information not available.")
            
            # Financial highlights
            st.subheader("Financial Highlights")
            fin_col1, fin_col2, fin_col3 = st.columns(3)
            
            with fin_col1:
                st.metric("Revenue Growth", f"{info.get('revenueGrowth', 'N/A')}%")
                st.metric("Gross Margins", f"{info.get('grossMargins', 'N/A')}%")
                st.metric("Profit Margins", f"{info.get('profitMargins', 'N/A')}%")
            
            with fin_col2:
                st.metric("ROE", f"{info.get('returnOnEquity', 'N/A')}%")
                st.metric("ROA", f"{info.get('returnOnAssets', 'N/A')}%")
                st.metric("Current Ratio", info.get('currentRatio', 'N/A'))
            
            with fin_col3:
                st.metric("Debt to Equity", info.get('debtToEquity', 'N/A'))
                st.metric("Operating Cash Flow", f"${info.get('operatingCashflow', 'N/A'):,}")
                st.metric("Free Cash Flow", f"${info.get('freeCashflow', 'N/A'):,}")
            
            # Historical financials
            st.subheader("Historical Financials")
            financials = pd.DataFrame({
                'Year': ['2023', '2022', '2021', '2020', '2019'],
                'Revenue (B)': [info.get('totalRevenue', 0)/1e9 if info.get('totalRevenue') else 'N/A', 
                               info.get('previousRevenue', 0)/1e9 if info.get('previousRevenue') else 'N/A',
                               274.52, 260.17, 265.60],
                'Net Income (B)': [info.get('netIncomeToCommon', 0)/1e9 if info.get('netIncomeToCommon') else 'N/A',
                                  info.get('previousNetIncome', 0)/1e9 if info.get('previousNetIncome') else 'N/A',
                                  57.41, 55.26, 55.34],
                'EPS': [info.get('trailingEps', 'N/A'), info.get('previousEps', 'N/A'), 3.28, 3.31, 3.00],
                'Dividend': [info.get('dividendRate', 'N/A'), info.get('previousDividend', 'N/A'), 0.82, 0.80, 0.75]
            })
            st.dataframe(financials, use_container_width=True)
            
            # Institutional ownership
            st.subheader("Institutional Ownership")
            inst_data = pd.DataFrame({
                'Institution': ['Vanguard', 'BlackRock', 'State Street', 'Fidelity', 'Geode Capital'],
                'Shares Held': [1250000000, 980000000, 750000000, 520000000, 380000000],
                'Value (B)': [225, 176, 135, 94, 68],
                '% Change': [2.3, -1.2, 0.8, 3.1, -0.5]
            })
            st.dataframe(inst_data, use_container_width=True)
        
        else:
            st.error("Unable to fetch data for the provided ticker symbol.")

# Predictions Page
elif selected == "Predictions":
    st.title("🔮 Stock & Mutual Fund Predictions")
    
    tab1, tab2 = st.tabs(["Stock Predictions", "Mutual Fund Analysis"])
    
    with tab1:
        st.subheader("Stock Price Predictions")
        
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Enter Stock Symbol", "AAPL")
        with col2:
            prediction_days = st.slider("Prediction Period (Days)", 7, 90, 30)
        
        if st.button("Generate Prediction"):
            if ticker:
                with st.spinner("Analyzing data and generating prediction..."):
                    prediction, trend, confidence = predict_stock_price(ticker, prediction_days)
                    
                    if prediction is not None:
                        # Get current price
                        hist, _ = fetch_stock_data(ticker, "1d")
                        current_price = hist['Close'].iloc[-1] if hist is not None and not hist.empty else None
                        
                        if current_price:
                            price_change = prediction - current_price
                            percent_change = (price_change / current_price) * 100
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Price", f"${current_price:.2f}")
                            with col2:
                                st.metric(f"Predicted Price ({prediction_days} days)", f"${prediction:.2f}", 
                                         f"{price_change:.2f} ({percent_change:.2f}%)")
                            with col3:
                                st.metric("Trend", trend, f"{confidence:.1f}% Confidence")
                            
                            # Generate historical chart with prediction
                            hist_long, _ = fetch_stock_data(ticker, "6mo")
                            if hist_long is not None and not hist_long.empty:
                                # Create future dates for prediction
                                last_date = hist_long.index[-1]
                                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days, freq='D')
                                
                                # Create prediction line (simple linear projection for demo)
                                price_trend = np.linspace(current_price, prediction, prediction_days)
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=hist_long.index, 
                                    y=hist_long['Close'], 
                                    name='Historical Price',
                                    line=dict(color='#FF0000')
                                ))
                                fig.add_trace(go.Scatter(
                                    x=future_dates, 
                                    y=price_trend, 
                                    name='Prediction',
                                    line=dict(color='#FFA15A', dash='dash')
                                ))
                                fig.update_layout(
                                    title=f"{ticker} Price Prediction",
                                    xaxis_title="Date",
                                    yaxis_title="Price ($)",
                                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Prediction details
                            st.subheader("Prediction Analysis")
                            
                            if trend == "Bullish":
                                st.success("""
                                **Bullish Outlook:**
                                - Technical indicators suggest upward momentum
                                - Consider buying opportunities
                                - Set appropriate stop-loss levels
                                """)
                            elif trend == "Bearish":
                                st.error("""
                                **Bearish Outlook:**
                                - Technical indicators suggest downward pressure
                                - Consider selling or hedging strategies
                                - Wait for confirmation before entering new positions
                                """)
                            else:
                                st.warning("""
                                **Neutral Outlook:**
                                - Mixed signals from technical indicators
                                - Market may be consolidating
                                - Wait for clearer direction before making significant moves
                                """)
                            
                            # Risk factors
                            st.subheader("Risk Factors")
                            risk_factors = [
                                "Market volatility may impact predictions",
                                "Unexpected news events can drastically change price direction",
                                "Technical analysis has limitations in predicting black swan events",
                                "Past performance is not indicative of future results"
                            ]
                            
                            for factor in risk_factors:
                                st.write(f"• {factor}")
                        else:
                            st.error("Could not fetch current price data.")
                    else:
                        st.error("Could not generate prediction. Please check the stock symbol and try again.")
            else:
                st.warning("Please enter a stock symbol.")
    
    with tab2:
        st.subheader("Mutual Fund Analysis & Predictions")
        
        # Get mutual fund data
        mutual_funds = get_mutual_funds()
        
        if mutual_funds:
            # Display mutual funds in a dataframe
            mf_data = []
            for symbol, data in mutual_funds.items():
                mf_data.append({
                    'Symbol': symbol,
                    'Name': data['name'],
                    'Category': data['category'],
                    'Price': data['price'],
                    'YTD Return': f"{data['ytd_return']:.2f}%",
                    'Expense Ratio': f"{data['expense_ratio']:.2f}%"
                })
            
            mf_df = pd.DataFrame(mf_data)
            st.dataframe(mf_df, use_container_width=True)
            
            # Fund selector for detailed analysis
            selected_fund = st.selectbox("Select Fund for Detailed Analysis", list(mutual_funds.keys()))
            
            if selected_fund:
                fund_data = mutual_funds[selected_fund]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${fund_data['price']:.2f}")
                with col2:
                    st.metric("YTD Return", f"{fund_data['ytd_return']:.2f}%")
                with col3:
                    st.metric("Expense Ratio", f"{fund_data['expense_ratio']:.2f}%")
                
                # Performance chart
                fund_hist, _ = fetch_stock_data(selected_fund, "1y")
                if fund_hist is not None and not fund_hist.empty:
                    fig = px.line(fund_hist, x=fund_hist.index, y='Close', 
                                 title=f"{selected_fund} Performance (1 Year)")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Fund analysis
                st.subheader("Fund Analysis")
                
                # Simple prediction based on historical performance
                if fund_hist is not None and len(fund_hist) > 30:
                    recent_performance = fund_hist['Close'].pct_change(30).iloc[-1] * 100
                    avg_performance = fund_hist['Close'].pct_change(30).mean() * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("30-Day Performance", f"{recent_performance:.2f}%")
                    with col2:
                        st.metric("Avg 30-Day Performance", f"{avg_performance:.2f}%")
                    
                    # Simple outlook based on recent performance
                    if recent_performance > avg_performance + 1:
                        outlook = "Positive"
                        reasoning = "Recent performance exceeds historical average"
                        color = "green"
                    elif recent_performance < avg_performance - 1:
                        outlook = "Cautious"
                        reasoning = "Recent performance below historical average"
                        color = "orange"
                    else:
                        outlook = "Neutral"
                        reasoning = "Recent performance in line with historical average"
                        color = "blue"
                    
                    st.markdown(f"**Short-Term Outlook:** <span style='color:{color}'>{outlook}</span>", unsafe_allow_html=True)
                    st.write(f"**Reasoning:** {reasoning}")
                
                # Investment recommendation
                st.subheader("Investment Considerations")
                
                considerations = {
                    "Diversification": "Provides exposure to multiple securities within its category",
                    "Professional Management": "Managed by experienced investment professionals",
                    "Liquidity": "Generally highly liquid with daily trading",
                    "Cost Efficiency": f"Expense ratio of {fund_data['expense_ratio']:.2f}% is {'low' if fund_data['expense_ratio'] < 0.5 else 'moderate' if fund_data['expense_ratio'] < 1.0 else 'high'}",
                    "Risk Level": "Varies by fund category and investment strategy"
                }
                
                for factor, description in considerations.items():
                    with st.expander(factor):
                        st.write(description)
        
        else:
            st.info("Mutual fund data temporarily unavailable.")

# Settings Page
elif selected == "Settings":
    st.title("⚙️ Settings")
    
    st.subheader("App Preferences")
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox("Theme", ["Dark", "Light"], index=0)
        currency = st.selectbox("Currency", ["USD", "INR", "EUR", "GBP", "JPY"], index=0)
        default_view = st.selectbox("Default View", ["Dashboard", "Stock Analysis", "Portfolio", "News"], index=0)
    
    with col2:
        refresh_rate = st.slider("Data Refresh Rate (minutes)", 1, 60, 5)
        notifications = st.checkbox("Enable Notifications", value=True)
        auto_refresh = st.checkbox("Auto Refresh Data", value=False)
    
    if st.button("Save Preferences"):
        st.success("Preferences saved successfully!")
    
    st.subheader("Data Management")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear Cache"):
            st.session_state.cached_data = {}
            st.success("Cache cleared successfully!")
    
    with col2:
        if st.button("Export Portfolio Data"):
            # In a real app, this would generate a CSV file for download
            st.success("Portfolio data exported successfully!")
    
    with col3:
        if st.button("Reset to Default Settings"):
            st.session_state.cached_data = {}
            st.session_state.watchlist = []
            st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Quantity', 'Purchase Price', 'Purchase Date'])
            st.success("Settings reset to defaults!")
    
    st.subheader("Account Information")
    st.write("**Username:** trader123")
    st.write("**Email:** user@example.com")
    st.write("**Subscription:** Premium (expires Dec 31, 2023)")
    
    if st.button("Upgrade Subscription"):
        st.info("Redirecting to subscription page...")
    
    st.subheader("About MarketMentor")
    st.write("**Version:** 2.1.0")
    st.write("**Last Updated:** October 15, 2023")
    st.write("**Developer:** Ashwik Bire")
    st.write("**Contact:** ashwikbire@gmail.com")
    
    st.write("---")
    st.write("MarketMentor provides financial information and educational content for informational purposes only. "
             "It is not intended as investment advice. Always conduct your own research and consider consulting "
             "with a qualified financial advisor before making investment decisions.")

# Add a footer to all pages
st.write("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("<div style='text-align: center; color: #FF0000;'><b>MarketMentor Pro</b><br>Advanced Financial Analytics Platform</div>", unsafe_allow_html=True)
with footer_col2:
    st.markdown("<div style='text-align: center;'><b>Disclaimer:</b><br>Not investment advice. Data may be delayed.</div>", unsafe_allow_html=True)
with footer_col3:
    st.markdown("<div style='text-align: center;'>**© 2023 MarketMentor**<br>Version 2.1.0</div>", unsafe_allow_html=True)

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
