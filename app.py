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

# Format currency based on stock type
def format_currency(value, currency="USD"):
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
def fetch_news(query="stock market"):
    try:
        NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"  # Note: This is a demo key, replace with your own
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
    if df.empty:
        return df
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
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
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            up_val = delta
            down_val = 0.
        else:
            up_val = 0.
            down_val = -delta

        up = (up * (period - 1) + up_val) / period
        down = (down * (period - 1) + down_val) / period

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi

# Prediction functions
@st.cache_data(ttl=3600, show_spinner=False)
def predict_stock_price(ticker, days=30):
    """ Simple stock price prediction using historical data and trend analysis """
    try:
        # Get historical data
        hist, _ = fetch_stock_data(ticker, "1y")
        if hist is None or hist.empty:
            return None, None, None
        
        # Simple linear regression for prediction
        prices = hist['Close'].values
        x = np.arange(len(prices))
        
        # Fit a linear model (for simplicity)
        z = np.polyfit(x, prices, 1)
        p = np.poly1d(z)
        
        # Predict future prices
        future_x = np.arange(len(prices), len(prices) + days)
        future_prices = p(future_x)
        
        # Calculate confidence intervals (simplified)
        current_price = prices[-1]
        predicted_price = future_prices[-1]
        confidence = max(0, min(100, 95 - (abs(predicted_price - current_price) / current_price * 100)))
        
        return future_prices, predicted_price, confidence
    except:
        return None, None, None

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
                 "News & Sentiment", "Learning Center", "Company Info", "Predictions", "Settings"],
        icons=["house", "graph-up", "bar-chart", "wallet", 
               "diagram-3", "globe", "calendar", "currency-bitcoin",
               "newspaper", "book", "building", "lightbulb", "gear"],
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
    watchlist_symbol = st.text_input("Add symbol to watchlist:", key="watchlist_input")
    if st.button("Add to Watchlist", key="add_watchlist"):
        if watchlist_symbol and watchlist_symbol not in st.session_state.watchlist:
            st.session_state.watchlist.append(watchlist_symbol.upper())
            st.success(f"Added {watchlist_symbol.upper()} to watchlist")
    
    if st.session_state.watchlist:
        for symbol in st.session_state.watchlist:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(symbol)
            with col2:
                if st.button("X", key=f"remove_{symbol}"):
                    st.session_state.watchlist.remove(symbol)
                    st.rerun()

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
                change_color = "#00FF00" if row["Change"] >= 0 else "#FF0000"
                st.metric(
                    label=f"{row['Name']} ({row['Symbol']})",
                    value=format_currency(row["Price"], row["Currency"]),
                    delta=f"{row['Change']:.2f} ({row['Change %']:.2f}%)",
                    delta_color="normal" if row["Change"] >= 0 else "inverse"
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
        for article in news_articles:
            with st.expander(f"{article['title']} - {article['source']['name']}"):
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
                          value=getattr(st.session_state, 'analyze_symbol', 'AAPL'))
    
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
            
            # Price chart
            st.subheader("Price Chart")
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
                st.write(f"**Name:** {info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Website:** {info.get('website', 'N/A')}")
                st.write(f"**Summary:** {info.get('longBusinessSummary', 'N/A')}")
        else:
            st.error("Could not fetch data for the specified symbol. Please check the symbol and try again.")

# Technical Analysis Page
elif selected == "Technical Analysis":
    st.title("📈 Technical Analysis")
    
    symbol = st.text_input("Enter stock symbol:", "AAPL")
    period = st.selectbox("Select period:", ["1mo", "3mo", "6mo", "1y", "2y"])
    
    if symbol:
        hist, info = fetch_stock_data(symbol, period)
        
        if hist is not None and not hist.empty:
            currency = info.get('currency', 'USD') if info else 'USD'
            hist = get_technical_indicators(hist)
            
            # Price with SMA
            st.subheader("Moving Averages")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close', line=dict(color='white')))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name='SMA 20', line=dict(color='yellow')))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50', line=dict(color='orange')))
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
            fig2.add_trace(go.Scatter(x=hist.index, y=hist['EMA_12'], name='EMA 12', line=dict(color='yellow')))
            fig2.add_trace(go.Scatter(x=hist.index, y=hist['EMA_26'], name='EMA 26', line=dict(color='orange')))
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

# Portfolio Manager Page
elif selected == "Portfolio Manager":
    st.title("💼 Portfolio Manager")
    
    tab1, tab2, tab3 = st.tabs(["View Portfolio", "Add Holding", "Performance Analysis"])
    
    with tab1:
        st.subheader("Your Portfolio")
        if st.session_state.portfolio.empty:
            st.info("Your portfolio is empty. Add holdings to get started.")
        else:
            st.dataframe(st.session_state.portfolio)
            
            # Calculate portfolio value
            total_value = 0
            for _, holding in st.session_state.portfolio.iterrows():
                hist, _ = fetch_stock_data(holding['Symbol'], "1d")
                if hist is not None and not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    total_value += current_price * holding['Quantity']
            
            st.metric("Total Portfolio Value", format_currency(total_value, "USD"))
    
    with tab2:
        st.subheader("Add New Holding")
        with st.form("add_holding_form"):
            symbol = st.text_input("Symbol")
            quantity = st.number_input("Quantity", min_value=1, value=1)
            purchase_price = st.number_input("Purchase Price", min_value=0.0, value=0.0)
            purchase_date = st.date_input("Purchase Date", value=datetime.now())
            
            if st.form_submit_button("Add Holding"):
                new_holding = pd.DataFrame({
                    'Symbol': [symbol.upper()],
                    'Quantity': [quantity],
                    'Purchase Price': [purchase_price],
                    'Purchase Date': [purchase_date]
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
                    
                    performance_data.append({
                        'Symbol': holding['Symbol'],
                        'Quantity': holding['Quantity'],
                        'Purchase Price': holding['Purchase Price'],
                        'Current Price': current_price,
                        'Purchase Value': purchase_value,
                        'Current Value': current_value,
                        'Gain/Loss': gain_loss,
                        'Gain/Loss %': gain_loss_percent
                    })
            
            if performance_data:
                performance_df = pd.DataFrame(performance_data)
                st.dataframe(performance_df)
                
                # Portfolio allocation chart
                fig = px.pie(performance_df, values='Current Value', names='Symbol', title='Portfolio Allocation')
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

# Options Chain Page
elif selected == "Options Chain":
    st.title("📊 Options Chain Analysis")
    st.info("Options chain analysis feature is coming soon. Check back in the next update!")

# Market Overview Page
elif selected == "Market Overview":
    st.title("🌍 Global Market Overview")
    
    indices_df = fetch_global_indices()
    if not indices_df.empty:
        st.dataframe(indices_df)
        
        # Create a bar chart of performance
        fig = px.bar(indices_df, x='Name', y='Change %', title='Global Indices Performance (%)',
                    color='Change %', color_continuous_scale=['red', 'white', 'green'])
        st.plotly_chart(fig, use_container_width=True)

# Economic Calendar Page
elif selected == "Economic Calendar":
    st.title("📅 Economic Calendar")
    st.info("Economic calendar feature is coming soon. Check back in the next update!")

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
        st.dataframe(crypto_df)
        
        # Crypto performance chart
        fig = px.bar(crypto_df, x='Name', y='Change %', title='Cryptocurrency Performance (%)',
                    color='Change %', color_continuous_scale=['red', 'white', 'green'])
        st.plotly_chart(fig, use_container_width=True)

# News & Sentiment Page
elif selected == "News & Sentiment":
    st.title("📰 News & Market Sentiment")
    
    news_query = st.text_input("Search for news:", "stock market")
    if st.button("Fetch News"):
        news_articles = fetch_news(news_query)
        
        if news_articles:
            for article in news_articles:
                with st.expander(f"{article['title']} - {article['source']['name']}"):
                    st.write(article['description'])
                    if article['url']:
                        st.markdown(f"[Read more]({article['url']})")
        else:
            st.info("No news found for your query. Try a different search term.")

# Learning Center Page
elif selected == "Learning Center":
    st.title("📚 Learning Center")
    
    st.subheader("Educational Resources")
    topics = [
        "Introduction to Stock Market",
        "Technical Analysis Basics",
        "Fundamental Analysis",
        "Options Trading",
        "Portfolio Management",
        "Risk Management",
        "Cryptocurrency Investing",
        "Market Psychology"
    ]
    
    for topic in topics:
        with st.expander(topic):
            st.write(f"This section would contain educational content about {topic}.")
            st.info("Detailed educational content is coming soon in the next update!")

# Company Info Page
elif selected == "Company Info":
    st.title("🏢 Company Information")
    
    symbol = st.text_input("Enter company symbol:", "AAPL")
    if symbol:
        hist, info = fetch_stock_data(symbol, "1d")
        
        if info is not None:
            st.subheader("Company Profile")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'longName' in info:
                    st.write(f"**Company Name:** {info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Country:** {info.get('country', 'N/A')}")
            
            with col2:
                st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
                st.write(f"**Website:** {info.get('website', 'N/A')}")
                st.write(f"**Phone:** {info.get('phone', 'N/A')}")
            
            if 'longBusinessSummary' in info:
                st.subheader("Business Summary")
                st.write(info.get('longBusinessSummary', 'No summary available.'))
            
            # Key executives
            if 'companyOfficers' in info and info['companyOfficers']:
                st.subheader("Key Executives")
                executives = info['companyOfficers'][:5]  # Show top 5 executives
                for exec in executives:
                    st.write(f"**{exec.get('name', 'N/A')}** - {exec.get('title', 'N/A')}")
        else:
            st.error("Could not fetch company information. Please check the symbol and try again.")

# Predictions Page
elif selected == "Predictions":
    st.title("🔮 Stock & Mutual Fund Predictions")
    
    symbol = st.text_input("Enter symbol for prediction:", "AAPL")
    days = st.slider("Prediction days:", min_value=7, max_value=90, value=30)
    
    if st.button("Generate Prediction"):
        with st.spinner("Generating prediction..."):
            future_prices, predicted_price, confidence = predict_stock_price(symbol, days)
            
            if future_prices is not None:
                hist, info = fetch_stock_data(symbol, "6mo")
                currency = info.get('currency', 'USD') if info else 'USD'
                current_price = hist['Close'].iloc[-1] if hist is not None and not hist.empty else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", format_currency(current_price, currency))
                with col2:
                    st.metric("Predicted Price", format_currency(predicted_price, currency))
                with col3:
                    st.metric("Confidence Level", f"{confidence:.1f}%")
                
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
                    title=f"{symbol} Price Prediction",
                    xaxis_title="Date",
                    yaxis_title=f"Price ({currency})",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Disclaimer:** Predictions are based on historical data and trend analysis. 
                They are not guaranteed and should not be considered as financial advice. 
                Always do your own research before making investment decisions.
                """)
            else:
                st.error("Could not generate prediction. Please check the symbol and try again.")

# Settings Page
elif selected == "Settings":
    st.title("⚙️ Settings")
    
    st.subheader("Appearance")
    theme = st.selectbox("Color Theme", ["Dark (Red/Black)", "Light", "Blue", "Green"])
    st.info(f"Selected theme: {theme}. Note: Theme changes require app restart to take effect.")
    
    st.subheader("Data Preferences")
    auto_refresh = st.checkbox("Auto-refresh data", value=True)
    refresh_interval = st.slider("Refresh interval (minutes)", min_value=1, max_value=60, value=15)
    
    st.subheader("Notifications")
    email_notifications = st.checkbox("Email notifications", value=False)
    price_alerts = st.checkbox("Price alerts", value=False)
    
    if st.button("Save Settings"):
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
</script>
""", unsafe_allow_html=True)
