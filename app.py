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

# Apply pure black theme with white text
st.markdown("""
<style>
    .main {background-color: #000000; color: #FFFFFF;}
    .stApp {background-color: #000000; color: #FFFFFF;}
    h1, h2, h3, h4, h5, h6 {color: #FFFFFF !important; border-bottom: 1px solid #333333; padding-bottom: 8px;}
    .stButton>button {background-color: #111111; color: white; border: 1px solid #333333; border-radius: 4px;}
    .stTextInput>div>div>input, .stSelectbox>div>div>select {background-color: #111111; color: white; border: 1px solid #333333;}
    .stMetric {background-color: #111111; border-radius: 5px; padding: 10px; border-left: 3px solid #444444;}
    .stDataFrame {background-color: #111111; color: white;}
    .streamlit-expanderHeader {background-color: #111111; border-radius: 4px; padding: 8px; color: white;}
    .stTabs {background-color: #000000;}
    div[data-baseweb="tab-list"] {background-color: #111111; gap: 2px;}
    div[data-baseweb="tab"] {background-color: #222222; color: white; padding: 10px 20px; border-radius: 4px 4px 0 0;}
    div[data-baseweb="tab"]:hover {background-color: #333333;}
    div[data-baseweb="tab"][aria-selected="true"] {background-color: #444444;}
    .stProgress > div > div > div {background-color: #444444;}
    .css-1d391kg {background-color: #000000;}
    .css-1lcbmhc {background-color: #000000;}
    .stAlert {background-color: #111111; color: white;}
    .st-bb {background-color: #111111;}
    .st-at {background-color: #222222;}
    .st-bh {background-color: #111111;}
    .st-bi {background-color: #111111;}
    .st-bj {background-color: #222222;}
    .st-bk {background-color: #111111;}
    .st-bl {background-color: #111111;}
    .st-bm {background-color: #222222;}
    table {color: white;}
    thead th {color: white !important;}
    tbody td {color: white !important;}
    .stDownloadButton>button {background-color: #111111; color: white; border: 1px solid #333333;}
    .stNumberInput>div>div>input {background-color: #111111; color: white; border: 1px solid #333333;}
    .stDateInput>div>div>input {background-color: #111111; color: white; border: 1px solid #333333;}
    .stTimeInput>div>div>input {background-color: #111111; color: white; border: 1px solid #333333;}
    .stSlider>div>div>div>div {background-color: #444444;}
    .stRadio>div {background-color: #111111; padding: 10px; border-radius: 4px;}
    .stCheckbox>div {background-color: #111111; padding: 10px; border-radius: 4px;}
    .stSelectbox>div>div>select {color: white;}
    .stTextArea>div>div>textarea {background-color: #111111; color: white; border: 1px solid #333333;}
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
        "^HSI": {"name": "Hang Seng", "currency": "HKD"}
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

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_precious_metals():
    # Precious metals symbols and conversion to INR (using approximate conversion rates)
    metals = {
        "GC=F": {"name": "Gold", "conversion": 82.5, "unit": "per 10g"},
        "SI=F": {"name": "Silver", "conversion": 82.5, "unit": "per kg"},
        "PL=F": {"name": "Platinum", "conversion": 82.5, "unit": "per gram"},
        "PA=F": {"name": "Palladium", "conversion": 82.5, "unit": "per gram"}
    }
    
    results = {}
    for symbol, data in metals.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                last_close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = ((last_close - prev_close) / prev_close) * 100
                # Convert to INR
                inr_price = last_close * data["conversion"]
                results[symbol] = {
                    "name": data["name"],
                    "price": inr_price,
                    "change": change,
                    "unit": data["unit"]
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

# Sidebar navigation
with st.sidebar:
    st.title("MarketMentor Pro")
    
    # Developer info
    st.markdown("---")
    st.markdown("### Developed by Ashwik Bire")
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ashwik_Bire-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/ashwik-bire-b2a000186)")
    
    selected = option_menu(
        "Navigation",
        ["Dashboard", "Stock Analysis", "Technical Analysis", "Portfolio Manager", 
         "Options Chain", "Market Overview", "Predictions", "Crypto Markets",
         "News & Sentiment", "Learning Center", "Company Info", "Settings"],
        icons=['speedometer2', 'graph-up', 'bar-chart-line', 'wallet', 
               'diagram-3', 'globe', 'activity', 'currency-bitcoin',
               'newspaper', 'book', 'building', 'gear'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#000000"},
            "icon": {"color": "#FFFFFF", "font-size": "16px"}, 
            "nav-link": {"color": "#FFFFFF", "font-size": "14px", "text-align": "left"},
            "nav-link-selected": {"background-color": "#333333"},
        }
    )
    
    # Watchlist section in sidebar
    st.markdown("---")
    st.subheader("📋 Watchlist")
    watchlist_symbol = st.text_input("Add symbol to watchlist", "AAPL")
    if st.button("Add to Watchlist") and watchlist_symbol:
        if watchlist_symbol not in st.session_state.watchlist:
            st.session_state.watchlist.append(watchlist_symbol)
            st.success(f"Added {watchlist_symbol} to watchlist")
    
    if st.session_state.watchlist:
        for symbol in st.session_state.watchlist:
            try:
                hist, _ = fetch_stock_data(symbol, "1d")
                if hist is not None and not hist.empty:
                    price = hist['Close'].iloc[-1]
                    st.write(f"{symbol}: ${price:.2f}")
            except:
                st.write(f"{symbol}: N/A")

# Dashboard Page
if selected == "Dashboard":
    st.title("📊 Market Dashboard")
    
    # Load global indices with caching
    indices_data = fetch_global_indices()
    
    # Display key indices
    st.subheader("🌍 Global Indices")
    cols = st.columns(5)
    index_count = 0
    for symbol, data in list(indices_data.items())[:5]:
        with cols[index_count % 5]:
            currency_symbol = "$" if data["currency"] == "USD" else "₹" if data["currency"] == "INR" else "€" if data["currency"] == "EUR" else "£"
            st.metric(
                label=data["name"],
                value=f"{currency_symbol}{data['price']:.2f}",
                delta=f"{data['change']:.2f}%"
            )
        index_count += 1
    
    # Precious metals section
    st.subheader("🥇 Precious Metals (in ₹)")
    metals_data = fetch_precious_metals()
    cols = st.columns(4)
    metal_count = 0
    for symbol, data in metals_data.items():
        with cols[metal_count % 4]:
            st.metric(
                label=data["name"],
                value=f"₹{data['price']:,.2f}",
                delta=f"{data['change']:.2f}%"
            )
            st.caption(data["unit"])
        metal_count += 1
    
    # Market overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sector Performance")
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
                     color_continuous_scale=px.colors.sequential.Blues_r)
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'font': {'color': 'white'}
        })
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Market Sentiment")
        sentiment_data = {"Bullish": 45, "Neutral": 30, "Bearish": 25}
        fig = px.pie(values=list(sentiment_data.values()), names=list(sentiment_data.keys()),
                     title="Market Sentiment Distribution",
                     color_discrete_map={"Bullish": "#00CC96", "Neutral": "#FFA15A", "Bearish": "#EF553B"})
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'font': {'color': 'white'}
        })
        st.plotly_chart(fig, use_container_width=True)
        
        # Market health indicators
        st.subheader("📈 Market Health")
        health_data = {
            "Volatility Index (VIX)": "18.5",
            "Advance/Decline Ratio": "1.2:1",
            "New Highs/Lows": "285/142",
            "Put/Call Ratio": "0.85"
        }
        for indicator, value in health_data.items():
            st.write(f"**{indicator}:** {value}")
    
    # Recent news with caching
    st.subheader("📰 Market News")
    news_articles = fetch_news()
    if news_articles:
        for article in news_articles[:3]:
            with st.expander(f"{article['title']} - {article['source']['name']}"):
                st.write(article.get('description', 'No description available'))
                st.markdown(f"[Read more]({article['url']})")
    else:
        st.info("News feed temporarily unavailable.")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    action_cols = st.columns(4)
    with action_cols[0]:
        if st.button("Screen Stocks"):
            st.session_state.selected = "Stock Analysis"
    with action_cols[1]:
        if st.button("Manage Portfolio"):
            st.session_state.selected = "Portfolio Manager"
    with action_cols[2]:
        if st.button("View Predictions"):
            st.session_state.selected = "Predictions"
    with action_cols[3]:
        if st.button("Check News"):
            st.session_state.selected = "News & Sentiment"

# Stock Analysis Page
elif selected == "Stock Analysis":
    st.title("📈 Stock Analysis")
    
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
                    
                    fig.update_layout({
                        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                        'font': {'color': 'white'}
                    })
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
                    fig.update_layout({
                        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                        'font': {'color': 'white'}
                    })
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
                if hist is not null and not hist.empty:
                    comparison_data[symbol] = {
                        'history': hist,
                        'info': info,
                        'current_price': hist['Close'].iloc[-1],
                        'prev_close': hist['Close'].iloc[-2] if len(hist) > 1 else hist['Close'].iloc[-1],
                        'change': ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
                    }
            
            if comparison_data:
                # Display comparison metrics
                st.subheader("Comparison Metrics")
                cols = st.columns(len(comparison_data))
                for i, (symbol, data) in enumerate(comparison_data.items()):
                    with cols[i]:
                        st.metric(
                            label=symbol,
                            value=f"${data['current_price']:.2f}",
                            delta=f"{data['change']:.2f}%"
                        )
                
                # Price comparison chart
                st.subheader("Price Comparison (Normalized)")
                fig = go.Figure()
                
                for symbol, data in comparison_data.items():
                    normalized_price = (data['history']['Close'] / data['history']['Close'].iloc[0]) * 100
                    fig.add_trace(go.Scatter(
                        x=data['history'].index,
                        y=normalized_price,
                        mode='lines',
                        name=symbol
                    ))
                
                fig.update_layout(
                    title="Normalized Price Comparison",
                    xaxis_title="Date",
                    yaxis_title="Normalized Price (%)",
                    hovermode='x unified',
                    showlegend=True
                )
                fig.update_layout({
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                    'font': {'color': 'white'}
                })
                st.plotly_chart(fig, use_container_width=True)
                
                # Key metrics comparison table
                st.subheader("Key Metrics Comparison")
                metrics_data = []
                for symbol, data in comparison_data.items():
                    info = data['info']
                    metrics_data.append({
                        'Symbol': symbol,
                        'Price': data['current_price'],
                        'Change (%)': data['change'],
                        'Market Cap (B)': info.get('marketCap', 0) / 1e9 if info.get('marketCap') else 'N/A',
                        'P/E Ratio': info.get('trailingPE', 'N/A'),
                        '52W High': info.get('fiftyTwoWeekHigh', 'N/A'),
                        '52W Low': info.get('fiftyTwoWeekLow', 'N/A'),
                        'Volume': info.get('volume', 'N/A')
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.error("Unable to fetch data for the provided symbols.")
    
    with tab3:
        st.subheader("Stock Screener")
        st.info("Advanced stock screening functionality will be implemented here.")
        
        # Basic screener options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_market_cap = st.number_input("Min Market Cap (Billion $)", value=10)
            max_pe = st.number_input("Max P/E Ratio", value=30)
        
        with col2:
            min_price = st.number_input("Min Price ($)", value=10)
            max_price = st.number_input("Max Price ($)", value=500)
        
        with col3:
            sector = st.selectbox("Sector", ["All", "Technology", "Healthcare", "Financial", "Consumer", "Energy"])
            min_volume = st.number_input("Min Volume (M)", value=1)
        
        if st.button("Run Screener"):
            # This would normally connect to a database or API for screening
            st.success("Screener ran successfully! (Demo mode)")
            
            # Sample results
            sample_stocks = [
                {"Symbol": "AAPL", "Price": 175.34, "Change": 1.2, "Market Cap": 2740, "P/E": 28.5, "Sector": "Technology"},
                {"Symbol": "MSFT", "Price": 337.69, "Change": 0.8, "Market Cap": 2510, "P/E": 32.1, "Sector": "Technology"},
                {"Symbol": "GOOGL", "Price": 139.93, "Change": 1.5, "Market Cap": 1760, "P/E": 24.8, "Sector": "Technology"},
                {"Symbol": "AMZN", "Price": 145.18, "Change": -0.3, "Market Cap": 1490, "P/E": 58.3, "Sector": "Consumer"},
                {"Symbol": "META", "Price": 329.88, "Change": 2.1, "Market Cap": 845, "P/E": 26.7, "Sector": "Technology"},
            ]
            
            screened_df = pd.DataFrame(sample_stocks)
            st.dataframe(screened_df, use_container_width=True)

# Technical Analysis Page
elif selected == "Technical Analysis":
    st.title("📊 Technical Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input("Enter Stock Symbol for Technical Analysis", "AAPL")
    with col2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"])
    
    if ticker:
        hist, info = fetch_stock_data(ticker, period)
        if hist is not None and not hist.empty:
            # Calculate technical indicators
            hist = get_technical_indicators(hist)
            
            st.subheader(f"Technical Analysis for {ticker.upper()}")
            
            # Price chart with indicators
            chart_type = st.selectbox("Chart Type", ["Line", "Candlestick", "OHLC"])
            
            if chart_type == "Candlestick":
                fig = go.Figure(data=[go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='Price'
                )])
            elif chart_type == "OHLC":
                fig = go.Figure(data=[go.Ohlc(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='Price'
                )])
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Price'))
            
            # Add moving averages
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], mode='lines', name='SMA 20', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], mode='lines', name='SMA 50', line=dict(dash='dash')))
            
            fig.update_layout(
                title=f"{ticker.upper()} Price with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified'
            )
            fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                'font': {'color': 'white'}
            })
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI Chart
            st.subheader("Relative Strength Index (RSI)")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], mode='lines', name='RSI'))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_rsi.update_layout(yaxis_range=[0, 100])
            fig_rsi.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                'font': {'color': 'white'}
            })
            st.plotly_chart(fir_rsi, use_container_width=True)
            
            # MACD Chart
            st.subheader("Moving Average Convergence Divergence (MACD)")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], mode='lines', name='MACD'))
            fig_macd.add_trace(go.Scatter(x=hist.index, y=hist['MACD_Signal'], mode='lines', name='Signal'))
            fig_macd.add_trace(go.Bar(x=hist.index, y=hist['MACD_Hist'], name='Histogram'))
            fig_macd.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                'font': {'color': 'white'}
            })
            st.plotly_chart(fig_macd, use_container_width=True)
            
            # Technical indicators summary
            st.subheader("Technical Indicators Summary")
            current_rsi = hist['RSI'].iloc[-1] if 'RSI' in hist.columns else None
            current_macd = hist['MACD'].iloc[-1] if 'MACD' in hist.columns else None
            current_macd_signal = hist['MACD_Signal'].iloc[-1] if 'MACD_Signal' in hist.columns else None
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if current_rsi:
                    st.metric("RSI", f"{current_rsi:.2f}")
                    if current_rsi > 70:
                        st.error("Overbought Territory")
                    elif current_rsi < 30:
                        st.success("Oversold Territory")
                    else:
                        st.info("Neutral Territory")
            
            with col2:
                if current_macd and current_macd_signal:
                    st.metric("MACD", f"{current_macd:.4f}")
                    st.metric("Signal", f"{current_macd_signal:.4f}")
                    if current_macd > current_macd_signal:
                        st.success("Bullish Signal (MACD > Signal)")
                    else:
                        st.error("Bearish Signal (MACD < Signal)")
            
            with col3:
                if 'SMA_20' in hist.columns and 'SMA_50' in hist.columns:
                    current_sma20 = hist['SMA_20'].iloc[-1]
                    current_sma50 = hist['SMA_50'].iloc[-1]
                    current_price = hist['Close'].iloc[-1]
                    
                    st.metric("SMA 20", f"{current_sma20:.2f}")
                    st.metric("SMA 50", f"{current_sma50:.2f}")
                    
                    if current_price > current_sma20 and current_price > current_sma50:
                        st.success("Price above both MAs")
                    elif current_price < current_sma20 and current_price < current_sma50:
                        st.error("Price below both MAs")
                    else:
                        st.warning("Mixed signals from MAs")
        
        else:
            st.error("Unable to fetch data for the provided ticker symbol.")

# Portfolio Manager Page
elif selected == "Portfolio Manager":
    st.title("💼 Portfolio Manager")
    
    tab1, tab2, tab3 = st.tabs(["Current Portfolio", "Add Holding", "Performance Analysis"])
    
    with tab1:
        st.subheader("Your Investment Portfolio")
        
        if st.session_state.portfolio.empty:
            st.info("Your portfolio is empty. Add holdings to get started.")
        else:
            # Calculate current values
            portfolio = st.session_state.portfolio.copy()
            current_prices = []
            current_values = []
            changes = []
            
            for _, holding in portfolio.iterrows():
                hist, _ = fetch_stock_data(holding['Symbol'], "1d")
                if hist is not None and not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    current_value = current_price * holding['Quantity']
                    purchase_value = holding['Purchase Price'] * holding['Quantity']
                    change = ((current_value - purchase_value) / purchase_value) * 100
                    
                    current_prices.append(current_price)
                    current_values.append(current_value)
                    changes.append(change)
                else:
                    current_prices.append(0)
                    current_values.append(0)
                    changes.append(0)
            
            portfolio['Current Price'] = current_prices
            portfolio['Current Value'] = current_values
            portfolio['Change (%)'] = changes
            portfolio['Gain/Loss'] = portfolio['Current Value'] - (portfolio['Purchase Price'] * portfolio['Quantity'])
            
            # Display portfolio
            st.dataframe(portfolio, use_container_width=True)
            
            # Portfolio summary
            total_investment = (portfolio['Purchase Price'] * portfolio['Quantity']).sum()
            total_current = portfolio['Current Value'].sum()
            total_change = ((total_current - total_investment) / total_investment) * 100 if total_investment > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Investment", f"${total_investment:,.2f}")
            with col2:
                st.metric("Current Value", f"${total_current:,.2f}")
            with col3:
                st.metric("Total Gain/Loss", f"${total_current - total_investment:,.2f}", f"{total_change:.2f}%")
            
            # Portfolio allocation chart
            st.subheader("Portfolio Allocation")
            fig = px.pie(portfolio, values='Current Value', names='Symbol', title="Portfolio Allocation by Symbol")
            fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                'font': {'color': 'white'}
            })
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Add New Holding")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            symbol = st.text_input("Symbol", "AAPL").upper()
        with col2:
            quantity = st.number_input("Quantity", min_value=1, value=10)
        with col3:
            purchase_price = st.number_input("Purchase Price ($)", min_value=0.01, value=150.0)
        with col4:
            purchase_date = st.date_input("Purchase Date", value=datetime.now())
        
        if st.button("Add to Portfolio"):
            # Check if symbol already exists in portfolio
            if symbol in st.session_state.portfolio['Symbol'].values:
                st.warning(f"{symbol} is already in your portfolio. Updating quantity and average price.")
                # Update existing holding (simplified - in reality would need to calculate average price)
                idx = st.session_state.portfolio[st.session_state.portfolio['Symbol'] == symbol].index[0]
                st.session_state.portfolio.at[idx, 'Quantity'] += quantity
                # This should really calculate a weighted average price
                st.session_state.portfolio.at[idx, 'Purchase Price'] = purchase_price
            else:
                # Add new holding
                new_holding = pd.DataFrame({
                    'Symbol': [symbol],
                    'Quantity': [quantity],
                    'Purchase Price': [purchase_price],
                    'Purchase Date': [purchase_date]
                })
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_holding], ignore_index=True)
            
            st.success(f"Added {quantity} shares of {symbol} to your portfolio.")
    
    with tab3:
        st.subheader("Portfolio Performance Analysis")
        
        if st.session_state.portfolio.empty:
            st.info("Your portfolio is empty. Add holdings to analyze performance.")
        else:
            # This would normally involve more complex calculations and historical data
            st.info("Advanced portfolio analysis features will be implemented here.")
            
            # Simulated performance chart
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            noise = np.random.normal(0, 2, 30)
            performance_data = {
                'Date': dates,
                'Portfolio Value': noise.cumsum() + 100000
            }
            performance_df = pd.DataFrame(performance_data)
            
            fig = px.line(performance_df, x='Date', y='Portfolio Value', title="Portfolio Performance Over Time")
            fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                'font': {'color': 'white'}
            })
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Volatility (30d)", "12.5%")
            with col2:
                st.metric("Sharpe Ratio", "1.2")
            with col3:
                st.metric("Max Drawdown", "-8.7%")

# Options Chain Page
elif selected == "Options Chain":
    st.title("📊 Options Chain Analysis")
    st.info("Options chain data functionality will be implemented here.")
    
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Enter Stock Symbol for Options", "AAPL")
    with col2:
        expiration = st.selectbox("Expiration Date", ["2023-12-15", "2024-01-19", "2024-02-16"])
    
    if ticker:
        st.subheader(f"Options Chain for {ticker.upper()} - {expiration}")
        
        # Simulated options data
        strikes = np.arange(150, 201, 5)
        calls = []
        puts = []
        
        for strike in strikes:
            calls.append({
                'Strike': strike,
                'Last Price': round(np.random.uniform(0.5, 15.0), 2),
                'Bid': round(np.random.uniform(0.5, 15.0), 2),
                'Ask': round(np.random.uniform(0.5, 15.0), 2),
                'Volume': np.random.randint(100, 1000),
                'Open Interest': np.random.randint(100, 2000),
                'IV': round(np.random.uniform(20, 50), 2)
            })
            
            puts.append({
                'Strike': strike,
                'Last Price': round(np.random.uniform(0.5, 15.0), 2),
                'Bid': round(np.random.uniform(0.5, 15.0), 2),
                'Ask': round(np.random.uniform(0.5, 15.0), 2),
                'Volume': np.random.randint(100, 1000),
                'Open Interest': np.random.randint(100, 2000),
                'IV': round(np.random.uniform(20, 50), 2)
            })
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Calls")
            calls_df = pd.DataFrame(calls)
            st.dataframe(calls_df, use_container_width=True)
        
        with col2:
            st.subheader("Puts")
            puts_df = pd.DataFrame(puts)
            st.dataframe(puts_df, use_container_width=True)
        
        # Options strategy builder
        st.subheader("Options Strategy Builder")
        strategy = st.selectbox("Select Strategy", ["Covered Call", "Cash-Secured Put", "Vertical Spread", "Iron Condor"])
        st.write(f"**{strategy} Strategy Details:**")
        
        if strategy == "Covered Call":
            st.write("- Sell 1 call option against 100 shares of stock")
            st.write("- Collect premium income but limit upside potential")
            st.write("- Best in neutral to slightly bullish markets")
        
        elif strategy == "Cash-Secured Put":
            st.write("- Sell 1 put option with cash reserved to buy shares")
            st.write("- Collect premium with obligation to buy stock at strike price")
            st.write("- Bullish strategy that generates income")

# Market Overview Page
elif selected == "Market Overview":
    st.title("🌍 Market Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Market Heatmap")
        sectors = ['Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer', 'Real Estate']
        performance = np.random.uniform(-3, 3, len(sectors))
        
        heatmap_data = pd.DataFrame({
            'Sector': sectors,
            'Performance': performance
        })
        
        fig = px.imshow([performance], 
                        labels=dict(x="Sector", y="", color="Performance (%)"),
                        x=sectors,
                        color_continuous_scale='RdYlGn',
                        aspect="auto")
        fig.update_layout(title="Sector Performance Heatmap")
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'font': {'color': 'white'}
        })
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Top Gainers")
        gainers = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMD'],
            'Price': [175.34, 337.69, 245.18, 456.78, 112.45],
            'Change %': [2.5, 1.8, 4.2, 3.7, 5.1]
        })
        st.dataframe(gainers, use_container_width=True)
        
        st.subheader("📉 Top Losers")
        losers = pd.DataFrame({
            'Symbol': ['NFLX', 'META', 'AMZN', 'GOOGL', 'DIS'],
            'Price': [345.67, 234.56, 123.45, 89.12, 78.90],
            'Change %': [-2.1, -1.7, -3.5, -2.8, -4.2]
        })
        st.dataframe(losers, use_container_width=True)
    
    st.subheader("📅 Economic Events")
    events = pd.DataFrame({
        'Date': ['2023-12-15', '2023-12-20', '2024-01-05', '2024-01-15'],
        'Event': ['FOMC Meeting', 'CPI Data Release', 'Jobs Report', 'Retail Sales'],
        'Impact': ['High', 'High', 'Medium', 'Medium']
    })
    st.dataframe(events, use_container_width=True)

# Predictions Page
elif selected == "Predictions":
    st.title("🔮 Market Predictions")
    
    tab1, tab2, tab3 = st.tabs(["Stock Predictions", "IPO Predictions", "Mutual Fund Predictions"])
    
    with tab1:
        st.subheader("📈 Stock Price Predictions")
        
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Enter Stock Symbol for Prediction", "AAPL")
            periods = st.slider("Prediction Period (Days)", 7, 90, 30)
        
        with col2:
            model = st.selectbox("Prediction Model", ["ARIMA", "LSTM", "Prophet", "Moving Average"])
            st.metric("Model Accuracy", "87.3%")
        
        if st.button("Generate Prediction"):
            st.success(f"Generating {periods}-day prediction for {ticker} using {model} model...")
            
            # Simulated prediction data
            hist, _ = fetch_stock_data(ticker, "3mo")
            if hist is not None and not hist.empty:
                last_date = hist.index[-1]
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
                
                # Simulate prediction with some noise
                last_price = hist['Close'].iloc[-1]
                trend = np.random.uniform(-0.005, 0.005, periods)
                noise = np.random.normal(0, 0.01, periods)
                predicted_prices = last_price * (1 + np.cumsum(trend + noise))
                
                # Create figure
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=hist.index, 
                    y=hist['Close'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='#1f77b4')
                ))
                
                # Prediction data
                fig.add_trace(go.Scatter(
                    x=future_dates, 
                    y=predicted_prices,
                    mode='lines',
                    name='Prediction',
                    line=dict(color='#ff7f0e', dash='dash')
                ))
                
                # Confidence interval (simulated)
                upper_bound = predicted_prices * 1.05
                lower_bound = predicted_prices * 0.95
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=upper_bound,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=lower_bound,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    line=dict(width=0),
                    name='Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f"{ticker} Price Prediction ({periods} days)",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified'
                )
                fig.update_layout({
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                    'font': {'color': 'white'}
                })
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction summary
                price_change = ((predicted_prices[-1] - last_price) / last_price) * 100
                st.metric("Predicted Price", f"${predicted_prices[-1]:.2f}", f"{price_change:.2f}%")
                
                if price_change > 0:
                    st.success(f"Bullish prediction: {ticker} is expected to increase by {price_change:.2f}% over {periods} days.")
                else:
                    st.error(f"Bearish prediction: {ticker} is expected to decrease by {abs(price_change):.2f}% over {periods} days.")
            else:
                st.error("Unable to fetch data for the provided ticker symbol.")
    
    with tab2:
        st.subheader("📊 IPO Predictions")
        
        st.info("Upcoming IPO predictions based on market conditions and company fundamentals.")
        
        # Sample IPO data
        ipos = pd.DataFrame({
            'Company': ['TechInnovate Inc.', 'BioHeal Labs', 'GreenEnergy Solutions', 'DataSecure Corp.'],
            'Expected Date': ['2024-01-15', '2024-02-10', '2024-03-05', '2024-04-20'],
            'Sector': ['Technology', 'Healthcare', 'Energy', 'Cybersecurity'],
            'Expected Price Range': ['$20-$25', '$15-$18', '$22-$28', '$30-$35'],
            'Predicted Performance': ['+15% to +25%', '+8% to +12%', '+20% to +30%', '+10% to +18%'],
            'Confidence': ['High', 'Medium', 'High', 'Medium']
        })
        
        st.dataframe(ipos, use_container_width=True)
        
        st.subheader("IPO Performance Factors")
        factors = pd.DataFrame({
            'Factor': ['Market Conditions', 'Company Fundamentals', 'Industry Trends', 'Investor Sentiment'],
            'Weight': ['35%', '25%', '20%', '20%'],
            'Impact': ['High', 'High', 'Medium', 'Medium']
        })
        
        st.dataframe(factors, use_container_width=True)
        
        st.subheader("IPO Recommendation")
        selected_ipo = st.selectbox("Select IPO for Analysis", ipos['Company'].tolist())
        
        if selected_ipo:
            st.success(f"Analysis for {selected_ipo}:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Recommendation", "BUY", "Strong")
            
            with col2:
                st.metric("Risk Level", "Medium", "-2%")
            
            with col3:
                st.metric("Expected Return", "+18%", "3%")
    
    with tab3:
        st.subheader("📊 Mutual Fund Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fund_category = st.selectbox("Fund Category", [
                "Large Cap", "Mid Cap", "Small Cap", "Sectoral", 
                "Index", "Debt", "Hybrid", "International"
            ])
        
        with col2:
            risk_appetite = st.select_slider("Risk Appetite", ["Low", "Medium", "High"])
        
        # Sample mutual fund data
        funds = pd.DataFrame({
            'Fund Name': ['Blue Chip Growth', 'Index Tracker', 'Technology Leaders', 'Balanced Advantage'],
            'Category': ['Large Cap', 'Index', 'Sectoral', 'Hybrid'],
            '1Y Return': ['18.5%', '12.3%', '22.7%', '15.2%'],
            '3Y CAGR': ['15.2%', '10.8%', '19.4%', '13.1%'],
            'Risk': ['Medium', 'Low', 'High', 'Medium'],
            'Predicted Return': ['+12% to +16%', '+8% to +12%', '+18% to +24%', '+10% to +14%']
        })
        
        # Filter based on selection
        filtered_funds = funds[funds['Category'] == fund_category]
        if risk_appetite != "All":
            filtered_funds = filtered_funds[filtered_funds['Risk'] == risk_appetite]
        
        st.dataframe(filtered_funds, use_container_width=True)
        
        st.subheader("Top Recommended Funds")
        recommended_funds = filtered_funds.head(3)
        
        for _, fund in recommended_funds.iterrows():
            with st.expander(f"{fund['Fund Name']} - {fund['Category']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("1Y Return", fund['1Y Return'])
                
                with col2:
                    st.metric("3Y CAGR", fund['3Y CAGR'])
                
                with col3:
                    st.metric("Predicted Return", fund['Predicted Return'])
                
                st.progress(75 if fund['Risk'] == 'High' else 50 if fund['Risk'] == 'Medium' else 25)
                st.caption(f"Risk Level: {fund['Risk']}")

# Crypto Markets Page
elif selected == "Crypto Markets":
    st.title("₿ Crypto Markets")
    
    tab1, tab2, tab3 = st.tabs(["Market Overview", "Crypto Analysis", "Portfolio"])
    
    with tab1:
        st.subheader("Cryptocurrency Market Overview")
        
        # Sample crypto data
        cryptocurrencies = [
            {"Name": "Bitcoin", "Symbol": "BTC-USD", "Price": 43256.78, "Change": 2.3, "Market Cap": 845.2},
            {"Name": "Ethereum", "Symbol": "ETH-USD", "Price": 2345.67, "Change": 1.8, "Market Cap": 285.4},
            {"Name": "Binance Coin", "Symbol": "BNB-USD", "Price": 345.67, "Change": -0.5, "Market Cap": 52.8},
            {"Name": "Cardano", "Symbol": "ADA-USD", "Price": 0.4567, "Change": 3.2, "Market Cap": 15.9},
            {"Name": "Solana", "Symbol": "SOL-USD", "Price": 98.76, "Change": 5.4, "Market Cap": 41.2},
        ]
        
        crypto_df = pd.DataFrame(cryptocurrencies)
        st.dataframe(crypto_df, use_container_width=True)
        
        # Crypto market cap distribution
        st.subheader("Market Cap Distribution")
        fig = px.pie(crypto_df, values='Market Cap', names='Name', title="Cryptocurrency Market Cap Distribution")
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'font': {'color': 'white'}
        })
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Cryptocurrency Analysis")
        
        crypto_symbol = st.selectbox("Select Cryptocurrency", ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD"])
        
        if crypto_symbol:
            # Simulated crypto data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            prices = np.random.normal(100, 20, 30).cumsum() + 1000
            
            fig = px.line(x=dates, y=prices, title=f"{crypto_symbol} Price History")
            fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
            fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                'font': {'color': 'white'}
            })
            st.plotly_chart(fig, use_container_width=True)
            
            # Crypto metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${prices[-1]:.2f}")
            with col2:
                st.metric("24h Volume", "$45.2B")
            with col3:
                st.metric("Market Cap", "$845.2B")
            with col4:
                st.metric("Circulating Supply", "19.2M BTC")
    
    with tab3:
        st.subheader("Crypto Portfolio")
        st.info("Track your cryptocurrency investments here.")
        
        # Sample crypto portfolio
        crypto_portfolio = pd.DataFrame({
            'Coin': ['Bitcoin', 'Ethereum', 'Cardano'],
            'Amount': [0.5, 3.2, 5000],
            'Avg. Price': [38500, 2100, 0.42],
            'Current Price': [43256, 2345, 0.4567],
            'Value': [21628, 7504, 2283.5]
        })
        
        st.dataframe(crypto_portfolio, use_container_width=True)
        
        total_value = crypto_portfolio['Value'].sum()
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")

# News & Sentiment Page
elif selected == "News & Sentiment":
    st.title("📰 News & Sentiment Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Market News", "Sentiment Analysis", "Social Media Trends"])
    
    with tab1:
        st.subheader("Latest Financial News")
        
        news_query = st.text_input("Search for news", "stock market")
        if st.button("Search News"):
            articles = fetch_news(news_query)
            if articles:
                for article in articles:
                    with st.expander(f"{article['title']} - {article['source']['name']}"):
                        st.write(article.get('description', 'No description available'))
                        if article.get('urlToImage'):
                            st.image(article['urlToImage'], width=300)
                        st.markdown(f"[Read full article]({article['url']})")
            else:
                st.info("No news articles found or news service unavailable.")
    
    with tab2:
        st.subheader("Market Sentiment Analysis")
        
        # Sentiment indicators
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Sentiment", "Bullish", "2.5%")
        with col2:
            st.metric("Fear & Greed Index", "64 (Greed)", "3")
        with col3:
            st.metric("Volatility", "Medium", "-1.2%")
        
        # Sentiment by sector
        st.subheader("Sector Sentiment")
        sector_sentiment = pd.DataFrame({
            'Sector': ['Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer'],
            'Sentiment': [75, 62, 45, 58, 67],
            'Change': [2.5, -1.2, -3.4, 1.7, 0.8]
        })
        
        fig = px.bar(sector_sentiment, x='Sector', y='Sentiment', title="Sector Sentiment Scores")
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'font': {'color': 'white'}
        })
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Social Media Trends")
        
        # Trending topics
        st.subheader("Trending Financial Topics")
        topics = [
            {"Topic": "#BitcoinETF", "Volume": 125000, "Sentiment": "Bullish"},
            {"Topic": "#FederalReserve", "Volume": 89000, "Sentiment": "Neutral"},
            {"Topic": "#EarningsSeason", "Volume": 76000, "Sentiment": "Bullish"},
            {"Topic": "#Inflation", "Volume": 68000, "Sentiment": "Bearish"},
            {"Topic": "#NFT", "Volume": 54000, "Sentiment": "Bearish"},
        ]
        
        topics_df = pd.DataFrame(topics)
        st.dataframe(topics_df, use_container_width=True)
        
        # Social media sentiment
        st.subheader("Social Media Sentiment")
        platforms = ['Twitter', 'Reddit', 'Stocktwits', 'YouTube']
        sentiment = np.random.uniform(40, 80, len(platforms))
        
        fig = px.bar(x=platforms, y=sentiment, title="Sentiment by Platform")
        fig.update_layout(xaxis_title="Platform", yaxis_title="Sentiment Score")
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'font': {'color': 'white'}
        })
        st.plotly_chart(fig, use_container_width=True)

# Learning Center Page
elif selected == "Learning Center":
    st.title("📚 Learning Center")
    
    tab1, tab2, tab3 = st.tabs(["Tutorials", "Glossary", "Resources"])
    
    with tab1:
        st.subheader("Investment Tutorials")
        
        tutorials = [
            {"Title": "Introduction to Stock Market", "Level": "Beginner", "Duration": "15 min"},
            {"Title": "Technical Analysis Basics", "Level": "Beginner", "Duration": "20 min"},
            {"Title": "Options Trading Strategies", "Level": "Intermediate", "Duration": "30 min"},
            {"Title": "Portfolio Diversification", "Level": "Intermediate", "Duration": "25 min"},
            {"Title": "Advanced Risk Management", "Level": "Advanced", "Duration": "45 min"},
        ]
        
        for tutorial in tutorials:
            with st.expander(f"{tutorial['Title']} ({tutorial['Level']} - {tutorial['Duration']})"):
                st.write("This tutorial will cover the fundamentals of this topic.")
                if st.button("Start Tutorial", key=tutorial['Title']):
                    st.success(f"Starting tutorial: {tutorial['Title']}")
    
    with tab2:
        st.subheader("Financial Glossary")
        
        terms = [
            {"Term": "P/E Ratio", "Definition": "Price-to-Earnings ratio, a valuation metric"},
            {"Term": "Market Cap", "Definition": "Total market value of a company's outstanding shares"},
            {"Term": "Dividend Yield", "Definition": "Annual dividend payment divided by stock price"},
            {"Term": "ETF", "Definition": "Exchange-Traded Fund, a basket of securities"},
            {"Term": "IPO", "Definition": "Initial Public Offering, when a company first sells shares to the public"},
        ]
        
        for term in terms:
            with st.expander(term['Term']):
                st.write(term['Definition'])
    
    with tab3:
        st.subheader("Learning Resources")
        
        resources = [
            {"Type": "E-book", "Title": "The Intelligent Investor", "Author": "Benjamin Graham"},
            {"Type": "Course", "Title": "Stock Market Fundamentals", "Provider": "Coursera"},
            {"Type": "Podcast", "Title": "Market Foolery", "Provider": "The Motley Fool"},
            {"Type": "YouTube Channel", "Title": "Investing with Rose", "Provider": "Rose Han"},
            {"Type": "Blog", "Title": "The Reformed Broker", "Author": "Josh Brown"},
        ]
        
        for resource in resources:
            st.write(f"**{resource['Type']}**: {resource['Title']} by {resource.get('Author', resource.get('Provider', 'Unknown'))}")
            st.progress(np.random.randint(20, 100))

# Company Info Page
elif selected == "Company Info":
    st.title("🏢 Company Information")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        company_symbol = st.text_input("Enter Company Symbol", "AAPL")
    with col2:
        info_type = st.selectbox("Information Type", ["Overview", "Financials", "Executives", "Holdings"])
    
    if company_symbol:
        st.subheader(f"Company Information for {company_symbol.upper()}")
        
        # Simulated company data
        if info_type == "Overview":
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Company Name:** Apple Inc.")
                st.write("**Sector:** Technology")
                st.write("**Industry:** Consumer Electronics")
                st.write("**Founded:** April 1, 1976")
            
            with col2:
                st.write("**CEO:** Tim Cook")
                st.write("**Employees:** 164,000")
                st.write("**Headquarters:** Cupertino, California")
                st.write("**Website:** www.apple.com")
            
            st.subheader("Business Description")
            st.write("Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The company also sells various related services.")
        
        elif info_type == "Financials":
            financials = pd.DataFrame({
                'Year': ['2023', '2022', '2021', '2020', '2019'],
                'Revenue (B)': [383.29, 365.82, 274.52, 260.17, 265.60],
                'Net Income (B)': [97.00, 94.68, 57.41, 55.26, 55.34],
                'EPS': [6.13, 5.67, 3.28, 3.31, 3.00],
                'Dividend': [0.96, 0.88, 0.82, 0.80, 0.75]
            })
            st.dataframe(financials, use_container_width=True)
        
        elif info_type == "Executives":
            executives = pd.DataFrame({
                'Name': ['Tim Cook', 'Jeff Williams', 'Luca Maestri', 'Katherine Adams'],
                'Title': ['CEO', 'COO', 'CFO', 'General Counsel'],
                'Salary ($)': [3M, 2.7M, 2.5M, 2.3M],
                'Age': [63, 60, 60, 56]
            })
            st.dataframe(executives, use_container_width=True)
        
        elif info_type == "Holdings":
            holdings = pd.DataFrame({
                'Holder': ['Vanguard', 'BlackRock', 'State Street', 'Berkshire Hathaway'],
                'Shares (M)': [1280, 1050, 680, 915],
                'Percentage': [7.8%, 6.4%, 4.1%, 5.6%]
            })
            st.dataframe(holdings, use_container_width=True)

# Settings Page
elif selected == "Settings":
    st.title("⚙️ Settings")
    
    st.subheader("Appearance")
    theme = st.selectbox("Theme", ["Dark", "Light", "System Default"])
    language = st.selectbox("Language", ["English", "Spanish", "French", "German"])
    
    st.subheader("Notifications")
    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("Price Alerts", value=True)
        st.checkbox("News Updates", value=True)
    with col2:
        st.checkbox("Portfolio Notifications", value=True)
        st.checkbox("Market News", value=True)
    
    st.subheader("Data Preferences")
    st.selectbox("Default Period", ["1 Month", "3 Months", "6 Months", "1 Year"])
    st.slider("Chart Animation", 0, 100, 50)
    
    st.subheader("Account")
    st.text_input("Name", "John Doe")
    st.text_input("Email", "john.doe@example.com")
    st.text_input("Phone", "+1 (555) 123-4567")
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
    
    st.subheader("Danger Zone")
    if st.button("Clear All Data", type="secondary"):
        st.session_state.watchlist = []
        st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Quantity', 'Purchase Price', 'Purchase Date'])
        st.success("All data cleared!")

else:
    st.title(f"{selected}")
    st.info(f"This page ({selected}) is under development and will be implemented soon.")
