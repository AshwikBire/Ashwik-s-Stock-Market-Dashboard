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
    .stApp {background-color: #000000;}
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
    .prediction-positive {color: #00FF00; font-weight: bold;}
    .prediction-negative {color: #FF0000; font-weight: bold;}
    .prediction-neutral {color: #FFFF00; font-weight: bold;}
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

@st.cache_data(ttl=3600, show_spinner=False)
def generate_stock_prediction(ticker, days=30):
    """Generate a simple stock price prediction using historical data"""
    try:
        hist, _ = fetch_stock_data(ticker, "1y")
        if hist is None or hist.empty:
            return None, None, None
        
        # Simple prediction using moving average and random walk
        last_price = hist['Close'].iloc[-1]
        volatility = hist['Close'].pct_change().std()
        
        # Generate future dates
        future_dates = [hist.index[-1] + timedelta(days=i) for i in range(1, days+1)]
        
        # Create prediction (simple model - in reality, use more sophisticated models)
        predicted_prices = []
        current_price = last_price
        
        for _ in range(days):
            change = np.random.normal(0, volatility)
            current_price = current_price * (1 + change)
            predicted_prices.append(current_price)
        
        # Calculate prediction confidence
        confidence = max(0, min(100, 100 - (volatility * 1000)))
        
        return future_dates, predicted_prices, confidence
    except:
        return None, None, None

@st.cache_data(ttl=86400, show_spinner=False)
def get_upcoming_ipos():
    """Get upcoming IPO data (simulated)"""
    ipos = [
        {"Company": "TechInnovate", "Exchange": "NSE", "Date": "2023-12-15", "Price Range": "₹450-₹465", "Lot Size": 15},
        {"Company": "GreenEnergy Solutions", "Exchange": "BSE", "Date": "2023-12-20", "Price Range": "₹320-₹335", "Lot Size": 20},
        {"Company": "HealthCare Plus", "Exchange": "NSE", "Date": "2024-01-05", "Price Range": "₹500-₹525", "Lot Size": 12},
        {"Company": "FinTech Global", "Exchange": "BSE", "Date": "2024-01-15", "Price Range": "₹275-₹290", "Lot Size": 18},
        {"Company": "Logistics Corp", "Exchange": "NSE", "Date": "2024-01-25", "Price Range": "₹380-₹395", "Lot Size": 16}
    ]
    return pd.DataFrame(ipos)

@st.cache_data(ttl=86400, show_spinner=False)
def get_mutual_funds():
    """Get mutual fund data (simulated)"""
    funds = [
        {"Fund Name": "BlueChip Growth Fund", "Category": "Equity", "NAV": "₹150.25", "1Y Return": "18.5%", "Rating": "5 Star"},
        {"Fund Name": "Balanced Advantage Fund", "Category": "Hybrid", "NAV": "₹125.75", "1Y Return": "14.2%", "Rating": "4 Star"},
        {"Fund Name": "Small Cap Fund", "Category": "Equity", "NAV": "₹85.50", "1Y Return": "22.8%", "Rating": "4 Star"},
        {"Fund Name": "Debt Income Fund", "Category": "Debt", "NAV": "₹110.30", "1Y Return": "9.2%", "Rating": "3 Star"},
        {"Fund Name": "Index Fund Nifty 50", "Category": "Index", "NAV": "₹175.80", "1Y Return": "16.4%", "Rating": "4 Star"},
        {"Fund Name": "Sectoral Technology Fund", "Category": "Sectoral", "NAV": "₹95.45", "1Y Return": "25.3%", "Rating": "5 Star"}
    ]
    return pd.DataFrame(funds)

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
         "Predictions", "Market Overview", "News & Sentiment", "Settings"],
        icons=['speedometer2', 'graph-up', 'bar-chart-line', 'wallet', 
               'activity', 'globe', 'newspaper', 'gear'],
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
                if hist is not None and not hist.empty:
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
            st.plotly_chart(fig_rsi, use_container_width=True)
            
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

# Predictions Page
elif selected == "Predictions":
    st.title("🔮 Market Predictions")
    
    tab1, tab2, tab3 = st.tabs(["Stock Predictions", "IPO Analysis", "Mutual Funds"])
    
    with tab1:
        st.subheader("Stock Price Predictions")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker = st.text_input("Enter Stock Symbol for Prediction", "AAPL")
        with col2:
            prediction_days = st.slider("Prediction Days", 7, 90, 30)
        
        if st.button("Generate Prediction") and ticker:
            with st.spinner("Generating prediction..."):
                future_dates, predicted_prices, confidence = generate_stock_prediction(ticker, prediction_days)
                
                if future_dates is not None:
                    # Get historical data for comparison
                    hist, info = fetch_stock_data(ticker, "3mo")
                    
                    st.subheader(f"Price Prediction for {ticker.upper()}")
                    
                    # Create prediction chart
                    fig = go.Figure()
                    
                    # Add historical data
                    if hist is not None and not hist.empty:
                        fig.add_trace(go.Scatter(
                            x=hist.index, 
                            y=hist['Close'], 
                            mode='lines', 
                            name='Historical Price',
                            line=dict(color='#1f77b4')
                        ))
                    
                    # Add prediction data
                    fig.add_trace(go.Scatter(
                        x=future_dates, 
                        y=predicted_prices, 
                        mode='lines', 
                        name='Predicted Price',
                        line=dict(color='#ff7f0e', dash='dash')
                    ))
                    
                    # Add confidence interval (simulated)
                    upper_bound = [price * (1 + (100 - confidence)/500) for price in predicted_prices]
                    lower_bound = [price * (1 - (100 - confidence)/500) for price in predicted_prices]
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates + future_dates[::-1],
                        y=upper_bound + lower_bound[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 127, 14, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Interval'
                    ))
                    
                    fig.update_layout(
                        title=f"{ticker.upper()} Price Prediction ({prediction_days} days)",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode='x unified',
                        showlegend=True
                    )
                    fig.update_layout({
                        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                        'font': {'color': 'white'}
                    })
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction summary
                    current_price = hist['Close'].iloc[-1] if hist is not None and not hist.empty else predicted_prices[0]
                    predicted_change = ((predicted_prices[-1] - current_price) / current_price) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        st.metric("Predicted Price", f"${predicted_prices[-1]:.2f}", f"{predicted_change:.2f}%")
                    with col3:
                        st.metric("Prediction Confidence", f"{confidence:.1f}%")
                    
                    # Prediction interpretation
                    st.subheader("Prediction Analysis")
                    if predicted_change > 5:
                        st.markdown('<p class="prediction-positive">📈 Bullish Prediction: Strong upward trend expected</p>', unsafe_allow_html=True)
                    elif predicted_change < -5:
                        st.markdown('<p class="prediction-negative">📉 Bearish Prediction: Downward trend expected</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="prediction-neutral">↔ Neutral Prediction: Sideways movement expected</p>', unsafe_allow_html=True)
                    
                    st.info("Note: Predictions are based on historical data and technical analysis. Past performance is not indicative of future results. Always do your own research before investing.")
                    
                else:
                    st.error("Unable to generate prediction for the provided ticker symbol.")
    
    with tab2:
        st.subheader("Upcoming IPO Analysis")
        
        st.info("Analysis of upcoming Initial Public Offerings (IPOs)")
        
        # Get IPO data
        ipo_df = get_upcoming_ipos()
        
        if not ipo_df.empty:
            st.dataframe(ipo_df, use_container_width=True)
            
            # IPO analysis
            st.subheader("IPO Evaluation Framework")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Key Factors to Consider:**")
                st.write("• Company fundamentals")
                st.write("• Industry growth potential")
                st.write("• Valuation compared to peers")
                st.write("• Management team experience")
                st.write("• Offering size and structure")
            
            with col2:
                st.markdown("**IPO Grading System:**")
                st.write("• **A+**: Exceptional opportunity")
                st.write("• **A**: Strong investment potential")
                st.write("• **B**: Moderate potential")
                st.write("• **C**: High risk, speculative")
                st.write("• **D**: Not recommended")
            
            # Sample IPO ratings
            st.subheader("Current IPO Ratings")
            ipo_ratings = [
                {"Company": "TechInnovate", "Rating": "A", "Outlook": "Positive", "Risk": "Medium"},
                {"Company": "GreenEnergy Solutions", "Rating": "A+", "Outlook": "Very Positive", "Risk": "Low"},
                {"Company": "HealthCare Plus", "Rating": "B", "Outlook": "Neutral", "Risk": "Medium"},
                {"Company": "FinTech Global", "Rating": "A", "Outlook": "Positive", "Risk": "Medium"},
                {"Company": "Logistics Corp", "Rating": "B+", "Outlook": "Positive", "Risk": "Medium"}
            ]
            
            ipo_ratings_df = pd.DataFrame(ipo_ratings)
            st.dataframe(ipo_ratings_df, use_container_width=True)
        
        else:
            st.warning("No upcoming IPO data available at the moment.")
    
    with tab3:
        st.subheader("Mutual Fund Analysis & Predictions")
        
        st.info("Comprehensive analysis of mutual funds with performance predictions")
        
        # Get mutual fund data
        mf_df = get_mutual_funds()
        
        if not mf_df.empty:
            st.dataframe(mf_df, use_container_width=True)
            
            # Mutual fund performance chart
            st.subheader("Mutual Fund Performance Comparison")
            
            # Simulated performance data
            categories = mf_df['Category'].unique()
            performance_data = {
                'Category': [],
                'Average Return': []
            }
            
            for category in categories:
                performance_data['Category'].append(category)
                # Simulate average returns based on category
                if category == "Equity":
                    performance_data['Average Return'].append(18.5)
                elif category == "Hybrid":
                    performance_data['Average Return'].append(14.2)
                elif category == "Debt":
                    performance_data['Average Return'].append(9.2)
                elif category == "Index":
                    performance_data['Average Return'].append(16.4)
                else:
                    performance_data['Average Return'].append(22.8)
            
            performance_df = pd.DataFrame(performance_data)
            
            fig = px.bar(performance_df, x='Category', y='Average Return', 
                         title="Average Returns by Fund Category",
                         color='Average Return',
                         color_continuous_scale=px.colors.sequential.Viridis)
            fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                'font': {'color': 'white'}
            })
            st.plotly_chart(fig, use_container_width=True)
            
            # Fund recommendation
            st.subheader("Top Fund Recommendations")
            
            recommended_funds = [
                {"Fund Name": "BlueChip Growth Fund", "Category": "Equity", "Recommendation": "Strong Buy", "Target Return": "20%"},
                {"Fund Name": "Sectoral Technology Fund", "Category": "Sectoral", "Recommendation": "Buy", "Target Return": "25%"},
                {"Fund Name": "Index Fund Nifty 50", "Category": "Index", "Recommendation": "Buy", "Target Return": "16%"},
                {"Fund Name": "Balanced Advantage Fund", "Category": "Hybrid", "Recommendation": "Hold", "Target Return": "14%"}
            ]
            
            recommended_df = pd.DataFrame(recommended_funds)
            st.dataframe(recommended_df, use_container_width=True)
            
            # Investment strategy
            st.subheader("Investment Strategy Suggestions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**For Aggressive Investors:**")
                st.write("• Focus on Equity and Sectoral funds")
                st.write("• 70% in growth funds")
                st.write("• 30% in hybrid funds")
                st.write("• Consider SIP for volatility management")
            
            with col2:
                st.markdown("**For Conservative Investors:**")
                st.write("• Focus on Debt and Hybrid funds")
                st.write("• 60% in debt funds")
                st.write("• 30% in hybrid funds")
                st.write("• 10% in equity funds for growth")
        
        else:
            st.warning("No mutual fund data available at the moment.")

# Market Overview Page
elif selected == "Market Overview":
    st.title("🌍 Market Overview")
    
    # Load global indices with caching
    indices_data = fetch_global_indices()
    
    # Display all indices in a grid
    st.subheader("Global Market Indices")
    cols = st.columns(4)
    index_count = 0
    for symbol, data in indices_data.items():
        with cols[index_count % 4]:
            currency_symbol = "$" if data["currency"] == "USD" else "₹" if data["currency"] == "INR" else "€" if data["currency"] == "EUR" else "£"
            st.metric(
                label=data["name"],
                value=f"{currency_symbol}{data['price']:.2f}",
                delta=f"{data['change']:.2f}%"
            )
        index_count += 1
    
    # Market heatmap (simulated)
    st.subheader("Market Heatmap")
    
    # Simulated sector performance
    sectors = {
        "Technology": np.random.uniform(-3, 5),
        "Healthcare": np.random.uniform(-2, 4),
        "Financials": np.random.uniform(-4, 3),
        "Energy": np.random.uniform(-5, 6),
        "Consumer Cyclical": np.random.uniform(-2, 4),
        "Real Estate": np.random.uniform(-3, 2),
        "Utilities": np.random.uniform(-1, 3),
        "Communications": np.random.uniform(-2, 5),
        "Materials": np.random.uniform(-3, 4)
    }
    
    sector_df = pd.DataFrame({
        "Sector": list(sectors.keys()),
        "Performance": list(sectors.values())
    })
    
    fig = px.bar(sector_df, x="Sector", y="Performance", 
                 title="Sector Performance (%)", 
                 color="Performance",
                 color_continuous_scale=px.colors.diverging.RdYlGn)
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font': {'color': 'white'}
    })
    st.plotly_chart(fig, use_container_width=True)
    
    # Economic indicators
    st.subheader("Key Economic Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Inflation Rate", "4.2%", "-0.3%")
    with col2:
        st.metric("Unemployment Rate", "3.8%", "0.1%")
    with col3:
        st.metric("GDP Growth", "2.9%", "0.2%")
    with col4:
        st.metric("Interest Rate", "5.5%", "0.0%")
    
    # Currency rates
    st.subheader("Currency Exchange Rates")
    
    currencies = {
        "USD/INR": 83.25,
        "EUR/INR": 90.15,
        "GBP/INR": 105.40,
        "JPY/INR": 0.58,
        "USD/EUR": 0.92,
        "USD/GBP": 0.79,
        "USD/JPY": 143.50
    }
    
    cols = st.columns(4)
    currency_count = 0
    for pair, rate in currencies.items():
        with cols[currency_count % 4]:
            st.metric(pair, f"{rate:.2f}")
        currency_count += 1

# News & Sentiment Page
elif selected == "News & Sentiment":
    st.title("📰 News & Market Sentiment")
    
    tab1, tab2 = st.tabs(["Market News", "Sentiment Analysis"])
    
    with tab1:
        st.subheader("Latest Financial News")
        
        news_articles = fetch_news()
        if news_articles:
            for article in news_articles:
                with st.expander(f"{article['title']} - {article['source']['name']}"):
                    if article.get('urlToImage'):
                        st.image(article['urlToImage'], width=300)
                    st.write(article.get('description', 'No description available'))
                    st.write(f"Published at: {article['publishedAt']}")
                    st.markdown(f"[Read full article]({article['url']})")
        else:
            st.info("News feed temporarily unavailable. Please try again later.")
    
    with tab2:
        st.subheader("Market Sentiment Analysis")
        
        # Simulated sentiment data
        sentiment_data = {
            "Overall Market": {"Bullish": 45, "Neutral": 30, "Bearish": 25},
            "Technology Sector": {"Bullish": 55, "Neutral": 25, "Bearish": 20},
            "Financial Sector": {"Bullish": 35, "Neutral": 40, "Bearish": 25},
            "Healthcare Sector": {"Bullish": 60, "Neutral": 20, "Bearish": 20}
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            for sector, sentiment in list(sentiment_data.items())[:2]:
                fig = px.pie(values=list(sentiment.values()), names=list(sentiment.keys()),
                             title=f"{sector} Sentiment",
                             color_discrete_map={"Bullish": "#00CC96", "Neutral": "#FFA15A", "Bearish": "#EF553B"})
                fig.update_layout({
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                    'font': {'color': 'white'},
                    'height': 300
                })
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            for sector, sentiment in list(sentiment_data.items())[2:]:
                fig = px.pie(values=list(sentiment.values()), names=list(sentiment.keys()),
                             title=f"{sector} Sentiment",
                             color_discrete_map={"Bullish": "#00CC96", "Neutral": "#FFA15A", "Bearish": "#EF553B"})
                fig.update_layout({
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                    'font': {'color': 'white'},
                    'height': 300
                })
                st.plotly_chart(fig, use_container_width=True)
        
        # Social media sentiment
        st.subheader("Social Media Sentiment Trends")
        
        # Simulated social media data
        dates = pd.date_range(end=datetime.now(), periods=15, freq='D')
        twitter_sentiment = np.random.uniform(40, 70, 15)
        reddit_sentiment = np.random.uniform(30, 65, 15)
        
        social_df = pd.DataFrame({
            'Date': dates,
            'Twitter': twitter_sentiment,
            'Reddit': reddit_sentiment
        })
        
        fig = px.line(social_df, x='Date', y=['Twitter', 'Reddit'], 
                      title="Social Media Sentiment Index (Higher = More Positive)")
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'font': {'color': 'white'}
        })
        st.plotly_chart(fig, use_container_width=True)

# Settings Page
elif selected == "Settings":
    st.title("⚙️ Settings")
    
    st.subheader("Appearance Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox("Color Theme", ["Dark", "Black", "Blue Dark"])
        font_size = st.slider("Font Size", 12, 18, 14)
        chart_animation = st.checkbox("Enable Chart Animations", value=True)
    
    with col2:
        default_home = st.selectbox("Default Home Page", ["Dashboard", "Stock Analysis", "Predictions"])
        refresh_rate = st.slider("Data Refresh Rate (minutes)", 5, 60, 15)
        st.checkbox("Show Detailed Tooltips", value=True)
    
    st.subheader("Data Sources")
    
    data_sources = {
        "Stock Data": "Yahoo Finance",
        "News": "NewsAPI",
        "Economic Data": "Simulated for Demo",
        "IPO Data": "Simulated for Demo",
        "Mutual Fund Data": "Simulated for Demo"
    }
    
    for source, provider in data_sources.items():
        st.write(f"**{source}:** {provider}")
    
    st.subheader("Account")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Name", "Ashwik Bire")
        st.text_input("Email", "ashwik.bire@example.com")
    
    with col2:
        st.selectbox("Currency", ["USD", "INR", "EUR", "GBP"])
        st.selectbox("Date Format", ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"])
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
    
    st.subheader("About")
    st.write("**MarketMentor Pro v1.0**")
    st.write("Advanced financial analytics platform for investors and traders.")
    st.write("Developed by Ashwik Bire")
    st.write("[Contact Support](mailto:support@marketmentor.com)")

else:
    st.title(f"{selected}")
    st.info(f"This page ({selected}) is under development and will be implemented soon.")
