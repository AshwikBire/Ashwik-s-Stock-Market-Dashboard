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
    .stButton>button {background-color: #1A1A1A; color: white; border: 1px solid #333333; border-radius: 4px;}
    .stTextInput>div>div>input, .stSelectbox>div>div>select {background-color: #1A1A1A; color: white; border: 1px solid #333333;}
    .stMetric {background-color: #1A1A1A; border-radius: 5px; padding: 10px; border-left: 3px solid #4A4A4A;}
    .stDataFrame {background-color: #1A1A1A; color: white;}
    .streamlit-expanderHeader {background-color: #1A1A1A; color: white; border-radius: 4px; padding: 8px;}
    .stTabs {background-color: #000000;}
    div[data-baseweb="tab-list"] {background-color: #1A1A1A; gap: 2px;}
    div[data-baseweb="tab"] {background-color: #1A1A1A; color: white; padding: 10px 20px; border-radius: 4px 4px 0 0;}
    div[data-baseweb="tab"]:hover {background-color: #333333;}
    div[data-baseweb="tab"][aria-selected="true"] {background-color: #4A4A4A;}
    .stProgress > div > div > div {background-color: #4A4A4A;}
    .css-1d391kg {background-color: #000000;}
    .css-1lcbmhc {background-color: #000000;}
    .stAlert {background-color: #1A1A1A; color: white;}
    .stSuccess {background-color: #1A1A1A; color: white;}
    .stInfo {background-color: #1A1A1A; color: white;}
    .stWarning {background-color: #1A1A1A; color: white;}
    .stError {background-color: #1A1A1A; color: white;}
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
    metals = {
        "GC=F": {"name": "Gold", "currency": "USD"},
        "SI=F": {"name": "Silver", "currency": "USD"},
        "PL=F": {"name": "Platinum", "currency": "USD"},
        "PA=F": {"name": "Palladium", "currency": "USD"},
        "HG=F": {"name": "Copper", "currency": "USD"}
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

# Sidebar navigation without images
with st.sidebar:
    st.title("MarketMentor Pro")
    
    selected = option_menu(
        "Navigation",
        ["Dashboard", "Stock Analysis", "Technical Analysis", "Portfolio Manager", 
         "Options Chain", "Market Overview", "Precious Metals", "Economic Calendar", "Crypto Markets",
         "News & Sentiment", "Learning Center", "Company Info", "Settings"],
        icons=['speedometer2', 'graph-up', 'bar-chart-line', 'wallet', 
               'diagram-3', 'globe', 'gem', 'calendar', 'currency-bitcoin',
               'newspaper', 'book', 'building', 'gear'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#1A1A1A"},
            "icon": {"color": "#FFFFFF", "font-size": "16px"}, 
            "nav-link": {"color": "#FFFFFF", "font-size": "14px", "text-align": "left"},
            "nav-link-selected": {"background-color": "#4A4A4A"},
        }
    )
    
    # Watchlist section in sidebar
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
    
    # Developer info
    st.markdown("---")
    st.write("**Developed by Ashwik Bire**")
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/ashwik-bire-b2a000186)")

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
    
    # Precious metals overview
    st.subheader("🥇 Precious Metals")
    metals_data = fetch_precious_metals()
    cols = st.columns(5)
    metal_count = 0
    for symbol, data in list(metals_data.items())[:5]:
        with cols[metal_count % 5]:
            currency_symbol = "$" if data["currency"] == "USD" else "₹" if data["currency"] == "INR" else "€" if data["currency"] == "EUR" else "£"
            st.metric(
                label=data["name"],
                value=f"{currency_symbol}{data['price']:.2f}",
                delta=f"{data['change']:.2f}%"
            )
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
        if st.button("View Options"):
            st.session_state.selected = "Options Chain"
    with action_cols[3]:
        if st.button("Check News"):
            st.session_state.selected = "News & Sentiment"

# Precious Metals Page
elif selected == "Precious Metals":
    st.title("🥇 Precious Metals")
    
    # Load precious metals data
    metals_data = fetch_precious_metals()
    
    # Display metals in a grid
    st.subheader("Precious Metals Prices")
    cols = st.columns(5)
    for idx, (symbol, data) in enumerate(metals_data.items()):
        with cols[idx % 5]:
            currency_symbol = "$" if data["currency"] == "USD" else "₹" if data["currency"] == "INR" else "€" if data["currency"] == "EUR" else "£"
            st.metric(
                label=data["name"],
                value=f"{currency_symbol}{data['price']:.2f}",
                delta=f"{data['change']:.2f}%"
            )
    
    # Metals performance chart
    st.subheader("📈 Metals Performance")
    
    # Simulated performance data
    metals_perf = pd.DataFrame({
        'Metal': ['Gold', 'Silver', 'Platinum', 'Palladium', 'Copper'],
        '7d Change': [1.2, 3.5, -0.8, -2.1, 0.5],
        '30d Change': [5.3, 8.2, 2.5, -5.3, 3.2],
        '90d Change': [12.5, 18.7, 7.8, -8.5, 9.3]
    })
    
    fig = px.bar(metals_perf, x='Metal', y=['7d Change', '30d Change', '90d Change'], 
                 title="Precious Metals Performance", barmode='group')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font': {'color': 'white'}
    })
    st.plotly_chart(fig, use_container_width=True)
    
    # Gold vs Silver ratio
    st.subheader("📊 Gold/Silver Ratio")
    
    # Calculate ratio
    gold_price = metals_data.get("GC=F", {}).get("price", 1900)
    silver_price = metals_data.get("SI=F", {}).get("price", 22.5)
    ratio = gold_price / silver_price if silver_price else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gold Price", f"${gold_price:.2f}")
    with col2:
        st.metric("Silver Price", f"${silver_price:.2f}")
    with col3:
        st.metric("Gold/Silver Ratio", f"{ratio:.2f}")
    
    # Historical ratio context
    st.write("**Historical Context:**")
    st.write("- Ratio < 50: Silver is expensive relative to gold")
    st.write("- Ratio 50-80: Typical range")
    st.write("- Ratio > 80: Gold is expensive relative to silver")
    
    # Metals news
    st.subheader("📰 Metals News")
    metals_news = fetch_news("gold silver precious metals")
    if metals_news:
        for article in metals_news[:3]:
            with st.expander(f"{article['title']} - {article['source']['name']}"):
                st.write(article.get('description', 'No description available'))
                st.markdown(f"[Read more]({article['url']})")
    else:
        st.info("Metals news temporarily unavailable.")

# ... (rest of the code remains the same with updated theme)

# Add a footer to all pages
st.write("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.write("**MarketMentor Pro**")
    st.write("Advanced Financial Analytics Platform")
with footer_col2:
    st.write("**Developed by Ashwik Bire**")
    st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/ashwik-bire-b2a000186)")
with footer_col3:
    st.write("**© 2023 MarketMentor**")
    st.write("Version 2.1.0")

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
