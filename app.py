import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
from plotly import graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from textblob import TextBlob
from xgboost import XGBRegressor
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import ta  # Technical analysis library
import warnings
import json
from bs4 import BeautifulSoup
import time
import calendar
from fredapi import Fred
import investpy
import mplfinance as mpf
import plotly.figure_factory as ff
from fpdf import FPDF
import base64
import pytz
import quantstats as qs
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import networkx as nx

warnings.filterwarnings('ignore')

# Set page config with dark theme
st.set_page_config(
    page_title="MarketMentor", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Apply black and red theme CSS
st.markdown("""
<style>
    .main {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #B22222;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF0000;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #1A1A1A;
        color: white;
        border: 1px solid #333333;
        border-radius: 4px;
    }
    .stSelectbox>div>div>select {
        background-color: #1A1A1A;
        color: white;
        border: 1px solid #333333;
        border-radius: 4px;
    }
    .stSlider>div>div>div>div {
        background-color: #B22222;
    }
    .stMetric {
        background-color: #1A1A1A;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(178, 34, 34, 0.3);
        border-left: 4px solid #B22222;
    }
    .css-1d391kg {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #1A1A1A;
        border-left: 4px solid #B22222;
    }
    .stProgress > div > div > div > div {
        background-color: #B22222;
    }
    .stAlert {
        background-color: #1A1A1A;
        border-left: 4px solid #B22222;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1A1A1A;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        border-bottom: 2px solid #333333;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1A1A1A;
        color: #FF0000;
        border-bottom: 2px solid #B22222;
    }
    div[data-testid="stSidebarUserContent"] {
        background-color: #000000;
    }
    .sidebar .sidebar-content {
        background-color: #000000;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #B22222;
        border-bottom: 1px solid #333333;
        padding-bottom: 0.3rem;
    }
    .stDataFrame {
        background-color: #1A1A1A;
    }
    header {
        background-color: #000000;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #1A1A1A;
    }
    .st-bh {
        background-color: transparent;
    }
    .st-ag {
        background-color: #000000;
    }
    .st-af {
        background-color: #000000;
    }
    .st-ae {
        background-color: #000000;
    }
    .card {
        background-color: #1A1A1A;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(178, 34, 34, 0.3);
        margin-bottom: 1rem;
        border-left: 4px solid #B22222;
    }
    .red-badge {
        background-color: #B22222;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .green-badge {
        background-color: #008000;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1A1A1A;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(178, 34, 34, 0.3);
        text-align: center;
        border-left: 4px solid #B22222;
    }
    .news-card {
        background-color: #1A1A1A;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #333333;
    }
    .news-card:hover {
        border-left: 4px solid #B22222;
        transition: all 0.3s ease;
    }
    .stock-card {
        background-color: #1A1A1A;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #333333;
    }
    .stock-card:hover {
        border-left: 4px solid #B22222;
        transition: all 0.3s ease;
    }
    .indicator-positive {
        color: #00FF00;
        font-weight: bold;
    }
    .indicator-negative {
        color: #FF0000;
        font-weight: bold;
    }
    .section-header {
        background: linear-gradient(90deg, #000000, #1A1A1A);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #B22222;
    }
</style>
""", unsafe_allow_html=True)

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"
FRED_API_KEY = "6c5b447e2f6e497d0f4d9b4b3b4f4e4a"  # You might need to get your own key
ALPHA_VANTAGE_KEY = "X86NOH6II01P7R24"  # Replace with your Alpha Vantage key

# Developer Info
developer_info = {
    "name": "Ashwik Bire",
    "linkedin": "https://www.linkedin.com/in/ashwik-bire-1b4530250/",
    "role": "Financial Data Scientist"
}

# Cache data functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker, period="1y"):
    return yf.Ticker(ticker).history(period=period)

@st.cache_data(ttl=3600)
def get_indices_data(indices_dict):
    data = {}
    for symbol, name in indices_dict.items():
        try:
            ticker_data = yf.Ticker(symbol).history(period="1d")
            if not ticker_data.empty:
                last_close = round(ticker_data['Close'].iloc[-1], 2)
                prev_close = round(ticker_data['Close'].iloc[-2] if len(ticker_data) > 1 else last_close, 2)
                change = round(last_close - prev_close, 2)
                percent_change = round((change / prev_close) * 100, 2)
                data[name] = {"value": last_close, "change": change, "percent_change": percent_change}
        except:
            continue
    return data

@st.cache_data(ttl=3600)
def get_news(query="stock market", page_size=10):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize={page_size}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("articles", [])
    except:
        pass
    return []

@st.cache_data(ttl=3600)
def get_company_info(ticker):
    try:
        company = yf.Ticker(ticker)
        info = company.info
        return info
    except:
        return None

@st.cache_data(ttl=3600)
def get_market_movers():
    try:
        # Get top gainers and losers
        gainers = pd.read_html("https://money.rediff.com/gainers/bse/daily/groupa")[0].head(10)
        losers = pd.read_html("https://money.rediff.com/losers/bse/daily/groupa")[0].head(10)
        return gainers, losers
    except:
        return None, None

@st.cache_data(ttl=3600)
def get_ipo_data():
    try:
        # Get upcoming IPOs
        ipo_url = "https://www.chittorgarh.com/report/upcoming-ipo-in-india/82/"
        ipo_data = pd.read_html(ipo_url)[0]
        return ipo_data
    except:
        return None

@st.cache_data(ttl=3600)
def get_mutual_funds():
    # Sample mutual fund data - in a real app, you'd use an API
    mf_data = {
        'Fund Name': ['SBI Bluechip Fund', 'HDFC Top 100 Fund', 'ICICI Pru Bluechip Fund', 
                     'Axis Long Term Equity Fund', 'Mirae Asset Emerging Bluechip Fund'],
        'Category': ['Large Cap', 'Large Cap', 'Large Cap', 'ELSS', 'Large & Mid Cap'],
        '1Y Return (%)': [18.5, 17.2, 19.1, 22.3, 26.7],
        '3Y Return (%)': [15.2, 14.8, 16.3, 18.9, 22.4],
        '5Y Return (%)': [13.7, 13.2, 14.5, 16.8, 19.2],
        'Risk': ['Moderate', 'Moderate', 'Moderate', 'High', 'High']
    }
    return pd.DataFrame(mf_data)

@st.cache_data(ttl=3600)
def get_global_indices():
    indices = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ',
        '^FTSE': 'FTSE 100',
        '^N225': 'Nikkei 225',
        '^HSI': 'Hang Seng',
        '^AXJO': 'ASX 200',
        '^GDAXI': 'DAX',
        '^FCHI': 'CAC 40',
        '^STOXX50E': 'Euro Stoxx 50'
    }
    return get_indices_data(indices)

@st.cache_data(ttl=3600)
def get_commodities():
    commodities = {
        'GC=F': 'Gold',
        'SI=F': 'Silver',
        'CL=F': 'Crude Oil',
        'NG=F': 'Natural Gas',
        'ZC=F': 'Corn',
        'ZS=F': 'Soybeans'
    }
    return get_indices_data(commodities)

@st.cache_data(ttl=3600)
def get_economic_calendar():
    # Sample economic calendar data
    events = {
        'Date': ['2023-06-15', '2023-06-16', '2023-06-20', '2023-06-22', '2023-06-25'],
        'Event': ['Fed Interest Rate Decision', 'Bank of Japan Policy Meeting', 
                 'UK Inflation Data', 'US Jobless Claims', 'Eurozone PMI'],
        'Impact': ['High', 'Medium', 'High', 'Medium', 'Medium'],
        'Previous': ['5.25%', '-0.1%', '8.7%', '230K', '54.2'],
        'Forecast': ['5.5%', '-0.1%', '8.4%', '225K', '54.5']
    }
    return pd.DataFrame(events)

@st.cache_data(ttl=3600)
def get_alpha_vantage_data(function, symbol, interval=None):
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_KEY
    }
    
    if interval:
        params["interval"] = interval
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        return data
    except:
        return None

@st.cache_data(ttl=3600)
def get_crypto_data(symbol="BTC-USD", period="1mo"):
    try:
        data = yf.Ticker(symbol).history(period=period)
        return data
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_sector_performance():
    # Sample sector performance data
    sectors = {
        'Technology': 2.5,
        'Healthcare': 1.8,
        'Financial Services': -0.7,
        'Consumer Cyclical': 1.2,
        'Communication Services': 3.1,
        'Industrials': 0.5,
        'Energy': -2.3,
        'Utilities': -1.2,
        'Real Estate': 0.8,
        'Materials': -0.5,
        'Consumer Defensive': 1.5
    }
    return sectors

@st.cache_data(ttl=3600)
def get_earnings_calendar():
    # Sample earnings calendar
    earnings = {
        'Date': ['2023-07-15', '2023-07-18', '2023-07-20', '2023-07-22', '2023-07-25'],
        'Company': ['Reliance Industries', 'Infosys', 'HDFC Bank', 'TCS', 'ICICI Bank'],
        'EPS Estimate': [25.4, 18.7, 22.1, 32.5, 16.8],
        'Revenue Estimate (Cr)': [225000, 125000, 98500, 155000, 87500]
    }
    return pd.DataFrame(earnings)

@st.cache_data(ttl=3600)
def get_insider_trading():
    # Sample insider trading data
    insider_data = {
        'Date': ['2023-06-15', '2023-06-14', '2023-06-10', '2023-06-08', '2023-06-05'],
        'Company': ['Reliance', 'Tata Motors', 'HDFC Bank', 'Infosys', 'Wipro'],
        'Insider': ['Mukesh Ambani', 'N Chandrasekaran', 'Aditya Puri', 'Salil Parekh', 'Thierry Delaporte'],
        'Relationship': ['CEO', 'Chairman', 'Ex-CEO', 'CEO', 'CEO'],
        'Transaction': ['Buy', 'Buy', 'Sell', 'Buy', 'Sell'],
        'Shares': [100000, 50000, 25000, 75000, 30000],
        'Value (Cr)': [250, 25, 18, 120, 15]
    }
    return pd.DataFrame(insider_data)

@st.cache_data(ttl=3600)
def get_dividend_calendar():
    # Sample dividend calendar
    dividends = {
        'Declaration Date': ['2023-06-15', '2023-06-18', '2023-06-20', '2023-06-22'],
        'Company': ['ITC', 'Coal India', 'HUL', 'Power Grid'],
        'Dividend (₹)': [6.5, 5.0, 22.0, 4.5],
        'Type': ['Interim', 'Final', 'Interim', 'Interim'],
        'Yield (%)': [3.2, 5.1, 1.8, 4.2]
    }
    return pd.DataFrame(dividends)

@st.cache_data(ttl=3600)
def get_etf_data():
    # Sample ETF data
    etfs = {
        'ETF': ['Nippon India ETF Nifty BeES', 'ICICI Pru Nifty ETF', 'SBI ETF Nifty 50', 
                'Kotak Nifty ETF', 'HDFC Nifty ETF'],
        'AUM (Cr)': [12500, 8500, 7200, 4500, 3800],
        'Expense Ratio (%)': [0.05, 0.07, 0.04, 0.06, 0.08],
        '1Y Return (%)': [15.2, 14.8, 15.5, 14.9, 15.1],
        'Tracking Error': [0.02, 0.03, 0.01, 0.04, 0.03]
    }
    return pd.DataFrame(etfs)

@st.cache_data(ttl=3600)
def get_bond_yields():
    # Sample bond yield data
    bonds = {
        'Tenure': ['1 Month', '3 Month', '6 Month', '1 Year', '5 Year', '10 Year', '30 Year'],
        'Yield (%)': [6.25, 6.45, 6.70, 6.85, 7.20, 7.35, 7.50],
        'Previous (%)': [6.20, 6.40, 6.65, 6.80, 7.15, 7.30, 7.45],
        'Change (bps)': [5, 5, 5, 5, 5, 5, 5]
    }
    return pd.DataFrame(bonds)

@st.cache_data(ttl=3600)
def get_market_sentiment():
    # Sample market sentiment data
    sentiment = {
        'Indicator': ['Fear & Greed Index', 'Put/Call Ratio', 'VIX', 'Advance/Decline', 'High/Low Ratio'],
        'Value': [45, 0.85, 18.5, 1.25, 0.85],
        'Interpretation': ['Fear', 'Neutral', 'Low Volatility', 'Bullish', 'Neutral']
    }
    return pd.DataFrame(sentiment)

@st.cache_data(ttl=3600)
def get_short_interest():
    # Sample short interest data
    short_data = {
        'Symbol': ['YESBANK', 'VEDL', 'TATASTEEL', 'ADANIPORTS', 'SBIN'],
        'Short Interest (%)': [8.5, 7.2, 6.8, 5.5, 4.9],
        'Days to Cover': [4.2, 3.8, 3.5, 2.9, 2.5],
        'Change (%)': [0.5, -0.2, 0.3, -0.4, 0.1]
    }
    return pd.DataFrame(short_data)

# Sidebar menu
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/stock.png", width=80)
    st.title("MarketMentor")
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "Company Overview", "Market Movers", "F&O", "Global Markets", 
                "Mutual Funds", "SIP Calculator", "IPO Tracker", "Predictions for Mutual Funds & IPOs", 
                "Mutual Fund NAV Viewer", "Sectors", "News", "Learning", "Volume Spike", 
                "Stock Screener", "Predictions", "Buy/Sell Predictor", "News Sentiment", 
                "Technical Analysis", "Portfolio Tracker", "Economic Calendar", "Crypto", 
                "Options Chain", "Bonds", "ETF", "Market Sentiment", "Earnings Calendar",
                "Insider Trading", "Dividend Calendar", "Short Interest"],
        icons=['house', 'building', 'graph-up', 'activity', 'globe', 
               'bank', 'calculator', 'rocket', 'lightning', 
               'bar-chart', 'grid', 'newspaper', 'book', 'activity', 
               'search', 'graph-up-arrow', 'currency-exchange', 'chat', 
               'graph-up', 'wallet', 'calendar', 'currency-bitcoin', 'link',
               'graph-up-arrow', 'collection', 'activity', 'calendar-event',
               'person-badge', 'cash-coin', 'arrow-down-right-circle'],
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#000000"},
            "icon": {"color": "#B22222", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#FFFFFF"},
            "nav-link-selected": {"background-color": "#1A1A1A", "color": "#B22222"},
        }
    )
    
    # Developer info in sidebar
    st.markdown("---")
    st.markdown("### Developer Info")
    st.markdown(f"**Name:** {developer_info['name']}")
    st.markdown(f"**Role:** {developer_info['role']}")
    st.markdown(f"[LinkedIn Profile]({developer_info['linkedin']})")

# Home - Market Overview
if selected == "Home":
    st.title("🏠 Home - Market Overview")
    
    # Market indices
    indices_dict = {
        "^NSEI": "Nifty 50",
        "^BSESN": "Sensex",
        "^NSEBANK": "Bank Nifty",
        "^CNXIT": "Nifty IT",
        "NSE_MIDCAP.NS": "Nifty Midcap",
        "NSEMDCP50.NS": "Nifty Next 50"
    }
    
    indices_data = get_indices_data(indices_dict)
    
    # Display indices in columns
    cols = st.columns(3)
    idx = 0
    for name, data in indices_data.items():
        with cols[idx % 3]:
            change_color = "green" if data["percent_change"] >= 0 else "red"
            change_icon = "📈" if data["percent_change"] >= 0 else "📉"
            st.metric(
                label=name,
                value=f"₹{data['value']:,}",
                delta=f"{change_icon} {data['percent_change']:.2f}% ({data['change']:.2f})",
                delta_color="normal" if data["percent_change"] >= 0 else "inverse"
            )
        idx += 1
    
    # Market overview tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Quick Stock Search", "Sector Performance", "Market Summary", "Watchlist", "Market Pulse"])
    
    with tab1:
        st.subheader("🔍 Quick Stock Search")
        ticker = st.text_input("Enter stock symbol (e.g., RELIANCE.NS, INFY.NS):", "RELIANCE.NS", key="home_search")
        
        if ticker:
            data = get_stock_data(ticker, "1mo")
            if not data.empty:
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
                    title=f"{ticker} Stock Price",
                    yaxis_title="Price (₹)",
                    xaxis_title="Date",
                    height=400,
                    template="plotly_dark",
                    plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    font=dict(color='#FFFFFF')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display key metrics
                current_price = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - prev_close
                change_percent = (change / prev_close) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"₹{current_price:.2f}")
                col2.metric("Change", f"₹{change:.2f}", f"{change_percent:.2f}%")
                col3.metric("Previous Close", f"₹{prev_close:.2f}")
    
    with tab2:
        st.subheader("📊 Sector Performance")
        sectors = get_sector_performance()
        
        # Create sector performance chart
        fig = px.bar(
            x=list(sectors.keys()),
            y=list(sectors.values()),
            title="Sector Performance (%)",
            color=list(sectors.values()),
            color_continuous_scale=["red", "white", "green"]
        )
        fig.update_layout(
            template="plotly_dark", 
            xaxis_title="Sector", 
            yaxis_title="Change (%)",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Market Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.info("**Market Breadth**")
            breadth_data = {
                "Advances": 1256,
                "Declines": 874,
                "Unchanged": 120
            }
            fig = px.pie(
                values=list(breadth_data.values()),
                names=list(breadth_data.keys()),
                title="Market Breadth"
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font=dict(color='#FFFFFF')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.info("**Volume Analysis**")
            volume_data = {
                "Nifty 50": "₹45,672 Cr",
                "Bank Nifty": "₹32,145 Cr",
                "Top Gainer": "RELIANCE (₹2,145 Cr)",
                "Top Loser": "HDFC Bank (₹1,876 Cr)"
            }
            for key, value in volume_data.items():
                st.write(f"{key}: {value}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.subheader("⭐ Watchlist")
        watchlist_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS"]
        watchlist_data = []
        
        for stock in watchlist_stocks:
            try:
                data = yf.Ticker(stock).history(period="1d")
                if not data.empty:
                    current = data['Close'].iloc[-1]
                    prev = data['Close'].iloc[-2] if len(data) > 1 else current
                    change = ((current - prev) / prev) * 100
                    watchlist_data.append({
                        "Stock": stock,
                        "Price": f"₹{current:.2f}",
                        "Change": f"{change:.2f}%",
                        "Color": "green" if change >= 0 else "red"
                    })
            except:
                continue
        
        if watchlist_data:
            for item in watchlist_data:
                st.markdown('<div class="stock-card">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([2, 1, 1])
                col1.write(item["Stock"])
                col2.write(item["Price"])
                col3.write(f":{item['Color']}[{item['Change']}]")
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.subheader("📊 Market Pulse")
        
        # Market sentiment
        sentiment_data = get_market_sentiment()
        st.dataframe(sentiment_data, use_container_width=True)
        
        # Market heatmap
        st.subheader("Market Heatmap")
        heatmap_data = {
            'Stock': ['RELIANCE', 'HDFC', 'INFY', 'TCS', 'ICICI', 'HUL', 'ITC', 'SBIN', 'BAJFIN', 'KOTAK'],
            'Change (%)': [2.5, -1.2, 3.1, 1.8, -0.7, 0.9, -1.5, 2.2, -0.3, 1.5]
        }
        heatmap_df = pd.DataFrame(heatmap_data)
        
        fig = px.treemap(heatmap_df, path=['Stock'], values='Change (%)',
                        color='Change (%)', color_continuous_scale=['red', 'white', 'green'])
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# Company Overview
elif selected == "Company Overview":
    st.title("🏢 Company Overview")
    
    # Stock search
    ticker = st.text_input("Enter stock symbol:", "RELIANCE.NS", key="company_overview")
    
    if ticker:
        info = get_company_info(ticker)
        if info:
            # Display company info in a structured way
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if 'longName' in info:
                    st.subheader(info['longName'])
                if 'sector' in info:
                    st.write(f"**Sector:** {info['sector']}")
                if 'industry' in info:
                    st.write(f"**Industry:** {info['industry']}")
                if 'website' in info:
                    st.write(f"**Website:** {info['website']}")
                if 'longBusinessSummary' in info:
                    with st.expander("Business Summary"):
                        st.write(info['longBusinessSummary'])
            
            with col2:
                # Display key metrics
                metrics_data = {}
                if 'marketCap' in info:
                    metrics_data['Market Cap'] = f"₹{info['marketCap']/10000000:.2f} Cr"
                if 'trailingPE' in info:
                    metrics_data['P/E Ratio'] = f"{info['trailingPE']:.2f}"
                if 'priceToBook' in info:
                    metrics_data['P/B Ratio'] = f"{info['priceToBook']:.2f}"
                if 'dividendYield' in info:
                    metrics_data['Dividend Yield'] = f"{info['dividendYield']*100 if info['dividendYield'] else 0:.2f}%"
                if 'profitMargins' in info:
                    metrics_data['Profit Margin'] = f"{info['profitMargins']*100:.2f}%"
                if 'returnOnEquity' in info:
                    metrics_data['ROE'] = f"{info['returnOnEquity']*100:.2f}%"
                if '52WeekChange' in info:
                    metrics_data['52W Change'] = f"{info['52WeekChange']*100:.2f}%"
                
                # Display metrics in a grid
                cols = st.columns(3)
                for i, (key, value) in enumerate(metrics_data.items()):
                    with cols[i % 3]:
                        st.markdown(f'<div class="metric-card"><h4>{key}</h4><h3>{value}</h3></div>', unsafe_allow_html=True)
            
            # Tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price Chart", "Financials", "Holdings", "Analysis", "Options"])
            
            with tab1:
                st.subheader("Price Chart")
                period = st.selectbox("Select period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
                data = get_stock_data(ticker, period)
                
                if not data.empty:
                    # Create a candlestick chart
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
                        title=f"{ticker} Stock Price",
                        yaxis_title="Price (₹)",
                        xaxis_title="Date",
                        height=500,
                        template="plotly_dark",
                        plot_bgcolor='#000000',
                        paper_bgcolor='#000000',
                        font=dict(color='#FFFFFF')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add technical indicators
                    st.subheader("Technical Indicators")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        # Calculate RSI
                        delta = data['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        st.metric("RSI (14)", f"{rsi.iloc[-1]:.2f}")
                    
                    with col2:
                        # Calculate MACD
                        exp12 = data['Close'].ewm(span=12, adjust=False).mean()
                        exp26 = data['Close'].ewm(span=26, adjust=False).mean()
                        macd = exp12 - exp26
                        signal = macd.ewm(span=9, adjust=False).mean()
                        st.metric("MACD", f"{macd.iloc[-1]:.2f}")
                    
                    with col3:
                        # Calculate Bollinger Bands
                        sma20 = data['Close'].rolling(window=20).mean()
                        std20 = data['Close'].rolling(window=20).std()
                        upper_band = sma20 + (std20 * 2)
                        lower_band = sma20 - (std20 * 2)
                        current_price = data['Close'].iloc[-1]
                        bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1]) * 100
                        st.metric("Bollinger Position", f"{bb_position:.2f}%")
                    
                    with col4:
                        # Calculate Volume
                        volume_avg = data['Volume'].rolling(window=20).mean()
                        volume_ratio = data['Volume'].iloc[-1] / volume_avg.iloc[-1] if volume_avg.iloc[-1] > 0 else 0
                        st.metric("Volume Ratio", f"{volume_ratio:.2f}")
            
            with tab2:
                st.subheader("Financial Metrics")
                # Display key financial metrics
                financials = {
                    'Revenue Growth (YoY)': '15.2%',
                    'Net Profit Margin': '18.7%',
                    'ROCE': '22.1%',
                    'Debt to Equity': '0.45',
                    'Current Ratio': '1.8',
                    'Dividend Payout': '35.2%'
                }
                
                cols = st.columns(3)
                for i, (key, value) in enumerate(financials.items()):
                    with cols[i % 3]:
                        st.markdown(f'<div class="metric-card"><h4>{key}</h4><h3>{value}</h3></div>', unsafe_allow_html=True)
                
                # Financial charts
                st.subheader("Revenue & Profit Trend")
                years = ['2019', '2020', '2021', '2022', '2023']
                revenue = [100, 120, 145, 175, 210]  # Sample data
                profit = [15, 18, 22, 28, 35]  # Sample data
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=years, y=revenue, name='Revenue (Cr)'))
                fig.add_trace(go.Scatter(x=years, y=profit, name='Profit (Cr)', yaxis='y2', mode='lines+markers'))
                
                fig.update_layout(
                    title="Revenue and Profit Trend",
                    yaxis=dict(title="Revenue (₹ Cr)"),
                    yaxis2=dict(title="Profit (₹ Cr)", overlaying='y', side='right'),
                    template="plotly_dark",
                    plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    font=dict(color='#FFFFFF')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Institutional Holdings")
                # Sample holdings data
                holdings = {
                    'Promoters': '45.2%',
                    'FIIs': '23.7%',
                    'DIIs': '18.5%',
                    'Public': '12.6%'
                }
                
                fig = px.pie(
                    values=[float(h.strip('%')) for h in holdings.values()],
                    names=list(holdings.keys()),
                    title="Shareholding Pattern"
                )
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    font=dict(color='#FFFFFF')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("Analyst Recommendations")
                # Sample analyst data
                recommendations = {
                    'Strong Buy': 12,
                    'Buy': 8,
                    'Hold': 3,
                    'Sell': 1,
                    'Strong Sell': 0
                }
                
                fig = px.bar(
                    x=list(recommendations.keys()),
                    y=list(recommendations.values()),
                    title="Analyst Recommendations",
                    color=list(recommendations.keys()),
                    color_discrete_sequence=['green', 'lightgreen', 'gray', 'lightcoral', 'red']
                )
                fig.update_layout(
                    template="plotly_dark", 
                    xaxis_title="Recommendation", 
                    yaxis_title="Number of Analysts",
                    plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    font=dict(color='#FFFFFF')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Price targets
                st.subheader("Price Targets")
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Target", "₹2,850")
                col2.metric("High Target", "₹3,200")
                col3.metric("Low Target", "₹2,500")
            
            with tab5:
                st.subheader("Options Chain")
                # Sample options chain data
                options_data = {
                    'Strike Price': [2800, 2850, 2900, 2950, 3000, 3050, 3100],
                    'Call OI': [1245, 987, 765, 543, 432, 321, 210],
                    'Call Change': [125, 87, 65, 43, 32, 21, 10],
                    'Put OI': [210, 321, 432, 543, 765, 987, 1245],
                    'Put Change': [10, 21, 32, 43, 65, 87, 125]
                }
                
                options_df = pd.DataFrame(options_data)
                st.dataframe(options_df, use_container_width=True)
                
                # Options chain visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(x=options_df['Strike Price'], y=options_df['Call OI'], name='Call OI', marker_color='red'))
                fig.add_trace(go.Bar(x=options_df['Strike Price'], y=options_df['Put OI'], name='Put OI', marker_color='green'))
                
                fig.update_layout(
                    title="Options Open Interest",
                    xaxis_title="Strike Price",
                    yaxis_title="Open Interest",
                    barmode='group',
                    template="plotly_dark",
                    plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    font=dict(color='#FFFFFF')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Export company info as JSON
            if st.button("Export Company Info as JSON"):
                # Create a simplified version for JSON export
                export_info = {
                    'name': info.get('longName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'marketCap': info.get('marketCap', ''),
                    'peRatio': info.get('trailingPE', ''),
                    'pbRatio': info.get('priceToBook', ''),
                    'dividendYield': info.get('dividendYield', ''),
                    'website': info.get('website', ''),
                    'summary': info.get('longBusinessSummary', '')
                }
                
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_info, indent=2),
                    file_name=f"{ticker}_company_info.json",
                    mime="application/json"
                )
        else:
            st.error("Could not fetch company information. Please check the stock symbol.")

# Market Movers - Top Gainers & Losers
elif selected == "Market Movers":
    st.title("📈 Market Movers")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Nifty 50", "Sectoral Indices", "Volume Shockers", "All-Time Highs/Lows"])
    
    with tab1:
        st.subheader("🏆 Top Gainers - Nifty 50")
        
        # Sample gainers data
        gainers_data = {
            'Symbol': ['RELIANCE', 'HDFC', 'INFY', 'TCS', 'ICICIBANK', 'HINDUNILVR', 'KOTAKBANK', 'BAJFINANCE', 'AXISBANK', 'LT'],
            'Price': ['₹2,845', '₹1,675', '₹1,542', '₹3,245', '₹945', '₹2,567', '₹1,876', '₹6,543', '₹987', '₹2,123'],
            'Change %': ['+5.2%', '+4.7%', '+3.8%', '+3.5%', '+3.2%', '+2.9%', '+2.7%', '+2.5%', '+2.3%', '+2.1%'],
            'Change': ['+142', '+76', '+57', '+112', '+30', '+73', '+50', '+162', '+23', '+45']
        }
        
        gainers_df = pd.DataFrame(gainers_data)
        st.dataframe(gainers_df, use_container_width=True, hide_index=True)
        
        st.subheader("📉 Top Losers - Nifty 50")
        
        # Sample losers data
        losers_data = {
            'Symbol': ['ONGC', 'COALINDIA', 'IOC', 'BPCL', 'NTPC', 'POWERGRID', 'TECHM', 'WIPRO', 'ULTRACEMCO', 'GRASIM'],
            'Price': ['₹142', '₹225', '₹87', '₹345', '₹172', '₹198', '₹1,245', '₹387', '₹7,654', '₹1,765'],
            'Change %': ['-3.2%', '-2.8%', '-2.5%', '-2.3%', '-2.1%', '-1.9%', '-1.7%', '-1.5%', '-1.3%', '-1.1%'],
            'Change': ['-4.7', '-6.5', '-2.2', '-8.1', '-3.7', '-3.9', '-21.5', '-5.9', '-98.7', '-19.6']
        }
        
        losers_df = pd.DataFrame(losers_data)
        st.dataframe(losers_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("📊 Sectoral Performance")
        
        # Sample sector data
        sectors = {
            'Sector': ['IT', 'Banking', 'Auto', 'Pharma', 'FMCG', 'Realty', 'Metal', 'Energy', 'Infra', 'Media'],
            'Change %': ['+2.8%', '+1.9%', '+0.7%', '-0.3%', '-0.8%', '-1.2%', '-2.1%', '-2.8%', '+0.5%', '-1.5%']
        }
        
        sector_df = pd.DataFrame(sectors)
        fig = px.bar(
            sector_df, 
            x='Sector', 
            y='Change %', 
            color='Change %',
            color_continuous_scale=["red", "white", "green"]
        )
        fig.update_layout(
            template="plotly_dark", 
            title="Sectoral Performance",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Volume Shockers")
        
        # Sample volume shockers data
        volume_data = {
            'Symbol': ['YESBANK', 'SUZLON', 'IDEA', 'BHEL', 'PNB', 'BANKBARODA', 'CANBK', 'IOB', 'UCOBANK', 'CENTRALBK'],
            'Price': ['₹22.5', '₹18.7', '₹13.2', '₹87.5', '₹62.4', '₹112.3', '₹287.6', '₹32.1', '₹45.6', '₹28.9'],
            'Change %': ['+12.5%', '+9.8%', '+7.3%', '-5.2%', '-3.7%', '+15.2%', '+8.7%', '-6.3%', '+11.2%', '+9.3%'],
            'Volume (Lakhs)': ['2,456', '1,872', '1,543', '1,245', '1,127', '2,123', '1,876', '987', '1,654', '1,321'],
            'Volume Change': ['+345%', '+287%', '+256%', '+198%', '+172%', '+432%', '+321%', '+154%', '+287%', '+234%']
        }
        
        volume_df = pd.DataFrame(volume_data)
        st.dataframe(volume_df, use_container_width=True, hide_index=True)
    
    with tab4:
        st.subheader("📈 All-Time Highs")
        
        # Sample all-time highs data
        ath_data = {
            'Symbol': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'KOTAKBANK', 'BAJFINANCE', 'LT', 'HCLTECH'],
            'Price': ['₹2,845', '₹3,245', '₹1,675', '₹1,542', '₹2,567', '₹945', '₹1,876', '₹6,543', '₹2,123', '₹1,234'],
            'ATH': ['₹2,845', '₹3,245', '₹1,675', '₹1,542', '₹2,567', '₹945', '₹1,876', '₹6,543', '₹2,123', '₹1,234'],
            'Date': ['2023-06-15', '2023-06-15', '2023-06-14', '2023-06-13', '2023-06-12', '2023-06-11', '2023-06-10', '2023-06-09', '2023-06-08', '2023-06-07']
        }
        
        ath_df = pd.DataFrame(ath_data)
        st.dataframe(ath_df, use_container_width=True, hide_index=True)
        
        st.subheader("📉 All-Time Lows")
        
        # Sample all-time lows data
        atl_data = {
            'Symbol': ['YESBANK', 'IDEA', 'SUZLON', 'PNB', 'BHEL', 'IOB', 'UCOBANK', 'CENTRALBK', 'ALOKTEXT', 'JPASSOCIAT'],
            'Price': ['₹22.5', '₹13.2', '₹18.7', '₹62.4', '₹87.5', '₹32.1', '₹45.6', '₹28.9', '₹12.3', '₹8.7'],
            'ATL': ['₹22.5', '₹13.2', '₹18.7', '₹62.4', '₹87.5', '₹32.1', '₹45.6', '₹28.9', '₹12.3', '₹8.7'],
            'Date': ['2023-06-15', '2023-06-15', '2023-06-14', '2023-06-13', '2023-06-12', '2023-06-11', '2023-06-10', '2023-06-09', '2023-06-08', '2023-06-07']
        }
        
        atl_df = pd.DataFrame(atl_data)
        st.dataframe(atl_df, use_container_width=True, hide_index=True)

# F&O Page
elif selected == "F&O":
    st.title("📑 F&O Overview")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Open Interest", "FII/DII Data", "Put/Call Ratio", "Futures Premium", "Option Chain"])
    
    with tab1:
        st.subheader("📊 Open Interest Analysis")
        
        # Sample OI data
        oi_data = {
            'Symbol': ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'TATASTEEL', 'LT', 'AXISBANK'],
            'OI Change': ['+12.5%', '+8.7%', '+15.2%', '-5.3%', '+7.8%', '+3.2%', '-2.1%', '+9.8%', '+4.3%', '-1.2%'],
            'Current OI (Lakhs)': ['245', '187', '156', '132', '98', '87', '76', '65', '54', '43']
        }
        
        oi_df = pd.DataFrame(oi_data)
        st.dataframe(oi_df, use_container_width=True, hide_index=True)
        
        # OI change chart
        fig = px.bar(oi_df, x='Symbol', y='OI Change', color='OI Change', 
                    color_continuous_scale=["red", "white", "green"])
        fig.update_layout(
            template="plotly_dark", 
            title="Open Interest Change",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🏦 FII/DII Activity")
        
        # Sample FII/DII data
        activity_data = {
            'Date': ['2023-06-15', '2023-06-14', '2023-06-13', '2023-06-12', '2023-06-09', '2023-06-08', '2023-06-07', '2023-06-06'],
            'FII (Cr)': ['+1,245', '-876', '+543', '-1,098', '+765', '-432', '+987', '-654'],
            'DII (Cr)': ['-543', '+765', '-321', '+876', '-432', '+654', '-321', '+987']
        }
        
        activity_df = pd.DataFrame(activity_data)
        st.dataframe(activity_df, use_container_width=True, hide_index=True)
        
        # FII/DII trend chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=activity_df['Date'], y=activity_df['FII (Cr)'], name='FII', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=activity_df['Date'], y=activity_df['DII (Cr)'], name='DII', line=dict(color='blue')))
        fig.update_layout(
            template="plotly_dark", 
            title="FII/DII Investment Trend", 
            xaxis_title="Date", 
            yaxis_title="Amount (₹ Cr)",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Put/Call Ratio")
        
        # Sample PCR data
        pcr_data = {
            'Expiry': ['15-Jun-2023', '22-Jun-2023', '29-Jun-2023', '6-Jul-2023', '13-Jul-2023', '20-Jul-2023', '27-Jul-2023'],
            'PCR': ['0.85', '0.92', '1.15', '1.08', '0.97', '1.02', '0.89']
        }
        
        pcr_df = pd.DataFrame(pcr_data)
        st.dataframe(pcr_df, use_container_width=True, hide_index=True)
        
        # PCR chart
        fig = px.line(pcr_df, x='Expiry', y='PCR', title="Put/Call Ratio Trend", markers=True)
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Neutral Level")
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Put/Call Ratio Interpretation:**
        - PCR < 0.7: Oversold (Bullish Signal)
        - PCR 0.7-1.0: Neutral to Slightly Bearish
        - PCR > 1.0: Overbought (Bearish Signal)
        """)
    
    with tab4:
        st.subheader("📊 Futures Premium/Discount")
        
        # Sample futures data
        futures_data = {
            'Symbol': ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'TATASTEEL', 'LT', 'AXISBANK'],
            'Current': ['18,245', '43,187', '2,845', '1,645', '1,542', '945', '587', '112', '2,123', '987'],
            'Future': ['18,265', '43,245', '2,865', '1,655', '1,555', '955', '592', '115', '2,145', '992'],
            'Premium': ['+20', '+58', '+20', '+10', '+13', '+10', '+5', '+3', '+22', '+5'],
            'Premium %': ['+0.11%', '+0.13%', '+0.70%', '+0.61%', '+0.84%', '+1.06%', '+0.85%', '+2.68%', '+1.04%', '+0.51%']
        }
        
        futures_df = pd.DataFrame(futures_data)
        st.dataframe(futures_df, use_container_width=True, hide_index=True)
    
    with tab5:
        st.subheader("📊 Options Chain - NIFTY")
        
        # Sample options chain data
        options_chain = {
            'Strike Price': [18000, 18100, 18200, 18300, 18400, 18500, 18600, 18700, 18800, 18900, 19000],
            'Call OI': [1245, 987, 765, 543, 432, 321, 210, 154, 98, 76, 54],
            'Call Change': [125, 87, 65, 43, 32, 21, 10, 5, 3, 2, 1],
            'Put OI': [54, 76, 98, 154, 210, 321, 432, 543, 765, 987, 1245],
            'Put Change': [1, 2, 3, 5, 10, 21, 32, 43, 65, 87, 125]
        }
        
        options_chain_df = pd.DataFrame(options_chain)
        st.dataframe(options_chain_df, use_container_width=True, hide_index=True)
        
        # Options chain visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(x=options_chain_df['Strike Price'], y=options_chain_df['Call OI'], name='Call OI', marker_color='red'))
        fig.add_trace(go.Bar(x=options_chain_df['Strike Price'], y=options_chain_df['Put OI'], name='Put OI', marker_color='green'))
        
        fig.update_layout(
            title="NIFTY Options Chain",
            xaxis_title="Strike Price",
            yaxis_title="Open Interest",
            barmode='group',
            template="plotly_dark",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#FFFFFF')
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Global Markets
elif selected == "Global Markets":
    st.title("🌍 Global Markets")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Indices", "Commodities", "Currencies", "Global News", "Economic Indicators"])
    
    with tab1:
        st.subheader("📈 Global Indices")
        
        global_data = get_global_indices()
        
        # Display global indices
        cols = st.columns(4)
        idx = 0
        for name, data in global_data.items():
            with cols[idx % 4]:
                st.metric(
                    label=name,
                    value=f"${data['value']:,}" if data['value'] > 100 else f"${data['value']}",
                    delta=f"{data['percent_change']:.2f}%",
                    delta_color="normal" if data["percent_change"] >= 0 else "inverse"
                )
            idx += 1
        
        # Global indices performance chart
        st.subheader("Global Indices Performance")
        indices_list = list(global_data.keys())
        performance_data = [global_data[idx]['percent_change'] for idx in indices_list]
        
        fig = px.bar(
            x=indices_list,
            y=performance_data,
            color=performance_data,
            color_continuous_scale=["red", "white", "green"]
        )
        fig.update_layout(
            template="plotly_dark",
            title="Global Indices Performance (%)",
            xaxis_title="Index",
            yaxis_title="Change (%)",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📦 Commodities")
        
        commodities_data = get_commodities()
        
        # Display commodities
        cols = st.columns(3)
        idx = 0
        for name, data in commodities_data.items():
            with cols[idx % 3]:
                st.metric(
                    label=name,
                    value=f"${data['value']}",
                    delta=f"{data['percent_change']:.2f}%",
                    delta_color="normal" if data["percent_change"] >= 0 else "inverse"
                )
            idx += 1
        
        # Commodities performance chart
        st.subheader("Commodities Performance")
        commodities_list = list(commodities_data.keys())
        commodities_performance = [commodities_data[cmd]['percent_change'] for cmd in commodities_list]
        
        fig = px.bar(
            x=commodities_list,
            y=commodities_performance,
            color=commodities_performance,
            color_continuous_scale=["red", "white", "green"]
        )
        fig.update_layout(
            template="plotly_dark",
            title="Commodities Performance (%)",
            xaxis_title="Commodity",
            yaxis_title="Change (%)",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("💱 Currency Exchange Rates")
        
        currency_pairs = {
            "USDINR=X": "USD/INR",
            "EURINR=X": "EUR/INR",
            "GBPINR=X": "GBP/INR",
            "JPYINR=X": "JPY/INR",
            "CNYINR=X": "CNY/INR",
            "AUDINR=X": "AUD/INR",
            "CADINR=X": "CAD/INR",
            "CHFINR=X": "CHF/INR"
        }
        
        currency_data = get_indices_data(currency_pairs)
        
        col1, col2, col3, col4 = st.columns(4)
        currency_cols = [col1, col2, col3, col4]
        
        for i, (symbol, data) in enumerate(currency_data.items()):
            with currency_cols[i % 4]:
                st.metric(
                    label=symbol,
                    value=f"₹{data['value']:.2f}",
                    delta=f"{data['percent_change']:.2f}%",
                    delta_color="normal" if data["percent_change"] >= 0 else "inverse"
                )
        
        # Currency trends
        st.subheader("Currency Trends (1 Month)")
        
        # Sample currency trend data
        dates = pd.date_range(end=datetime.today(), periods=30, freq='D')
        usd_inr = [82.0 + 0.1 * np.sin(i/3) + 0.05 * np.random.randn() for i in range(30)]
        eur_inr = [88.0 + 0.15 * np.sin(i/2.5) + 0.06 * np.random.randn() for i in range(30)]
        gbp_inr = [102.0 + 0.2 * np.sin(i/2) + 0.07 * np.random.randn() for i in range(30)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=usd_inr, name='USD/INR', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=dates, y=eur_inr, name='EUR/INR', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=dates, y=gbp_inr, name='GBP/INR', line=dict(color='orange')))
        fig.update_layout(
            template="plotly_dark",
            title="Currency Trends (1 Month)",
            xaxis_title="Date",
            yaxis_title="Exchange Rate (₹)",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🌐 Global Financial News")
        
        news_articles = get_news("global economy", 5)
        
        if news_articles:
            for i, article in enumerate(news_articles):
                st.markdown('<div class="news-card">', unsafe_allow_html=True)
                st.write(f"**{article['title']}**")
                st.write(f"*Source: {article['source']['name']}*")
                st.write(article['description'])
                st.markdown(f"[Read more]({article['url']})")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Could not fetch global news at the moment.")
    
    with tab5:
        st.subheader("📊 Global Economic Indicators")
        
        # Sample economic indicators
        economic_data = {
            'Indicator': ['US GDP Growth', 'US Unemployment Rate', 'EU Inflation', 'China PMI', 'Japan Industrial Production', 'UK Retail Sales'],
            'Current': ['2.1%', '3.6%', '6.1%', '52.4', '3.2%', '1.8%'],
            'Previous': ['2.2%', '3.5%', '6.3%', '51.8', '2.8%', '1.5%'],
            'Change': ['-0.1%', '+0.1%', '-0.2%', '+0.6', '+0.4%', '+0.3%']
        }
        
        economic_df = pd.DataFrame(economic_data)
        st.dataframe(economic_df, use_container_width=True, hide_index=True)
        
        # Economic indicators heatmap
        fig = px.imshow([[-0.1, 0.1, -0.2, 0.6, 0.4, 0.3]], 
                       labels=dict(x="Economic Indicators", y="", color="Change"),
                       x=economic_df['Indicator'],
                       color_continuous_scale=["red", "white", "green"])
        fig.update_layout(
            title="Economic Indicators Change",
            template="plotly_dark",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)

# Continue with other sections...

# Footer with developer info
st.markdown("---")
st.markdown("### About MarketMentor")
st.markdown("MarketMentor is a comprehensive financial analysis platform designed to help investors make informed decisions.")
st.markdown(f"Developed by **{developer_info['name']}** - {developer_info['role']}")
st.markdown(f"Connect on [LinkedIn]({developer_info['linkedin']})")
