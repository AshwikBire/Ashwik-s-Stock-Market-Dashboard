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

warnings.filterwarnings('ignore')

# Set page config with dark theme
st.set_page_config(
    page_title="MarketMentor", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Apply dark theme CSS
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #1F77B4;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    .stSelectbox>div>div>select {
        background-color: #262730;
        color: white;
    }
    .stSlider>div>div>div>div {
        background-color: #1F77B4;
    }
    .stMetric {
        background-color: #1C1C25;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .css-1d391kg {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #1C1C25;
    }
    .stProgress > div > div > div > div {
        background-color: #1F77B4;
    }
    .stAlert {
        background-color: #1C1C25;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1C1C25;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1F77B4;
    }
    div[data-testid="stSidebarUserContent"] {
        background-color: #0E1117;
    }
    .sidebar .sidebar-content {
        background-color: #0E1117;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1F77B4;
    }
    .stDataFrame {
        background-color: #1C1C25;
    }
    header {
        background-color: #0E1117;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #1C1C25;
    }
    .st-bh {
        background-color: transparent;
    }
    .st-ag {
        background-color: #0E1117;
    }
    .st-af {
        background-color: #0E1117;
    }
    .st-ae {
        background-color: #0E1117;
    }
</style>
""", unsafe_allow_html=True)

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"
FRED_API_KEY = "6c5b447e2f6e497d0f4d9b4b3b4f4e4a"  # You might need to get your own key

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
                "Technical Analysis", "Portfolio Tracker", "Economic Calendar"],
        icons=['house', 'building', 'graph-up', 'activity', 'globe', 
               'bank', 'calculator', 'rocket', 'lightning', 
               'bar-chart', 'grid', 'newspaper', 'book', 'activity', 
               'search', 'graph-up-arrow', 'currency-exchange', 'chat', 
               'graph-up', 'wallet', 'calendar'],
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#0E1117"},
            "icon": {"color": "#1F77B4", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#FAFAFA"},
            "nav-link-selected": {"background-color": "#1F77B4"},
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
    tab1, tab2, tab3, tab4 = st.tabs(["Quick Stock Search", "Sector Performance", "Market Summary", "Watchlist"])
    
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
                    template="plotly_dark"
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
        sectors = {
            "NIFTY IT": "^CNXIT",
            "NIFTY BANK": "^NSEBANK",
            "NIFTY AUTO": "^CNXAUTO",
            "NIFTY FMCG": "^CNXFMCG",
            "NIFTY PHARMA": "^CNXPHARMA",
            "NIFTY REALTY": "^CNXREALTY"
        }
        
        sector_data = {}
        for name, symbol in sectors.items():
            try:
                data = yf.Ticker(symbol).history(period="1d")
                if not data.empty:
                    last_close = data['Close'].iloc[-1]
                    prev_close = data['Close'].iloc[-2] if len(data) > 1 else last_close
                    change = ((last_close - prev_close) / prev_close) * 100
                    sector_data[name] = change
            except:
                continue
        
        # Create sector performance chart
        if sector_data:
            fig = px.bar(
                x=list(sector_data.keys()),
                y=list(sector_data.values()),
                title="Sector Performance (%)",
                color=list(sector_data.values()),
                color_continuous_scale=["red", "white", "green"]
            )
            fig.update_layout(template="plotly_dark", xaxis_title="Sector", yaxis_title="Change (%)")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Market Summary")
        col1, col2 = st.columns(2)
        
        with col1:
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
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.info("**Volume Analysis**")
            volume_data = {
                "Nifty 50": "₹45,672 Cr",
                "Bank Nifty": "₹32,145 Cr",
                "Top Gainer": "RELIANCE (₹2,145 Cr)",
                "Top Loser": "HDFC Bank (₹1,876 Cr)"
            }
            for key, value in volume_data.items():
                st.write(f"{key}: {value}")
    
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
                col1, col2, col3 = st.columns([2, 1, 1])
                col1.write(item["Stock"])
                col2.write(item["Price"])
                col3.write(f":{item['Color']}[{item['Change']}]")

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
                        st.metric(key, value)
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Financials", "Holdings", "Analysis"])
            
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
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add technical indicators
                    st.subheader("Technical Indicators")
                    col1, col2, col3 = st.columns(3)
                    
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
                        st.metric(key, value)
                
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
                    template="plotly_dark"
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
                fig.update_layout(template="plotly_dark")
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
                fig.update_layout(template="plotly_dark", xaxis_title="Recommendation", yaxis_title="Number of Analysts")
                st.plotly_chart(fig, use_container_width=True)
                
                # Price targets
                st.subheader("Price Targets")
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Target", "₹2,850")
                col2.metric("High Target", "₹3,200")
                col3.metric("Low Target", "₹2,500")
            
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
    
    tab1, tab2, tab3 = st.tabs(["Nifty 50", "Sectoral Indices", "Volume Shockers"])
    
    with tab1:
        st.subheader("🏆 Top Gainers - Nifty 50")
        
        # Sample gainers data
        gainers_data = {
            'Symbol': ['RELIANCE', 'HDFC', 'INFY', 'TCS', 'ICICIBANK'],
            'Price': ['₹2,845', '₹1,675', '₹1,542', '₹3,245', '₹945'],
            'Change %': ['+5.2%', '+4.7%', '+3.8%', '+3.5%', '+3.2%'],
            'Change': ['+142', '+76', '+57', '+112', '+30']
        }
        
        gainers_df = pd.DataFrame(gainers_data)
        st.dataframe(gainers_df, use_container_width=True, hide_index=True)
        
        st.subheader("📉 Top Losers - Nifty 50")
        
        # Sample losers data
        losers_data = {
            'Symbol': ['ONGC', 'COALINDIA', 'IOC', 'BPCL', 'NTPC'],
            'Price': ['₹142', '₹225', '₹87', '₹345', '₹172'],
            'Change %': ['-3.2%', '-2.8%', '-2.5%', '-2.3%', '-2.1%'],
            'Change': ['-4.7', '-6.5', '-2.2', '-8.1', '-3.7']
        }
        
        losers_df = pd.DataFrame(losers_data)
        st.dataframe(losers_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("📊 Sectoral Performance")
        
        # Sample sector data
        sectors = {
            'Sector': ['IT', 'Banking', 'Auto', 'Pharma', 'FMCG', 'Realty', 'Metal', 'Energy'],
            'Change %': ['+2.8%', '+1.9%', '+0.7%', '-0.3%', '-0.8%', '-1.2%', '-2.1%', '-2.8%']
        }
        
        sector_df = pd.DataFrame(sectors)
        fig = px.bar(
            sector_df, 
            x='Sector', 
            y='Change %', 
            color='Change %',
            color_continuous_scale=["red", "white", "green"]
        )
        fig.update_layout(template="plotly_dark", title="Sectoral Performance")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Volume Shockers")
        
        # Sample volume shockers data
        volume_data = {
            'Symbol': ['YESBANK', 'SUZLON', 'IDEA', 'BHEL', 'PNB'],
            'Price': ['₹22.5', '₹18.7', '₹13.2', '₹87.5', '₹62.4'],
            'Change %': ['+12.5%', '+9.8%', '+7.3%', '-5.2%', '-3.7%'],
            'Volume (Lakhs)': ['2,456', '1,872', '1,543', '1,245', '1,127'],
            'Volume Change': ['+345%', '+287%', '+256%', '+198%', '+172%']
        }
        
        volume_df = pd.DataFrame(volume_data)
        st.dataframe(volume_df, use_container_width=True, hide_index=True)

# F&O Page
elif selected == "F&O":
    st.title("📑 F&O Overview")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Open Interest", "FII/DII Data", "Put/Call Ratio", "Futures Premium"])
    
    with tab1:
        st.subheader("📊 Open Interest Analysis")
        
        # Sample OI data
        oi_data = {
            'Symbol': ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'HDFCBANK', 'INFY'],
            'OI Change': ['+12.5%', '+8.7%', '+15.2%', '-5.3%', '+7.8%'],
            'Current OI (Lakhs)': ['245', '187', '156', '132', '98']
        }
        
        oi_df = pd.DataFrame(oi_data)
        st.dataframe(oi_df, use_container_width=True, hide_index=True)
        
        # OI change chart
        fig = px.bar(oi_df, x='Symbol', y='OI Change', color='OI Change', 
                    color_continuous_scale=["red", "white", "green"])
        fig.update_layout(template="plotly_dark", title="Open Interest Change")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🏦 FII/DII Activity")
        
        # Sample FII/DII data
        activity_data = {
            'Date': ['2023-06-15', '2023-06-14', '2023-06-13', '2023-06-12', '2023-06-09'],
            'FII (Cr)': ['+1,245', '-876', '+543', '-1,098', '+765'],
            'DII (Cr)': ['-543', '+765', '-321', '+876', '-432']
        }
        
        activity_df = pd.DataFrame(activity_data)
        st.dataframe(activity_df, use_container_width=True, hide_index=True)
        
        # FII/DII trend chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=activity_df['Date'], y=activity_df['FII (Cr)'], name='FII', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=activity_df['Date'], y=activity_df['DII (Cr)'], name='DII', line=dict(color='blue')))
        fig.update_layout(template="plotly_dark", title="FII/DII Investment Trend", xaxis_title="Date", yaxis_title="Amount (₹ Cr)")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Put/Call Ratio")
        
        # Sample PCR data
        pcr_data = {
            'Expiry': ['15-Jun-2023', '22-Jun-2023', '29-Jun-2023', '6-Jul-2023'],
            'PCR': ['0.85', '0.92', '1.15', '1.08']
        }
        
        pcr_df = pd.DataFrame(pcr_data)
        st.dataframe(pcr_df, use_container_width=True, hide_index=True)
        
        # PCR chart
        fig = px.line(pcr_df, x='Expiry', y='PCR', title="Put/Call Ratio Trend", markers=True)
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Neutral Level")
        fig.update_layout(template="plotly_dark")
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
            'Symbol': ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'HDFCBANK', 'INFY'],
            'Current': ['18,245', '43,187', '2,845', '1,645', '1,542'],
            'Future': ['18,265', '43,245', '2,865', '1,655', '1,555'],
            'Premium': ['+20', '+58', '+20', '+10', '+13'],
            'Premium %': ['+0.11%', '+0.13%', '+0.70%', '+0.61%', '+0.84%']
        }
        
        futures_df = pd.DataFrame(futures_data)
        st.dataframe(futures_df, use_container_width=True, hide_index=True)

# Global Markets
elif selected == "Global Markets":
    st.title("🌍 Global Markets")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Indices", "Commodities", "Currencies", "Global News"])
    
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
            yaxis_title="Change (%)"
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
            yaxis_title="Change (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("💱 Currency Exchange Rates")
        
        currency_pairs = {
            "USDINR=X": "USD/INR",
            "EURINR=X": "EUR/INR",
            "GBPINR=X": "GBP/INR",
            "JPYINR=X": "JPY/INR",
            "CNYINR=X": "CNY/INR"
        }
        
        currency_data = get_indices_data(currency_pairs)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        currency_cols = [col1, col2, col3, col4, col5]
        
        for i, (symbol, data) in enumerate(currency_data.items()):
            with currency_cols[i]:
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
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=usd_inr, name='USD/INR', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=dates, y=eur_inr, name='EUR/INR', line=dict(color='blue')))
        fig.update_layout(
            template="plotly_dark",
            title="Currency Trends (1 Month)",
            xaxis_title="Date",
            yaxis_title="Exchange Rate (₹)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🌐 Global Financial News")
        
        news_articles = get_news("global economy", 5)
        
        if news_articles:
            for i, article in enumerate(news_articles):
                with st.expander(f"{article['title']}"):
                    st.write(f"**Source:** {article['source']['name']}")
                    st.write(f"**Published At:** {article['publishedAt']}")
                    st.write(article['description'])
                    st.markdown(f"[Read more]({article['url']})")
        else:
            st.info("Could not fetch global news at the moment.")

# Continue with other sections...

# Footer with developer info
st.markdown("---")
st.markdown("### About MarketMentor")
st.markdown("MarketMentor is a comprehensive financial analysis platform designed to help investors make informed decisions.")
st.markdown(f"Developed by **{developer_info['name']}** - {developer_info['role']}")
st.markdown(f"Connect on [LinkedIn]({developer_info['linkedin']})")
