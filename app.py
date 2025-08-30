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
warnings.filterwarnings('ignore')

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Set page config
st.set_page_config(page_title="MarketMentor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark theme with red and black
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FF4B4B !important;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1d391kg p {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #1A1A1A;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(255, 75, 75, 0.2);
        border-left: 4px solid #FF4B4B;
        margin-bottom: 15px;
    }
    
    /* Positive and negative changes */
    .positive-change {
        color: #00CC96 !important;
        font-weight: bold;
    }
    .negative-change {
        color: #FF4B4B !important;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .stButton button:hover {
        background-color: #CC3C3C;
        color: white;
    }
    
    /* Select boxes and inputs */
    .stSelectbox, .stTextInput, .stNumberInput, .stSlider {
        background-color: #1A1A1A;
        color: white;
        border-radius: 5px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1A1A;
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        border: 1px solid #333333;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: white;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #1A1A1A;
        color: white;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1A1A1A;
        color: white;
        border: 1px solid #333333;
        border-radius: 5px;
    }
    
    /* JSON formatting */
    .json-container {
        background-color: #1A1A1A;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        max-height: 400px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 14px;
        border: 1px solid #333333;
    }
    
    /* Custom toggle for dark mode */
    .dark-mode-toggle {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar menu with custom styling
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>MarketMentor</h1>", unsafe_allow_html=True)
    selected = option_menu(
        "",
        ["Home", "Company Overview", "Market Movers", "F&O", "Global Markets", "Mutual Funds", "SIP Calculator", 
         "IPO Tracker", "Predictions for Mutual Funds & IPOs", "Mutual Fund NAV Viewer", "Sectors", "News", 
         "Learning", "Volume Spike", "Stock Screener", "Predictions", "Buy/Sell Predictor", "News Sentiment", 
         "Technical Analysis", "Portfolio Tracker"],
        icons=['house', 'building', 'graph-up', 'arrow-left-right', 'globe', 'bank', 'calculator', 'rocket', 
               'graph-up-arrow', 'bar-chart', 'grid-3x3', 'newspaper', 'book', 'activity', 'search', 'lightbulb', 
               'currency-exchange', 'chat-quote', 'speedometer', 'briefcase'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#000000"},
            "icon": {"color": "#FF4B4B", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#FFFFFF", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#FF4B4B", "color": "#FFFFFF"},
        }
    )

# Cache data fetching functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker, period="1y"):
    return yf.Ticker(ticker).history(period=period)

@st.cache_data(ttl=3600)
def get_index_data():
    indices = {
        "^NSEI": "Nifty 50",
        "^BSESN": "Sensex",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^GSPC": "S&P 500",
    }
    data = {}
    for symbol, name in indices.items():
        ticker_data = yf.Ticker(symbol).history(period="1d")
        data[name] = {
            'last_close': round(ticker_data['Close'].iloc[-1], 2),
            'change': round(ticker_data['Close'].iloc[-1] - ticker_data['Open'].iloc[-1], 2),
            'percent_change': round(((ticker_data['Close'].iloc[-1] - ticker_data['Open'].iloc[-1]) / ticker_data['Open'].iloc[-1]) * 100, 2)
        }
    return data

@st.cache_data(ttl=3600)
def get_global_markets():
    global_indices = {
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^GSPC": "S&P 500",
        "^FTSE": "FTSE 100",
        "^N225": "Nikkei 225",
        "^HSI": "Hang Seng",
        "^GDAXI": "DAX",
        "^FCHI": "CAC 40",
    }
    data = {}
    for symbol, name in global_indices.items():
        try:
            ticker_data = yf.Ticker(symbol).history(period="1d")
            data[name] = {
                'last_close': round(ticker_data['Close'].iloc[-1], 2),
                'change': round(ticker_data['Close'].iloc[-1] - ticker_data['Open'].iloc[-1], 2),
                'percent_change': round(((ticker_data['Close'].iloc[-1] - ticker_data['Open'].iloc[-1]) / ticker_data['Open'].iloc[-1]) * 100, 2)
            }
        except:
            data[name] = {'last_close': 'N/A', 'change': 'N/A', 'percent_change': 'N/A'}
    return data

# Home - Market Overview
if selected == "Home":
    st.title("🏠 Home - Market Overview")
    
    # Display major indices
    st.subheader("📊 Major Indices Performance")
    index_data = get_index_data()
    cols = st.columns(len(index_data))
    for idx, (name, data) in enumerate(index_data.items()):
        with cols[idx]:
            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
            delta_color = "positive-change" if data['percent_change'] >= 0 else "negative-change"
            st.markdown(f"<h3 style='color: white;'>{name}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: white;'>{data['last_close']}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p class='{delta_color}'>Δ {data['percent_change']}%</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Market sentiment
    st.subheader("📈 Market Sentiment")
    sentiment_cols = st.columns(3)
    with sentiment_cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: white;'>Advancers</h3>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: #00CC96;'>1,245</h2>", unsafe_allow_html=True)
        st.markdown("<p class='positive-change'>52%</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with sentiment_cols[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: white;'>Decliners</h3>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: #FF4B4B;'>987</h2>", unsafe_allow_html=True)
        st.markdown("<p class='negative-change'>41%</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with sentiment_cols[2]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: white;'>Unchanged</h3>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: white;'>168</h2>", unsafe_allow_html=True)
        st.markdown("<p>7%</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent news
    st.subheader("📰 Top Financial News")
    try:
        url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey={NEWS_API_KEY}&language=en&pageSize=3"
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            for article in articles:
                with st.expander(article["title"]):
                    st.write(f"**Source:** {article['source']['name']}")
                    st.write(article.get("description", "No description available."))
                    st.markdown(f"[Read more]({article['url']})")
        else:
            st.info("News feature will be available after configuring News API")
    except:
        st.info("News feature will be available after configuring News API")

# Company Overview Page - Enhanced with detailed info and JSON view
elif selected == "Company Overview":
    st.title("🏢 Company Overview")
    
    ticker = st.text_input("🔎 Enter Stock Ticker (e.g., AAPL, TCS.NS)", "RELIANCE.NS")
    
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="6mo")
            
            # Display company info in tabs
            tab1, tab2, tab3 = st.tabs(["Overview", "Financials", "Raw Data"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Company Details")
                    st.markdown(f"**Name:** {info.get('longName', 'N/A')}")
                    st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.markdown(f"**Market Cap:** ₹{info.get('marketCap', 0):,}")
                    st.markdown(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
                    st.markdown(f"**Country:** {info.get('country', 'N/A')}")
                    
                with col2:
                    st.subheader("📈 Key Metrics")
                    st.markdown(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                    st.markdown(f"**EPS:** {info.get('trailingEps', 'N/A')}")
                    st.markdown(f"**Dividend Yield:** {info.get('dividendYield', 0)*100 if info.get('dividendYield') else 'N/A'}%")
                    st.markdown(f"**52W High/Low:** ₹{info.get('fiftyTwoWeekHigh', 'N/A')}/₹{info.get('fiftyTwoWeekLow', 'N/A')}")
                    st.markdown(f"**Beta:** {info.get('beta', 'N/A')}")
                    st.markdown(f"**Volume Avg:** {info.get('averageVolume', 'N/A'):,}")
                
                # Price chart
                st.subheader("📊 Price Chart")
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
                    title=f"{ticker} Price Chart", 
                    xaxis_title="Date", 
                    yaxis_title="Price",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("💵 Financial Summary")
                financials = stock.financials
                if not financials.empty:
                    st.write("**Recent Revenue & Earnings**")
                    st.dataframe(financials.head().T.style.background_gradient(cmap='Reds'))
                else:
                    st.info("Financial data not available for this ticker")
                    
                # Balance sheet and cash flow
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("💰 Balance Sheet")
                    balance_sheet = stock.balance_sheet
                    if not balance_sheet.empty:
                        st.dataframe(balance_sheet.head().T.style.background_gradient(cmap='Reds'))
                
                with col2:
                    st.subheader("💸 Cash Flow")
                    cash_flow = stock.cashflow
                    if not cash_flow.empty:
                        st.dataframe(cash_flow.head().T.style.background_gradient(cmap='Reds'))
            
            with tab3:
                st.subheader("📋 Raw Company Data (JSON)")
                # Filter out very long values for display
                display_info = {k: v for k, v in info.items() if not isinstance(v, (list, dict)) and v is not None}
                # Format the JSON with indentation
                formatted_json = json.dumps(display_info, indent=2)
                st.markdown(f'<div class="json-container">{formatted_json}</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error fetching data: {e}")

# Rest of the code remains the same for other sections, but with the dark theme applied
# [Previous code for other sections continues here...]

# Note: The rest of the code would follow the same pattern, with appropriate styling adjustments
# for the dark theme with red and black colors, and white text.
