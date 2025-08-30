import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
from plotly import graph_objects as go
from streamlit_option_menu import option_menu
from textblob import TextBlob
from xgboost import XGBRegressor
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
import json
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Set page config with dark theme
st.set_page_config(
    page_title="MarketMentor",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for dark theme with red accents
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 75, 75, 0.3);
    }
    .stSelectbox>div>div>select {
        background-color: #262730;
        color: white;
        border: 1px solid #444;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
        border: 1px solid #444;
    }
    .stSlider>div>div>div>div {
        background-color: #FF4B4B;
    }
    .stProgress>div>div>div {
        background-color: #FF4B4B;
    }
    .stMetric {
        background-color: #262730;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stAlert {
        background-color: #262730;
        border-left: 4px solid #FF4B4B;
    }
    .css-1d391kg {
        background-color: #0E1117;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #FAFAFA;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: #0E1117;
        font-weight: bold;
    }
    .stDataFrame {
        background-color: #262730;
    }
    .stTable {
        background-color: #262730;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    header {
        background-color: #0E1117;
    }
    .card {
        background-color: #262730;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .card h3 {
        margin-top: 0;
        color: #FF4B4B;
    }
    .positive {
        color: #00C853;
    }
    .negative {
        color: #FF4B4B;
    }
    .neutral {
        color: #FFD700;
    }
    .company-header {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    .company-header img {
        width: 60px;
        height: 60px;
        margin-right: 1rem;
        border-radius: 8px;
    }
    .company-header h1 {
        margin-bottom: 0;
    }
    .financial-metric {
        text-align: center;
        padding: 1rem;
        background-color: #1E2229;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .financial-metric h4 {
        margin: 0 0 0.5rem 0;
        color: #FF4B4B;
        font-size: 0.9rem;
    }
    .financial-metric p {
        margin: 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #0E1117;
    }
</style>
""", unsafe_allow_html=True)

# Load company data from JSON file
@st.cache_data
def load_company_data():
    # Check if JSON file exists, if not create it with sample data
    json_file = Path("company_data.json")
    
    if not json_file.exists():
        sample_data = {
            "AAPL": {
                "name": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "employees": 154000,
                "founded": 1976,
                "ceo": "Tim Cook",
                "headquarters": "Cupertino, California",
                "website": "https://www.apple.com",
                "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
                "financials": {
                    "market_cap": "2.5T",
                    "revenue": "394.3B",
                    "net_income": "99.8B",
                    "pe_ratio": 28.5,
                    "dividend_yield": 0.5,
                    "eps": 6.13,
                    "beta": 1.21,
                    "52_week_high": 197.89,
                    "52_week_low": 124.17
                },
                "key_statistics": {
                    "shares_outstanding": "15.9B",
                    "float": "15.8B",
                    "insider_ownership": "0.07%",
                    "institutional_ownership": "59.2%"
                }
            },
            "MSFT": {
                "name": "Microsoft Corporation",
                "sector": "Technology",
                "industry": "Software—Infrastructure",
                "employees": 221000,
                "founded": 1975,
                "ceo": "Satya Nadella",
                "headquarters": "Redmond, Washington",
                "website": "https://www.microsoft.com",
                "description": "Microsoft Corporation develops, licenses, and supports software, services, devices, and solutions worldwide.",
                "financials": {
                    "market_cap": "2.3T",
                    "revenue": "211.9B",
                    "net_income": "72.4B",
                    "pe_ratio": 31.2,
                    "dividend_yield": 0.8,
                    "eps": 9.72,
                    "beta": 0.89,
                    "52_week_high": 366.78,
                    "52_week_low": 213.43
                },
                "key_statistics": {
                    "shares_outstanding": "7.43B",
                    "float": "7.41B",
                    "insider_ownership": "0.12%",
                    "institutional_ownership": "72.5%"
                }
            },
            "GOOGL": {
                "name": "Alphabet Inc.",
                "sector": "Communication Services",
                "industry": "Internet Content & Information",
                "employees": 156500,
                "founded": 1998,
                "ceo": "Sundar Pichai",
                "headquarters": "Mountain View, California",
                "website": "https://www.abc.xyz",
                "description": "Alphabet Inc. provides online advertising services, cloud computing, software, and hardware.",
                "financials": {
                    "market_cap": "1.7T",
                    "revenue": "307.4B",
                    "net_income": "76.0B",
                    "pe_ratio": 24.8,
                    "dividend_yield": 0.0,
                    "eps": 5.80,
                    "beta": 1.06,
                    "52_week_high": 153.78,
                    "52_week_low": 83.34
                },
                "key_statistics": {
                    "shares_outstanding": "12.5B",
                    "float": "12.4B",
                    "insider_ownership": "0.15%",
                    "institutional_ownership": "68.3%"
                }
            },
            "AMZN": {
                "name": "Amazon.com Inc.",
                "sector": "Consumer Cyclical",
                "industry": "Internet Retail",
                "employees": 1541000,
                "founded": 1994,
                "ceo": "Andy Jassy",
                "headquarters": "Seattle, Washington",
                "website": "https://www.amazon.com",
                "description": "Amazon.com Inc. engages in the retail sale of consumer products and subscriptions through online and physical stores.",
                "financials": {
                    "market_cap": "1.5T",
                    "revenue": "574.8B",
                    "net_income": "3.0B",
                    "pe_ratio": 85.4,
                    "dividend_yield": 0.0,
                    "eps": 0.28,
                    "beta": 1.14,
                    "52_week_high": 145.86,
                    "52_week_low": 81.43
                },
                "key_statistics": {
                    "shares_outstanding": "10.2B",
                    "float": "10.1B",
                    "insider_ownership": "0.21%",
                    "institutional_ownership": "61.7%"
                }
            },
            "TSLA": {
                "name": "Tesla, Inc.",
                "sector": "Automotive",
                "industry": "Auto Manufacturers",
                "employees": 127855,
                "founded": 2003,
                "ceo": "Elon Musk",
                "headquarters": "Austin, Texas",
                "website": "https://www.tesla.com",
                "description": "Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, energy generation and storage systems.",
                "financials": {
                    "market_cap": "750.0B",
                    "revenue": "96.8B",
                    "net_income": "15.0B",
                    "pe_ratio": 65.2,
                    "dividend_yield": 0.0,
                    "eps": 4.30,
                    "beta": 2.01,
                    "52_week_high": 299.29,
                    "52_week_low": 101.81
                },
                "key_statistics": {
                    "shares_outstanding": "3.18B",
                    "float": "3.16B",
                    "insider_ownership": "0.43%",
                    "institutional_ownership": "44.8%"
                }
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(sample_data, f, indent=4)
        
        return sample_data
    
    else:
        with open(json_file, 'r') as f:
            return json.load(f)

# Cache stock data download
@st.cache_data
def load_stock_data(symbol, period="1y"):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    return hist

# Fetch news for a company
@st.cache_data
def fetch_news(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if data['status'] == 'ok':
            return data['articles'][:5]  # Return top 5 articles
        else:
            return []
    except:
        return []

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate MACD
def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

# Sidebar menu
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-share.png", width=80)
    st.title("MarketMentor")
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Company Overview", "Market Movers", "Global Markets", 
                "Mutual Funds", "Sectors", "News", "Learning", "Volume Spike", 
                "News Sentiment", "Predictions", "Buy/Sell Predictor", "Stock Screener",
                "F&O", "SIP Calculator", "IPO Tracker", "Watchlist", "Portfolio Tracker"],
        icons=["house", "building", "graph-up", "globe", "piggy-bank", "grid-3x3", 
               "newspaper", "book", "activity", "bar-chart", "lightbulb", "arrow-left-right",
               "search", "graph-up-arrow", "calculator", "megaphone", "star", "wallet"],
        menu_icon="cast",
        default_index=0
    )

# Load company data
company_data = load_company_data()

# Home - Market Overview
if selected == "Home":
    st.title("🏠 Home - Market Overview")
    
    # Market summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("S&P 500", "4,891.23", "0.45%")
    with col2:
        st.metric("NASDAQ", "15,628.04", "0.75%")
    with col3:
        st.metric("Dow Jones", "38,654.42", "0.15%")
    with col4:
        st.metric("Russell 2000", "1,978.23", "-0.25%")
    
    # Top gainers and losers
    st.subheader("📈 Top Gainers")
    gainers_data = {
        "Symbol": ["NVDA", "META", "TSLA", "AMD", "SHOP"],
        "Name": ["NVIDIA Corp", "Meta Platforms", "Tesla Inc", "Advanced Micro Devices", "Shopify Inc"],
        "Price": ["$610.32", "$468.25", "$245.67", "$178.90", "$82.45"],
        "Change": ["+5.2%", "+4.8%", "+4.1%", "+3.9%", "+3.5%"]
    }
    gainers_df = pd.DataFrame(gainers_data)
    st.dataframe(gainers_df, use_container_width=True, hide_index=True)
    
    st.subheader("📉 Top Losers")
    losers_data = {
        "Symbol": ["PFE", "DASH", "CVNA", "RIVN", "LCID"],
        "Name": ["Pfizer Inc", "DoorDash Inc", "Carvana Co", "Rivian Automotive", "Lucid Group Inc"],
        "Price": ["$27.45", "$102.36", "$56.78", "$16.23", "$3.45"],
        "Change": ["-3.8%", "-3.2%", "-2.9%", "-2.7%", "-2.5%"]
    }
    losers_df = pd.DataFrame(losers_data)
    st.dataframe(losers_df, use_container_width=True, hide_index=True)
    
    # Market heatmap
    st.subheader("🌡️ Sector Performance Heatmap")
    sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer Cyclical', 'Energy', 
               'Industrials', 'Communication', 'Utilities', 'Real Estate']
    performance = [2.5, 1.2, -0.3, 1.8, -1.5, 0.7, 1.1, -0.8, 0.3]
    
    fig, ax = plt.subfaces(figsize=(10, 6))
    colors = ['#FF4B4B' if x < 0 else '#00C853' for x in performance]
    ax.barh(sectors, performance, color=colors)
    ax.set_xlabel('Percentage Change (%)')
    ax.set_title('Sector Performance')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    st.pyplot(fig)

# Company Overview - Detailed stock analysis
elif selected == "Company Overview":
    st.title("🏢 Company Overview")
    
    # Stock selector
    symbols = list(company_data.keys())
    selected_symbol = st.selectbox("Select a stock symbol", symbols)
    
    if selected_symbol:
        company_info = company_data[selected_symbol]
        stock_data = load_stock_data(selected_symbol, "1y")
        
        # Company header with logo and basic info
        st.markdown(f"""
        <div class="company-header">
            <img src="https://logo.clearbit.com/{company_info['website'].replace('https://www.', '')}?size=120" 
                 onerror="this.src='https://via.placeholder.com/120x120/262730/FF4B4B?text={selected_symbol}'">
            <div>
                <h1>{company_info['name']} ({selected_symbol})</h1>
                <p>{company_info['sector']} • {company_info['industry']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Financial metrics
        st.subheader("📊 Financial Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="financial-metric">
                <h4>Market Cap</h4>
                <p>{company_info['financials']['market_cap']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="financial-metric">
                <h4>P/E Ratio</h4>
                <p>{company_info['financials']['pe_ratio']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="financial-metric">
                <h4>Revenue</h4>
                <p>{company_info['financials']['revenue']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="financial-metric">
                <h4>Net Income</h4>
                <p>{company_info['financials']['net_income']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="financial-metric">
                <h4>Dividend Yield</h4>
                <p>{company_info['financials']['dividend_yield']}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional financial metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="financial-metric">
                <h4>EPS</h4>
                <p>{company_info['financials']['eps']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="financial-metric">
                <h4>Beta</h4>
                <p>{company_info['financials']['beta']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="financial-metric">
                <h4>52W High</h4>
                <p>${company_info['financials']['52_week_high']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="financial-metric">
                <h4>52W Low</h4>
                <p>${company_info['financials']['52_week_low']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Company information tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price Chart", "Company Info", "Financials", "Key Statistics", "News"])
        
        with tab1:
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Price'
            ))
            fig.update_layout(
                title=f"{selected_symbol} Stock Price",
                yaxis_title="Price (USD)",
                xaxis_title="Date",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators
            st.subheader("📈 Technical Indicators")
            
            # Calculate RSI
            stock_data['RSI'] = calculate_rsi(stock_data)
            
            # Calculate MACD
            macd, signal, histogram = calculate_macd(stock_data)
            
            # Create subplots
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price', 'MACD', 'RSI'),
                row_width=[0.2, 0.2, 0.6]
            )
            
            # Price
            fig.add_trace(go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Price'
            ), row=1, col=1)
            
            # MACD
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=macd,
                name='MACD',
                line=dict(color='#FF4B4B')
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=signal,
                name='Signal',
                line=dict(color='#00C853')
            ), row=2, col=1)
            
            fig.add_trace(go.Bar(
                x=stock_data.index,
                y=histogram,
                name='Histogram',
                marker_color=np.where(histogram < 0, '#FF4B4B', '#00C853')
            ), row=2, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['RSI'],
                name='RSI',
                line=dict(color='#FFD700')
            ), row=3, col=1)
            
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="#FF4B4B", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#00C853", row=3, col=1)
            
            fig.update_layout(
                height=800,
                showlegend=True,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Company information
            st.subheader("🏢 Company Details")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="card">
                    <h3>Company Information</h3>
                    <p><strong>CEO:</strong> {company_info['ceo']}</p>
                    <p><strong>Founded:</strong> {company_info['founded']}</p>
                    <p><strong>Employees:</strong> {company_info['employees']:,}</p>
                    <p><strong>Headquarters:</strong> {company_info['headquarters']}</p>
                    <p><strong>Website:</strong> <a href="{company_info['website']}" target="_blank">{company_info['website']}</a></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="card">
                    <h3>Business Description</h3>
                    <p>{company_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            # Financial statements (simulated)
            st.subheader("💵 Financial Statements")
            
            # Income statement
            st.markdown("#### Income Statement (in millions)")
            income_data = {
                '2023': ['394,328', '223,546', '170,782', '63,090', '107,692', '7,891', '99,801'],
                '2022': ['365,817', '207,489', '158,328', '58,665', '99,663', '6,543', '93,120'],
                '2021': ['347,155', '195,515', '151,640', '54,760', '96,880', '5,890', '90,990']
            }
            income_df = pd.DataFrame(income_data, index=[
                'Total Revenue', 'Cost of Revenue', 'Gross Profit', 
                'Operating Expenses', 'Operating Income', 'Interest Expense', 
                'Net Income'
            ])
            st.dataframe(income_df, use_container_width=True)
            
            # Balance sheet
            st.markdown("#### Balance Sheet (in millions)")
            balance_data = {
                '2023': ['135,405', '246,674', '382,079', '287,912', '94,167'],
                '2022': ['124,308', '229,836', '354,144', '267,564', '86,580'],
                '2021': ['112,645', '214,218', '326,863', '251,084', '75,779']
            }
            balance_df = pd.DataFrame(balance_data, index=[
                'Current Assets', 'Non-Current Assets', 'Total Assets',
                'Total Liabilities', 'Total Equity'
            ])
            st.dataframe(balance_df, use_container_width=True)
        
        with tab4:
            # Key statistics
            st.subheader("📈 Key Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="card">
                    <h3>Share Statistics</h3>
                    <p><strong>Shares Outstanding:</strong> {company_info['key_statistics']['shares_outstanding']}</p>
                    <p><strong>Float:</strong> {company_info['key_statistics']['float']}</p>
                    <p><strong>Insider Ownership:</strong> {company_info['key_statistics']['insider_ownership']}</p>
                    <p><strong>Institutional Ownership:</strong> {company_info['key_statistics']['institutional_ownership']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="card">
                    <h3>Valuation Measures</h3>
                    <p><strong>Market Cap:</strong> {company_info['financials']['market_cap']}</p>
                    <p><strong>P/E Ratio:</strong> {company_info['financials']['pe_ratio']}</p>
                    <p><strong>EPS (TTM):</strong> {company_info['financials']['eps']}</p>
                    <p><strong>Beta (5Y Monthly):</strong> {company_info['financials']['beta']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab5:
            # News
            st.subheader("📰 Latest News")
            news_articles = fetch_news(company_info['name'])
            
            if news_articles:
                for article in news_articles:
                    st.markdown(f"""
                    <div class="card">
                        <h3>{article['title']}</h3>
                        <p><strong>Source:</strong> {article['source']['name']} • {article['publishedAt'][:10]}</p>
                        <p>{article['description']}</p>
                        <p><a href="{article['url']}" target="_blank">Read more</a></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent news articles found.")

# Market Movers - Top Gainers & Losers
elif selected == "Market Movers":
    st.title("📈 Market Movers - Active Stocks, Top Gainers & Losers")
    
    # Top gainers
    st.subheader("🚀 Top Gainers")
    gainers_data = {
        "Symbol": ["NVDA", "META", "TSLA", "AMD", "SHOP", "NET", "SNOW", "DDOG", "CRWD", "ZS"],
        "Name": ["NVIDIA Corp", "Meta Platforms", "Tesla Inc", "Advanced Micro Devices", 
                 "Shopify Inc", "Cloudflare Inc", "Snowflake Inc", "Datadog Inc", 
                 "CrowdStrike Holdings", "Zscaler Inc"],
        "Price": ["$610.32", "$468.25", "$245.67", "$178.90", "$82.45", 
                  "$95.60", "$185.32", "$125.67", "$320.45", "$185.90"],
        "Change %": ["+5.2%", "+4.8%", "+4.1%", "+3.9%", "+3.5%", 
                     "+3.2%", "+2.9%", "+2.7%", "+2.5%", "+2.3%"],
        "Volume": ["45.2M", "38.7M", "52.1M", "28.9M", "15.4M", 
                   "12.8M", "10.5M", "8.7M", "7.9M", "6.5M"]
    }
    gainers_df = pd.DataFrame(gainers_data)
    st.dataframe(gainers_df, use_container_width=True, hide_index=True)
    
    # Top losers
    st.subheader("🔻 Top Losers")
    losers_data = {
        "Symbol": ["PFE", "DASH", "CVNA", "RIVN", "LCID", "NKLA", "AFRM", "UPST", "HOOD", "COIN"],
        "Name": ["Pfizer Inc", "DoorDash Inc", "Carvana Co", "Rivian Automotive", 
                 "Lucid Group Inc", "Nikola Corp", "Affirm Holdings", "Upstart Holdings", 
                 "Robinhood Markets", "Coinbase Global"],
        "Price": ["$27.45", "$102.36", "$56.78", "$16.23", "$3.45", 
                  "$0.87", "$35.67", "$28.90", "$12.34", "$145.67"],
        "Change %": ["-3.8%", "-3.2%", "-2.9%", "-2.7%", "-2.5%", 
                     "-2.4%", "-2.1%", "-1.9%", "-1.7%", "-1.5%"],
        "Volume": ["32.1M", "18.9M", "15.7M", "12.8M", "10.5M", 
                   "8.7M", "7.9M", "6.5M", "5.8M", "4.9M"]
    }
    losers_df = pd.DataFrame(losers_data)
    st.dataframe(losers_df, use_container_width=True, hide_index=True)
    
    # Most active
    st.subheader("🔥 Most Active")
    active_data = {
        "Symbol": ["TSLA", "AAPL", "NVDA", "AMD", "AMZN", "META", "GOOGL", "MSFT", "SPY", "QQQ"],
        "Name": ["Tesla Inc", "Apple Inc", "NVIDIA Corp", "Advanced Micro Devices", 
                 "Amazon.com Inc", "Meta Platforms", "Alphabet Inc", "Microsoft Corp", 
                 "SPDR S&P 500", "Invesco QQQ Trust"],
        "Price": ["$245.67", "$188.50", "$610.32", "$178.90", "$175.25", 
                  "$468.25", "$150.45", "$407.65", "$489.12", "$426.78"],
        "Change %": ["+4.1%", "+0.8%", "+5.2%", "+3.9%", "+1.5%", 
                     "+4.8%", "+1.2%", "+0.9%", "+0.5%", "+0.7%"],
        "Volume": ["52.1M", "48.7M", "45.2M", "28.9M", "25.6M", 
                   "38.7M", "22.4M", "20.8M", "78.9M", "45.6M"]
    }
    active_df = pd.DataFrame(active_data)
    st.dataframe(active_df, use_container_width=True, hide_index=True)

# Global Markets - Major Indices
elif selected == "Global Markets":
    st.title("🌍 Global Markets Status")
    
    # World indices
    indices_data = {
        "Index": ["S&P 500", "Dow Jones", "NASDAQ", "Russell 2000", "FTSE 100", 
                  "DAX", "CAC 40", "Nikkei 225", "Hang Seng", "Shanghai Composite", 
                  "ASX 200", "TSX Composite", "BSE Sensex", "Nifty 50"],
        "Price": ["4,891.23", "38,654.42", "15,628.04", "1,978.23", "7,654.32", 
                  "16,789.10", "7,432.10", "36,543.21", "16,789.54", "3,245.67", 
                  "7,654.32", "21,098.76", "72,345.67", "21,876.54"],
        "Change %": ["+0.45%", "+0.15%", "+0.75%", "-0.25%", "+0.32%", 
                     "+0.67%", "+0.23%", "-0.12%", "-0.45%", "-0.32%", 
                     "+0.54%", "+0.21%", "+1.23%", "+1.05%"]
    }
    indices_df = pd.DataFrame(indices_data)
    st.dataframe(indices_df, use_container_width=True, hide_index=True)
    
    # World map visualization
    st.subheader("🌎 Global Market Performance Heatmap")
    
    # Create a sample heatmap data
    country_data = {
        "Country": ["United States", "United Kingdom", "Germany", "France", "Japan", 
                    "China", "Australia", "Canada", "India", "Brazil"],
        "Code": ["USA", "GBR", "DEU", "FRA", "JPN", "CHN", "AUS", "CAN", "IND", "BRA"],
        "Performance": [0.45, 0.32, 0.67, 0.23, -0.12, -0.32, 0.54, 0.21, 1.23, -0.45]
    }
    country_df = pd.DataFrame(country_data)
    
    fig = px.choropleth(country_df, locations="Code", color="Performance",
                        hover_name="Country", 
                        color_continuous_scale=px.colors.diverging.RdYlGn,
                        title="Global Market Performance")
    st.plotly_chart(fig, use_container_width=True)

# Other sections would follow the same pattern with enhanced UI and functionality

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>MarketMentor - Stock Market Dashboard | Developed with ❤️ using Streamlit</p>
    <p>Disclaimer: This is a simulation for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
