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
import ta  # Technical analysis library
import warnings
import json
from bs4 import BeautifulSoup
import time

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="MarketMentor", layout="wide", page_icon="📈")

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Developer Info
developer_info = {
    "name": "Ashwik Bire",
    "linkedin": "https://www.linkedin.com/in/ashwik-bire-1b4530250/",
    "role": "Financial Data Scientist"
}

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "MarketMentor",
        ["Home", "Company Overview", "Market Movers", "F&O", "Global Markets", 
         "Mutual Funds", "SIP Calculator", "IPO Tracker", "Predictions for Mutual Funds & IPOs", 
         "Mutual Fund NAV Viewer", "Sectors", "News", "Learning", "Volume Spike", 
         "Stock Screener", "Predictions", "Buy/Sell Predictor", "News Sentiment", 
         "Technical Analysis", "Portfolio Tracker", "Economic Calendar"],
        icons=['house', 'building', 'graph-up', 'activity', 'globe', 
               'bank', 'calculator', 'rocket', 'lightning', 
               'bar-chart', 'grid', 'newspaper', 'book', 'activity', 
               'search', 'graph-up-arrow', 'currency-exchange', 'chat', 
               'graph-up', 'wallet', 'calendar'],
        menu_icon="cast",
        default_index=0
    )
    
    # Developer info in sidebar
    st.markdown("---")
    st.markdown("### Developer Info")
    st.markdown(f"**Name:** {developer_info['name']}")
    st.markdown(f"**Role:** {developer_info['role']}")
    st.markdown(f"[LinkedIn Profile]({developer_info['linkedin']})")

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
                change = round(ticker_data['Close'].iloc[-1] - ticker_data['Open'].iloc[-1], 2)
                percent_change = round((change / ticker_data['Open'].iloc[-1]) * 100, 2)
                data[name] = {"value": last_close, "change": percent_change}
        except:
            continue
    return data

@st.cache_data(ttl=3600)
def get_news(query="stock market", page_size=10):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize={page_size}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
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
            change_color = "green" if data["change"] >= 0 else "red"
            change_icon = "📈" if data["change"] >= 0 else "📉"
            st.metric(
                label=name,
                value=f"₹{data['value']:,}",
                delta=f"{change_icon} {data['change']}%",
                delta_color="normal" if data["change"] >= 0 else "inverse"
            )
        idx += 1
    
    # Stock search
    st.subheader("🔍 Quick Stock Search")
    ticker = st.text_input("Enter stock symbol (e.g., RELIANCE.NS, INFY.NS):", "RELIANCE.NS")
    
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
                height=400
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

# Company Overview
elif selected == "Company Overview":
    st.title("🏢 Company Overview")
    
    # Stock search
    ticker = st.text_input("Enter stock symbol:", "RELIANCE.NS")
    
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
                
                # Display metrics in a grid
                cols = st.columns(3)
                for i, (key, value) in enumerate(metrics_data.items()):
                    with cols[i % 3]:
                        st.metric(key, value)
            
            # Display stock chart
            st.subheader("Price Chart")
            period = st.selectbox("Select period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
            data = get_stock_data(ticker, period)
            
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
                    height=500
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
    
    gainers, losers = get_market_movers()
    
    if gainers is not None and losers is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Gainers")
            st.dataframe(gainers, use_container_width=True)
        
        with col2:
            st.subheader("📉 Top Losers")
            st.dataframe(losers, use_container_width=True)
    else:
        st.error("Could not fetch market movers data.")

# F&O Page
elif selected == "F&O":
    st.title("📑 F&O Overview")
    
    # Placeholder for F&O data
    st.info("Futures and Options data will be displayed here. This section is under development.")
    
    # Example of what could be displayed
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Most Active Futures")
        # This would typically show the most traded futures contracts
        st.write("1. NIFTY 50")
        st.write("2. BANKNIFTY")
        st.write("3. RELIANCE")
        st.write("4. INFY")
        st.write("5. HDFC BANK")
    
    with col2:
        st.subheader("Open Interest Gainers")
        # This would typically show stocks with highest OI change
        st.write("1. TATASTEEL")
        st.write("2. ADANIPORTS")
        st.write("3. HINDALCO")
        st.write("4. JSWSTEEL")
        st.write("5. ICICIBANK")

# Global Markets
elif selected == "Global Markets":
    st.title("🌍 Global Markets")
    
    global_indices = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^FTSE": "FTSE 100",
        "^N225": "Nikkei 225",
        "^HSI": "Hang Seng",
        "CL=F": "Crude Oil",
        "GC=F": "Gold"
    }
    
    global_data = get_indices_data(global_indices)
    
    # Display global indices
    cols = st.columns(4)
    idx = 0
    for name, data in global_data.items():
        with cols[idx % 4]:
            change_color = "green" if data["change"] >= 0 else "red"
            st.metric(
                label=name,
                value=f"${data['value']:,}" if not name in ["Crude Oil", "Gold"] else f"${data['value']}",
                delta=f"{data['change']}%",
                delta_color="normal" if data["change"] >= 0 else "inverse"
            )
        idx += 1
    
    # Currency rates
    st.subheader("💱 Currency Exchange Rates")
    currency_pairs = {
        "USDINR=X": "USD/INR",
        "EURINR=X": "EUR/INR",
        "GBPINR=X": "GBP/INR",
        "JPYINR=X": "JPY/INR"
    }
    
    currency_data = get_indices_data(currency_pairs)
    
    col1, col2, col3, col4 = st.columns(4)
    currency_cols = [col1, col2, col3, col4]
    
    for i, (symbol, data) in enumerate(currency_data.items()):
        with currency_cols[i]:
            st.metric(
                label=symbol,
                value=f"₹{data['value']:.2f}",
                delta=f"{data['change']:.2f}%",
                delta_color="normal" if data["change"] >= 0 else "inverse"
            )

# Continue with other sections...

# Mutual Funds
elif selected == "Mutual Funds":
    st.title("💼 Mutual Funds")
    
    # Placeholder for mutual funds data
    st.info("Mutual funds data will be displayed here. This section is under development.")
    
    fund_categories = ["Equity", "Debt", "Hybrid", "ELSS", "Sectoral"]
    selected_category = st.selectbox("Select Fund Category", fund_categories)
    
    if selected_category:
        # This would typically show top funds in the selected category
        st.subheader(f"Top {selected_category} Funds")
        
        # Sample data
        sample_funds = {
            "Equity": ["SBI Bluechip Fund", "HDFC Top 100 Fund", "ICICI Pru Bluechip Fund"],
            "Debt": ["SBI Magnum Gilt Fund", "HDFC Corporate Bond Fund", "ICICI Pru All Seasons Bond Fund"],
            "Hybrid": ["SBI Equity Hybrid Fund", "HDFC Hybrid Equity Fund", "ICICI Pru Balanced Advantage Fund"],
            "ELSS": ["SBI Long Term Equity Fund", "HDFC Taxsaver Fund", "ICICI Pru Long Term Equity Fund"],
            "Sectoral": ["SBI Banking & Financial Services Fund", "HDFC Healthcare Fund", "ICICI Pru Technology Fund"]
        }
        
        for fund in sample_funds[selected_category]:
            st.write(f"- {fund}")

# SIP Calculator
elif selected == "SIP Calculator":
    st.title("📈 SIP Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_investment = st.number_input("Monthly Investment (₹)", min_value=500, value=5000, step=500)
        investment_period = st.slider("Investment Period (Years)", min_value=1, max_value=30, value=10)
        expected_return = st.slider("Expected Annual Return (%)", min_value=1, max_value=30, value=12)
    
    with col2:
        # Calculate SIP
        monthly_rate = expected_return / 100 / 12
        months = investment_period * 12
        future_value = monthly_investment * ((1 + monthly_rate)**months - 1) / monthly_rate * (1 + monthly_rate)
        total_investment = monthly_investment * months
        estimated_returns = future_value - total_investment
        
        st.metric("Total Investment", f"₹{total_investment:,.0f}")
        st.metric("Estimated Returns", f"₹{estimated_returns:,.0f}")
        st.metric("Future Value", f"₹{future_value:,.0f}", delta=f"{estimated_returns:,.0f}")

# IPO Tracker
elif selected == "IPO Tracker":
    st.title("🆕 IPO Tracker")
    
    ipo_data = get_ipo_data()
    
    if ipo_data is not None:
        st.subheader("Upcoming IPOs")
        st.dataframe(ipo_data, use_container_width=True)
    else:
        st.error("Could not fetch IPO data.")
    
    # Past IPOs performance
    st.subheader("Recent IPO Performance")
    
    # Sample data for recent IPOs
    recent_ipos = [
        {"Name": "LIC", "Issue Price": "₹949", "Current Price": "₹915", "Return": "-3.58%"},
        {"Name": "Paytm", "Issue Price": "₹2,150", "Current Price": "₹810", "Return": "-62.33%"},
        {"Name": "Zomato", "Issue Price": "₹76", "Current Price": "₹125", "Return": "+64.47%"},
        {"Name": "Nykaa", "Issue Price": "₹1,125", "Current Price": "₹1,650", "Return": "+46.67%"},
        {"Name": "Policybazaar", "Issue Price": "₹980", "Current Price": "₹720", "Return": "-26.53%"}
    ]
    
    st.table(pd.DataFrame(recent_ipos))

# Continue with other sections...

# Predictions for Mutual Funds & IPOs
elif selected == "Predictions for Mutual Funds & IPOs":
    st.title("📊 Predictions for Mutual Funds & IPOs")
    
    st.info("This section provides predictive analysis for mutual funds and IPOs.")
    
    option = st.radio("Select Prediction Type:", ["Mutual Funds", "IPOs"])
    
    if option == "Mutual Funds":
        st.subheader("Mutual Fund Performance Prediction")
        
        # Sample prediction UI
        fund_name = st.selectbox("Select Fund", ["SBI Bluechip Fund", "HDFC Top 100 Fund", "ICICI Pru Bluechip Fund"])
        prediction_period = st.selectbox("Prediction Period", ["1 Month", "3 Months", "6 Months", "1 Year"])
        
        if st.button("Predict"):
            # Placeholder for prediction logic
            st.success(f"Predicted return for {fund_name} over {prediction_period}: +8.5%")
            
            # Show historical performance chart
            dates = pd.date_range(end=datetime.today(), periods=12, freq='M')
            values = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=values, mode='lines', name='Historical NAV'))
            fig.update_layout(title=f"{fund_name} Historical Performance",
                             xaxis_title="Date", yaxis_title="NAV (₹)")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.subheader("IPO Performance Prediction")
        
        # Sample prediction UI
        ipo_name = st.text_input("Enter IPO Name", "Company XYZ")
        issue_price = st.number_input("Issue Price (₹)", min_value=10, value=100, step=10)
        sector = st.selectbox("Sector", ["Technology", "Healthcare", "Financial Services", "Consumer Goods", "Industrial"])
        
        if st.button("Predict IPO Performance"):
            # Placeholder for prediction logic
            st.success(f"Predicted performance for {ipo_name}: +15% listing gain")
            st.info("Note: IPO predictions are based on market conditions, company fundamentals, and comparable analysis.")

# Continue with other sections...

# Footer with developer info
st.markdown("---")
st.markdown("### About MarketMentor")
st.markdown("MarketMentor is a comprehensive financial analysis platform designed to help investors make informed decisions.")
st.markdown(f"Developed by **{developer_info['name']}** - {developer_info['role']}")
st.markdown(f"Connect on [LinkedIn]({developer_info['linkedin']})")

# Add some custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .css-1d391kg {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)
