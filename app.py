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
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Set page config with theme
st.set_page_config(
    page_title="MarketMentor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for navy blue + dark blue theme
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #1F4E79;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #1F4E79;
        color: white;
    }
    .stSelectbox>div>div>select {
        background-color: #1F4E79;
        color: white;
    }
    .stSlider>div>div>div>div {
        background-color: #1F4E79;
    }
    .sidebar .sidebar-content {
        background-color: #1F4E79;
    }
    .css-1d391kg {
        background-color: #1F4E79;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #4A90E2;
    }
    .stMetric {
        background-color: #1F4E79;
        border-radius: 5px;
        padding: 10px;
    }
    .stDataFrame {
        background-color: #1F4E79;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "MarketMentor",
        ["Home","Company Overview", "Market Movers", "F&O", "Global Markets", "Mutual Funds", "SIP Calculator","IPO Tracker","Predictions for Mutual Funds & IPOs","Mutual Fund NAV Viewer","Sectors", "News", "Learning", "Volume Spike", "Stock Screener", "Predictions", "Buy/Sell Predictor", "News Sentiment"],
        icons=['house', 'graph-up', 'globe', 'bank', 'boxes', 'newspaper', 'building', 'book', 'activity', 'search'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#1F4E79"},
            "icon": {"color": "white", "font-size": "18px"}, 
            "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#4A90E2"},
        }
    )

# Helper function to get currency symbol based on ticker
def get_currency(ticker):
    if ticker.endswith('.NS') or ticker.endswith('.BO'):
        return '₹'
    else:
        return '$'

# Home - Market Overview with Learning Materials
if selected == "Home":
    st.title("🏠 Home - Market Overview")
    
    # Learning Materials Section
    with st.expander("📚 Learning Materials - Start Your Investment Journey"):
        st.markdown("""
        ### Beginner's Guide to Stock Market Investing
        
        **1. Understanding the Basics**
        - What are stocks and how do they work?
        - Different types of investments: stocks, bonds, mutual funds
        - Risk vs. return: finding your investment style
        
        **2. Fundamental Analysis**
        - How to read financial statements
        - Key financial ratios: P/E, PEG, ROE, Debt-to-Equity
        - Evaluating company management and competitive advantage
        
        **3. Technical Analysis**
        - Reading stock charts: candlestick patterns
        - Important technical indicators: Moving Averages, RSI, MACD
        - Support and resistance levels
        
        **4. Investment Strategies**
        - Value investing: finding undervalued stocks
        - Growth investing: identifying high-potential companies
        - Dividend investing: building passive income
        
        **5. Risk Management**
        - Diversification: don't put all eggs in one basket
        - Position sizing: how much to invest in each stock
        - Setting stop-losses to protect your capital
        
        **6. Psychology of Investing**
        - Controlling emotions: fear and greed
        - Long-term thinking vs. short-term speculation
        - Developing a disciplined investment approach
        
        *More resources will be added regularly. Check back often!*
        """)
    
    indices = {
        "^NSEI": "Nifty 50",
        "^BSESN": "Sensex",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^GSPC": "S&P 500",
    }
    st.subheader("Major Indices Performance")
    cols = st.columns(len(indices))
    for idx, (symbol, name) in enumerate(indices.items()):
        data = yf.Ticker(symbol).history(period="1d")
        last_close = round(data['Close'].iloc[-1], 2)
        change = round(data['Close'].iloc[-1] - data['Open'].iloc[-1], 2)
        percent_change = round((change / data['Open'].iloc[-1]) * 100, 2)
        currency = '₹' if symbol in ["^NSEI", "^BSESN"] else '$'
        cols[idx].metric(label=name, value=f"{currency}{last_close}", delta=f"{percent_change}%")

# Market Movers - Top Gainers & Losers
elif selected == "Market Movers":
    st.title("📈 Market Movers - Active Stocks, Top Gainers & Losers")

    # Active Stocks (Example: Nifty 50 stocks)
    tickers_list = 'RELIANCE.NS TCS.NS INFY.NS HDFCBANK.NS ICICIBANK.NS'
    nifty = yf.Tickers(tickers_list)

    # Fetching recent closing prices
    data = {ticker: nifty.tickers[ticker].history(period="1d")['Close'].iloc[-1] for ticker in nifty.tickers}

    # Sorting stocks for gainers and losers
    gainers = sorted(data.items(), key=lambda x: x[1], reverse=True)
    losers = sorted(data.items(), key=lambda x: x[1])

    # Displaying Active Stocks
    st.subheader("📊 Active Stocks (Recent Close Prices)")
    active_stocks = pd.DataFrame(data.items(), columns=["Stock", "Price"])
    active_stocks['Price'] = active_stocks['Price'].apply(lambda x: f'₹{x:.2f}')
    st.dataframe(active_stocks)

    # Top Gainers
    st.subheader("🚀 Top Gainers")
    top_gainers = pd.DataFrame(gainers, columns=['Stock', 'Price'])
    top_gainers['Price'] = top_gainers['Price'].apply(lambda x: f'₹{x:.2f}')
    st.dataframe(top_gainers)

    # Top Losers
    st.subheader("📉 Top Losers")
    top_losers = pd.DataFrame(losers, columns=['Stock', 'Price'])
    top_losers['Price'] = top_losers['Price'].apply(lambda x: f'₹{x:.2f}')
    st.dataframe(top_losers)

# Global Markets - Major Indices
elif selected == "Global Markets":
    st.title("🌍 Global Markets Status")
    global_indices = {
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^GSPC": "S&P 500",
        "^FTSE": "FTSE 100",
        "^N225": "Nikkei 225",
        "^HSI": "Hang Seng",
    }
    st.subheader("Major Global Indices")
    cols = st.columns(3)
    for idx, (symbol, name) in enumerate(global_indices.items()):
        data = yf.Ticker(symbol).history(period="1d")
        last_close = round(data['Close'].iloc[-1], 2)
        change = round(data['Close'].iloc[-1] - data['Open'].iloc[-1], 2)
        percent_change = round((change / data['Open'].iloc[-1]) * 100, 2)
        currency = '$' if symbol in ["^DJI", "^IXIC", "^GSPC"] else '¥' if symbol == "^N225" else 'HK$' if symbol == "^HSI" else '£'
        cols[idx % 3].metric(label=name, value=f"{currency}{last_close}", delta=f"{percent_change}%")

# Company Overview Page
elif selected == "Company Overview":
    st.title("🏢 Company Overview")
    
    ticker = st.text_input("🔎 Enter Stock Ticker (e.g., AAPL, TCS.NS)", "TCS.NS")

    if ticker:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            info = stock.info
            
            # Determine currency symbol
            currency = get_currency(ticker)
            
            # Live metrics
            st.markdown("### 📌 Key Market Metrics")
            with st.container():
                col1, col2, col3 = st.columns(3)
                current_price = info.get('regularMarketPrice', hist['Close'].iloc[-1] if not hist.empty else 'N/A')
                day_high = info.get('dayHigh', hist['High'].iloc[-1] if not hist.empty else 'N/A')
                day_low = info.get('dayLow', hist['Low'].iloc[-1] if not hist.empty else 'N/A')
                
                col1.metric("💰 Current Price", f"{currency}{current_price:.2f}" if isinstance(current_price, float) else current_price)
                col2.metric("📈 Day High", f"{currency}{day_high:.2f}" if isinstance(day_high, float) else day_high)
                col3.metric("📉 Day Low", f"{currency}{day_low:.2f}" if isinstance(day_low, float) else day_low)

            st.markdown("---")

            # Interactive price chart
            st.markdown("### 📈 Price Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name="Close Price", line=dict(color='#4A90E2')))
            fig.update_layout(
                title=f"{ticker.upper()} Historical Price Chart",
                xaxis_title="Date",
                yaxis_title=f"Price ({currency})",
                template="plotly_dark",
                hovermode="x unified",
                height=400,
                plot_bgcolor='#0E1117',
                paper_bgcolor='#0E1117',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Organized info display
            st.markdown("### 🏢 Company Snapshot")
            with st.expander("📘 General Information", expanded=True):
                st.markdown(f"**Name:** {info.get('longName', 'N/A')}")
                st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
                st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
                st.markdown(f"**Website:** [{info.get('website', 'N/A')}]({info.get('website', '#')})")
                st.markdown(f"**Headquarters:** {info.get('city', 'N/A')}, {info.get('country', 'N/A')}")
                st.markdown(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")

            with st.expander("📄 Business Description"):
                st.write(info.get("longBusinessSummary", "No summary available."))

            with st.expander("📊 Key Financials"):
                col1, col2 = st.columns(2)
                market_cap = info.get('marketCap', 'N/A')
                if market_cap != 'N/A':
                    market_cap = f"{currency}{market_cap/1e9:.2f}B" if market_cap < 1e12 else f"{currency}{market_cap/1e12:.2f}T"
                col1.metric("Market Cap", market_cap)
                col2.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
                
                col1.metric("EPS", info.get('trailingEps', 'N/A'))
                col2.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A')

        except Exception as e:
            st.error(f"Error retrieving data: {e}")

# ... [Rest of the code remains similar with currency symbol adjustments where needed]

# Learning - Stock Market Resources
elif selected == "Learning":
    st.title("📘 Learn the Stock Market")

    st.markdown("""
    <div style="background-color: #1F4E79; padding: 20px; border-radius: 10px;">
    <h2 style="color: #4A90E2;">Comprehensive Stock Market Learning Center</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📚 Fundamental Analysis
        - **Financial Statements**: Learn to read balance sheets, income statements, and cash flow statements
        - **Valuation Methods**: DCF, P/E ratio, PEG ratio, and other valuation metrics
        - **Economic Indicators**: How macroeconomic factors affect stock prices
        - **Sector Analysis**: Understanding different industry sectors and their dynamics
        
        ### 📈 Technical Analysis
        - **Chart Patterns**: Head and shoulders, double tops/bottoms, triangles
        - **Technical Indicators**: Moving averages, RSI, MACD, Bollinger Bands
        - **Volume Analysis**: How trading volume confirms price movements
        - **Support and Resistance**: Identifying key price levels
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Algorithmic Trading
        - **Introduction to Algo Trading**: Basics of algorithmic strategies
        - **Backtesting**: How to test your trading strategies
        - **Risk Management**: Position sizing and risk control in algo trading
        - **Execution Strategies**: VWAP, TWAP, and other execution algorithms
        
        ### 📊 Investment Strategies
        - **Value Investing**: Finding undervalued companies
        - **Growth Investing**: Identifying high-growth potential stocks
        - **Dividend Investing**: Building income-generating portfolios
        - **Sector Rotation**: Adjusting portfolios based on economic cycles
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎥 Video Tutorials
    - **Beginner's Guide to Stock Market**: [Watch Here](#)
    - **Technical Analysis Masterclass**: [Watch Here](#)
    - **Fundamental Analysis Deep Dive**: [Watch Here](#)
    - **Options Trading Explained**: [Watch Here](#)
    
    ### 📖 Recommended Books
    - The Intelligent Investor by Benjamin Graham
    - A Random Walk Down Wall Street by Burton Malkiel
    - Common Stocks and Uncommon Profits by Philip Fisher
    - The Little Book of Common Sense Investing by John C. Bogle
    """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background-color: #1F4E79; padding: 15px; border-radius: 10px;">
    <h3 style="color: #4A90E2;">Connect with the Creator</h3>
    <p>This platform is created by <strong>Ashwik Bire</strong>, a finance enthusiast passionate about making market education accessible to everyone.</p>
    <p><a href="https://www.linkedin.com/in/ashwik-bire-b2a000186/" style="color: #4A90E2;">🔗 Connect on LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True)

# ... [Rest of the code remains similar with currency symbol adjustments where needed]

# Predictions - Stock Price Prediction
elif selected == "Predictions":
    st.title("📈 Stock Price Predictions")

    ticker = st.text_input("Enter Company Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")
    currency = get_currency(ticker)

    if ticker:
        try:
            # Fetch stock data from Yahoo Finance
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")  # 1 year of data

            if hist.empty:
                st.warning("No data available for this ticker.")
            else:
                # Show the most recent data
                st.subheader(f"Recent Stock Data for {ticker}")
                st.write(hist.tail())

                # Plot the stock's historical closing price
                st.subheader("📊 Stock Price History")
                st.line_chart(hist["Close"])

                # Calculate a simple moving average (SMA) for predictions
                sma50 = hist["Close"].rolling(window=50).mean()
                sma200 = hist["Close"].rolling(window=200).mean()

                st.subheader("📉 Moving Averages")
                st.line_chart(pd.DataFrame({
                    "50-Day SMA": sma50,
                    "200-Day SMA": sma200
                }))

                # Determine Buy/Sell signal based on SMA
                st.subheader("🔍 Buy/Sell Signal")
                current_price = hist["Close"].iloc[-1]
                if sma50.iloc[-1] > sma200.iloc[-1]:
                    st.success(
                        f"📈 Signal: **BUY** - 50-day SMA is above 200-day SMA (Current Price: {currency}{current_price:.2f})")
                elif sma50.iloc[-1] < sma200.iloc[-1]:
                    st.error(
                        f"📉 Signal: **SELL** - 50-day SMA is below 200-day SMA (Current Price: {currency}{current_price:.2f})")
                else:
                    st.warning(f"⏸️ Signal: **HOLD** - No clear trend (Current Price: {currency}{current_price:.2f})")

                # Show price data vs moving averages
                st.subheader("📈 Price vs. Moving Averages")
                st.line_chart(hist[["Close"]].join(pd.DataFrame({
                    "50-Day SMA": sma50,
                    "200-Day SMA": sma200
                })))

        except Exception as e:
            st.error(f"Error retrieving data: {e}")

# ... [Continue with the rest of your code, making similar currency adjustments]
