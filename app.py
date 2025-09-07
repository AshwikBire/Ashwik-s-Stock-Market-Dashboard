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
    page_title="MarketMentor - Advanced Financial Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply pure dark theme with minimal CSS
st.markdown("""
<style>
    .main {background-color: #0A0F2D; color: #E0E0E0;}
    h1, h2, h3, h4, h5, h6 {color: #4A90E2 !important; border-bottom: 1px solid #1A1F3B; padding-bottom: 8px;}
    .stButton>button {background-color: #1A1F3B; color: white; border: 1px solid #2D6BA1;}
    .stTextInput>div>div>input, .stSelectbox>div>div>select {background-color: #1A1F3B; color: white; border: 1px solid #2D6BA1;}
    .stMetric {background-color: #13274F; border-radius: 5px; padding: 10px; border-left: 3px solid #4A90E2;}
    .stDataFrame {background-color: #13274F;}
    .streamlit-expanderHeader {background-color: #13274F; border-radius: 4px; padding: 8px;}
</style>
""", unsafe_allow_html=True)

# Initialize session state for caching
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = {}

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
        "^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ",
        "^NSEI": "Nifty 50", "^BSESN": "Sensex", "^FTSE": "FTSE 100"
    }
    results = {}
    for symbol, name in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                last_close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = ((last_close - prev_close) / prev_close) * 100
                results[symbol] = {"name": name, "price": last_close, "change": change}
        except:
            continue
    return results

@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(query="stock market"):
    try:
        NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&pageSize=3"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json().get("articles", [])
        return []
    except:
        return []

# Sidebar navigation without images
with st.sidebar:
    selected = option_menu(
        "MarketMentor",
        ["Dashboard", "Stock Analysis", "Market Overview", "Technical Analysis", 
         "Options Chain", "Portfolio Tracker", "Economic Calendar", "Crypto Markets",
         "News & Sentiment", "Learning Center", "Settings"],
        icons=['speedometer2', 'graph-up', 'globe', 'bar-chart-line', 
               'diagram-3', 'wallet', 'calendar', 'currency-bitcoin',
               'newspaper', 'book', 'gear'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#13274F"},
            "icon": {"color": "#4A90E2", "font-size": "16px"}, 
            "nav-link": {"color": "#E0E0E0", "font-size": "14px", "text-align": "left"},
            "nav-link-selected": {"background-color": "#4A90E2"},
        }
    )

# Dashboard Page
if selected == "Dashboard":
    st.title("📊 Market Dashboard")
    
    # Load global indices with caching
    indices_data = fetch_global_indices()
    
    # Display key indices
    st.subheader("🌍 Global Indices")
    cols = st.columns(4)
    for idx, (symbol, data) in enumerate(indices_data.items()):
        if idx < 4:  # Limit for performance
            with cols[idx % 4]:
                currency = '$' if symbol not in ["^NSEI", "^BSESN"] else '₹'
                st.metric(
                    label=data["name"],
                    value=f"{currency}{data['price']:.2f}",
                    delta=f"{data['change']:.2f}%"
                )
    
    # Market overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sector Performance")
        sectors = {
            "Technology": "+2.3%", "Healthcare": "+1.5%", "Financials": "-0.8%",
            "Energy": "+3.2%", "Consumer Cyclical": "+0.7%"
        }
        sector_df = pd.DataFrame({
            "Sector": list(sectors.keys()),
            "Performance": [float(x.strip('%')) for x in sectors.values()]
        })
        fig = px.bar(sector_df, x="Performance", y="Sector", orientation='h',
                     title="Sector Performance (%)", color="Performance")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Market Sentiment")
        sentiment_data = {"Bullish": 45, "Neutral": 30, "Bearish": 25}
        fig = px.pie(values=list(sentiment_data.values()), names=list(sentiment_data.keys()),
                     title="Market Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent news with caching
    st.subheader("📰 Market News")
    news_articles = fetch_news()
    if news_articles:
        for article in news_articles:
            with st.expander(f"{article['title']} - {article['source']['name']}"):
                st.write(article.get('description', 'No description available'))
                st.markdown(f"[Read more]({article['url']})")

# Stock Analysis Page
elif selected == "Stock Analysis":
    st.title("📈 Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input("🔍 Enter Stock Symbol", "AAPL")
    with col2:
        analysis_type = st.selectbox("Analysis Type", ["Overview", "Financials", "Holdings"])
    
    if ticker:
        hist, info = fetch_stock_data(ticker)
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
                    st.metric("Market Cap", f"${market_cap/1e9:.2f}B" if market_cap >= 1e9 else f"${market_cap/1e6:.2f}M")
            with col4:
                st.metric("P/E Ratio", f"{pe_ratio:.2f}" if isinstance(pe_ratio, float) else pe_ratio)
            
            # Price chart
            st.subheader("Price Chart")
            fig = px.line(hist, x=hist.index, y='Close', title=f"{ticker.upper()} Price History")
            st.plotly_chart(fig, use_container_width=True)
            
            # Financial statements (simulated)
            if analysis_type == "Financials":
                st.subheader("Financial Statements")
                financials = pd.DataFrame({
                    'Year': ['2023', '2022', '2021'],
                    'Revenue': [383.29, 365.82, 274.52],
                    'Net Income': [97.00, 94.68, 57.41],
                    'EPS': [6.13, 5.67, 3.28]
                })
                st.dataframe(financials, use_container_width=True)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=financials['Year'], y=financials['Revenue'], name='Revenue'))
                fig.add_trace(go.Bar(x=financials['Year'], y=financials['Net Income'], name='Net Income'))
                fig.update_layout(title="Revenue vs Net Income (Billions $)")
                st.plotly_chart(fig, use_container_width=True)

# Technical Analysis Page
elif selected == "Technical Analysis":
    st.title("📊 Technical Analysis")
    
    ticker = st.text_input("Enter Stock Symbol for Technical Analysis", "AAPL")
    if ticker:
        hist, info = fetch_stock_data(ticker, "3mo")
        if hist is not None and not hist.empty:
            # Calculate basic technical indicators
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['RSI'] = 100 - (100 / (1 + (hist['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                           hist['Close'].diff().clip(upper=0).abs().rolling(14).mean())))
            
            indicator = st.selectbox("Select Technical Indicator", 
                                    ["Moving Averages", "RSI", "MACD", "Bollinger Bands"])
            
            if indicator == "Moving Averages":
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='#4A90E2')))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name='SMA 20', line=dict(color='#FFA15A')))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50', line=dict(color='#00CC96')))
                fig.update_layout(title="Moving Averages")
                st.plotly_chart(fig, use_container_width=True)
                
            elif indicator == "RSI":
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI', line=dict(color='#4A90E2')))
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig.update_layout(title="Relative Strength Index (RSI)")
                st.plotly_chart(fig, use_container_width=True)

# Options Chain Page
elif selected == "Options Chain":
    st.title("📋 Options Chain Analysis")
    
    st.subheader("Options Data (Simulated)")
    options_data = pd.DataFrame({
        'Strike': [150, 155, 160, 165, 170],
        'Call OI': [1200, 950, 780, 620, 480],
        'Put OI': [850, 720, 1080, 920, 680],
        'Call Volume': [450, 320, 280, 210, 180],
        'Put Volume': [380, 290, 420, 350, 270]
    })
    
    st.dataframe(options_data.style.background_gradient(cmap="Blues"), use_container_width=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=options_data['Strike'], y=options_data['Call OI'], name='Call OI'))
    fig.add_trace(go.Bar(x=options_data['Strike'], y=options_data['Put OI'], name='Put OI'))
    fig.update_layout(title="Open Interest Analysis", barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Options Strategy Builder")
    strategy = st.selectbox("Select Strategy", ["Long Call", "Long Put", "Covered Call", "Iron Condor"])
    st.write(f"**{strategy} Strategy Overview:**")
    st.write("- Maximum Profit: Unlimited" if strategy == "Long Call" else "- Maximum Profit: Limited")
    st.write("- Maximum Loss: Premium Paid" if strategy in ["Long Call", "Long Put"] else "- Maximum Loss: Limited")

# Portfolio Tracker Page
elif selected == "Portfolio Tracker":
    st.title("💰 Portfolio Tracker")
    
    # Initialize portfolio in session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Quantity', 'Purchase Price'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Add Holding")
        symbol = st.text_input("Stock Symbol", "AAPL")
        quantity = st.number_input("Quantity", min_value=1, value=10)
        price = st.number_input("Purchase Price", min_value=0.01, value=150.0)
        
        if st.button("Add to Portfolio"):
            new_holding = pd.DataFrame({'Symbol': [symbol], 'Quantity': [quantity], 'Purchase Price': [price]})
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_holding], ignore_index=True)
    
    with col2:
        st.subheader("Current Portfolio")
        if not st.session_state.portfolio.empty:
            # Simulate current prices
            st.session_state.portfolio['Current Price'] = st.session_state.portfolio['Symbol'].apply(
                lambda x: 150.0 if x == 'AAPL' else 280.0 if x == 'MSFT' else 100.0
            )
            st.session_state.portfolio['Value'] = st.session_state.portfolio['Quantity'] * st.session_state.portfolio['Current Price']
            st.session_state.portfolio['Gain/Loss'] = st.session_state.portfolio['Value'] - (
                st.session_state.portfolio['Quantity'] * st.session_state.portfolio['Purchase Price'])
            
            st.dataframe(st.session_state.portfolio, use_container_width=True)
            
            total_value = st.session_state.portfolio['Value'].sum()
            total_gain = st.session_state.portfolio['Gain/Loss'].sum()
            st.metric("Total Portfolio Value", f"${total_value:,.2f}")
            st.metric("Total Gain/Loss", f"${total_gain:,.2f}", delta=f"{((total_gain/(total_value-total_gain))*100):.2f}%")
        else:
            st.info("No holdings in portfolio. Add some stocks to get started.")

# Economic Calendar Page
elif selected == "Economic Calendar":
    st.title("📅 Economic Calendar")
    
    # Simulated economic events
    economic_events = pd.DataFrame({
        'Date': ['2023-09-15', '2023-09-20', '2023-09-25', '2023-10-01'],
        'Event': ['Fed Interest Rate Decision', 'CPI Data Release', 'Unemployment Claims', 'Non-Farm Payrolls'],
        'Impact': ['High', 'High', 'Medium', 'High'],
        'Previous': ['5.25%', '3.2%', '230K', '187K'],
        'Forecast': ['5.5%', '3.4%', '225K', '190K']
    })
    
    st.dataframe(economic_events, use_container_width=True)
    
    st.subheader("Central Bank Watch")
    rates_data = pd.DataFrame({
        'Central Bank': ['Federal Reserve', 'ECB', 'Bank of England', 'Bank of Japan'],
        'Current Rate': ['5.25%', '4.25%', '5.00%', '-0.10%'],
        'Next Meeting': ['Sept 20, 2023', 'Oct 5, 2023', 'Sept 21, 2023', 'Oct 12, 2023'],
        'Expected Change': ['+0.25%', '+0.25%', '+0.25%', 'No Change']
    })
    
    st.dataframe(rates_data, use_container_width=True)

# Crypto Markets Page
elif selected == "Crypto Markets":
    st.title("₿ Crypto Markets")
    
    # Simulated crypto data
    crypto_data = pd.DataFrame({
        'Cryptocurrency': ['Bitcoin', 'Ethereum', 'Binance Coin', 'Cardano', 'Solana'],
        'Price': [25800, 1650, 215, 0.25, 32.50],
        '24h Change': [-1.2, 2.5, -0.8, 3.2, -2.1],
        'Market Cap (B)': [500, 200, 35, 9, 12]
    })
    
    st.dataframe(crypto_data, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Crypto Fear & Greed Index")
        current_index = 45  # Neutral
        st.metric("Current Index", f"{current_index}/100", "Neutral")
        st.progress(current_index / 100)
        
    with col2:
        st.subheader("Bitcoin Dominance")
        dominance = 48.5  # Percentage
        st.metric("BTC Dominance", f"{dominance}%")
        st.progress(dominance / 100)

# News & Sentiment Page
elif selected == "News & Sentiment":
    st.title("📰 News & Market Sentiment")
    
    news_query = st.text_input("Search Financial News", "stock market")
    if news_query:
        news_articles = fetch_news(news_query)
        if news_articles:
            for article in news_articles:
                with st.expander(f"{article['title']} - {article['source']['name']}"):
                    st.write(article.get('description', 'No description available'))
                    st.markdown(f"[Read more]({article['url']})")
    
    st.subheader("Market Sentiment Analysis")
    sentiment_tabs = st.tabs(["Stocks", "Sectors", "Market Overview"])
    
    with sentiment_tabs[0]:
        stock_sentiment = pd.DataFrame({
            'Stock': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'Sentiment': ['Bullish', 'Neutral', 'Bullish', 'Bearish', 'Neutral'],
            'Confidence': [85, 60, 78, 72, 55]
        })
        st.dataframe(stock_sentiment, use_container_width=True)
    
    with sentiment_tabs[1]:
        sector_sentiment = pd.DataFrame({
            'Sector': ['Technology', 'Healthcare', 'Financials', 'Energy', 'Consumer'],
            'Sentiment': ['Bullish', 'Neutral', 'Bearish', 'Bullish', 'Neutral'],
            'Trend': ['Improving', 'Stable', 'Worsening', 'Improving', 'Stable']
        })
        st.dataframe(sector_sentiment, use_container_width=True)

# Learning Center Page
elif selected == "Learning Center":
    st.title("📚 Learning Center")
    
    learning_tabs = st.tabs(["Beginner Guides", "Technical Analysis", "Options Trading", "Portfolio Management"])
    
    with learning_tabs[0]:
        st.subheader("Getting Started with Investing")
        st.write("""
        ### 1. Understanding the Basics
        - What are stocks and how do they work?
        - Different types of investments: stocks, bonds, mutual funds
        - Risk vs. return: finding your investment style
        
        ### 2. Fundamental Analysis
        - How to read financial statements
        - Key financial ratios: P/E, PEG, ROE, Debt-to-Equity
        - Evaluating company management and competitive advantage
        """)
    
    with learning_tabs[1]:
        st.subheader("Technical Analysis Fundamentals")
        st.write("""
        ### 1. Chart Patterns
        - Support and resistance levels
        - Trend lines and channels
        - Common patterns: head and shoulders, double tops/bottoms
        
        ### 2. Technical Indicators
        - Moving averages (SMA, EMA)
        - Relative Strength Index (RSI)
        - Moving Average Convergence Divergence (MACD)
        """)
    
    with learning_tabs[2]:
        st.subheader("Options Trading Strategies")
        st.write("""
        ### 1. Basic Strategies
        - Covered calls: Generating income from existing holdings
        - Protective puts: Insurance against downside risk
        - Long calls/puts: Directional bets with limited risk
        
        ### 2. Advanced Strategies
        - Iron condors: Neutral strategy for range-bound markets
        - Straddles/strangles: Volatility plays
        - Vertical spreads: Defined risk directional plays
        """)

# Settings Page
elif selected == "Settings":
    st.title("⚙️ Settings")
    
    st.subheader("App Preferences")
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox("Theme", ["Dark", "Light"], index=0)
        currency = st.selectbox("Currency", ["USD", "INR", "EUR", "GBP"], index=0)
        default_view = st.selectbox("Default View", ["Dashboard", "Stock Analysis", "Portfolio"], index=0)
    
    with col2:
        refresh_rate = st.slider("Data Refresh Rate (minutes)", 1, 60, 5)
        notifications = st.checkbox("Enable Notifications", value=True)
        st.button("Save Preferences")
    
    st.subheader("Data Management")
    st.button("Clear Cache")
    st.button("Export Portfolio Data")
    st.button("Reset to Default Settings")
