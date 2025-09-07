import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from plotly import graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
import talib
import warnings
warnings.filterwarnings('ignore')

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Set page config with minimal options for faster loading
st.set_page_config(
    page_title="MarketMentor - Financial Analytics", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply optimized CSS for faster rendering
st.markdown("""
<style>
    .main {
        background-color: #0A0F2D;
        color: #E0E0E0;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #4A90E2;
    }
    
    .stButton>button {
        background-color: #1F4E79;
        color: white;
        border-radius: 5px;
    }
    
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #1A1F3B;
        color: white;
        border: 1px solid #2D6BA1;
    }
    
    .stMetric {
        background-color: #13274F;
        border-radius: 5px;
        padding: 10px;
        border-left: 4px solid #4A90E2;
    }
    
    .stDataFrame {
        border-radius: 5px;
    }
    
    .streamlit-expanderHeader {
        background-color: #13274F;
        border-radius: 5px;
        padding: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for caching
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = {}
if 'currency' not in st.session_state:
    st.session_state.currency = 'USD'

# Sidebar menu without image for faster loading
with st.sidebar:
    st.title("MarketMentor")
    
    # User preferences
    with st.expander("Preferences", expanded=False):
        st.session_state.currency = st.selectbox("Currency", ["USD", "INR", "EUR", "GBP"], index=0)
        
    # Navigation menu
    selected = option_menu(
        "Navigation",
        ["Dashboard", "Company Analysis", "Market Analysis", "Global Markets", 
         "Mutual Funds", "SIP Calculator", "IPO Tracker", "Sectors", 
         "News", "Learning", "Watchlist", "Stock Screener"],
        icons=['speedometer2', 'building', 'graph-up', 'globe', 
               'bank', 'calculator', 'megaphone', 'grid-3x3', 
               'newspaper', 'book', 'star', 'search'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#13274F"},
            "icon": {"color": "#4A90E2", "font-size": "16px"}, 
            "nav-link": {"color": "#E0E0E0", "font-size": "14px"},
            "nav-link-selected": {"background-color": "#4A90E2"},
        }
    )

# Helper functions with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_currency_symbol(currency_code):
    symbols = {
        'USD': '$',
        'INR': '₹',
        'EUR': '€',
        'GBP': '£'
    }
    return symbols.get(currency_code, '$')

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(ticker, period="1mo"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_indices_data():
    indices = {
        "^GSPC": {"name": "S&P 500", "currency": "USD"},
        "^DJI": {"name": "Dow Jones", "currency": "USD"},
        "^IXIC": {"name": "NASDAQ", "currency": "USD"},
        "^NSEI": {"name": "Nifty 50", "currency": "INR"},
        "^BSESN": {"name": "Sensex", "currency": "INR"},
    }
    
    results = {}
    for symbol, data in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                last_close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = last_close - prev_close
                percent_change = (change / prev_close) * 100
                
                results[symbol] = {
                    "name": data["name"],
                    "price": last_close,
                    "change": percent_change,
                    "currency": data["currency"]
                }
        except:
            continue
    
    return results

# Dashboard Page
if selected == "Dashboard":
    st.title("Market Dashboard")
    
    # Market indices with cached data
    st.subheader("Global Indices")
    indices_data = get_indices_data()
    
    cols = st.columns(4)
    for idx, (symbol, data) in enumerate(indices_data.items()):
        with cols[idx % 4]:
            st.metric(
                label=data["name"],
                value=f"{get_currency_symbol(data['currency'])}{data['price']:.2f}",
                delta=f"{data['change']:.2f}%"
            )
    
    # Simple market overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sector Performance")
        sectors = {
            "Technology": "+2.3%",
            "Healthcare": "+1.5%",
            "Financials": "-0.8%",
            "Energy": "+3.2%",
        }
        
        for sector, perf in sectors.items():
            st.write(f"{sector}: {perf}")
    
    with col2:
        st.subheader("Market Sentiment")
        sentiment_data = {
            "Bullish": 45,
            "Neutral": 30,
            "Bearish": 25
        }
        
        # Simple bar chart instead of pie chart for faster rendering
        fig = px.bar(x=list(sentiment_data.keys()), y=list(sentiment_data.values()),
                     title="Market Sentiment", color=list(sentiment_data.keys()),
                     color_discrete_map={"Bullish": "#00CC96", "Neutral": "#FFA15A", "Bearish": "#EF553B"})
        st.plotly_chart(fig, use_container_width=True)

# Company Analysis Page
elif selected == "Company Analysis":
    st.title("Company Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input("Enter Stock Symbol", "AAPL")
    
    with col2:
        analysis_type = st.selectbox("Analysis Type", ["Overview", "Technical"])
    
    if ticker:
        hist, info = fetch_stock_data(ticker)
        
        if hist is not None and info is not None:
            currency_symbol = get_currency_symbol(st.session_state.currency)
            
            if analysis_type == "Overview":
                st.subheader(f"{info.get('longName', 'N/A')} ({ticker.upper()})")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = info.get('regularMarketPrice', hist['Close'].iloc[-1] if not hist.empty else 'N/A')
                previous_close = info.get('regularMarketPreviousClose', 'N/A')
                market_cap = info.get('marketCap', 'N/A')
                pe_ratio = info.get('trailingPE', 'N/A')
                
                with col1:
                    st.metric("Current Price", f"{currency_symbol}{current_price:.2f}" if isinstance(current_price, float) else current_price)
                with col2:
                    if isinstance(previous_close, float) and isinstance(current_price, float):
                        change = current_price - previous_close
                        change_percent = (change / previous_close) * 100
                        st.metric("Previous Close", f"{currency_symbol}{previous_close:.2f}", 
                                 f"{change:.2f} ({change_percent:.2f}%)")
                with col3:
                    st.metric("Market Cap", f"{currency_symbol}{market_cap/1e9:.2f}B" if isinstance(market_cap, (int, float)) else market_cap)
                with col4:
                    st.metric("P/E Ratio", f"{pe_ratio:.2f}" if isinstance(pe_ratio, float) else pe_ratio)
                
                # Price chart
                st.subheader("Price Chart")
                fig = px.line(hist, x=hist.index, y='Close', title=f"{ticker.upper()} Price History")
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Technical":
                st.subheader("Technical Indicators")
                
                # Calculate simple technical indicators
                hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                
                # Select indicator to display
                indicator = st.selectbox("Select Indicator", ["Moving Averages", "RSI"])
                
                if indicator == "Moving Averages":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='#4A90E2')))
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name='SMA 20', line=dict(color='#FFA15A')))
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50', line=dict(color='#00CC96')))
                    fig.update_layout(title="Moving Averages")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif indicator == "RSI":
                    # Calculate RSI
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    hist['RSI'] = 100 - (100 / (1 + rs))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI', line=dict(color='#4A90E2')))
                    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig.update_layout(title="Relative Strength Index (RSI)")
                    st.plotly_chart(fig, use_container_width=True)

# Market Analysis Page
elif selected == "Market Analysis":
    st.title("Market Analysis")
    
    analysis_tab, correlation_tab = st.tabs(["Sector Analysis", "Correlation"])
    
    with analysis_tab:
        st.subheader("Sector Performance")
        
        sectors = {
            "Technology": ["AAPL", "MSFT", "GOOGL"],
            "Healthcare": ["JNJ", "PFE", "UNH"],
            "Financials": ["JPM", "BAC", "V"],
        }
        
        selected_sector = st.selectbox("Select Sector", list(sectors.keys()))
        
        if selected_sector:
            st.write(f"Top stocks in {selected_sector} sector:")
            
            cols = st.columns(len(sectors[selected_sector]))
            performance_data = []
            
            for idx, ticker in enumerate(sectors[selected_sector]):
                with cols[idx]:
                    try:
                        hist, info = fetch_stock_data(ticker, "1mo")
                        if hist is not None and not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                            prev_price = hist['Close'].iloc[0]
                            change = ((current_price - prev_price) / prev_price) * 100
                            
                            st.metric(ticker, f"${current_price:.2f}", f"{change:.2f}%")
                            performance_data.append({
                                "Ticker": ticker,
                                "Performance": change
                            })
                    except:
                        st.error(f"Error loading {ticker}")
    
    with correlation_tab:
        st.subheader("Correlation Matrix")
        
        # Fetch data for multiple stocks
        correlation_tickers = st.multiselect("Select stocks for correlation", 
                                            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                                            default=["AAPL", "MSFT", "GOOGL"])
        
        if len(correlation_tickers) >= 2:
            correlation_data = {}
            
            for ticker in correlation_tickers:
                try:
                    hist, info = fetch_stock_data(ticker, "3mo")
                    if hist is not None and not hist.empty:
                        correlation_data[ticker] = hist['Close'].pct_change().dropna()
                except:
                    st.error(f"Error loading data for {ticker}")
            
            if correlation_data:
                corr_df = pd.DataFrame(correlation_data)
                correlation_matrix = corr_df.corr()
                
                fig = px.imshow(correlation_matrix, 
                                color_continuous_scale=px.colors.diverging.RdBu_r,
                                aspect="auto",
                                title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)

# Global Markets Page
elif selected == "Global Markets":
    st.title("Global Markets")
    
    regions = st.selectbox("Select Region", ["Americas", "Europe", "Asia-Pacific"])
    
    if regions == "Americas":
        indices = {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones",
            "^IXIC": "NASDAQ",
        }
    elif regions == "Europe":
        indices = {
            "^FTSE": "FTSE 100",
            "^GDAXI": "DAX",
            "^FCHI": "CAC 40",
        }
    else:
        indices = {
            "^N225": "Nikkei 225",
            "^HSI": "Hang Seng",
            "000001.SS": "Shanghai Composite",
        }
    
    cols = st.columns(3)
    
    for idx, (symbol, name) in enumerate(indices.items()):
        try:
            hist, info = fetch_stock_data(symbol, "2d")
            
            if hist is not None and len(hist) >= 2:
                last_close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = last_close - prev_close
                percent_change = (change / prev_close) * 100
                
                with cols[idx % 3]:
                    currency = 'USD' if regions != 'Asia-Pacific' else '¥'
                    st.metric(
                        label=name,
                        value=f"{get_currency_symbol(currency)}{last_close:.2f}",
                        delta=f"{percent_change:.2f}%"
                    )
        except:
            pass

# Additional simplified sections
elif selected == "Mutual Funds":
    st.title("Mutual Funds")
    st.info("Mutual fund data loading...")

elif selected == "SIP Calculator":
    st.title("SIP Calculator")
    
    monthly_investment = st.number_input("Monthly Investment", value=5000)
    years = st.slider("Investment Duration (Years)", 1, 30, 10)
    expected_return = st.slider("Expected Annual Return (%)", 1, 25, 12)
    
    months = years * 12
    monthly_rate = expected_return / 12 / 100
    
    future_value = monthly_investment * (((1 + monthly_rate)**months - 1) * (1 + monthly_rate)) / monthly_rate
    invested = monthly_investment * months
    gain = future_value - invested
    
    st.metric("Future Value", f"₹{future_value:,.2f}")
    st.metric("Total Invested", f"₹{invested:,.2f}")
    st.metric("Estimated Gain", f"₹{gain:,.2f}")

elif selected == "News":
    st.title("Financial News")
    
    news_query = st.text_input("Search News:", "stock market")
    
    if news_query:
        try:
            url = f"https://newsapi.org/v2/everything?q={news_query}&apiKey={NEWS_API_KEY}&language=en&pageSize=5"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                for article in articles:
                    with st.expander(f"{article['title']} - {article['source']['name']}"):
                        st.write(article.get("description", "No description available"))
                        st.markdown(f"[Read more]({article['url']})")
            else:
                st.info("News feed temporarily unavailable.")
        except:
            st.info("News feed temporarily unavailable.")

# Add other sections with similar optimizations
elif selected == "IPO Tracker":
    st.title("IPO Tracker")
    st.info("IPO data loading...")

elif selected == "Sectors":
    st.title("Sectors")
    st.info("Sector data loading...")

elif selected == "Learning":
    st.title("Learning Center")
    st.info("Educational content loading...")

elif selected == "Watchlist":
    st.title("Watchlist")
    st.info("Watchlist functionality loading...")

elif selected == "Stock Screener":
    st.title("Stock Screener")
    st.info("Screener functionality loading...")

# Add a footer
st.markdown("---")
st.markdown("MarketMentor - Simplified Financial Analytics | Data may be delayed")
