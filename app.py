import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from plotly import graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import talib
import warnings
warnings.filterwarnings('ignore')

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Set page config with optimized settings
st.set_page_config(
    page_title="MarketMentor - Financial Analytics", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply optimized dark theme CSS
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #0A0F2D;
        color: #E0E0E0;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #4A90E2 !important;
        font-weight: 600;
        padding-bottom: 8px;
        border-bottom: 1px solid #1A1F3B;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #13274F !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1A1F3B;
        color: white;
        border: 1px solid #2D6BA1;
        border-radius: 4px;
        padding: 8px 16px;
    }
    
    /* Input widgets */
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stSlider>div>div>div>div {
        background-color: #1A1F3B;
        color: white;
        border: 1px solid #2D6BA1;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #13274F;
        border-radius: 5px;
        padding: 10px;
        border-left: 3px solid #4A90E2;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #13274F;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #13274F;
        border-radius: 4px;
        padding: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for caching
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = {}

# Helper functions with caching
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker, period="1mo"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_global_indices():
    indices = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^NSEI": "Nifty 50",
        "^BSESN": "Sensex",
        "^FTSE": "FTSE 100",
        "^GDAXI": "DAX",
        "^FCHI": "CAC 40",
        "^N225": "Nikkei 225",
        "^HSI": "Hang Seng"
    }
    
    results = {}
    for symbol, name in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                last_close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = last_close - prev_close
                percent_change = (change / prev_close) * 100
                results[symbol] = {
                    "name": name,
                    "price": last_close,
                    "change": percent_change
                }
        except:
            continue
    
    return results

@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(query="stock market"):
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize=5"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json().get("articles", [])
        return []
    except:
        return []

# Sidebar menu without images
with st.sidebar:
    selected = option_menu(
        "MarketMentor",
        ["Dashboard", "Company Analysis", "Market Analysis", "Global Markets", 
         "Mutual Funds", "SIP Calculator", "IPO Tracker", "Predictions", 
         "Sectors", "News", "Learning"],
        icons=['speedometer2', 'building', 'graph-up', 'globe', 'bank', 
               'calculator', 'megaphone', 'robot', 'grid-3x3', 'newspaper', 'book'],
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
        if idx < 8:  # Limit to 8 indices for performance
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
            "Technology": "+2.3%",
            "Healthcare": "+1.5%",
            "Financials": "-0.8%",
            "Energy": "+3.2%",
            "Consumer Cyclical": "+0.7%"
        }
        
        sector_df = pd.DataFrame({
            "Sector": list(sectors.keys()),
            "Performance": [float(x.strip('%')) for x in sectors.values()]
        })
        
        fig = px.bar(sector_df, x="Performance", y="Sector", orientation='h',
                     title="Sector Performance (%)", color="Performance",
                     color_continuous_scale=px.colors.sequential.Blues_r)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Market Sentiment")
        sentiment_data = {
            "Bullish": 45,
            "Neutral": 30,
            "Bearish": 25
        }
        
        fig = px.pie(values=list(sentiment_data.values()), 
                     names=list(sentiment_data.keys()),
                     title="Market Sentiment Distribution",
                     color=list(sentiment_data.keys()),
                     color_discrete_map={"Bullish": "#00CC96", "Neutral": "#FFA15A", "Bearish": "#EF553B"})
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent news with caching
    st.subheader("📰 Market News")
    news_articles = fetch_news()
    
    if news_articles:
        for article in news_articles[:3]:  # Limit to 3 articles for performance
            with st.expander(f"{article['title']} - {article['source']['name']}"):
                st.write(article.get('description', 'No description available'))
                st.markdown(f"[Read more]({article['url']})")
    else:
        st.info("News feed temporarily unavailable.")

# Company Analysis Page
elif selected == "Company Analysis":
    st.title("🏢 Company Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input("🔍 Enter Stock Symbol", "AAPL")
    
    with col2:
        analysis_type = st.selectbox("Analysis Type", ["Overview", "Technical"])
    
    if ticker:
        # Use cached data fetching
        hist, info = fetch_stock_data(ticker)
        
        if hist is not None and info is not None:
            currency_symbol = '$'
            
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
                    else:
                        st.metric("Previous Close", f"{currency_symbol}{previous_close:.2f}" if isinstance(previous_close, float) else previous_close)
                with col3:
                    if isinstance(market_cap, (int, float)):
                        if market_cap >= 1e12:
                            st.metric("Market Cap", f"{currency_symbol}{market_cap/1e12:.2f}T")
                        elif market_cap >= 1e9:
                            st.metric("Market Cap", f"{currency_symbol}{market_cap/1e9:.2f}B")
                        else:
                            st.metric("Market Cap", f"{currency_symbol}{market_cap:,.2f}")
                    else:
                        st.metric("Market Cap", market_cap)
                with col4:
                    st.metric("P/E Ratio", f"{pe_ratio:.2f}" if isinstance(pe_ratio, float) else pe_ratio)
                
                # Price chart
                st.subheader("Price Chart")
                fig = px.line(hist, x=hist.index, y='Close', title=f"{ticker.upper()} Price History")
                st.plotly_chart(fig, use_container_width=True)
                
                # Company info
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("Company Details"):
                        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                        st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
                        st.write(f"**Country:** {info.get('country', 'N/A')}")
                
                with col2:
                    with st.expander("Trading Information"):
                        st.write(f"**52W High:** {currency_symbol}{info.get('fiftyTwoWeekHigh', 'N/A')}")
                        st.write(f"**52W Low:** {currency_symbol}{info.get('fiftyTwoWeekLow', 'N/A')}")
                        st.write(f"**Volume (Avg):** {info.get('averageVolume', 'N/A'):,}")
                        st.write(f"**Beta:** {info.get('beta', 'N/A')}")
            
            elif analysis_type == "Technical":
                st.subheader("Technical Analysis")
                
                # Calculate technical indicators
                if not hist.empty:
                    hist['SMA_20'] = talib.SMA(hist['Close'], timeperiod=20)
                    hist['SMA_50'] = talib.SMA(hist['Close'], timeperiod=50)
                    hist['RSI'] = talib.RSI(hist['Close'], timeperiod=14)
                
                # Select indicator to display
                indicator = st.selectbox("Select Indicator", 
                                        ["Moving Averages", "RSI"])
                
                if indicator == "Moving Averages":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='#4A90E2')))
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name='SMA 20', line=dict(color='#FFA15A')))
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50', line=dict(color='#00CC96')))
                    fig.update_layout(title="Moving Averages")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("Shorter-term moving averages crossing above longer-term ones may indicate bullish trends, and vice versa.")
                
                elif indicator == "RSI":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI', line=dict(color='#4A90E2')))
                    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig.update_layout(title="Relative Strength Index (RSI)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("RSI above 70 indicates overbought conditions, while below 30 indicates oversold conditions.")
        
        else:
            st.error("Unable to fetch data for the provided ticker symbol.")

# Market Analysis Page
elif selected == "Market Analysis":
    st.title("📈 Market Analysis")
    
    analysis_tab, correlation_tab = st.tabs(["Sector Analysis", "Correlation Matrix"])
    
    with analysis_tab:
        st.subheader("Sector Performance Analysis")
        
        sectors = {
            "Technology": ["AAPL", "MSFT", "GOOGL"],
            "Healthcare": ["JNJ", "PFE", "UNH"],
            "Financials": ["JPM", "BAC", "V"],
            "Energy": ["XOM", "CVX", "COP"],
            "Consumer Cyclical": ["AMZN", "TSLA", "HD"]
        }
        
        selected_sector = st.selectbox("Select Sector", list(sectors.keys()))
        
        if selected_sector:
            st.write(f"Top stocks in {selected_sector} sector:")
            
            cols = st.columns(len(sectors[selected_sector]))
            performance_data = []
            
            for idx, ticker in enumerate(sectors[selected_sector]):
                with cols[idx]:
                    try:
                        # Use cached data fetching
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
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                fig = px.bar(perf_df, x='Ticker', y='Performance', 
                             title=f"Performance of {selected_sector} Stocks",
                             color='Performance', color_continuous_scale=px.colors.sequential.Blues_r)
                st.plotly_chart(fig, use_container_width=True)
    
    with correlation_tab:
        st.subheader("Correlation Matrix")
        
        # Fetch data for multiple stocks
        correlation_tickers = st.multiselect("Select stocks for correlation", 
                                            ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"],
                                            default=["AAPL", "MSFT", "GOOGL"])
        
        if len(correlation_tickers) >= 2:
            correlation_data = {}
            
            for ticker in correlation_tickers:
                try:
                    # Use cached data fetching
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
                                title="Correlation Matrix of Selected Stocks")
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("Correlation values range from -1 (perfect negative correlation) to +1 (perfect positive correlation).")

# Global Markets Page
elif selected == "Global Markets":
    st.title("🌍 Global Markets")
    
    regions = st.selectbox("Select Region", ["Americas", "Europe", "Asia-Pacific"])
    
    # Use cached data
    indices_data = fetch_global_indices()
    
    if regions == "Americas":
        filtered_indices = {k: v for k, v in indices_data.items() if k in ["^GSPC", "^DJI", "^IXIC"]}
    elif regions == "Europe":
        filtered_indices = {k: v for k, v in indices_data.items() if k in ["^FTSE", "^GDAXI", "^FCHI"]}
    else:
        filtered_indices = {k: v for k, v in indices_data.items() if k in ["^N225", "^HSI"]}
    
    cols = st.columns(3)
    
    for idx, (symbol, data) in enumerate(filtered_indices.items()):
        with cols[idx % 3]:
            currency = '$' if symbol not in ["^N225", "^HSI"] else '¥'
            st.metric(
                label=data["name"],
                value=f"{currency}{data['price']:.2f}",
                delta=f"{data['change']:.2f}%"
            )
    
    # Global market performance chart
    if filtered_indices:
        index_df = pd.DataFrame([
            {"Index": data["name"], "Change": data["change"]} 
            for data in filtered_indices.values()
        ])
        fig = px.bar(index_df, x="Index", y="Change", title=f"{regions} Market Performance",
                     color="Change", color_continuous_scale=px.colors.diverging.RdYlGn)
        st.plotly_chart(fig, use_container_width=True)

# Mutual Funds Page
elif selected == "Mutual Funds":
    st.title("🏦 Mutual Funds Insights")
    
    # Simulated mutual fund data
    mf_data = {
        "Fund Name": ["Axis Bluechip Fund", "Mirae Asset Large Cap", "Parag Parikh Flexi Cap", "UTI Nifty Index"],
        "1Y Return": ["15.0%", "13.2%", "17.5%", "12.0%"],
        "3Y Return": ["12.5%", "11.8%", "15.2%", "10.5%"],
        "5Y Return": ["14.2%", "13.5%", "16.8%", "11.2%"],
        "Risk": ["Moderate", "Moderate", "High", "Low"]
    }
    
    mf_df = pd.DataFrame(mf_data)
    st.dataframe(mf_df, use_container_width=True)
    
    # Fund performance chart
    fig = px.bar(mf_df, x="Fund Name", y=[float(x.strip('%')) for x in mf_df["1Y Return"]],
                 title="1-Year Returns (%)", color="Fund Name",
                 color_discrete_sequence=px.colors.qualitative.Dark2)
    st.plotly_chart(fig, use_container_width=True)

# SIP Calculator Page
elif selected == "SIP Calculator":
    st.title("📈 SIP Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_investment = st.number_input("Monthly Investment (₹)", min_value=500, value=5000, step=500)
        years = st.slider("Investment Duration (Years)", 1, 30, 10)
    
    with col2:
        expected_return = st.slider("Expected Annual Return (%)", 1, 25, 12)
        inflation_rate = st.slider("Expected Inflation Rate (%)", 0, 10, 6)
    
    # Calculate SIP
    months = years * 12
    monthly_rate = expected_return / 12 / 100
    
    future_value = monthly_investment * (((1 + monthly_rate)**months - 1) * (1 + monthly_rate)) / monthly_rate
    invested = monthly_investment * months
    gain = future_value - invested
    
    # Adjust for inflation
    real_value = future_value / (1 + inflation_rate/100)**years
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💰 Future Value", f"₹{future_value:,.2f}")
    with col2:
        st.metric("📊 Amount Invested", f"₹{invested:,.2f}")
    with col3:
        st.metric("📈 Estimated Gains", f"₹{gain:,.2f}")
    
    st.info(f"**Inflation-adjusted value:** ₹{real_value:,.2f} (in today's purchasing power)")

# IPO Tracker Page
elif selected == "IPO Tracker":
    st.title("🆕 IPO Tracker")
    
    # Simulated IPO data
    ipo_data = {
        "Company": ["ABC Tech", "SmartFin Ltd", "GreenPower", "NetPay Corp"],
        "Issue Price (₹)": [100, 240, 150, 280],
        "Current Price (₹)": [145, 190, 170, 260],
        "Gain/Loss (%)": [45, -20.8, 13.3, -7.1],
        "Listing Date": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05"]
    }
    
    ipo_df = pd.DataFrame(ipo_data)
    st.dataframe(ipo_df, use_container_width=True)
    
    # IPO performance chart
    fig = px.bar(ipo_df, x="Company", y="Gain/Loss (%)", 
                 title="IPO Performance", color="Gain/Loss (%)",
                 color_continuous_scale=px.colors.diverging.RdYlGn)
    st.plotly_chart(fig, use_container_width=True)

# Predictions Page
elif selected == "Predictions":
    st.title("🔮 Market Predictions")
    
    st.subheader("Stock Price Forecast")
    ticker = st.text_input("Enter Stock Ticker for Prediction", "AAPL")
    
    if ticker:
        # Simulated prediction data
        hist, info = fetch_stock_data(ticker, "3mo")
        
        if hist is not None and not hist.empty:
            last_date = hist.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
            # Simple prediction based on recent trend
            recent_trend = np.mean(hist['Close'][-5:]) - np.mean(hist['Close'][-10:-5])
            base_price = hist['Close'].iloc[-1]
            predicted_prices = [base_price + (i * recent_trend / 30) for i in range(1, 31)]
            
            # Create prediction chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Historical', line=dict(color='#4A90E2')))
            fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices, name='Predicted', line=dict(color='#FFA15A', dash='dash')))
            fig.update_layout(title=f"{ticker} Price Prediction")
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("Note: This is a simplified prediction for demonstration purposes. Always consult with financial advisors before making investment decisions.")

# Sectors Page
elif selected == "Sectors":
    st.title("📊 Sector Wise Performance")
    
    sector_performance = {
        "Banking": "+1.8%",
        "IT": "-0.5%",
        "Energy": "+2.1%",
        "FMCG": "+0.9%",
        "Pharma": "-1.2%",
        "Auto": "+1.0%",
    }
    
    sector_df = pd.DataFrame({
        "Sector": list(sector_performance.keys()),
        "Performance": [float(x.strip('%')) for x in sector_performance.values()]
    })
    
    fig = px.bar(sector_df, x="Sector", y="Performance", 
                 title="Sector Performance", color="Performance",
                 color_continuous_scale=px.colors.diverging.RdYlGn)
    st.plotly_chart(fig, use_container_width=True)

# News Page
elif selected == "News":
    st.title("📰 Financial News")
    
    news_query = st.text_input("Search Financial News:", "stock market")
    
    if news_query:
        # Use cached news fetching
        news_articles = fetch_news(news_query)
        
        if news_articles:
            for article in news_articles:
                with st.expander(f"{article['title']} - {article['source']['name']}"):
                    st.write(article.get('description', 'No description available'))
                    st.markdown(f"[Read more]({article['url']})")
        else:
            st.info("No articles found or news feed temporarily unavailable.")

# Learning Page
elif selected == "Learning":
    st.title("📘 Stock Market Learning Center")
    
    with st.expander("Beginner's Guide to Stock Market Investing"):
        st.markdown("""
        ### 1. Understanding the Basics
        - What are stocks and how do they work?
        - Different types of investments: stocks, bonds, mutual funds
        - Risk vs. return: finding your investment style
        
        ### 2. Fundamental Analysis
        - How to read financial statements
        - Key financial ratios: P/E, PEG, ROE, Debt-to-Equity
        - Evaluating company management and competitive advantage
        
        ### 3. Technical Analysis
        - Reading stock charts: candlestick patterns
        - Important technical indicators: Moving Averages, RSI, MACD
        - Support and resistance levels
        """)
    
    with st.expander("Investment Strategies"):
        st.markdown("""
        ### Value Investing
        Finding undervalued stocks with strong fundamentals
        
        ### Growth Investing
        Identifying companies with high growth potential
        
        ### Dividend Investing
        Building passive income through dividend-paying stocks
        """)
    
    with st.expander("Risk Management"):
        st.markdown("""
        ### Diversification
        Don't put all your eggs in one basket
        
        ### Position Sizing
        How much to invest in each stock
        
        ### Stop-Loss Orders
        Protecting your capital from significant losses
        """)
