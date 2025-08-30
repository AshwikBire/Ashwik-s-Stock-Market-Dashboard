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
warnings.filterwarnings('ignore')

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Set page config
st.set_page_config(page_title="MarketMentor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .positive-change {
        color: green;
        font-weight: bold;
    }
    .negative-change {
        color: red;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "MarketMentor",
        ["Home", "Company Overview", "Market Movers", "F&O", "Global Markets", "Mutual Funds", "SIP Calculator", 
         "IPO Tracker", "Predictions for Mutual Funds & IPOs", "Mutual Fund NAV Viewer", "Sectors", "News", 
         "Learning", "Volume Spike", "Stock Screener", "Predictions", "Buy/Sell Predictor", "News Sentiment", 
         "Technical Analysis", "Portfolio Tracker"],
        icons=['house', 'building', 'graph-up', 'arrow-left-right', 'globe', 'bank', 'calculator', 'rocket', 
               'graph-up-arrow', 'bar-chart', 'grid-3x3', 'newspaper', 'book', 'activity', 'search', 'lightbulb', 
               'currency-exchange', 'chat-quote', 'speedometer', 'briefcase'],
        menu_icon="cast",
        default_index=0
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
            st.metric(label=name, value=f"{data['last_close']}", delta=f"{data['percent_change']}%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Market sentiment
    st.subheader("📈 Market Sentiment")
    sentiment_cols = st.columns(3)
    with sentiment_cols[0]:
        st.metric("Advancers", "1,245", "52%")
    with sentiment_cols[1]:
        st.metric("Decliners", "987", "41%")
    with sentiment_cols[2]:
        st.metric("Unchanged", "168", "7%")
    
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

# Market Movers - Top Gainers & Losers
elif selected == "Market Movers":
    st.title("📈 Market Movers - Active Stocks, Top Gainers & Losers")
    
    # Add time period selector
    period = st.selectbox("Select Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=0)
    
    # Predefined list of popular Indian stocks
    popular_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
                     'HINDUNILVR.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS']
    
    # Fetch data
    data = {}
    progress_bar = st.progress(0)
    for i, ticker in enumerate(popular_stocks):
        try:
            hist = get_stock_data(ticker, period)
            if not hist.empty:
                initial_price = hist['Close'].iloc[0]
                current_price = hist['Close'].iloc[-1]
                change = current_price - initial_price
                percent_change = (change / initial_price) * 100
                data[ticker] = {
                    'current_price': current_price,
                    'change': change,
                    'percent_change': percent_change
                }
            progress_bar.progress((i + 1) / len(popular_stocks))
        except:
            continue
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    df.reset_index(inplace=True)
    df.columns = ['Stock', 'Current Price', 'Change', 'Percent Change']
    
    # Display gainers and losers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Top Gainers")
        gainers = df.nlargest(5, 'Percent Change')
        for _, row in gainers.iterrows():
            st.metric(row['Stock'], f"₹{row['Current Price']:.2f}", 
                     f"{row['Percent Change']:.2f}%")
    
    with col2:
        st.subheader("📉 Top Losers")
        losers = df.nsmallest(5, 'Percent Change')
        for _, row in losers.iterrows():
            st.metric(row['Stock'], f"₹{row['Current Price']:.2f}", 
                     f"{row['Percent Change']:.2f}%")
    
    # Display all stocks
    st.subheader("📊 All Tracked Stocks")
    st.dataframe(df.sort_values('Percent Change', ascending=False).reset_index(drop=True))

# Global Markets - Major Indices
elif selected == "Global Markets":
    st.title("🌍 Global Markets Status")
    
    global_data = get_global_markets()
    cols = st.columns(4)
    for idx, (name, data) in enumerate(global_data.items()):
        with cols[idx % 4]:
            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
            if data['last_close'] != 'N/A':
                st.metric(label=name, value=f"{data['last_close']}", 
                         delta=f"{data['percent_change']}%")
            else:
                st.metric(label=name, value="N/A", delta="N/A")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Currency rates
    st.subheader("💱 Currency Exchange Rates")
    forex_cols = st.columns(4)
    forex_rates = {
        "USD/INR": 83.25,
        "EUR/INR": 89.50,
        "GBP/INR": 105.75,
        "JPY/INR": 0.55
    }
    for idx, (pair, rate) in enumerate(forex_rates.items()):
        with forex_cols[idx]:
            st.metric(pair, f"{rate:.2f}")

# Mutual Funds - Insights
elif selected == "Mutual Funds":
    st.title("🏦 Mutual Funds Insights")
    
    # Add category filter
    category = st.selectbox("Select Category", 
                           ["All", "Large Cap", "Mid Cap", "Small Cap", "Sectoral", "ELSS"])
    
    # Sample mutual fund data
    mf_data = {
        "Fund Name": ["Axis Bluechip Fund", "Mirae Asset Large Cap Fund", 
                     "Parag Parikh Flexi Cap Fund", "UTI Nifty Index Fund",
                     "SBI Small Cap Fund", "HDFC Mid Cap Opportunities"],
        "Category": ["Large Cap", "Large Cap", "Flexi Cap", "Index", "Small Cap", "Mid Cap"],
        "1Y Return": [15.0, 13.2, 17.5, 12.0, 24.3, 18.7],
        "3Y Return": [14.2, 12.8, 16.9, 11.5, 26.4, 17.2],
        "Risk": ["Moderate", "Moderate", "Moderately High", "Low", "High", "Moderately High"],
        "Rating": [4.5, 4.3, 4.7, 4.0, 4.8, 4.4]
    }
    
    mf_df = pd.DataFrame(mf_data)
    
    # Filter by category
    if category != "All":
        mf_df = mf_df[mf_df["Category"] == category]
    
    # Display mutual funds
    st.dataframe(mf_df)
    
    # Performance chart
    st.subheader("📈 Category-wise Average Returns")
    category_avg = mf_df.groupby("Category")["1Y Return"].mean().reset_index()
    fig = px.bar(category_avg, x="Category", y="1Y Return", 
                 title="1 Year Average Returns by Category")
    st.plotly_chart(fig)

# Sectors - Sector Performance
elif selected == "Sectors":
    st.title("📊 Sector Wise Performance")
    
    # Sector data
    sector_data = {
        "Sector": ["Banking", "IT", "Energy", "FMCG", "Pharma", "Auto", "Real Estate", "Metals"],
        "Performance": [1.8, -0.5, 2.1, 0.9, -1.2, 1.0, 2.5, -0.8],
        "1M Change": [3.2, 1.5, 4.2, 2.1, -0.5, 2.8, 5.2, -2.1],
        "Outlook": ["Positive", "Neutral", "Positive", "Positive", "Negative", "Positive", "Positive", "Negative"]
    }
    
    sector_df = pd.DataFrame(sector_data)
    
    # Display sector performance
    st.dataframe(sector_df)
    
    # Performance chart
    fig = px.bar(sector_df, x="Sector", y="Performance", 
                 title="Sector Performance (%)", color="Performance",
                 color_continuous_scale="RdYlGn")
    st.plotly_chart(fig)
    
    # Sector outlook
    st.subheader("🔮 Sector Outlook")
    outlook_cols = st.columns(4)
    for idx, (_, row) in enumerate(sector_df.iterrows()):
        with outlook_cols[idx % 4]:
            if row["Outlook"] == "Positive":
                st.success(f"{row['Sector']} ✅")
            elif row["Outlook"] == "Negative":
                st.error(f"{row['Sector']} ❌")
            else:
                st.warning(f"{row['Sector']} ⏸️")

# News - Latest Financial News
elif selected == "News":
    st.title("📰 Latest Financial News")
    
    news_query = st.text_input("Search Financial News:", "stock market")
    date_filter = st.selectbox("Date Range", ["Last 24 hours", "Last week", "Last month"])
    
    if news_query:
        try:
            # Calculate date based on filter
            if date_filter == "Last 24 hours":
                from_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            elif date_filter == "Last week":
                from_date = (datetime.now() - timedelta(weeks=1)).strftime("%Y-%m-%d")
            else:
                from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                
            url = f"https://newsapi.org/v2/everything?q={news_query}&from={from_date}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize=10"
            response = requests.get(url)

            if response.status_code == 200:
                articles = response.json().get("articles", [])
                if articles:
                    for article in articles:
                        with st.container():
                            st.markdown("---")
                            st.subheader(article["title"])
                            st.write(f"*{article['source']['name']} - {article['publishedAt'].split('T')[0]}*")
                            st.write(article.get("description", "No description available."))
                            st.markdown(f"[🔗 Read More]({article['url']})")
                else:
                    st.warning("No articles found.")
            else:
                st.error("Unable to fetch news articles. Please check API or query.")
        except:
            st.info("News feature will be available after configuring News API")

# Learning - Stock Market Resources
elif selected == "Learning":
    st.title("📚 Learn the Stock Market")
    
    # Add topics selection
    topic = st.selectbox("Select a Topic", 
                        ["Basics of Stock Market", "Technical Analysis", "Fundamental Analysis", 
                         "Options Trading", "Mutual Funds", "IPOs", "Risk Management"])
    
    st.markdown(f"""
    Welcome to the **Learning Hub** of the Smart Stock Market Dashboard!  
    This section is crafted to help **beginners, enthusiasts, and investors** understand how the stock market works.
    
    ### 📖 Current Topic: {topic}
    """)
    
    # Display content based on selected topic
    if topic == "Basics of Stock Market":
        st.markdown("""
        - **What is a Stock?** Ownership in a company
        - **Stock Exchanges:** NSE, BSE, NYSE, NASDAQ
        - **How Trading Works:** Bid-Ask spread, order types
        - **Market Hours:** When markets open and close
        """)
    elif topic == "Technical Analysis":
        st.markdown("""
        - **Charts:** Line, Bar, Candlestick
        - **Indicators:** Moving Averages, RSI, MACD
        - **Patterns:** Head and Shoulders, Double Tops/Bottoms
        - **Support and Resistance:** Key price levels
        """)
    elif topic == "Fundamental Analysis":
        st.markdown("""
        - **Financial Statements:** Balance Sheet, Income Statement, Cash Flow
        - **Ratios:** P/E, P/B, Debt-to-Equity, ROE
        - **Valuation Methods:** DCF, Comparable Analysis
        - **Economic Indicators:** GDP, Inflation, Interest Rates
        """)
    
    # Resources section
    st.markdown("""
    ### 🔗 Additional Resources
    - [Investopedia](https://www.investopedia.com/)
    - [NSE India](https://www.nseindia.com/)
    - [BSE India](https://www.bseindia.com/)
    - [SEC EDGAR Database](https://www.sec.gov/edgar.shtml)
    """)

# Volume Spike Detector
elif selected == "Volume Spike":
    st.title("📈 Volume Spike Detector")
    st.markdown("This tool detects unusual volume surges in a stock based on a 10-day rolling average.")

    ticker = st.text_input("🔎 Enter Stock Ticker (e.g., TCS.NS, INFY.NS):", "TCS.NS")
    days = st.slider("🗓️ Select Days of Historical Data:", 30, 365, 90)

    if ticker:
        try:
            # Download historical stock data
            data = yf.download(ticker, period=f"{days}d")

            if data.empty:
                st.warning("⚠️ No data found. Please check the ticker symbol.")
            else:
                # Compute rolling average & spike detection
                data["Avg_Volume"] = data["Volume"].rolling(window=10).mean()
                data["Spike"] = data["Volume"] > (1.5 * data["Avg_Volume"])
                data.dropna(inplace=True)

                # --- Chart Section ---
                st.subheader("📊 Volume Trend with Spike Detection")
                fig = go.Figure()

                # Volume line
                fig.add_trace(go.Scatter(
                    x=data.index, y=data["Volume"],
                    mode='lines', name='Daily Volume',
                    line=dict(color='royalblue')
                ))

                # 10-Day Avg Volume line
                fig.add_trace(go.Scatter(
                    x=data.index, y=data["Avg_Volume"],
                    mode='lines', name='10-Day Avg Volume',
                    line=dict(color='orange')
                ))

                # Volume spikes
                spikes = data[data["Spike"]]
                fig.add_trace(go.Scatter(
                    x=spikes.index, y=spikes["Volume"],
                    mode='markers', name='Spikes',
                    marker=dict(size=10, color='red', symbol='star')
                ))

                fig.update_layout(
                    title=f"🔍 Volume Spike Detection for {ticker.upper()}",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    legend_title="Legend",
                    template="plotly_dark",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # --- Spike Events Table ---
                st.subheader("📌 Detected Volume Spike Events")
                st.dataframe(
                    spikes[["Volume", "Avg_Volume"]]
                    .rename(columns={"Volume": "Actual Volume", "Avg_Volume": "10-Day Avg"})
                    .style.format("{:,.0f}"),
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Error occurred: {e}")

# News Sentiment - Sentiment Analysis of News
elif selected == "News Sentiment":
    st.title("🔍 News Sentiment Analysis")
    ticker = st.text_input("Enter Stock Ticker to analyze news sentiment:", "AAPL")

    if ticker:
        st.info(f"Fetching and analyzing recent news sentiment for {ticker.upper()}...")
        try:
            url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}&language=en&pageSize=10"
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                sentiments = []
                for article in articles:
                    title = article["title"]
                    description = article.get("description", "")
                    text = f"{title}. {description}"
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    sentiments.append(polarity)
                    
                    with st.expander(title):
                        st.write(f"**Source:** {article['source']['name']}")
                        st.write(f"**Sentiment Score:** {round(polarity, 3)}")
                        st.write(description)
                        st.markdown(f"[Read full article]({article['url']})")

                if sentiments:
                    avg_sentiment = round(np.mean(sentiments), 3)
                    st.success(f"📊 **Average Sentiment Score** for {ticker.upper()}: {avg_sentiment}")
                    
                    # Sentiment gauge chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = avg_sentiment,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Overall Sentiment"},
                        delta = {'reference': 0},
                        gauge = {
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [-1, -0.2], 'color': "red"},
                                {'range': [-0.2, 0.2], 'color': "yellow"},
                                {'range': [0.2, 1], 'color': "green"}],
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to fetch news articles.")
        except:
            st.info("News sentiment analysis will be available after configuring News API")

# Predictions - Stock Price Prediction
elif selected == "Predictions":
    st.title("🔮 Stock Price Predictions")

    ticker = st.text_input("Enter Company Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")
    period = st.selectbox("Select Prediction Period", ["7 days", "30 days", "90 days"])

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

                # Calculate moving averages
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['SMA_200'] = hist['Close'].rolling(window=200).mean()

                st.subheader("📈 Moving Averages")
                st.line_chart(hist[['Close', 'SMA_50', 'SMA_200']])

                # Simple prediction using moving average crossover
                last_50_avg = hist['SMA_50'].iloc[-1]
                last_200_avg = hist['SMA_200'].iloc[-1]
                current_price = hist['Close'].iloc[-1]

                st.subheader("📋 Prediction Summary")
                if last_50_avg > last_200_avg:
                    st.success(f"📈 Bullish Trend Detected: 50-day SMA ({last_50_avg:.2f}) is above 200-day SMA ({last_200_avg:.2f})")
                    prediction = "UP"
                    confidence = 65
                else:
                    st.error(f"📉 Bearish Trend Detected: 50-day SMA ({last_50_avg:.2f}) is below 200-day SMA ({last_200_avg:.2f})")
                    prediction = "DOWN"
                    confidence = 60

                # Display prediction
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Direction", prediction)
                with col2:
                    st.metric("Confidence Level", f"{confidence}%")

        except Exception as e:
            st.error(f"Error retrieving data: {e}")

# Buy/Sell Predictor - Predict Buy or Sell Signal
elif selected == "Buy/Sell Predictor":
    st.title("💹 Buy/Sell Predictor")

    # Input: Ticker symbol
    ticker = st.text_input("Enter Company Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")

    if ticker:
        try:
            # Fetch stock data from Yahoo Finance
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")  # Fetch 1 year of data

            if hist.empty:
                st.warning("No data available for this ticker.")
            else:
                # Calculate technical indicators
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
                
                # Calculate RSI
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                hist['RSI'] = 100 - (100 / (1 + rs))

                # Current values
                current_price = hist['Close'].iloc[-1]
                current_rsi = hist['RSI'].iloc[-1]
                sma_50 = hist['SMA_50'].iloc[-1]
                sma_200 = hist['SMA_200'].iloc[-1]

                # Determine signal
                signal = "HOLD"
                reason = ""
                
                if sma_50 > sma_200 and current_rsi < 40:
                    signal = "BUY"
                    reason = "Bullish trend with oversold conditions"
                elif sma_50 < sma_200 and current_rsi > 60:
                    signal = "SELL"
                    reason = "Bearish trend with overbought conditions"
                else:
                    signal = "HOLD"
                    reason = "No clear trend or neutral conditions"

                # Display results
                st.subheader("📊 Technical Indicators")
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"₹{current_price:.2f}")
                col2.metric("RSI", f"{current_rsi:.2f}")
                col3.metric("50/200 SMA", f"₹{sma_50:.2f}/₹{sma_200:.2f}")

                st.subheader("🎯 Trading Signal")
                if signal == "BUY":
                    st.success(f"**{signal}** - {reason}")
                elif signal == "SELL":
                    st.error(f"**{signal}** - {reason}")
                else:
                    st.warning(f"**{signal}** - {reason}")

                # RSI chart
                st.subheader("📈 RSI Chart")
                st.line_chart(hist['RSI'])

        except Exception as e:
            st.error(f"Error retrieving data: {e}")

# Stock Screener
elif selected == "Stock Screener":
    st.title("📊 Stock Screener")
    
    # Add filters
    st.sidebar.header("🛠️ Screening Criteria")
    market_cap = st.sidebar.slider("Market Cap (Cr)", 0, 2000000, (1000, 100000))
    pe_ratio = st.sidebar.slider("P/E Ratio", 0.0, 100.0, (10.0, 30.0))
    dividend_yield = st.sidebar.slider("Dividend Yield (%)", 0.0, 10.0, (0.5, 5.0))
    
    # Predefined list of companies
    default_companies = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'HINDUNILVR.NS',
        'BAJAJFINSV.NS', 'HDFC.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'AXISBANK.NS', 'MARUTI.NS', 'LT.NS'
    ]
    
    # Display results
    st.subheader("📋 Screening Results")
    
    # Simulated screening results
    screened_data = {
        "Company": ["Reliance", "TCS", "HDFC Bank", "Infosys", "ICICI Bank"],
        "Ticker": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"],
        "Market Cap (Cr)": [1500000, 1200000, 800000, 600000, 500000],
        "P/E Ratio": [25.3, 30.1, 20.5, 22.8, 18.9],
        "Dividend Yield (%)": [0.8, 1.5, 1.2, 2.1, 1.8],
        "Sector": ["Energy", "IT", "Banking", "IT", "Banking"]
    }
    
    screened_df = pd.DataFrame(screened_data)
    st.dataframe(screened_df)
    
    # Visualization
    st.subheader("📊 Sector-wise P/E Ratio")
    fig = px.box(screened_df, x="Sector", y="P/E Ratio", title="P/E Ratio by Sector")
    st.plotly_chart(fig)

# Mutual Funds - NAV Viewer
elif selected == "Mutual Fund NAV Viewer":
    st.title("📈 Mutual Fund NAV Viewer")

    # Default scheme code for Axis Bluechip Fund
    scheme_code = st.text_input("Enter Mutual Fund Scheme Code (e.g. 118550)", "118550")

    if scheme_code:
        try:
            api_url = f"https://api.mfapi.in/mf/{scheme_code}"
            response = requests.get(api_url)

            if response.status_code == 200:
                nav_data = response.json()
                st.subheader(f"🔷 {nav_data['meta']['scheme_name']}")

                # Prepare NAV DataFrame
                nav_df = pd.DataFrame(nav_data['data'])
                nav_df['nav'] = nav_df['nav'].astype(float)
                nav_df['date'] = pd.to_datetime(nav_df['date'])
                nav_df = nav_df.sort_values(by='date', ascending=False)

                # Show latest NAV
                st.metric(label="📊 Latest NAV", value=f"₹{nav_df.iloc[0]['nav']}", delta=None)

                # Line Chart for NAV
                st.subheader("📉 NAV Trend (Last 30 Days)")
                st.line_chart(nav_df.set_index('date')['nav'].head(30).sort_index())

                # Show Data Table
                with st.expander("🔍 View All NAVs"):
                    st.dataframe(nav_df[['date', 'nav']].rename(columns={'date': 'Date', 'nav': 'NAV'}))

            else:
                st.error("⚠️ Failed to fetch mutual fund data. Please check the scheme code.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# SIP Calculator
elif selected == "SIP Calculator":
    st.title("📈 SIP Calculator")

    col1, col2 = st.columns(2)
    
    with col1:
        monthly_investment = st.number_input("Monthly Investment (₹)", value=5000, min_value=500, step=500)
        years = st.slider("Investment Duration (Years)", 1, 30, 10)
    
    with col2:
        expected_return = st.slider("Expected Annual Return (%)", 1, 25, 12)
        inflation_rate = st.slider("Expected Inflation Rate (%)", 1, 10, 6)

    months = years * 12
    monthly_rate = expected_return / 12 / 100

    future_value = monthly_investment * (((1 + monthly_rate)**months - 1) * (1 + monthly_rate)) / monthly_rate
    invested = monthly_investment * months
    gain = future_value - invested
    
    # Adjust for inflation
    real_value = future_value / (1 + inflation_rate/100)**years

    st.success(f"📊 Future Value: ₹{future_value:,.2f}")
    st.info(f"💰 Total Invested: ₹{invested:,.2f}")
    st.warning(f"📈 Estimated Gains: ₹{gain:,.2f}")
    st.error(f"🎯 Real Value (after inflation): ₹{real_value:,.2f}")
    
    # Yearly breakdown
    yearly_data = []
    for year in range(1, years + 1):
        yearly_fv = monthly_investment * 12 * (((1 + expected_return/100)**year - 1) / (expected_return/100)) * (1 + expected_return/100)
        yearly_data.append(yearly_fv)
    
    breakdown_df = pd.DataFrame({
        'Year': range(1, years + 1),
        'Value': yearly_data
    })
    
    st.subheader("📅 Year-wise Growth")
    st.line_chart(breakdown_df.set_index('Year'))

# IPO Tracker
elif selected == "IPO Tracker":
    st.title("🆕 IPO Tracker")
    
    # Add time filter
    ipo_status = st.selectbox("Filter by Status", ["All", "Upcoming", "Ongoing", "Completed"])
    
    # Sample IPO data
    ipo_data = pd.DataFrame({
        "Company": ["ABC Tech", "SmartFin Ltd", "GreenPower", "NetPay Corp", "HealthPlus"],
        "Issue Size (Cr)": [1200, 800, 1500, 600, 900],
        "Price Range": ["₹100-110", "₹240-250", "₹150-160", "₹280-290", "₹180-190"],
        "Listing Date": ["2023-06-15", "2023-07-01", "2023-05-20", "2023-08-10", "2023-09-05"],
        "Status": ["Completed", "Completed", "Completed", "Upcoming", "Ongoing"],
        "Gain/Loss (%)": [45, -20.8, 13.3, -7.1, "N/A"]
    })
    
    # Filter data
    if ipo_status != "All":
        ipo_data = ipo_data[ipo_data["Status"] == ipo_status]
    
    st.dataframe(ipo_data)
    
    # Visualization for completed IPOs
    if ipo_status in ["All", "Completed"]:
        completed_ipos = ipo_data[ipo_data["Status"] == "Completed"]
        if not completed_ipos.empty:
            st.subheader("📈 Performance of Completed IPOs")
            fig = px.bar(completed_ipos, x="Company", y="Gain/Loss (%)", 
                         title="IPO Performance (%)", color="Gain/Loss (%)",
                         color_continuous_scale="RdYlGn")
            st.plotly_chart(fig)

# Predictions for Mutual Funds & IPOs
elif selected == "Predictions for Mutual Funds & IPOs":
    st.title("🔮 Predictions for Mutual Funds & IPOs")
    
    tab1, tab2 = st.tabs(["Mutual Funds", "IPOs"])
    
    with tab1:
        st.subheader("📊 Mutual Fund NAV Forecast")
        
        # Simulated data
        dates = pd.date_range(start=pd.to_datetime("2023-01-01"), periods=12, freq='M')
        navs = np.linspace(100, 160, 12) + np.random.normal(0, 2, 12)
        
        nav_forecast = pd.DataFrame({'Month': dates, 'Predicted NAV': navs})
        nav_forecast.set_index("Month", inplace=True)
        st.line_chart(nav_forecast)
        
        # Fund recommendations
        st.subheader("🏆 Top Recommended Funds")
        recommended_funds = pd.DataFrame({
            "Fund": ["Axis Bluechip", "Mirae Asset Emerging", "Parag Parikh Flexi Cap"],
            "Category": ["Large Cap", "Mid Cap", "Flexi Cap"],
            "Expected Return (%)": [12.5, 15.2, 14.8],
            "Risk Level": ["Medium", "High", "Medium-High"]
        })
        st.dataframe(recommended_funds)
    
    with tab2:
        st.subheader("🚀 IPO Price Movement Prediction")
        
        # IPO predictions
        ipo_prediction = pd.DataFrame({
            "IPO": ["ABC Tech", "SmartFin Ltd", "GreenPower"],
            "Expected Listing Gain (%)": [20.5, -5.2, 12.7],
            "Confidence": ["High", "Medium", "High"],
            "Recommendation": ["Subscribe", "Avoid", "Subscribe"]
        })
        st.dataframe(ipo_prediction)
        
        # Subscription trends
        st.subheader("📊 Historical Subscription Rates")
        subscription_data = pd.DataFrame({
            "IPO": ["Tech IPO 1", "Tech IPO 2", "Finance IPO 1", "Finance IPO 2"],
            "Retail": [12.5, 8.7, 15.2, 6.8],
            "HNI": [24.8, 18.9, 32.5, 12.4],
            "QIB": [45.2, 32.7, 52.8, 28.9]
        })
        fig = px.bar(subscription_data, x="IPO", y=["Retail", "HNI", "QIB"], 
                     title="Subscription Rates by Category", barmode="group")
        st.plotly_chart(fig)

# F&O Page
elif selected == "F&O":
    st.title("📑 F&O Stocks - Live Overview")

    # Simulated F&O Data
    fo_data = {
        "Symbol": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "HINDUNILVR", "BAJFINANCE"],
        "LTP": [2820.5, 3480.7, 1463.2, 1640.0, 1103.5, 585.5, 2450.0, 7250.0],
        "Volume": [1250000, 850000, 650000, 920000, 870000, 1250000, 450000, 350000],
        "OI": [12500000, 8500000, 6500000, 9200000, 8700000, 12500000, 4500000, 3500000],
        "Change in OI": [12.5, -8.3, 5.7, -3.2, 7.8, 10.2, -5.5, 15.3],
        "Sector": ["Energy", "IT", "IT", "Banking", "Banking", "Banking", "FMCG", "Financial Services"]
    }

    df = pd.DataFrame(fo_data)

    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    sectors = st.sidebar.multiselect("Select Sector", df["Sector"].unique(), default=df["Sector"].unique())
    min_volume = st.sidebar.slider("Minimum Volume", 0, int(df["Volume"].max()), 500000)

    filtered_df = df[
        (df["Sector"].isin(sectors)) &
        (df["Volume"] >= min_volume)
    ]

    st.subheader("📊 Filtered F&O Stocks")
    st.dataframe(filtered_df)

    # OI Analysis
    st.subheader("📈 Open Interest Analysis")
    oi_cols = st.columns(2)
    
    with oi_cols[0]:
        st.metric("Total Open Interest", f"{filtered_df['OI'].sum():,}")
    with oi_cols[1]:
        avg_oi_change = filtered_df['Change in OI'].mean()
        st.metric("Average OI Change", f"{avg_oi_change:.2f}%")

    # OI Change chart
    fig = px.bar(filtered_df, x="Symbol", y="Change in OI", 
                 title="Change in Open Interest", color="Change in OI",
                 color_continuous_scale="RdYlGn")
    st.plotly_chart(fig)

    # Price vs OI correlation
    st.subheader("🔗 Price vs Open Interest Correlation")
    fig = px.scatter(filtered_df, x="LTP", y="OI", size="Volume", color="Sector",
                     hover_name="Symbol", title="Price vs Open Interest")
    st.plotly_chart(fig)

# Company Overview Page
elif selected == "Company Overview":
    st.title("🏢 Company Overview")
    
    ticker = st.text_input("🔎 Enter Stock Ticker (e.g., AAPL, TCS.NS)", "RELIANCE.NS")
    
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="6mo")
            
            # Display company info
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Company Details")
                st.write(f"**Name:** {info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Market Cap:** ₹{info.get('marketCap', 0):,}")
                
            with col2:
                st.subheader("📈 Key Metrics")
                st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                st.write(f"**EPS:** {info.get('trailingEps', 'N/A')}")
                st.write(f"**Dividend Yield:** {info.get('dividendYield', 0)*100 if info.get('dividendYield') else 'N/A'}%")
                st.write(f"**52W High/Low:** ₹{info.get('fiftyTwoWeekHigh', 'N/A')}/₹{info.get('fiftyTwoWeekLow', 'N/A')}")
            
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
            fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
            
            # Financials
            st.subheader("💵 Financial Summary")
            financials = stock.financials
            if not financials.empty:
                st.write("**Recent Revenue & Earnings**")
                st.dataframe(financials.head().T)
            else:
                st.info("Financial data not available for this ticker")
                
        except Exception as e:
            st.error(f"Error fetching data: {e}")

# Technical Analysis Page
elif selected == "Technical Analysis":
    st.title("📊 Technical Analysis")
    
    ticker = st.text_input("Enter Stock Ticker", "RELIANCE.NS")
    period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"], index=2)
    
    if ticker:
        try:
            data = get_stock_data(ticker, period)
            
            # Calculate technical indicators
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            data['MACD'] = ta.trend.macd_diff(data['Close'])
            
            # Display charts
            st.subheader("📈 Price with Moving Averages")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50'))
            fig.update_layout(title=f"{ticker} Price with Moving Averages")
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI chart
            st.subheader("📊 RSI Indicator")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
            fig.add_hline(y=70, line_dash="dash", line_color="red")
            fig.add_hline(y=30, line_dash="dash", line_color="green")
            fig.update_layout(title="RSI (14 days)")
            st.plotly_chart(fig, use_container_width=True)
            
            # MACD chart
            st.subheader("📉 MACD Indicator")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD'))
            fig.add_hline(y=0, line_color="gray")
            fig.update_layout(title="MACD")
            st.plotly_chart(fig, use_container_width=True)
            
            # Current values
            st.subheader("📋 Current Indicator Values")
            col1, col2, col3 = st.columns(3)
            col1.metric("RSI", f"{data['RSI'].iloc[-1]:.2f}")
            col2.metric("MACD", f"{data['MACD'].iloc[-1]:.4f}")
            col3.metric("Price vs SMA 20", f"{(data['Close'].iloc[-1] - data['SMA_20'].iloc[-1])/data['SMA_20'].iloc[-1]*100:.2f}%")
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")

# Portfolio Tracker
elif selected == "Portfolio Tracker":
    st.title("💼 Portfolio Tracker")
    
    # Initialize session state for portfolio
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame(columns=['Ticker', 'Quantity', 'Buy Price', 'Current Price', 'Change', 'Value'])
    
    # Add stock to portfolio
    st.subheader("➕ Add Stock to Portfolio")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_ticker = st.text_input("Ticker")
    with col2:
        new_quantity = st.number_input("Quantity", min_value=1, value=10)
    with col3:
        new_price = st.number_input("Buy Price", min_value=0.0, value=0.0, step=0.1)
    
    if st.button("Add to Portfolio") and new_ticker and new_quantity and new_price:
        try:
            # Get current price
            current_data = yf.Ticker(new_ticker).history(period="1d")
            current_price = current_data['Close'].iloc[-1]
            
            # Add to portfolio
            new_row = {
                'Ticker': new_ticker,
                'Quantity': new_quantity,
                'Buy Price': new_price,
                'Current Price': current_price,
                'Change': (current_price - new_price) / new_price * 100,
                'Value': new_quantity * current_price
            }
            
            st.session_state.portfolio = st.session_state.portfolio.append(new_row, ignore_index=True)
            st.success(f"Added {new_ticker} to portfolio")
        except:
            st.error("Error adding stock to portfolio")
    
    # Display portfolio
    st.subheader("📊 Your Portfolio")
    if not st.session_state.portfolio.empty:
        # Update current prices
        for idx, row in st.session_state.portfolio.iterrows():
            try:
                current_data = yf.Ticker(row['Ticker']).history(period="1d")
                current_price = current_data['Close'].iloc[-1]
                st.session_state.portfolio.at[idx, 'Current Price'] = current_price
                st.session_state.portfolio.at[idx, 'Change'] = (current_price - row['Buy Price']) / row['Buy Price'] * 100
                st.session_state.portfolio.at[idx, 'Value'] = row['Quantity'] * current_price
            except:
                pass
        
        # Display portfolio
        st.dataframe(st.session_state.portfolio)
        
        # Portfolio summary
        total_value = st.session_state.portfolio['Value'].sum()
        total_invested = (st.session_state.portfolio['Quantity'] * st.session_state.portfolio['Buy Price']).sum()
        total_change = (total_value - total_invested) / total_invested * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Value", f"₹{total_value:,.2f}")
        col2.metric("Total Invested", f"₹{total_invested:,.2f}")
        col3.metric("Total Gain/Loss", f"₹{total_value - total_invested:,.2f}", f"{total_change:.2f}%")
        
        # Portfolio allocation chart
        st.subheader("📈 Portfolio Allocation")
        fig = px.pie(st.session_state.portfolio, values='Value', names='Ticker', title='Portfolio Allocation by Value')
        st.plotly_chart(fig)
    else:
        st.info("Your portfolio is empty. Add some stocks to get started.")
