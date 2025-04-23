import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Make sure this line is included at the top
import seaborn as sns
import yfinance as yf  # Ensure that yfinance is imported for stock data
import requests
from plotly import graph_objects as go
from streamlit_option_menu import option_menu  # Import the option_menu
from textblob import TextBlob
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import timedelta

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Set page config
st.set_page_config(page_title="Smart Stock Market Dashboard", layout="wide")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "Smart Market Dashboard",
        ["Home", "Market Movers", "Global Markets", "Mutual Funds", "Sectors", "News", "Company Info", "Learning", "Volume Spike", "Screener", "Predictions", "Buy/Sell Predictor", "News Sentiment"],
        icons=['house', 'graph-up', 'globe', 'bank', 'boxes', 'newspaper', 'building', 'book', 'activity', 'search'],
        menu_icon="cast",
        default_index=0
    )

# Home - Market Overview
if selected == "Home":
    st.title("\U0001F3E0 Home - Market Overview")
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
        cols[idx].metric(label=name, value=f"{last_close}", delta=f"{percent_change}%")

# Market Movers - Top Gainers & Losers
elif selected == "Market Movers":
    st.title("\U0001F4C8 Market Movers - Top Gainers & Losers")
    tickers_list = 'RELIANCE.NS TCS.NS INFY.NS HDFCBANK.NS ICICIBANK.NS'
    nifty = yf.Tickers(tickers_list)
    data = {ticker: nifty.tickers[ticker].history(period="1d")['Close'].iloc[-1] for ticker in nifty.tickers}
    gainers = sorted(data.items(), key=lambda x: x[1], reverse=True)
    losers = sorted(data.items(), key=lambda x: x[1])

    st.subheader("Top Gainers")
    st.dataframe(pd.DataFrame(gainers, columns=['Stock', 'Price']))
    st.subheader("Top Losers")
    st.dataframe(pd.DataFrame(losers, columns=['Stock', 'Price']))

# Global Markets - Major Indices
elif selected == "Global Markets":
    st.title("\U0001F30E Global Markets Status")
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
        cols[idx % 3].metric(label=name, value=f"{last_close}", delta=f"{percent_change}%")

# Mutual Funds - Insights
elif selected == "Mutual Funds":
    st.title("\U0001F3E6 Mutual Funds Insights")
    mf_data = {
        "Axis Bluechip Fund": "15% Returns",
        "Mirae Asset Large Cap Fund": "13.2% Returns",
        "Parag Parikh Flexi Cap Fund": "17.5% Returns",
        "UTI Nifty Index Fund": "12% Returns",
    }
    st.dataframe(pd.DataFrame(mf_data.items(), columns=['Mutual Fund', '1Y Return']))
    st.info("Live Mutual Fund API integration coming soon!")

# Sectors - Sector Performance
elif selected == "Sectors":
    st.title("\U0001F4CA Sector Wise Performance")
    sector_performance = {
        "Banking": "+1.8%",
        "IT": "-0.5%",
        "Energy": "+2.1%",
        "FMCG": "+0.9%",
        "Pharma": "-1.2%",
        "Auto": "+1.0%",
    }
    st.dataframe(pd.DataFrame(sector_performance.items(), columns=['Sector', 'Performance']))

# News - Latest Financial News
elif selected == "News":
    st.title("\U0001F4F0 Latest Financial News")
    url = f"https://newsapi.org/v2/top-headlines?country=in&category=business&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        for article in response.json().get("articles", [])[:10]:
            st.subheader(article['title'])
            st.write(article['description'])
            st.markdown(f"[Read more]({article['url']})")
            st.markdown("---")
    else:
        st.error("Failed to fetch news, try again later.")

# Company Info - Stock Details
elif selected == "Company Info":
    st.title("\U0001F3E2 Company Info & Insights")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TCS.NS)", "AAPL")
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y")

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${stock.info.get('regularMarketPrice', 'N/A')}")
            col2.metric("Day High", f"${stock.info.get('dayHigh', 'N/A')}")
            col3.metric("Day Low", f"${stock.info.get('dayLow', 'N/A')}")

            st.subheader("\U0001F4C8 Price Trend (2 Years)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name="Close Price"))
            fig.update_layout(title=f"{ticker.upper()} Price Chart", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("\U0001F4C4 Company Profile")
            profile_data = {
                "Name": stock.info.get('longName', 'N/A'),
                "Sector": stock.info.get('sector', 'N/A'),
                "Industry": stock.info.get('industry', 'N/A'),
                "Market Cap": stock.info.get('marketCap', 'N/A'),
                "PE Ratio": stock.info.get('trailingPE', 'N/A'),
                "EPS": stock.info.get('trailingEps', 'N/A'),
                "Revenue": stock.info.get('totalRevenue', 'N/A'),
                "Profit Margins": stock.info.get('profitMargins', 'N/A'),
                "Dividend Yield": stock.info.get('dividendYield', 'N/A'),
                "Earnings Growth": stock.info.get('earningsGrowth', 'N/A'),
                "Revenue Growth": stock.info.get('revenueGrowth', 'N/A'),
                "52 Week High": stock.info.get('fiftyTwoWeekHigh', 'N/A'),
                "52 Week Low": stock.info.get('fiftyTwoWeekLow', 'N/A'),
                "Employees": stock.info.get('fullTimeEmployees', 'N/A')
            }
            st.dataframe(pd.DataFrame(list(profile_data.items()), columns=['Attribute', 'Value']))
        except Exception as e:
            st.error(f"Failed to fetch company data. Error: {e}")

# Learning - Stock Market Resources
elif selected == "Learning":
    st.title("📘 Learn the Stock Market")

    st.markdown("""
    Welcome to the **Learning Hub** of the Smart Stock Market Dashboard by [Ashwik Bire](https://www.linkedin.com/in/ashwik-bire-b2a000186/)!  
    This section is crafted to help **beginners, enthusiasts, and investors** understand how the stock market works — with a strong foundation in both **technical and fundamental analysis**, along with insights from **AI and machine learning**.

    ### 🎯 Purpose:
    - To **educate** users with curated stock market knowledge.
    - To **simplify complex concepts** like indicators, price action, technical patterns, and financial ratios.
    - To share **AI-powered learning resources** that explain how stock prediction models work.

    ### 🧠 What You’ll Learn:
    - 📈 Basics of Stock Market, Trading, and Investing  
    - 🧾 Financial Statements and Ratio Analysis  
    - 🧮 Technical Analysis (Indicators, Patterns, Volume)  
    - 🤖 AI/ML in the Stock Market  
    - 🛠 Tools & Resources for Smarter Investing

    ### 🔗 Connect with Ashwik Bire:
    [![LinkedIn](https://img.shields.io/badge/Connect%20with%20me-LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/ashwik-bire-b2a000186/)

    Stay tuned! We’re continuously updating this section with **videos, articles, and interactive tutorials**.
    """)


# Volume Spike - Stock Volume Insights
elif selected == "Volume Spike":
    st.title("\U0001F4C8 Volume Spike Analysis")
    st.info("Volume spike analysis is coming soon!")

# Stock Screener - Filter Stocks Based on Criteria
elif selected == "Screener":
    st.title("\U0001F50E Stock Screener")
    st.info("Stock screener is under development. Stay tuned for more!")

# Predictions - AI-Powered Stock Predictions
elif selected == "Predictions":
    st.title("🤖 AI-Based Stock Predictions")

    ticker = st.text_input("Enter Stock Ticker for Prediction:", "AAPL")

    if ticker:
        st.info(f"Fetching and predicting for {ticker.upper()}...")

        # Fetch data
        data = yf.download(ticker, period="6mo", interval="1d")
        if data.empty:
            st.warning("No data found for the ticker.")
        else:
            # Feature engineering
            data['Return'] = data['Close'].pct_change()
            data['Lag1'] = data['Close'].shift(1)
            data['Lag2'] = data['Close'].shift(2)
            data['Lag3'] = data['Close'].shift(3)
            data.dropna(inplace=True)

            # Define features and target
            features = data[['Lag1', 'Lag2', 'Lag3']]
            target = data['Close']

            # Split into train and test
            X_train, X_test = features[:-1], features[-1:]
            y_train = target[:-1]

            # Train a simple XGBoost regressor
            model = XGBRegressor(n_estimators=100, max_depth=3)
            model.fit(X_train, y_train)

            # Predict
            predicted_price = model.predict(X_test)[0]
            st.success(f"📈 Predicted Next Close Price for {ticker.upper()}: **${predicted_price:.2f}**")

            # Plot
            st.subheader("📊 Historical vs Predicted Price")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Close'))
            fig.add_trace(go.Scatter(x=[data.index[-1] + timedelta(days=1)],
                                     y=[predicted_price], mode='markers+text',
                                     text=["Predicted"], name='Prediction',
                                     marker=dict(color='red', size=10)))
            st.plotly_chart(fig, use_container_width=True)

# Buy/Sell Predictor - AI-Based Buy/Sell Predictor
elif selected == "Buy/Sell Predictor":
    st.title("\U0001F91D Buy/Sell Predictor - AI Recommendation")
    ticker = st.text_input("Enter Stock Ticker for Buy/Sell Analysis:", "AAPL")
    if ticker:
        try:
            data = yf.download(ticker, period="3mo", interval="1d")
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            data['SMA50'] = data['Close'].rolling(window=50).mean()

            st.subheader("📊 Price Chart with SMA20 & SMA50")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price"))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name="SMA 20"))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name="SMA 50"))
            fig.update_layout(title=f"{ticker.upper()} Buy/Sell Strategy", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            latest_close = data['Close'].iloc[-1]
            latest_sma20 = data['SMA20'].iloc[-1]
            latest_sma50 = data['SMA50'].iloc[-1]

            if latest_sma20 > latest_sma50 and latest_close > latest_sma20:
                st.success("✅ Recommendation: **BUY** (Uptrend confirmed)")
            elif latest_sma20 < latest_sma50 and latest_close < latest_sma20:
                st.error("❌ Recommendation: **SELL** (Downtrend detected)")
            else:
                st.warning("⚠️ Recommendation: **HOLD** (No clear trend)")

            st.markdown(f"**Current Price:** ${round(latest_close, 2)}")
            st.markdown(f"**SMA 20:** ${round(latest_sma20, 2)}")
            st.markdown(f"**SMA 50:** ${round(latest_sma50, 2)}")

        except Exception as e:
            st.error(f"Error fetching data: {e}")

# News Sentiment - Sentiment Analysis of News
elif selected == "News Sentiment":
    st.title("\U0001F50D News Sentiment Analysis")
    ticker = st.text_input("Enter Stock Ticker to analyze news sentiment:", "AAPL")

    if ticker:
        st.info(f"Fetching and analyzing recent news sentiment for {ticker.upper()}...")
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
                st.write(f"📰 **{title}**")
                st.write(f"🧠 Sentiment Score: {round(polarity, 3)}")
                st.markdown("---")

            if sentiments:
                avg_sentiment = round(np.mean(sentiments), 3)
                st.success(f"📊 **Average Sentiment Score** for {ticker.upper()}: {avg_sentiment}")
                if avg_sentiment > 0.2:
                    st.markdown("**📈 Overall Sentiment: Positive**")
                elif avg_sentiment < -0.2:
                    st.markdown("**📉 Overall Sentiment: Negative**")
                else:
                    st.markdown("**➖ Overall Sentiment: Neutral**")
        else:
            st.error("Failed to fetch news articles.")
