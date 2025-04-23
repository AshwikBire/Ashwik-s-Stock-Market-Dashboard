# Updated Smart Stock Market Dashboard with Advanced Features
# Install required packages first
# pip install streamlit yfinance requests streamlit-option-menu plotly scikit-learn

import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from streamlit_option_menu import option_menu
import plotly.graph_objs as go

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Set page config
st.set_page_config(page_title="Smart Stock Market Dashboard", layout="wide")

# Sidebar
with st.sidebar:
    selected = option_menu(
        "Smart Market Dashboard",
        ["Home", "Market Movers", "Global Markets", "Mutual Funds", "Sectors", "News", "Company Info", "Learning", "Volume Spike", "Screener"],
        icons=['house', 'graph-up', 'globe', 'bank', 'boxes', 'newspaper', 'building', 'book', 'activity', 'search'],
        menu_icon="cast",
        default_index=0
    )

# ---------------- PAGE LOGIC ----------------

# Home
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

# Market Movers
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

# Global Markets
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

# Mutual Funds
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

# Sectors
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

# News
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

# Company Info
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

# Learning
elif selected == "Learning":
    st.title("\U0001F4DA Learning Materials & Purpose")
    st.markdown("""
    ## \U0001F4C8 Purpose
    Empower yourself with real-time market insights, stock fundamentals, and advanced financial literacy resources.

    ## \U0001F4D6 Recommended Resources
    - [Investopedia - Stock Market Basics](https://www.investopedia.com/terms/s/stockmarket.asp)
    - [Yahoo Finance Learning](https://finance.yahoo.com/education/)
    - [MarketWatch - Trading Education](https://www.marketwatch.com/tools/trading-education)
    - [NSE India Market Tutorials](https://www.nseindia.com/learn)
    """)

# Volume Spike Detector
elif selected == "Volume Spike":
    st.title("\U0001F50A Volume Spike Detector")
    ticker = st.text_input("Enter Stock Ticker for Volume Spike Detection", "AAPL")
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="30d")
            avg_volume = hist['Volume'].mean()
            latest_volume = hist['Volume'].iloc[-1]

            st.metric(label="Average 30d Volume", value=f"{avg_volume:,.0f}")
            st.metric(label="Latest Volume", value=f"{latest_volume:,.0f}", delta=f"{((latest_volume - avg_volume) / avg_volume) * 100:.2f}%")

            if latest_volume > 1.5 * avg_volume:
                st.success("Volume Spike Detected! \U0001F680")
            else:
                st.info("No significant volume spike.")
        except Exception as e:
            st.error(f"Failed to fetch volume data. Error: {e}")

# Screener (New Feature)
elif selected == "Screener":
    st.title("\U0001F50D Stock Screener")
    st.write("(Coming Soon) Advanced Stock Screener for finding the best opportunities!")

# Footer
st.markdown("---")
st.markdown("Built By Ashwik Bire")
