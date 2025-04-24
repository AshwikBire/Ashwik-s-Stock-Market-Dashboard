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
from xgboost import XGBRegressor
from datetime import timedelta
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import streamlit as st

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Set page config
st.set_page_config(page_title="Smart Stock Market Dashboard", layout="wide")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "Smart Market Dashboard",
        ["Home", "Company Info", "Market Movers", "Global Markets", "Mutual Funds", "SIP Calculator","IPO Tracker","Predictions for Mutual Funds & IPOs","Mutual Fund NAV Viewer","Sectors", "News", "Learning", "Volume Spike", "Stock Screener", "Predictions", "Buy/Sell Predictor", "News Sentiment"],
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
# Market Movers - Active Stocks, Top Gainers & Losers
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
    st.dataframe(active_stocks)

    # Top Gainers
    st.subheader("🚀 Top Gainers")
    top_gainers = pd.DataFrame(gainers, columns=['Stock', 'Price'])
    st.dataframe(top_gainers)

    # Top Losers
    st.subheader("📉 Top Losers")
    top_losers = pd.DataFrame(losers, columns=['Stock', 'Price'])
    st.dataframe(top_losers)

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
    st.title("📰 Latest Financial News")
    news_query = st.text_input("Search Financial News:", "stock market")

    if news_query:
        url = f"https://newsapi.org/v2/everything?q={news_query}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize=10"
        response = requests.get(url)

        if response.status_code == 200:
            articles = response.json().get("articles", [])
            if articles:
                for article in articles:
                    st.markdown("----")
                    st.subheader(article["title"])
                    st.write(f"*{article['source']['name']} - {article['publishedAt'].split('T')[0]}*")
                    st.write(article.get("description", "No description available."))
                    st.markdown(f"[🔗 Read More]({article['url']})")
            else:
                st.warning("No articles found.")
        else:
            st.error("Unable to fetch news articles. Please check API or query.")


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
    st.title("📈 Volume Spike Detector")

    ticker = st.text_input("Enter Stock Ticker (e.g., TCS.NS, INFY.NS):", "TCS.NS")
    days = st.slider("Select Days of Historical Data:", 30, 365, 90)

    if ticker:
        try:
            # Download historical data
            data = yf.download(ticker, period=f"{days}d")

            if data.empty:
                st.warning("No data found. Please check the ticker symbol.")
            else:
                # Compute 10-day average volume
                data["Avg_Volume"] = data["Volume"].rolling(window=10).mean()
                data["Spike"] = data["Volume"] > (1.5 * data["Avg_Volume"])
                data.dropna(subset=["Avg_Volume"], inplace=True)

                st.markdown("### 🚀 Volume Spike Chart")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data["Volume"],
                                         mode='lines', name='Volume', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=data.index, y=data["Avg_Volume"],
                                         mode='lines', name='10-Day Avg Volume', line=dict(color='orange')))

                # Highlight spikes
                spike_data = data[data["Spike"]]
                fig.add_trace(go.Scatter(x=spike_data.index, y=spike_data["Volume"],
                                         mode='markers', name='Volume Spikes',
                                         marker=dict(size=10, color='red', symbol='star')))

                fig.update_layout(title=f"Volume Spike Analysis for {ticker}",
                                  xaxis_title="Date",
                                  yaxis_title="Volume",
                                  template="plotly_dark",
                                  height=500)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 🔍 Spike Events")
                st.dataframe(spike_data[["Volume", "Avg_Volume"]].style.format("{:,.0f}"), use_container_width=True)

        except Exception as e:
            st.error(f"Error occurred: {e}")



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
#company info----------------------------#
import streamlit as st
import yfinance as yf
import pandas as pd

# ✅ Define the function first
def company_info_page():
    st.title("📊 Company Info")

    ticker = st.text_input("Enter Company Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")

    if ticker:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Company Overview
            st.header(f"🏢 {info.get('longName', 'N/A')} ({info.get('symbol', ticker)})")

            # Basic Info
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📌 Basic Info")
                st.write(pd.DataFrame({
                    "Detail": [
                        "Exchange", "Sector", "Industry", "Country", "Market Cap", "Volume",
                        "52W High", "52W Low", "Dividend Yield", "Book Value", "Face Value"
                    ],
                    "Value": [
                        info.get("exchange", "N/A"), info.get("sector", "N/A"), info.get("industry", "N/A"),
                        info.get("country", "N/A"), info.get("marketCap", "N/A"), info.get("volume", "N/A"),
                        info.get("fiftyTwoWeekHigh", "N/A"), info.get("fiftyTwoWeekLow", "N/A"),
                        info.get("dividendYield", "N/A"), info.get("bookValue", "N/A"), info.get("faceValue", "N/A")
                    ]
                }))

            with col2:
                st.subheader("👔 Executive Info")
                st.markdown(f"**CEO**: {info.get('CEO', 'N/A')}")
                st.markdown(f"**Employees**: {info.get('fullTimeEmployees', 'N/A')}")
                website = info.get('website', '')
                if website:
                    st.markdown(f"**Website**: [{website}]({website})")

                st.subheader("📉 Financial Ratios (Simulated)")
                st.write(pd.DataFrame({
                    "Metric": ["PE Ratio", "PB Ratio", "EPS", "ROE", "ROCE", "Debt to Equity"],
                    "Value": [22.5, 4.2, 85.3, "18.5%", "22.1%", "0.35"]
                }))

            # Shareholding Pattern (Simulated)
            st.subheader("📊 Shareholding Pattern")
            share_pattern = {
                "Promoters": 49.5,
                "FIIs": 24.2,
                "DIIs": 13.4,
                "Retail": 12.9
            }
            st.bar_chart(pd.Series(share_pattern))

            # Company Summary
            st.subheader("📘 About the Company")
            st.write(info.get("longBusinessSummary", "No description available."))

            # Competitor Comparison (Simulated)
            st.subheader("🏁 Compare with Competitors")
            competitors = [ticker, "TCS.NS", "INFY.NS"]
            comp_data = {
                "Company": competitors,
                "PE Ratio": [22.5, 25.3, 30.1],
                "PB Ratio": [4.2, 5.0, 6.2],
                "ROE (%)": [18.5, 17.2, 19.5],
                "ROCE (%)": [22.1, 21.3, 22.0],
                "Debt/Equity": [0.35, 0.4, 0.3]
            }
            st.dataframe(pd.DataFrame(comp_data))

            # Coming Soon
            st.subheader("🗞️ News & Sentiment")
            st.info("Live news & sentiment analysis coming in next version 🚀")

        except Exception as e:
            st.error(f"❌ Could not retrieve data for ticker: {e}")

# ✅ THEN call the function in navigation
if selected == "Company Info":
    company_info_page()



#------------------predictions page---------------------------------#
# Predictions - Stock Price Prediction
elif selected == "Predictions":
    st.title("📈 Stock Price Predictions")

    ticker = st.text_input("Enter Company Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")

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
                        f"📈 Signal: **BUY** - 50-day SMA is above 200-day SMA (Current Price: ₹{current_price:.2f})")
                elif sma50.iloc[-1] < sma200.iloc[-1]:
                    st.error(
                        f"📉 Signal: **SELL** - 50-day SMA is below 200-day SMA (Current Price: ₹{current_price:.2f})")
                else:
                    st.warning(f"⏸️ Signal: **HOLD** - No clear trend (Current Price: ₹{current_price:.2f})")

                # Show price data vs moving averages
                st.subheader("📈 Price vs. Moving Averages")
                st.line_chart(hist[["Close"]].join(pd.DataFrame({
                    "50-Day SMA": sma50,
                    "200-Day SMA": sma200
                })))

                # Optional: Machine learning-based predictions can be added here.
                # For example, using a regression model or an LSTM for stock price prediction.

        except Exception as e:
            st.error(f"Error retrieving data: {e}")

#----------------------Buy/Sell Predictor-------------#

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
                # Show the most recent data
                st.subheader(f"Recent Stock Data for {ticker}")
                st.write(hist.tail())

                # Plot the stock's historical closing price
                st.subheader("📊 Stock Price History")
                st.line_chart(hist["Close"])

                # Calculate Simple Moving Averages (SMA)
                sma50 = hist["Close"].rolling(window=50).mean()
                sma200 = hist["Close"].rolling(window=200).mean()

                st.subheader("📉 Moving Averages")
                st.line_chart(pd.DataFrame({
                    "50-Day SMA": sma50,
                    "200-Day SMA": sma200
                }))

                # Calculate Relative Strength Index (RSI) for additional signal
                delta = hist["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                st.subheader("📈 RSI (Relative Strength Index)")
                st.line_chart(rsi)

                # Calculate Buy/Sell signal
                current_price = hist["Close"].iloc[-1]
                signal = ""

                # Simple Buy/Sell logic based on Moving Averages and RSI
                if sma50.iloc[-1] > sma200.iloc[-1] and rsi.iloc[-1] < 30:
                    signal = "Buy"
                    st.success(f"📈 Signal: **BUY** (Current Price: ₹{current_price:.2f}) - 50-day SMA is above 200-day SMA and RSI is below 30.")
                elif sma50.iloc[-1] < sma200.iloc[-1] and rsi.iloc[-1] > 70:
                    signal = "Sell"
                    st.error(f"📉 Signal: **SELL** (Current Price: ₹{current_price:.2f}) - 50-day SMA is below 200-day SMA and RSI is above 70.")
                else:
                    signal = "Hold"
                    st.warning(f"⏸️ Signal: **HOLD** (Current Price: ₹{current_price:.2f}) - No clear trend.")

                # Show price data vs moving averages and RSI
                st.subheader("📊 Price vs. Indicators")
                st.line_chart(hist[["Close"]].join(pd.DataFrame({
                    "50-Day SMA": sma50,
                    "200-Day SMA": sma200,
                    "RSI": rsi
                })))

        except Exception as e:
            st.error(f"Error retrieving data: {e}")

# Stock Screener - Default 15 companies, or user input for custom tickers
elif selected == "Stock Screener":
    st.title("📊 Stock Screener")

    # Predefined list of 15 companies (Nifty 50 or a custom list of top companies)
    default_companies = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'HINDUNILVR.NS',
        'BAJAJFINSV.NS', 'HDFC.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'AXISBANK.NS', 'MARUTI.NS', 'LT.NS'
    ]

    # Ask user whether they want to use the default list or input custom tickers
    choice = st.radio("Choose an option:", ("Use Default List", "Input Custom Tickers"))

    if choice == "Use Default List":
        # Display the stock data for the default 15 companies
        st.subheader("Showing 15 Default Companies")
        data = {}

        for ticker in default_companies:
            stock_data = yf.Ticker(ticker).history(period="1d")['Close']
            if not stock_data.empty:
                data[ticker] = stock_data.iloc[-1]
            else:
                data[ticker] = "No Data"

        # Display the data as a dataframe
        st.dataframe(pd.DataFrame(data.items(), columns=["Stock", "Price"]))

    elif choice == "Input Custom Tickers":
        # Input box for user to enter their own tickers
        tickers_input = st.text_area("Enter stock tickers (separated by space or comma):", "")
        if tickers_input:
            tickers_list = [ticker.strip() for ticker in tickers_input.split() if ticker.strip()]
            if len(tickers_list) > 0:
                st.subheader("Showing Custom Tickers")
                data = {}

                for ticker in tickers_list:
                    stock_data = yf.Ticker(ticker).history(period="1d")['Close']
                    if not stock_data.empty:
                        data[ticker] = stock_data.iloc[-1]
                    else:
                        data[ticker] = "No Data"

                # Display the custom tickers data
                st.dataframe(pd.DataFrame(data.items(), columns=["Stock", "Price"]))
            else:
                st.warning("Please enter valid stock tickers.")

#----------------Mutual Fund----------------#

elif selected == "Mutual Funds":
    st.title("💼 Mutual Funds Overview")

    import requests

    scheme_code = "118550"
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        scheme_name = data['meta']['scheme_name']
        nav_data = data['data']
        nav_df = pd.DataFrame(nav_data)
        nav_df['nav'] = nav_df['nav'].astype(float)
        nav_df['date'] = pd.to_datetime(nav_df['date'], format='%d-%m-%Y')
        nav_df = nav_df.sort_values('date')

        st.subheader(f"{scheme_name}")
        st.write(f"Latest NAV: ₹{nav_df.iloc[-1]['nav']} as of {nav_df.iloc[-1]['date'].date()}")
        st.line_chart(nav_df.set_index('date')['nav'])
    else:
        st.error("Failed to fetch mutual fund data.")


        #----------------------SIP Calculator--------------------------------#

elif selected == "SIP Calculator":
    st.title("📈 SIP Calculator")

    monthly_investment = st.number_input("Monthly Investment (₹)", value=5000)
    years = st.slider("Investment Duration (Years)", 1, 30, 10)
    expected_return = st.slider("Expected Annual Return (%)", 1, 25, 12)

    months = years * 12
    monthly_rate = expected_return / 12 / 100

    future_value = monthly_investment * (((1 + monthly_rate)**months - 1) * (1 + monthly_rate)) / monthly_rate
    invested = monthly_investment * months
    gain = future_value - invested

    st.success(f"📊 Future Value: ₹{future_value:,.2f}")
    st.info(f"💰 Invested: ₹{invested:,.2f}")
    st.warning(f"📈 Estimated Gains: ₹{gain:,.2f}")

#--------------------------IP0 Tracker------------------------------------------#

elif selected == "IPO Tracker":
    st.title("🆕 IPO Tracker")

    ipo_data = pd.DataFrame({
        "Company": ["ABC Tech", "SmartFin Ltd", "GreenPower", "NetPay Corp"],
        "Issue Price (₹)": [100, 240, 150, 280],
        "Current Price (₹)": [145, 190, 170, 260],
        "Gain/Loss (%)": [45, -20.8, 13.3, -7.1],
        "Sentiment": ["Bullish", "Bearish", "Neutral", "Bearish"]
    })

    st.dataframe(ipo_data)
    st.bar_chart(ipo_data.set_index("Company")["Gain/Loss (%)"])

#------------------------------Predictions for Mutual Funds & IPOs------------------------------------#
elif selected == "Predictions for Mutual Funds & IPOs":
    st.title("🔮 Predictions for Mutual Funds & IPOs")

    st.subheader("📊 Mutual Fund NAV Forecast (Simulated)")
    import numpy as np
    dates = pd.date_range(start=pd.to_datetime("2023-01-01"), periods=12, freq='M')
    navs = np.linspace(100, 160, 12) + np.random.normal(0, 2, 12)

    nav_forecast = pd.DataFrame({'Month': dates, 'Predicted NAV': navs})
    nav_forecast.set_index("Month", inplace=True)
    st.line_chart(nav_forecast)

    st.subheader("🚀 IPO Price Movement Prediction (Simulated)")
    ipo_prediction = pd.DataFrame({
        "IPO": ["ABC Tech", "SmartFin Ltd", "GreenPower"],
        "Predicted Return (%)": [20.5, -5.2, 12.7]
    })
    st.dataframe(ipo_prediction)


#---------------------------------------------------Mutual Fund Nav Viewver code-------------------------------------#
# 📈 Mutual Funds - Live NAV
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



#-------------------------------Company Management Deatils Info---------------------------#
def company_info_page():
    st.title("📊 Company Info with Employee & Management Details")

    # Add unique key to avoid duplicate element error
    ticker = st.text_input("Enter Company Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS", key="company_ticker_input")

    if ticker:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Company Overview
            st.header(f"🏢 {info.get('longName', 'N/A')} ({info.get('symbol', ticker)})")

            # Basic Info
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📌 Basic Info")
                st.write(pd.DataFrame({
                    "Detail": [
                        "Exchange", "Sector", "Industry", "Country", "Market Cap", "Volume",
                        "52W High", "52W Low", "Dividend Yield", "Book Value", "Face Value"
                    ],
                    "Value": [
                        info.get("exchange", "N/A"), info.get("sector", "N/A"), info.get("industry", "N/A"),
                        info.get("country", "N/A"), info.get("marketCap", "N/A"), info.get("volume", "N/A"),
                        info.get("fiftyTwoWeekHigh", "N/A"), info.get("fiftyTwoWeekLow", "N/A"),
                        info.get("dividendYield", "N/A"), info.get("bookValue", "N/A"), info.get("faceValue", "N/A")
                    ]
                }))

            with col2:
                st.subheader("👔 Executive Info")
                st.markdown(f"**CEO**: {info.get('CEO', 'N/A')}")
                st.markdown(f"**Employees**: {info.get('fullTimeEmployees', 'N/A')}")
                website = info.get('website', '')
                if website:
                    st.markdown(f"**Website**: [{website}]({website})")

            # Simulated Higher Management Details (Age, Salary, Contact, LinkedIn, etc.)
            st.subheader("🏢 Higher Management")
            higher_management = {
                "Name": ["John Doe", "Jane Smith", "Michael Johnson"],
                "Position": ["CEO", "CFO", "COO"],
                "Age": [48, 52, 45],
                "Salary (in ₹)": ["₹2,50,00,000", "₹2,10,00,000", "₹1,80,00,000"],
                "Email": ["johndoe@company.com", "janesmith@company.com", "michaelj@company.com"],
                "Contact": ["+91 123 456 7890", "+91 987 654 3210", "+91 555 123 4567"],
                "LinkedIn": [
                    "https://www.linkedin.com/in/johndoe/",
                    "https://www.linkedin.com/in/janesmith/",
                    "https://www.linkedin.com/in/michaeljohnson/"
                ]
            }

            # Create DataFrame for Higher Management Details
            management_df = pd.DataFrame(higher_management)

            st.dataframe(management_df)

            # Shareholding Pattern (Simulated)
            st.subheader("📊 Shareholding Pattern")
            share_pattern = {
                "Promoters": 49.5,
                "FIIs": 24.2,
                "DIIs": 13.4,
                "Retail": 12.9
            }
            st.bar_chart(pd.Series(share_pattern))

            # Company Summary
            st.subheader("📘 About the Company")
            st.write(info.get("longBusinessSummary", "No description available."))

            # Competitor Comparison (Simulated)
            st.subheader("🏁 Compare with Competitors")
            competitors = [ticker, "TCS.NS", "INFY.NS"]
            comp_data = {
                "Company": competitors,
                "PE Ratio": [22.5, 25.3, 30.1],
                "PB Ratio": [4.2, 5.0, 6.2],
                "ROE (%)": [18.5, 17.2, 19.5],
                "ROCE (%)": [22.1, 21.3, 22.0],
                "Debt/Equity": [0.35, 0.4, 0.3]
            }
            st.dataframe(pd.DataFrame(comp_data))

            # Coming Soon
            st.subheader("🗞️ News & Sentiment")
            st.info("Live news & sentiment analysis coming in next version 🚀")

        except Exception as e:
            st.error(f"❌ Could not retrieve data for ticker: {e}")

# Add this part in your navigation section:
if selected == "Company Info":
    company_info_page()
