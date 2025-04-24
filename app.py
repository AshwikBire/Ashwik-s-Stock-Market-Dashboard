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

# Stock Screener - Filter Stocks Based on Criteria
elif selected == "Screener":
    st.title("🧠 Smart Stock Screener")

    # Screener criteria inputs
    st.sidebar.header("📌 Filter Criteria")
    market_cap = st.sidebar.selectbox("Market Cap", ["All", "Large Cap", "Mid Cap", "Small Cap"])
    pe_min = st.sidebar.number_input("Min PE Ratio", value=0.0)
    pe_max = st.sidebar.number_input("Max PE Ratio", value=50.0)
    price_range = st.sidebar.slider("Price Range (₹)", 10, 5000, (50, 1000))
    volume_min = st.sidebar.number_input("Minimum Daily Volume", value=100000)

    # Example NSE tickers list (replace with a full list or CSV)
    tickers = {
        "RELIANCE.NS": "Reliance",
        "TCS.NS": "TCS",
        "INFY.NS": "Infosys",
        "HDFCBANK.NS": "HDFC Bank",
        "ICICIBANK.NS": "ICICI Bank",
        "LT.NS": "L&T",
        "SBIN.NS": "SBI",
    }

    results = []

    st.info("🔍 Scanning Stocks... Please wait.")
    for ticker, name in tickers.items():
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            pe = info.get("trailingPE", None)
            mc = info.get("marketCap", 0)
            price = info.get("regularMarketPrice", 0)
            volume = info.get("volume", 0)

            if not pe or pe < pe_min or pe > pe_max:
                continue
            if price < price_range[0] or price > price_range[1]:
                continue
            if volume < volume_min:
                continue
            if market_cap == "Large Cap" and mc < 200000000000:
                continue
            if market_cap == "Mid Cap" and (mc < 50000000000 or mc > 200000000000):
                continue
            if market_cap == "Small Cap" and mc > 50000000000:
                continue

            results.append({
                "Ticker": ticker,
                "Name": name,
                "Price": price,
                "PE Ratio": pe,
                "Volume": volume,
                "Market Cap": mc
            })

        except:
            continue

    if results:
        df = pd.DataFrame(results)

# Predictions - AI-Powered Stock Predictions
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import streamlit as st


# Function to create the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))  # Output layer (1 unit for predicting closing price)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Function for LSTM prediction
def predict_with_lstm(data):
    # Step 1: Preprocess data
    data = data[['Close']]  # We only need the closing prices for prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Step 2: Prepare the data for LSTM input
    X = []
    y = []
    look_back = 60  # Look-back window of 60 days

    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshape X to be in the shape [samples, time steps, features] for LSTM input
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Step 3: Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Step 4: Create and train the LSTM model
    model = create_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Step 5: Predict the next 30 days closing prices
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Rescale to original values

    # Prepare the predicted data as a DataFrame
    prediction_dates = data.index[-len(predictions):]
    prediction_df = pd.DataFrame(predictions, columns=['Predicted'], index=prediction_dates)

    return prediction_df


# Streamlit app UI
st.title("📈 AI-Powered Stock Predictions")

# Prediction Model Selection
model_type = st.selectbox("Choose Prediction Model", ["LSTM (Long Short Term Memory)", "XGBoost"])

# Stock Ticker input
ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")
st.sidebar.info("The models predict next 30 days closing price for the given stock.")

if ticker_input:
    st.info(f"🔍 Fetching data for {ticker_input}...")

    # Fetch Stock Data
    data = yf.download(ticker_input, period="1y", interval="1d")

    # Preprocess Data for Predictions
    data['Date'] = pd.to_datetime(data.index)
    data.set_index('Date', inplace=True)

    # AI Model Predictions
    if model_type == "LSTM (Long Short Term Memory)":
        st.write("Using LSTM model for prediction...")
        prediction = predict_with_lstm(data)  # This will predict using the LSTM model
        st.write("Predicted Closing Prices for the Next 30 Days:")
        st.dataframe(prediction)

        # Visualize LSTM prediction
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prediction.index, y=prediction['Predicted'], name='Predicted', mode='lines'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Actual', mode='lines', line=dict(color='red')))
        fig.update_layout(title=f"LSTM Prediction vs Actual for {ticker_input}", xaxis_title="Date",
                          yaxis_title="Price", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# This is the Streamlit app structure, which integrates LSTM-based AI predictions

# Buy/Sell Predictor - AI-Based Buy/Sell Predictor
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import streamlit as st


# Function to fetch and prepare data
def prepare_data(ticker):
    data = yf.download(ticker, period="1y", interval="1d")
    data['Date'] = pd.to_datetime(data.index)
    data.set_index('Date', inplace=True)

    # Feature engineering: Adding technical indicators (e.g., moving averages, RSI)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = 100 - (100 / (1 + (data['Close'].diff(1).gt(0).rolling(window=14).mean() /
                                     data['Close'].diff(1).lt(0).rolling(window=14).mean())))

    # Drop missing values
    data.dropna(inplace=True)

    # Defining the target: 1 if price increases, 0 if price decreases
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    # Features and target
    X = data[['Close', 'SMA_50', 'SMA_200', 'RSI']]
    y = data['Target']

    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, data


# XGBoost model for Buy/Sell Prediction
def train_buy_sell_model(X_train, y_train):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model


# Prediction using XGBoost
def predict_buy_sell(model, X_test):
    return model.predict(X_test)


# Streamlit UI for Buy/Sell Prediction
st.title("📈 AI-Based Buy/Sell Stock Predictor")

# Stock Ticker input
ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL", key="stock_ticker_input")
st.sidebar.info("The model predicts Buy/Sell signal based on stock price trends.")

if ticker_input:
    st.info(f"🔍 Fetching data for {ticker_input}...")

    # Prepare Data
    X, y, data = prepare_data(ticker_input)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train the model
    model = train_buy_sell_model(X_train, y_train)

    # Make predictions
    predictions = predict_buy_sell(model, X_test)

    # Convert predictions to Buy/Sell signals
    prediction_df = pd.DataFrame({'Date': data.index[-len(predictions):], 'Prediction': predictions})
    prediction_df['Prediction'] = prediction_df['Prediction'].map({1: 'Buy', 0: 'Sell'})

    st.write("Buy/Sell Predictions for the Stock:")
    st.dataframe(prediction_df)

    # Visualize the Buy/Sell predictions on stock price chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Stock Price', line=dict(color='blue')))

    buy_signals = prediction_df[prediction_df['Prediction'] == 'Buy']
    sell_signals = prediction_df[prediction_df['Prediction'] == 'Sell']

    fig.add_trace(go.Scatter(x=buy_signals['Date'], y=data.loc[buy_signals['Date']]['Close'],
                             mode='markers', name='Buy Signal',
                             marker=dict(symbol='triangle-up', color='green', size=10)))

    fig.add_trace(go.Scatter(x=sell_signals['Date'], y=data.loc[sell_signals['Date']]['Close'],
                             mode='markers', name='Sell Signal',
                             marker=dict(symbol='triangle-down', color='red', size=10)))

    fig.update_layout(title=f"Buy/Sell Predictions for {ticker_input}",
                      xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

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
