# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from newspaper import Article
import time

# Set page configuration
st.set_page_config(
    page_title="StockInsight - AI-Powered Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme
def apply_dark_theme():
    st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    .stSelectbox>div>div>select {
        background-color: #262730;
        color: white;
    }
    .css-1d391kg, .css-12oz5g7 {
        background-color: #0E1117;
    }
    </style>
    """, unsafe_allow_html=True)

apply_dark_theme()

# News API key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Cache data to improve performance
@st.cache_data(ttl=3600)
def get_stock_data(symbol, period="1y"):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except:
        return None, None

@st.cache_data(ttl=3600)
def get_company_news(company_name):
    try:
        url = f"https://newsapi.org/v2/everything?q={company_name}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        data = response.json()
        return data['articles'][:5]  # Return top 5 articles
    except:
        return []

@st.cache_data(ttl=3600)
def get_top_gainers_losers():
    try:
        # Using yfinance to get market movers
        tickers = yf.Tickers("^GSPC ^IXIC ^DJI")
        data = {}
        for symbol, ticker in tickers.tickers.items():
            info = ticker.info
            data[symbol] = {
                'regularMarketPrice': info.get('regularMarketPrice', 'N/A'),
                'regularMarketChange': info.get('regularMarketChange', 'N/A'),
                'regularMarketChangePercent': info.get('regularMarketChangePercent', 'N/A')
            }
        return data
    except:
        return {}

@st.cache_data(ttl=3600)
def predict_stock_price(symbol, days=30):
    try:
        # Get historical data
        stock = yf.Ticker(symbol)
        hist = stock.history(period="2y")
        
        # Prepare features
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
        hist['EMA_12'] = hist['Close'].ewm(span=12).mean()
        hist['EMA_26'] = hist['Close'].ewm(span=26).mean()
        hist['MACD'] = hist['EMA_12'] - hist['EMA_26']
        hist['RSI'] = compute_rsi(hist['Close'])
        
        # Drop NaN values
        hist = hist.dropna()
        
        # Features and target
        features = ['Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'MACD', 'RSI']
        X = hist[features]
        y = hist['Close']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        last_data = X.iloc[-1:].values
        future_predictions = []
        
        for _ in range(days):
            next_pred = model.predict(last_data)[0]
            future_predictions.append(next_pred)
            # Update last_data with the prediction for next day
            last_data = np.roll(last_data, -1)
            last_data[0, -1] = next_pred  # Update the Close price with prediction
        
        # Create future dates
        last_date = hist.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
        
        return future_dates, future_predictions, model.score(X_test, y_test)
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return [], [], 0

def compute_rsi(prices, window=14):
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1. + rs)

    for i in range(window, len(prices)):
        delta = deltas[i-1]  # cause the diff is 1 shorter

        if delta > 0:
            up_val = delta
            down_val = 0.
        else:
            up_val = 0.
            down_val = -delta

        up = (up*(window-1) + up_val)/window
        down = (down*(window-1) + down_val)/window

        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi

# SIP Calculator
def sip_calculator(monthly_investment, annual_return, years):
    monthly_return = annual_return / 12 / 100
    months = years * 12
    future_value = monthly_investment * (((1 + monthly_return) ** months - 1) / monthly_return) * (1 + monthly_return)
    return future_value

# Learning Materials
learning_materials = {
    "Basics": [
        {"title": "Introduction to Stock Market", "content": "The stock market is where investors buy and sell shares of companies..."},
        {"title": "Understanding Market Indicators", "content": "Key indicators like GDP, inflation, and employment rates affect market performance..."},
    ],
    "Technical Analysis": [
        {"title": "Moving Averages", "content": "Moving averages help smooth price data to identify trends..."},
        {"title": "RSI Indicator", "content": "The Relative Strength Index identifies overbought or oversold conditions..."},
    ],
    "Investment Strategies": [
        {"title": "Value Investing", "content": "Value investors seek stocks they believe are undervalued..."},
        {"title": "Growth Investing", "content": "Growth investors look for companies with strong growth potential..."},
    ]
}

# Main App
def main():
    st.sidebar.title("StockInsight 📈")
    menu = st.sidebar.selectbox("Navigation", [
        "Dashboard", 
        "Company Overview", 
        "Stock Prediction", 
        "SIP Calculator", 
        "IPO Prediction", 
        "Market Overview",
        "Mutual Funds",
        "Learning Center"
    ])
    
    if menu == "Dashboard":
        st.title("StockInsight Dashboard")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT):", "AAPL")
            
        with col2:
            period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
            
        if symbol:
            hist, info = get_stock_data(symbol, period)
            
            if hist is not None and not hist.empty:
                # Display basic info
                st.subheader(f"{info.get('longName', symbol)} ({symbol})")
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = hist['Close'][-1]
                prev_close = hist['Close'][-2] if len(hist) > 1 else current_price
                price_change = current_price - prev_close
                percent_change = (price_change / prev_close) * 100
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}", 
                             f"{price_change:.2f} ({percent_change:.2f}%)")
                
                with col2:
                    st.metric("Open", f"${hist['Open'][-1]:.2f}")
                
                with col3:
                    st.metric("Day's Range", f"${hist['Low'][-1]:.2f} - ${hist['High'][-1]:.2f}")
                
                with col4:
                    st.metric("Volume", f"{hist['Volume'][-1]:,}")
                
                # Price chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='Price'
                ))
                fig.update_layout(
                    title=f"{symbol} Stock Price",
                    yaxis_title="Price (USD)",
                    xaxis_title="Date",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional charts
                st.subheader("Technical Indicators")
                indicator = st.selectbox("Select Indicator", ["MACD", "RSI", "Moving Averages"])
                
                if indicator == "MACD":
                    exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
                    exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
                    macd = exp12 - exp26
                    signal = macd.ewm(span=9, adjust=False).mean()
                    
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        name='Price'
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=macd,
                        name='MACD',
                        line=dict(color='orange')
                    ), row=2, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=signal,
                        name='Signal',
                        line=dict(color='blue')
                    ), row=2, col=1)
                    
                    fig.update_layout(
                        title="MACD Indicator",
                        template="plotly_dark",
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif indicator == "RSI":
                    rsi = compute_rsi(hist['Close'])
                    
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        name='Price'
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=rsi,
                        name='RSI',
                        line=dict(color='purple')
                    ), row=2, col=1)
                    
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.update_layout(
                        title="RSI Indicator",
                        template="plotly_dark",
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:  # Moving Averages
                    hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                    hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        name='Price'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['SMA_50'],
                        name='50-Day SMA',
                        line=dict(color='orange')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['SMA_200'],
                        name='200-Day SMA',
                        line=dict(color='purple')
                    ))
                    
                    fig.update_layout(
                        title="Moving Averages",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # News section
                st.subheader("Latest News")
                company_name = info.get('longName', symbol)
                news_articles = get_company_news(company_name)
                
                for article in news_articles:
                    with st.expander(article['title']):
                        st.write(f"**Source:** {article['source']['name']}")
                        st.write(f"**Published At:** {article['publishedAt']}")
                        st.write(article['description'])
                        st.write(f"[Read more]({article['url']})")
            else:
                st.error("Invalid stock symbol or data not available. Please try another symbol.")
    
    elif menu == "Company Overview":
        st.title("Company Overview")
        symbol = st.text_input("Enter Stock Symbol:", "AAPL")
        
        if symbol:
            hist, info = get_stock_data(symbol, "1y")
            
            if hist is not None and not hist.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(f"{info.get('longName', symbol)} ({symbol})")
                    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.write(f"**Country:** {info.get('country', 'N/A')}")
                    st.write(f"**Website:** {info.get('website', 'N/A')}")
                    
                    st.subheader("Key Metrics")
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        st.write(f"**Market Cap:** ${info.get('marketCap', 'N/A'):,}")
                        st.write(f"**PE Ratio:** {info.get('trailingPE', 'N/A')}")
                        st.write(f"**EPS:** {info.get('trailingEps', 'N/A')}")
                        st.write(f"**52 Week High:** ${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}")
                    
                    with metrics_col2:
                        st.write(f"**Enterprise Value:** ${info.get('enterpriseValue', 'N/A'):,}")
                        st.write(f"**PEG Ratio:** {info.get('pegRatio', 'N/A')}")
                        st.write(f"**Dividend Yield:** {info.get('dividendYield', 'N/A')}")
                        st.write(f"**52 Week Low:** ${info.get('fiftyTwoWeekLow', 'N/A'):.2f}")
                
                with col2:
                    # Display company logo if available
                    if 'logo_url' in info:
                        st.image(info['logo_url'], width=150)
                    
                    # Download JSON data
                    st.download_button(
                        label="Download Company Data as JSON",
                        data=json.dumps(info, indent=2),
                        file_name=f"{symbol}_company_data.json",
                        mime="application/json"
                    )
                
                # Financials
                st.subheader("Financial Highlights")
                financials = yf.Ticker(symbol).financials
                if not financials.empty:
                    st.dataframe(financials.head())
                else:
                    st.info("Financial data not available for this company.")
                
                # Company description
                st.subheader("Business Description")
                st.write(info.get('longBusinessSummary', 'No description available.'))
            else:
                st.error("Invalid stock symbol or data not available. Please try another symbol.")
    
    elif menu == "Stock Prediction":
        st.title("Stock Price Prediction")
        symbol = st.text_input("Enter Stock Symbol:", "AAPL")
        days = st.slider("Prediction Period (days)", 7, 90, 30)
        
        if st.button("Predict"):
            with st.spinner("Training model and making predictions..."):
                future_dates, future_predictions, accuracy = predict_stock_price(symbol, days)
                
                if future_dates:
                    st.success(f"Model Accuracy: {accuracy:.2%}")
                    
                    # Get historical data for chart
                    hist, _ = get_stock_data(symbol, "6mo")
                    
                    # Create prediction chart
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        name='Historical Price',
                        line=dict(color='blue')
                    ))
                    
                    # Prediction data
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_predictions,
                        name='Predicted Price',
                        line=dict(color='orange', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} Price Prediction",
                        yaxis_title="Price (USD)",
                        xaxis_title="Date",
                        template="plotly_dark"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show prediction table
                    prediction_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_predictions
                    })
                    
                    st.subheader("Daily Predictions")
                    st.dataframe(prediction_df)
                else:
                    st.error("Could not generate predictions. Please try a different stock symbol.")
    
    elif menu == "SIP Calculator":
        st.title("SIP Calculator")
        st.write("Calculate the future value of your Systematic Investment Plan (SIP)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_investment = st.number_input("Monthly Investment ($)", min_value=10, value=1000, step=100)
            years = st.slider("Investment Period (years)", 1, 30, 10)
        
        with col2:
            annual_return = st.slider("Expected Annual Return (%)", 1.0, 30.0, 12.0)
        
        if st.button("Calculate"):
            future_value = sip_calculator(monthly_investment, annual_return, years)
            total_investment = monthly_investment * 12 * years
            wealth_gain = future_value - total_investment
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Investment", f"${total_investment:,.2f}")
            
            with col2:
                st.metric("Future Value", f"${future_value:,.2f}")
            
            with col3:
                st.metric("Wealth Gain", f"${wealth_gain:,.2f}")
            
            # Chart showing growth over time
            years_list = list(range(1, years+1))
            values = [sip_calculator(monthly_investment, annual_return, y) for y in years_list]
            investments = [monthly_investment * 12 * y for y in years_list]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years_list,
                y=values,
                name='Future Value',
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=years_list,
                y=investments,
                name='Total Investment',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title="SIP Growth Over Time",
                xaxis_title="Years",
                yaxis_title="Amount ($)",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif menu == "IPO Prediction":
        st.title("IPO Prediction")
        st.info("This feature uses machine learning to predict IPO performance based on historical data and market conditions.")
        
        # Placeholder for IPO prediction - in a real app, you would implement this
        st.write("IPO prediction functionality is under development.")
        st.write("Check back soon for updates!")
    
    elif menu == "Market Overview":
        st.title("Market Overview")
        st.subheader("Top Gainers & Losers")
        
        market_data = get_top_gainers_losers()
        
        if market_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Gainers")
                gainers = []
                
                for symbol, data in market_data.items():
                    if isinstance(data.get('regularMarketChangePercent', 0), (int, float)) and data['regularMarketChangePercent'] > 0:
                        gainers.append({
                            'Symbol': symbol,
                            'Price': data.get('regularMarketPrice', 'N/A'),
                            'Change': data.get('regularMarketChange', 'N/A'),
                            'Change %': f"{data.get('regularMarketChangePercent', 'N/A'):.2f}%"
                        })
                
                if gainers:
                    st.table(pd.DataFrame(gainers))
                else:
                    st.info("No gainers data available.")
            
            with col2:
                st.subheader("Top Losers")
                losers = []
                
                for symbol, data in market_data.items():
                    if isinstance(data.get('regularMarketChangePercent', 0), (int, float)) and data['regularMarketChangePercent'] < 0:
                        losers.append({
                            'Symbol': symbol,
                            'Price': data.get('regularMarketPrice', 'N/A'),
                            'Change': data.get('regularMarketChange', 'N/A'),
                            'Change %': f"{data.get('regularMarketChangePercent', 'N/A'):.2f}%"
                        })
                
                if losers:
                    st.table(pd.DataFrame(losers))
                else:
                    st.info("No losers data available.")
        else:
            st.error("Market data not available at the moment.")
        
        # Market indices
        st.subheader("Major Indices")
        indices = ['^GSPC', '^IXIC', '^DJI', '^N225', '^FTSE']
        
        index_data = {}
        for index in indices:
            try:
                ticker = yf.Ticker(index)
                info = ticker.info
                index_data[index] = {
                    'Name': info.get('shortName', index),
                    'Price': info.get('regularMarketPrice', 'N/A'),
                    'Change': info.get('regularMarketChange', 'N/A'),
                    'Change %': f"{info.get('regularMarketChangePercent', 'N/A'):.2f}%"
                }
            except:
                index_data[index] = {
                    'Name': index,
                    'Price': 'N/A',
                    'Change': 'N/A',
                    'Change %': 'N/A'
                }
        
        st.table(pd.DataFrame.from_dict(index_data, orient='index'))
    
    elif menu == "Mutual Funds":
        st.title("Mutual Fund Analysis")
        st.info("This section provides analysis and information about various mutual funds.")
        
        # Placeholder for mutual fund data
        fund_categories = ["Equity", "Debt", "Hybrid", "ELSS", "Sectoral"]
        selected_category = st.selectbox("Select Fund Category", fund_categories)
        
        if selected_category:
            st.write(f"Displaying top {selected_category} funds...")
            st.write("Mutual fund data loading functionality will be implemented here.")
    
    elif menu == "Learning Center":
        st.title("Stock Market Learning Center")
        
        category = st.selectbox("Select Category", list(learning_materials.keys()))
        
        if category:
            materials = learning_materials[category]
            
            for material in materials:
                with st.expander(material['title']):
                    st.write(material['content'])
        
        # Additional resources
        st.subheader("Additional Resources")
        st.write("""
        - [Investopedia](https://www.investopedia.com/) - Comprehensive investing education
        - [SEC EDGAR Database](https://www.sec.gov/edgar.shtml) - Company filings and reports
        - [Yahoo Finance](https://finance.yahoo.com/) - Market news and data
        - [Bloomberg Markets](https://www.bloomberg.com/markets) - Global financial news
        """)

if __name__ == "__main__":
    main()
