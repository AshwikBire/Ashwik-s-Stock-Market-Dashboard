import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import base64
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="StockSense - AI-Powered Stock Prediction & Learning",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme
def apply_dark_theme():
    # Custom CSS for dark theme
    st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #4CAF50;
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
    .stSlider>div>div>div>div {
        background-color: #4CAF50;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

apply_dark_theme()

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'ticker_data' not in st.session_state:
    st.session_state.ticker_data = None
if 'ticker_info' not in st.session_state:
    st.session_state.ticker_info = None
if 'news_data' not in st.session_state:
    st.session_state.news_data = None

# News API key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Function to fetch stock data
def fetch_stock_data(ticker, period='1y'):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        info = stock.info
        return data, info
    except:
        return None, None

# Function to fetch news
def fetch_news(ticker=None):
    try:
        if ticker:
            url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"
        else:
            url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey={NEWS_API_KEY}"
        
        response = requests.get(url)
        news_data = response.json()
        return news_data['articles'][:10]  # Return top 10 articles
    except:
        return []

# Function to train prediction model
def train_prediction_model(data, days=30):
    # Prepare data for prediction
    df = data.copy()
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'])
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    df['Price'] = df['Close']
    
    # Create features
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df = df.dropna()
    
    # Features and target
    features = ['Days', 'MA10', 'MA50', 'RSI']
    X = df[features]
    y = df['Price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    mae = mean_absolute_error(y_test, test_predictions)
    
    # Future prediction
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
    future_days = [(date - df['Date'].min()).days for date in future_dates]
    
    # Prepare future features (this is simplified)
    last_ma10 = df['MA10'].iloc[-1]
    last_ma50 = df['MA50'].iloc[-1]
    last_rsi = df['RSI'].iloc[-1]
    
    future_features = []
    for day in future_days:
        future_features.append([day, last_ma10, last_ma50, last_rsi])
    
    future_predictions = model.predict(future_features)
    
    return {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'mae': mae,
        'future_dates': future_dates,
        'future_predictions': future_predictions
    }

# Function to compute RSI
def compute_rsi(prices, window=14):
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1. + rs)
    
    for i in range(window, len(prices)):
        delta = deltas[i-1]
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

# Function to display company overview
def display_company_overview(info):
    st.subheader("Company Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Company Name", info.get('longName', 'N/A'))
        st.metric("Sector", info.get('sector', 'N/A'))
        st.metric("Industry", info.get('industry', 'N/A'))
    
    with col2:
        st.metric("Market Cap", f"${info.get('marketCap', 0):,}" if info.get('marketCap') else 'N/A')
        st.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
        st.metric("EPS", info.get('trailingEps', 'N/A'))
    
    with col3:
        st.metric("52 Week High", info.get('fiftyTwoWeekHigh', 'N/A'))
        st.metric("52 Week Low", info.get('fiftyTwoWeekLow', 'N/A'))
        st.metric("Volume", f"{info.get('volume', 0):,}" if info.get('volume') else 'N/A')
    
    # Display JSON data
    st.subheader("Raw Company Data (JSON)")
    with st.expander("View JSON Data"):
        st.json(info)

# Function to display stock chart
def display_stock_chart(data, predictions=None):
    fig = go.Figure()
    
    # Add actual price data
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Add predictions if available
    if predictions:
        future_dates = predictions['future_dates']
        future_predictions = predictions['future_predictions']
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions,
            mode='lines',
            name='Predictions',
            line=dict(color='#FFA500', dash='dash')
        ))
    
    fig.update_layout(
        title='Stock Price with Predictions',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Function to display news
def display_news(news_data, title="Latest Financial News"):
    st.subheader(title)
    
    for i, article in enumerate(news_data):
        with st.expander(f"{i+1}. {article['title']}"):
            st.write(f"**Source:** {article['source']['name']}")
            st.write(f"**Published At:** {article['publishedAt']}")
            st.write(article['description'])
            st.markdown(f"[Read more]({article['url']})")

# Function for SIP calculator
def sip_calculator():
    st.subheader("SIP Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_investment = st.number_input("Monthly Investment (₹)", min_value=500, value=5000, step=500)
        investment_period = st.slider("Investment Period (Years)", min_value=1, max_value=30, value=10)
        expected_return = st.slider("Expected Annual Return (%)", min_value=1, max_value=30, value=12)
    
    with col2:
        # Calculate SIP
        monthly_rate = expected_return / 100 / 12
        months = investment_period * 12
        future_value = monthly_investment * ((1 + monthly_rate)**months - 1) / monthly_rate * (1 + monthly_rate)
        total_investment = monthly_investment * months
        estimated_returns = future_value - total_investment
        
        st.metric("Total Investment", f"₹{total_investment:,.2f}")
        st.metric("Estimated Returns", f"₹{estimated_returns:,.2f}")
        st.metric("Future Value", f"₹{future_value:,.2f}")

# Function for IPO prediction (simplified)
def ipo_prediction():
    st.subheader("IPO Prediction")
    
    st.info("This feature uses machine learning to predict IPO performance based on historical data and market conditions.")
    
    # Sample IPO data (in a real app, this would come from an API)
    ipo_data = {
        'Company': ['TechInnovate', 'GreenEnergy Inc', 'BioPharma Solutions', 'NextGen Retail'],
        'Sector': ['Technology', 'Energy', 'Healthcare', 'Retail'],
        'Issue Size (Cr)': [1200, 800, 1500, 900],
        'Price Band (₹)': ['300-320', '200-210', '450-465', '180-190'],
        'Predicted Gain (%)': [25.4, 18.7, 32.1, 12.3],
        'Confidence': ['High', 'Medium', 'High', 'Medium']
    }
    
    df = pd.DataFrame(ipo_data)
    st.dataframe(df.style.highlight_max(subset=['Predicted Gain (%)'], color='#2ecc71'))
    
    st.write("""
    **Disclaimer:** IPO predictions are based on historical data and market conditions. 
    Past performance is not indicative of future results. Investing in IPOs carries risks.
    """)

# Function for top gainers and losers
def top_gainers_losers():
    st.subheader("Top Gainers & Losers")
    
    # Sample data (in a real app, this would come from an API)
    gainers_data = {
        'Symbol': ['RELIANCE', 'HDFC', 'INFY', 'TCS', 'BAJFIN'],
        'Price (₹)': [2750.50, 1620.75, 1785.25, 3315.80, 7452.60],
        'Change (%)': [5.2, 4.8, 4.1, 3.9, 3.5],
        'Volume': ['12.5M', '8.2M', '10.1M', '7.8M', '5.3M']
    }
    
    losers_data = {
        'Symbol': ['YESBANK', 'VEDL', 'TATA', 'ONGC', 'IOC'],
        'Price (₹)': [14.25, 235.40, 425.80, 135.75, 95.60],
        'Change (%)': [-4.8, -4.2, -3.7, -3.2, -2.9],
        'Volume': ['22.1M', '18.7M', '15.3M', '12.8M', '10.5M']
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top Gainers**")
        gainers_df = pd.DataFrame(gainers_data)
        st.dataframe(gainers_df.style.applymap(lambda x: 'color: #2ecc71' if isinstance(x, (int, float)) and x > 0 else '', subset=['Change (%)']))
    
    with col2:
        st.write("**Top Losers**")
        losers_df = pd.DataFrame(losers_data)
        st.dataframe(losers_df.style.applymap(lambda x: 'color: #e74c3c' if isinstance(x, (int, float)) and x < 0 else '', subset=['Change (%)']))

# Function for mutual funds
def mutual_funds():
    st.subheader("Mutual Fund Analysis")
    
    fund_type = st.selectbox("Select Fund Type", [
        "Equity Funds", 
        "Debt Funds", 
        "Hybrid Funds", 
        "ELSS", 
        "Index Funds"
    ])
    
    # Sample fund data (in a real app, this would come from an API)
    funds_data = {
        'Fund Name': [
            'BlueChip Equity Fund', 
            'Growth Opportunities Fund', 
            'Conservative Hybrid Fund', 
            'Tax Saver ELSS', 
            'Nifty 50 Index Fund'
        ],
        '1Y Return (%)': [18.5, 22.1, 12.3, 19.7, 16.8],
        '3Y Return (%)': [15.2, 18.7, 10.5, 16.9, 14.3],
        '5Y Return (%)': [13.8, 16.4, 9.2, 14.7, 12.6],
        'Risk': ['Moderate', 'High', 'Low', 'Moderate', 'Low'],
        'Rating': ['★★★★★', '★★★★', '★★★★☆', '★★★★★', '★★★★']
    }
    
    df = pd.DataFrame(funds_data)
    st.dataframe(df)
    
    # Fund selector for detailed analysis
    selected_fund = st.selectbox("Select Fund for Detailed Analysis", df['Fund Name'].tolist())
    
    if selected_fund:
        st.write(f"**Detailed Analysis for {selected_fund}**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("1 Year Return", "18.5%")
            st.metric("Expense Ratio", "0.65%")
        
        with col2:
            st.metric("3 Year Return", "15.2%")
            st.metric("Assets (Cr)", "2,450")
        
        with col3:
            st.metric("5 Year Return", "13.8%")
            st.metric("Risk", "Moderate")
        
        # Performance chart
        performance_data = {
            'Year': [2018, 2019, 2020, 2021, 2022, 2023],
            'Return (%)': [8.2, 12.5, -2.1, 18.7, 15.3, 18.5]
        }
        
        fig = px.line(performance_data, x='Year', y='Return (%)', title=f'{selected_fund} Performance')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

# Function for learning materials
def learning_materials():
    st.subheader("Stock Market Learning Center")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Basics", 
        "Technical Analysis", 
        "Fundamental Analysis", 
        "Advanced Strategies"
    ])
    
    with tab1:
        st.write("""
        ## Stock Market Basics
        
        ### What is a Stock?
        A stock represents ownership in a corporation and constitutes a claim on part of the corporation's assets and earnings.
        
        ### How Stock Markets Work
        Stock markets are where buyers and sellers meet to trade shares of public companies. Prices are determined by supply and demand.
        
        ### Key Concepts:
        - **Bull Market**: A period of rising stock prices
        - **Bear Market**: A period of falling stock prices
        - **Dividends**: Payments made by a corporation to its shareholders
        - **Market Capitalization**: Total value of a company's outstanding shares
        """)
        
        st.video("https://www.youtube.com/embed/F3QpgXBtDeo")
    
    with tab2:
        st.write("""
        ## Technical Analysis
        
        ### What is Technical Analysis?
        Technical analysis is the study of historical market data, including price and volume, to predict future price movements.
        
        ### Common Indicators:
        - **Moving Averages**: Identify trends by smoothing price data
        - **RSI (Relative Strength Index)**: Measures speed and change of price movements
        - **MACD**: Shows relationship between two moving averages
        - **Bollinger Bands**: Volatility bands placed above and below a moving average
        
        ### Chart Patterns:
        - Head and Shoulders
        - Double Top/Double Bottom
        - Triangles (Ascending, Descending, Symmetrical)
        """)
        
        st.image("https://www.investopedia.com/thmb/4KJGrn-MvC3kLp_9gXnGJcRr7_c=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/dotdash_Final_Technical_Analysis_Aug_2020-01-0a9ee9d376e84d5e8f5f51acc32b67c0.jpg", width=500)
    
    with tab3:
        st.write("""
        ## Fundamental Analysis
        
        ### What is Fundamental Analysis?
        Fundamental analysis involves evaluating a company's financial statements to determine the fair value of the business.
        
        ### Key Financial Ratios:
        - **P/E Ratio**: Price-to-Earnings ratio
        - **P/B Ratio**: Price-to-Book ratio
        - **Debt-to-Equity**: Measures financial leverage
        - **ROE**: Return on Equity
        - **Current Ratio**: Measures liquidity
        
        ### Important Statements:
        - Income Statement
        - Balance Sheet
        - Cash Flow Statement
        """)
        
        st.download_button(
            label="Download Fundamental Analysis Guide",
            data="Fundamental analysis is the cornerstone of investing. This guide covers...",
            file_name="fundamental_analysis_guide.txt",
            mime="text/plain"
        )
    
    with tab4:
        st.write("""
        ## Advanced Strategies
        
        ### Options Trading
        Options give the buyer the right, but not the obligation, to buy or sell an asset at a set price on or before a given date.
        
        ### Swing Trading
        Swing traders hold positions for several days or weeks to capitalize on expected upward or downward market shifts.
        
        ### Value Investing
        Value investors look for securities that appear underpriced by some form of fundamental analysis.
        
        ### Growth Investing
        Growth investors seek companies that offer strong earnings growth potential.
        """)
        
        st.write("**Recommended Books:**")
        st.write("- The Intelligent Investor by Benjamin Graham")
        st.write("- A Random Walk Down Wall Street by Burton Malkiel")
        st.write("- Common Stocks and Uncommon Profits by Philip Fisher")

# Main app function
def main():
    # Sidebar navigation
    st.sidebar.image("https://img.icons8.com/dusk/64/000000/stock.png", width=64)
    st.sidebar.title("StockSense")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "Home", 
            "Stock Analysis", 
            "SIP Calculator", 
            "IPO Prediction", 
            "Top Gainers/Losers", 
            "Mutual Funds", 
            "Learning Center"
        ]
    )
    
    # Display selected page
    if page == "Home":
        st.title("StockSense - AI-Powered Stock Prediction & Learning")
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("""
            ### Welcome to StockSense!
            
            StockSense is a comprehensive platform for stock market analysis, prediction, and education.
            Use our AI-powered tools to make informed investment decisions and expand your knowledge
            about the stock market.
            
            **Features:**
            - Real-time stock data and analysis
            - AI-powered price predictions
            - SIP calculator for investment planning
            - IPO performance predictions
            - Top gainers and losers tracking
            - Mutual fund analysis
            - Comprehensive learning resources
            """)
        
        with col2:
            st.image("https://img.icons8.com/dusk/200/000000/stock.png")
        
        # Quick stock lookup
        st.subheader("Quick Stock Lookup")
        quick_ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, RELIANCE.NS):", "RELIANCE.NS")
        
        if st.button("Get Quick Analysis"):
            with st.spinner("Fetching data..."):
                data, info = fetch_stock_data(quick_ticker)
                news = fetch_news(quick_ticker.split('.')[0])
                
                if data is not None and not data.empty:
                    st.success("Data fetched successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"₹{data['Close'].iloc[-1]:.2f}")
                    
                    with col2:
                        change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                        pct_change = (change / data['Close'].iloc[-2]) * 100
                        st.metric("Change", f"₹{change:.2f}", f"{pct_change:.2f}%")
                    
                    with col3:
                        st.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
                    
                    # Display mini chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index, 
                        y=data['Close'], 
                        mode='lines', 
                        name='Price',
                        line=dict(color='#4CAF50')
                    ))
                    fig.update_layout(
                        title=f'{quick_ticker} Price History',
                        template='plotly_dark',
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show latest news
                    if news:
                        display_news(news, f"Latest News about {quick_ticker}")
                else:
                    st.error("Could not fetch data for the specified symbol.")
    
    elif page == "Stock Analysis":
        st.title("Stock Analysis & Prediction")
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ticker = st.text_input("Enter Stock Symbol:", "RELIANCE.NS")
        
        with col2:
            period = st.selectbox("Select Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        
        if st.button("Analyze Stock"):
            with st.spinner("Fetching data and training model..."):
                data, info = fetch_stock_data(ticker, period)
                news = fetch_news(ticker.split('.')[0])
                
                if data is not None and not data.empty:
                    st.session_state.ticker_data = data
                    st.session_state.ticker_info = info
                    st.session_state.news_data = news
                    
                    st.success("Data fetched successfully!")
                else:
                    st.error("Could not fetch data for the specified symbol.")
                    return
            
        if st.session_state.ticker_data is not None:
            data = st.session_state.ticker_data
            info = st.session_state.ticker_info
            news = st.session_state.news_data
            
            # Display company overview
            display_company_overview(info)
            
            # Make predictions
            with st.spinner("Training prediction model..."):
                predictions = train_prediction_model(data)
            
            # Display stock chart with predictions
            display_stock_chart(data, predictions)
            
            # Display prediction metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Training RMSE", f"{predictions['train_rmse']:.2f}")
            
            with col2:
                st.metric("Test RMSE", f"{predictions['test_rmse']:.2f}")
            
            with col3:
                st.metric("Mean Absolute Error", f"{predictions['mae']:.2f}")
            
            # Display future predictions
            st.subheader("Future Price Predictions")
            future_df = pd.DataFrame({
                'Date': predictions['future_dates'],
                'Predicted Price': predictions['future_predictions']
            })
            st.dataframe(future_df)
            
            # Display news
            if news:
                display_news(news, f"Latest News about {ticker}")
    
    elif page == "SIP Calculator":
        st.title("SIP Calculator")
        st.markdown("---")
        sip_calculator()
    
    elif page == "IPO Prediction":
        st.title("IPO Prediction")
        st.markdown("---")
        ipo_prediction()
    
    elif page == "Top Gainers/Losers":
        st.title("Top Gainers & Losers")
        st.markdown("---")
        top_gainers_losers()
    
    elif page == "Mutual Funds":
        st.title("Mutual Fund Analysis")
        st.markdown("---")
        mutual_funds()
    
    elif page == "Learning Center":
        st.title("Stock Market Learning Center")
        st.markdown("---")
        learning_materials()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
            <p>Disclaimer: This application is for educational purposes only. 
            Stock market investments are subject to market risks. 
            Past performance is not indicative of future results.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
