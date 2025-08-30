import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
from plotly import graph_objects as go
from streamlit_option_menu import option_menu
from textblob import TextBlob
from xgboost import XGBRegressor
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import plotly.express as px
import ta  # Technical analysis library
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="MarketMentor", layout="wide", page_icon="📈")

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "MarketMentor",
        ["Home", "Company Overview", "Market Movers", "F&O", "Global Markets", 
         "Mutual Funds", "SIP Calculator", "IPO Tracker", "Predictions for Mutual Funds & IPOs", 
         "Mutual Fund NAV Viewer", "Sectors", "News", "Learning", "Volume Spike", 
         "Stock Screener", "Predictions", "Buy/Sell Predictor", "News Sentiment", 
         "Technical Analysis", "Portfolio Tracker", "Economic Calendar"],
        icons=['house', 'building', 'graph-up', 'activity', 'globe', 
               'bank', 'calculator', 'rocket', 'lightning', 
               'bar-chart', 'grid', 'newspaper', 'book', 'activity', 
               'search', 'graph-up-arrow', 'currency-exchange', 'chat', 
               'graph-up', 'wallet', 'calendar'],
        menu_icon="cast",
        default_index=0
    )

# Cache data functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker, period="1y"):
    return yf.Ticker(ticker).history(period=period)

@st.cache_data(ttl=3600)
def get_indices_data(indices_dict):
    data = {}
    for symbol, name in indices_dict.items():
        try:
            ticker_data = yf.Ticker(symbol).history(period="1d")
            if not ticker_data.empty:
                last_close = round(ticker_data['Close'].iloc[-1], 2)
                change = round(ticker_data['Close'].iloc[-1] - ticker_data['Open'].iloc[-1], 2)
                percent_change = round((change / ticker_data['Open'].iloc[-1]) * 100, 2)
                data[name] = {"value": last_close, "change": percent_change}
        except:
            continue
    return data

@st.cache_data(ttl=3600)
def get_news(query="stock market", page_size=10):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize={page_size}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    return []

# Home - Market Overview
if selected == "Home":
    st.title("🏠 Home - Market Overview")
    
    # Display major indices
    indices = {
        "^NSEI": "Nifty 50",
        "^BSESN": "Sensex",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^GSPC": "S&P 500",
    }
    
    st.subheader("Major Indices Performance")
    indices_data = get_indices_data(indices)
    
    cols = st.columns(len(indices_data))
    for idx, (name, data) in enumerate(indices_data.items()):
        cols[idx].metric(label=name, value=f"{data['value']}", delta=f"{data['change']}%")
    
    # Market sentiment gauge
    st.subheader("Market Sentiment")
    avg_sentiment = sum(data['change'] for data in indices_data.values()) / len(indices_data)
    
    if avg_sentiment > 1:
        sentiment = "Bullish 🐂"
        color = "green"
    elif avg_sentiment < -1:
        sentiment = "Bearish 🐻"
        color = "red"
    else:
        sentiment = "Neutral 😐"
        color = "gray"
        
    st.markdown(f"<h3 style='text-align: center; color: {color};'>{sentiment}</h3>", unsafe_allow_html=True)
    
    # Top news
    st.subheader("Top Market News")
    articles = get_news("stock market", 3)
    for article in articles:
        with st.expander(f"{article['title']} - {article['source']['name']}"):
            st.write(article.get('description', 'No description available'))
            st.markdown(f"[Read more]({article['url']})")

# Company Overview
elif selected == "Company Overview":
    st.title("🏢 Company Overview")
    
    ticker = st.text_input("Enter Company Ticker (e.g., RELIANCE.NS, AAPL):", "RELIANCE.NS")
    
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")
            
            if hist.empty:
                st.warning("No data available for this ticker.")
            else:
                # Display company info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Company Information")
                    st.write(f"**Name:** {info.get('longName', 'N/A')}")
                    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.write(f"**Market Cap:** ${info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "**Market Cap:** N/A")
                    st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                
                with col2:
                    st.subheader("Stock Performance")
                    current_price = info.get('regularMarketPrice', hist['Close'].iloc[-1] if not hist.empty else 'N/A')
                    prev_close = info.get('regularMarketPreviousClose', 'N/A')
                    
                    if current_price != 'N/A' and prev_close != 'N/A':
                        change = current_price - prev_close
                        percent_change = (change / prev_close) * 100
                        st.metric("Current Price", f"${current_price:.2f}" if isinstance(current_price, (int, float)) else current_price, 
                                 f"{percent_change:.2f}%")
                    else:
                        st.write("**Current Price:** N/A")
                    
                    st.write(f"**52 Week High:** ${info.get('fiftyTwoWeekHigh', 'N/A')}")
                    st.write(f"**52 Week Low:** ${info.get('fiftyTwoWeekLow', 'N/A')}")
                
                # Price chart
                st.subheader("Price Chart")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=hist.index,
                                            open=hist['Open'],
                                            high=hist['High'],
                                            low=hist['Low'],
                                            close=hist['Close'],
                                            name='Price'))
                fig.update_layout(title=f"{ticker} Price Chart",
                                 xaxis_title="Date",
                                 yaxis_title="Price ($)",
                                 height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Financials
                st.subheader("Financial Highlights")
                
                try:
                    financials = stock.financials
                    if not financials.empty:
                        st.write("**Recent Financials**")
                        st.dataframe(financials.head().T)
                except:
                    st.info("Financial data not available for this company.")
                
        except Exception as e:
            st.error(f"Error retrieving data: {e}")

# Market Movers - Top Gainers & Losers
elif selected == "Market Movers":
    st.title("📈 Market Movers")
    
    # Predefined list of popular Indian stocks
    popular_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
                     'SBIN.NS', 'HINDUNILVR.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'ITC.NS',
                     'KOTAKBANK.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS']
    
    # Fetch data for all stocks
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_stocks_data(stocks):
        data = {}
        for ticker in stocks:
            try:
                stock_data = yf.Ticker(ticker).history(period="2d")
                if not stock_data.empty and len(stock_data) >= 2:
                    prev_close = stock_data['Close'].iloc[-2]
                    current_price = stock_data['Close'].iloc[-1]
                    change = current_price - prev_close
                    percent_change = (change / prev_close) * 100
                    data[ticker] = {
                        'price': current_price,
                        'change': change,
                        'percent_change': percent_change,
                        'volume': stock_data['Volume'].iloc[-1]
                    }
            except:
                continue
        return data
    
    stocks_data = get_stocks_data(popular_stocks)
    
    if stocks_data:
        # Create DataFrame
        df = pd.DataFrame.from_dict(stocks_data, orient='index')
        df.index.name = 'Ticker'
        df.reset_index(inplace=True)
        
        # Top Gainers
        st.subheader("🚀 Top Gainers")
        gainers = df.nlargest(5, 'percent_change')
        for _, row in gainers.iterrows():
            st.metric(row['Ticker'], f"₹{row['price']:.2f}", f"{row['percent_change']:.2f}%")
        
        # Top Losers
        st.subheader("📉 Top Losers")
        losers = df.nsmallest(5, 'percent_change')
        for _, row in losers.iterrows():
            st.metric(row['Ticker'], f"₹{row['price']:.2f}", f"{row['percent_change']:.2f}%")
        
        # Most Active
        st.subheader("🔥 Most Active (Volume)")
        active = df.nlargest(5, 'volume')
        for _, row in active.iterrows():
            st.metric(row['Ticker'], f"₹{row['price']:.2f}", f"Volume: {row['volume']:,.0f}")
    else:
        st.warning("Could not fetch market data. Please try again later.")

# F&O Page
elif selected == "F&O":
    st.title("📑 F&O Overview")
    
    # Predefined list of F&O stocks
    fo_stocks = ['RELIANCE.NS', 'TATASTEEL.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
                'SBIN.NS', 'BAJFINANCE.NS', 'HINDUNILVR.NS', 'BHARTIARTL.NS', 'ITC.NS']
    
    # Fetch F&O data
    @st.cache_data(ttl=300)
    def get_fo_data(stocks):
        data = {}
        for ticker in stocks:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="5d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_close
                    percent_change = (change / prev_close) * 100
                    
                    data[ticker] = {
                        'price': current_price,
                        'change': change,
                        'percent_change': percent_change,
                        'volume': hist['Volume'].iloc[-1],
                        'open_interest': info.get('openInterest', 'N/A'),
                        'implied_volatility': info.get('impliedVolatility', 'N/A')
                    }
            except:
                continue
        return data
    
    fo_data = get_fo_data(fo_stocks)
    
    if fo_data:
        df = pd.DataFrame.from_dict(fo_data, orient='index')
        df.index.name = 'Ticker'
        df.reset_index(inplace=True)
        
        # Display F&O stocks
        st.subheader("F&O Stocks")
        st.dataframe(df[['Ticker', 'price', 'change', 'percent_change', 'volume']].rename(
            columns={'price': 'Price', 'change': 'Change', 'percent_change': '% Change', 'volume': 'Volume'}
        ), use_container_width=True)
        
        # OI analysis
        st.subheader("Open Interest Analysis")
        if 'open_interest' in df.columns and not all(df['open_interest'] == 'N/A'):
            oi_fig = px.bar(df, x='Ticker', y='open_interest', title='Open Interest by Stock')
            st.plotly_chart(oi_fig, use_container_width=True)
        else:
            st.info("Open interest data not available for these stocks.")
        
        # Volume analysis
        st.subheader("Volume Analysis")
        vol_fig = px.bar(df, x='Ticker', y='Volume', title='Trading Volume by Stock')
        st.plotly_chart(vol_fig, use_container_width=True)
        
    else:
        st.warning("Could not fetch F&O data. Please try again later.")

# Global Markets
elif selected == "Global Markets":
    st.title("🌍 Global Markets")
    
    global_indices = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^FTSE": "FTSE 100",
        "^N225": "Nikkei 225",
        "^HSI": "Hang Seng",
        "^AXJO": "ASX 200",
        "^STOXX50E": "Euro Stoxx 50",
    }
    
    indices_data = get_indices_data(global_indices)
    
    st.subheader("Global Indices")
    cols = st.columns(4)
    for idx, (name, data) in enumerate(indices_data.items()):
        cols[idx % 4].metric(label=name, value=f"{data['value']}", delta=f"{data['change']}%")
    
    # Currency rates
    st.subheader("Currency Exchange Rates")
    currencies = {
        "USDINR=X": "USD/INR",
        "EURINR=X": "EUR/INR",
        "GBPINR=X": "GBP/INR",
        "JPYINR=X": "JPY/INR",
    }
    
    currency_data = get_indices_data(currencies)
    
    curr_cols = st.columns(4)
    for idx, (name, data) in enumerate(currency_data.items()):
        curr_cols[idx % 4].metric(label=name, value=f"{data['value']:.2f}", delta=f"{data['change']:.2f}%")
    
    # Commodities
    st.subheader("Commodities")
    commodities = {
        "GC=F": "Gold",
        "SI=F": "Silver",
        "CL=F": "Crude Oil",
        "NG=F": "Natural Gas",
    }
    
    comm_data = get_indices_data(commodities)
    
    comm_cols = st.columns(4)
    for idx, (name, data) in enumerate(comm_data.items()):
        comm_cols[idx % 4].metric(label=name, value=f"${data['value']:.2f}", delta=f"{data['change']:.2f}%")

# Mutual Funds
elif selected == "Mutual Funds":
    st.title("💼 Mutual Funds")
    
    # Sample mutual fund data
    mf_categories = ["Large Cap", "Mid Cap", "Small Cap", "Flexi Cap", "Sectoral", "ELSS"]
    selected_category = st.selectbox("Select Category", mf_categories)
    
    # Sample fund data based on category
    mf_data = {
        "Large Cap": {
            "Axis Bluechip Fund": {"1Y Return": "15.2%", "3Y Return": "12.5%", "5Y Return": "14.1%", "Risk": "Moderate"},
            "Mirae Asset Large Cap Fund": {"1Y Return": "13.8%", "3Y Return": "11.9%", "5Y Return": "13.5%", "Risk": "Moderate"},
            "SBI Bluechip Fund": {"1Y Return": "14.5%", "3Y Return": "12.1%", "5Y Return": "13.8%", "Risk": "Moderate"},
        },
        "Mid Cap": {
            "Kotak Emerging Equity Fund": {"1Y Return": "18.2%", "3Y Return": "15.5%", "5Y Return": "16.8%", "Risk": "High"},
            "Axis Midcap Fund": {"1Y Return": "17.5%", "3Y Return": "14.9%", "5Y Return": "16.2%", "Risk": "High"},
        },
        "Small Cap": {
            "Nippon India Small Cap Fund": {"1Y Return": "22.1%", "3Y Return": "18.5%", "5Y Return": "19.8%", "Risk": "Very High"},
            "SBI Small Cap Fund": {"1Y Return": "21.5%", "3Y Return": "17.9%", "5Y Return": "19.2%", "Risk": "Very High"},
        }
    }
    
    if selected_category in mf_data:
        st.subheader(f"Top {selected_category} Funds")
        funds = mf_data[selected_category]
        funds_df = pd.DataFrame.from_dict(funds, orient='index')
        st.dataframe(funds_df, use_container_width=True)
    else:
        st.info("Detailed data coming soon for this category.")
    
    # Mutual fund comparison tool
    st.subheader("Compare Funds")
    col1, col2 = st.columns(2)
    
    with col1:
        fund1 = st.selectbox("Select Fund 1", list(mf_data.get("Large Cap", {}).keys()))
    
    with col2:
        fund2 = st.selectbox("Select Fund 2", list(mf_data.get("Large Cap", {}).keys()))
    
    if fund1 and fund2:
        comparison_data = {
            'Metric': ['1Y Return', '3Y Return', '5Y Return', 'Risk'],
            fund1: [mf_data['Large Cap'][fund1]['1Y Return'], 
                   mf_data['Large Cap'][fund1]['3Y Return'], 
                   mf_data['Large Cap'][fund1]['5Y Return'],
                   mf_data['Large Cap'][fund1]['Risk']],
            fund2: [mf_data['Large Cap'][fund2]['1Y Return'], 
                   mf_data['Large Cap'][fund2]['3Y Return'], 
                   mf_data['Large Cap'][fund2]['5Y Return'],
                   mf_data['Large Cap'][fund2]['Risk']]
        }
        
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True)

# SIP Calculator
elif selected == "SIP Calculator":
    st.title("📈 SIP Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_investment = st.number_input("Monthly Investment (₹)", min_value=500, value=5000, step=500)
        years = st.slider("Investment Period (Years)", 1, 30, 10)
    
    with col2:
        expected_return = st.slider("Expected Annual Return (%)", 1, 25, 12)
        step_up = st.slider("Annual Step-Up (%)", 0, 20, 10)
    
    # Calculate SIP
    months = years * 12
    monthly_rate = expected_return / 12 / 100
    step_up_rate = step_up / 100
    
    future_value = 0
    investment = monthly_investment
    
    for i in range(months):
        future_value += investment
        future_value *= (1 + monthly_rate)
        if (i+1) % 12 == 0:
            investment *= (1 + step_up_rate)
    
    invested = monthly_investment * ((1 + step_up_rate)**years - 1) / step_up_rate if step_up_rate > 0 else monthly_investment * years * 12
    gain = future_value - invested
    
    # Display results
    st.subheader("SIP Projection")
    col1, col2, col3 = st.columns(3)
    col1.metric("Invested Amount", f"₹{invested:,.0f}")
    col2.metric("Estimated Returns", f"₹{gain:,.0f}")
    col3.metric("Total Value", f"₹{future_value:,.0f}")
    
    # Yearly breakdown
    st.subheader("Yearly Breakdown")
    yearly_data = []
    yearly_investment = 0
    yearly_value = 0
    
    for year in range(1, years+1):
        for month in range(1, 13):
            monthly_inv = monthly_investment * (1 + step_up_rate)**(year-1)
            yearly_investment += monthly_inv
            yearly_value = (yearly_value + monthly_inv) * (1 + monthly_rate)
        
        yearly_data.append({
            'Year': year,
            'Invested': yearly_investment,
            'Value': yearly_value,
            'Returns': yearly_value - yearly_investment
        })
    
    yearly_df = pd.DataFrame(yearly_data)
    st.line_chart(yearly_df.set_index('Year')[['Invested', 'Value']])

# IPO Tracker
elif selected == "IPO Tracker":
    st.title("🆕 IPO Tracker")
    
    # Sample IPO data
    ipo_data = [
        {"Company": "ABC Tech", "Issue Date": "2023-05-15", "Price Range": "₹100-110", "Lot Size": 50, "Listing Date": "2023-05-30", "Status": "Listed", "Current Price": "₹145", "Gain": "45%"},
        {"Company": "SmartFin Ltd", "Issue Date": "2023-06-01", "Price Range": "₹230-240", "Lot Size": 40, "Listing Date": "2023-06-16", "Status": "Listed", "Current Price": "₹190", "Gain": "-20.8%"},
        {"Company": "GreenPower", "Issue Date": "2023-06-20", "Price Range": "₹140-150", "Lot Size": 60, "Listing Date": "2023-07-05", "Status": "Listed", "Current Price": "₹170", "Gain": "13.3%"},
        {"Company": "NetPay Corp", "Issue Date": "2023-07-10", "Price Range": "₹270-280", "Lot Size": 35, "Listing Date": "2023-07-25", "Status": "Listed", "Current Price": "₹260", "Gain": "-7.1%"},
        {"Company": "HealthPlus", "Issue Date": "2023-08-05", "Price Range": "₹180-190", "Lot Size": 55, "Listing Date": "2023-08-20", "Status": "Upcoming", "Current Price": "NA", "Gain": "NA"},
    ]
    
    ipo_df = pd.DataFrame(ipo_data)
    
    # Filter IPOs by status
    status_filter = st.selectbox("Filter by Status", ["All", "Listed", "Upcoming"])
    
    if status_filter != "All":
        filtered_df = ipo_df[ipo_df["Status"] == status_filter]
    else:
        filtered_df = ipo_df
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # IPO performance chart
    if status_filter == "Listed":
        listed_df = filtered_df.copy()
        listed_df['Gain'] = listed_df['Gain'].str.replace('%', '').astype(float)
        
        fig = px.bar(listed_df, x='Company', y='Gain', title='IPO Performance (%)')
        st.plotly_chart(fig, use_container_width=True)

# Predictions for Mutual Funds & IPOs
elif selected == "Predictions for Mutual Funds & IPOs":
    st.title("🔮 Predictions for Mutual Funds & IPOs")
    
    tab1, tab2 = st.tabs(["Mutual Fund NAV Forecast", "IPO Performance Prediction"])
    
    with tab1:
        st.subheader("Mutual Fund NAV Forecast")
        
        # Simulated NAV data
        nav_dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
        nav_values = [100 + i*5 + np.random.normal(0, 3) for i in range(12)]
        
        # Create forecast
        last_value = nav_values[-1]
        forecast_dates = pd.date_range(start=nav_dates[-1] + pd.DateOffset(months=1), periods=6, freq='M')
        forecast_values = [last_value * (1.01)**i + np.random.normal(0, 2) for i in range(1, 7)]
        
        # Combine historical and forecast
        all_dates = list(nav_dates) + list(forecast_dates)
        all_values = nav_values + forecast_values
        is_forecast = [False] * len(nav_values) + [True] * len(forecast_values)
        
        nav_df = pd.DataFrame({
            'Date': all_dates,
            'NAV': all_values,
            'Type': is_forecast
        })
        
        # Plot
        fig = px.line(nav_df, x='Date', y='NAV', color='Type', 
                     title='Mutual Fund NAV Forecast (Simulated)',
                     labels={'NAV': 'NAV Value', 'Date': 'Date'},
                     color_discrete_map={False: 'blue', True: 'red'})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction details
        current_nav = nav_values[-1]
        predicted_nav = forecast_values[-1]
        growth = ((predicted_nav - current_nav) / current_nav) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current NAV", f"₹{current_nav:.2f}")
        col2.metric("Predicted NAV (6M)", f"₹{predicted_nav:.2f}")
        col3.metric("Expected Growth", f"{growth:.2f}%")
    
    with tab2:
        st.subheader("IPO Performance Prediction")
        
        # Sample IPO prediction data
        ipo_predictions = [
            {"IPO": "TechNova", "Sector": "Technology", "Issue Size": "₹500 Cr", "Predicted Listing Gain": "15-25%", "Confidence": "High"},
            {"IPO": "GreenEnergy", "Sector": "Renewable Energy", "Issue Size": "₹750 Cr", "Predicted Listing Gain": "20-30%", "Confidence": "Medium"},
            {"IPO": "HealthPlus", "Sector": "Healthcare", "Issue Size": "₹300 Cr", "Predicted Listing Gain": "10-20%", "Confidence": "High"},
            {"IPO": "Finserve", "Sector": "Financial Services", "Issue Size": "₹600 Cr", "Predicted Listing Gain": "5-15%", "Confidence": "Medium"},
        ]
        
        ipo_pred_df = pd.DataFrame(ipo_predictions)
        st.dataframe(ipo_pred_df, use_container_width=True)
        
        # Sector-wise analysis
        sector_analysis = ipo_pred_df.groupby('Sector').size().reset_index(name='Count')
        fig = px.pie(sector_analysis, values='Count', names='Sector', title='IPO Distribution by Sector')
        st.plotly_chart(fig, use_container_width=True)

# Mutual Fund NAV Viewer
elif selected == "Mutual Fund NAV Viewer":
    st.title("📊 Mutual Fund NAV Viewer")
    
    # Sample mutual fund schemes
    mf_schemes = {
        "118550": "Axis Bluechip Fund Direct Growth",
        "120465": "Mirae Asset Emerging Bluechip Fund Direct Growth",
        "125350": "Parag Parikh Flexi Cap Fund Direct Growth",
        "122639": "SBI Small Cap Fund Direct Growth",
    }
    
    scheme_code = st.selectbox("Select Mutual Fund Scheme", list(mf_schemes.keys()), 
                              format_func=lambda x: f"{mf_schemes[x]} ({x})")
    
    if scheme_code:
        # Simulated NAV data
        dates = pd.date_range(end=pd.Timestamp.today(), periods=90, freq='D')
        navs = [100 + i*0.5 + np.random.normal(0, 1) for i in range(90)]
        
        nav_df = pd.DataFrame({'Date': dates, 'NAV': navs})
        nav_df = nav_df.sort_values('Date', ascending=False)
        
        # Display latest NAV
        latest_nav = nav_df.iloc[0]['NAV']
        prev_nav = nav_df.iloc[1]['NAV']
        change = latest_nav - prev_nav
        percent_change = (change / prev_nav) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Latest NAV", f"₹{latest_nav:.2f}")
        col2.metric("Change", f"₹{change:.2f}")
        col3.metric("Change %", f"{percent_change:.2f}%")
        
        # NAV chart
        st.subheader("NAV Trend (Last 90 Days)")
        fig = px.line(nav_df.sort_values('Date'), x='Date', y='NAV', title=f"{mf_schemes[scheme_code]} - NAV History")
        st.plotly_chart(fig, use_container_width=True)
        
        # NAV data table
        st.subheader("Historical NAV Data")
        st.dataframe(nav_df.head(30), use_container_width=True)
        
        # Download button
        csv = nav_df.to_csv(index=False)
        st.download_button(
            label="Download NAV Data as CSV",
            data=csv,
            file_name=f"nav_data_{scheme_code}.csv",
            mime="text/csv",
        )

# Sectors
elif selected == "Sectors":
    st.title("📊 Sector Wise Performance")
    
    # Sample sector data
    sector_data = {
        "Sector": ["Banking", "IT", "Energy", "FMCG", "Pharma", "Auto", "Realty", "Metal"],
        "1D Change": [1.8, -0.5, 2.1, 0.9, -1.2, 1.0, 2.5, -0.8],
        "1W Change": [3.2, 1.5, 4.2, 2.1, -0.5, 2.8, 5.1, 1.2],
        "1M Change": [8.5, 5.2, 12.1, 6.8, 3.2, 7.5, 15.2, 4.5],
    }
    
    sector_df = pd.DataFrame(sector_data)
    
    # Display sector performance
    st.subheader("Sector Performance")
    st.dataframe(sector_df, use_container_width=True)
    
    # Sector performance chart
    fig = px.bar(sector_df, x='Sector', y='1M Change', title='1-Month Sector Performance (%)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Top stocks in selected sector
    st.subheader("Top Stocks by Sector")
    selected_sector = st.selectbox("Select Sector", sector_df['Sector'].tolist())
    
    # Sample top stocks for each sector
    sector_stocks = {
        "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
        "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
        "Energy": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "GAIL.NS"],
        "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
        "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS"],
        "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS"],
        "Realty": ["DLF.NS", "PRESTIGE.NS", "SOBHA.NS", "GODREJPROP.NS", "BRIGADE.NS"],
        "Metal": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "SAIL.NS"],
    }
    
    if selected_sector in sector_stocks:
        st.write(f"Top {selected_sector} Stocks:")
        for stock in sector_stocks[selected_sector]:
            st.write(f"- {stock}")

# News
elif selected == "News":
    st.title("📰 Latest Financial News")
    
    news_query = st.text_input("Search Financial News:", "stock market")
    news_limit = st.slider("Number of Articles", 5, 20, 10)
    
    if news_query:
        articles = get_news(news_query, news_limit)
        
        if articles:
            for article in articles:
                with st.expander(f"{article['title']} - {article['source']['name']}"):
                    published = article['publishedAt'].split('T')[0] if 'publishedAt' in article else 'N/A'
                    st.write(f"**Published:** {published}")
                    
                    if article.get('urlToImage'):
                        st.image(article['urlToImage'], width=300)
                    
                    st.write(article.get('description', 'No description available'))
                    st.markdown(f"[Read full article]({article['url']})")
        else:
            st.warning("No articles found. Please try a different search term.")
    else:
        st.info("Enter a search term to find news articles.")

# Learning
elif selected == "Learning":
    st.title("📚 Learning Center")
    
    st.markdown("""
    ## Welcome to the MarketMentor Learning Center!
    
    Here you'll find educational resources to enhance your understanding of financial markets and investment strategies.
    """)
    
    # Learning topics
    topics = [
        {"title": "Stock Market Basics", "level": "Beginner", "duration": "30 min"},
        {"title": "Technical Analysis", "level": "Intermediate", "duration": "45 min"},
        {"title": "Fundamental Analysis", "level": "Intermediate", "duration": "60 min"},
        {"title": "Options Trading", "level": "Advanced", "duration": "90 min"},
        {"title": "Portfolio Management", "level": "Intermediate", "duration": "60 min"},
        {"title": "Risk Management", "level": "Intermediate", "duration": "45 min"},
    ]
    
    st.subheader("Available Learning Modules")
    for topic in topics:
        with st.expander(f"{topic['title']} ({topic['level']} - {topic['duration']})"):
            st.markdown(f"""
            **Overview:**
            This module covers the essential concepts of {topic['title'].lower()}.
            
            **What you'll learn:**
            - Key concepts and terminology
            - Practical applications
            - Common strategies
            - Risk factors
            
            **Prerequisites:** None
            """)
            
            if st.button(f"Start Learning {topic['title']}", key=topic['title']):
                st.success(f"Starting {topic['title']} module!")
    
    # Glossary of terms
    st.subheader("Financial Terms Glossary")
    
    financial_terms = {
        "IPO": "Initial Public Offering - when a company first sells its shares to the public",
        "SIP": "Systematic Investment Plan - regular investment in mutual funds",
        "NAV": "Net Asset Value - per-unit value of a mutual fund",
        "ETF": "Exchange-Traded Fund - basket of securities traded like a stock",
        "P/E Ratio": "Price-to-Earnings Ratio - valuation ratio of a company",
        "Market Cap": "Market Capitalization - total value of a company's outstanding shares",
    }
    
    term = st.selectbox("Select a term to learn", list(financial_terms.keys()))
    st.info(f"**{term}**: {financial_terms[term]}")

# Volume Spike
elif selected == "Volume Spike":
    st.title("📈 Volume Spike Detector")
    
    ticker = st.text_input("Enter Stock Ticker (e.g., TCS.NS, INFY.NS):", "TCS.NS")
    days = st.slider("Select Days of Historical Data:", 30, 365, 90)
    threshold = st.slider("Volume Spike Threshold (x times average):", 1.0, 5.0, 1.5, 0.1)
    
    if ticker:
        try:
            data = yf.download(ticker, period=f"{days}d")
            
            if data.empty:
                st.warning("No data found. Please check the ticker symbol.")
            else:
                # Calculate rolling average and detect spikes
                data["Avg_Volume"] = data["Volume"].rolling(window=10).mean()
                data["Spike"] = data["Volume"] > (threshold * data["Avg_Volume"])
                data.dropna(inplace=True)
                
                # Plot
                st.subheader("Volume Trend with Spike Detection")
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=data.index, y=data["Volume"],
                    mode='lines', name='Daily Volume',
                    line=dict(color='royalblue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=data.index, y=data["Avg_Volume"],
                    mode='lines', name=f'{10}-Day Avg Volume',
                    line=dict(color='orange')
                ))
                
                spikes = data[data["Spike"]]
                fig.add_trace(go.Scatter(
                    x=spikes.index, y=spikes["Volume"],
                    mode='markers', name='Spikes',
                    marker=dict(size=8, color='red', symbol='diamond')
                ))
                
                fig.update_layout(
                    title=f"Volume Spike Detection for {ticker.upper()}",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    legend_title="Legend",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display spike events
                st.subheader("Volume Spike Events")
                if not spikes.empty:
                    spike_data = spikes[["Volume", "Avg_Volume"]].copy()
                    spike_data["Spike_Multiple"] = spike_data["Volume"] / spike_data["Avg_Volume"]
                    spike_data = spike_data.sort_values("Spike_Multiple", ascending=False)
                    
                    st.dataframe(
                        spike_data.rename(columns={
                            "Volume": "Actual Volume", 
                            "Avg_Volume": f"{10}-Day Avg",
                            "Spike_Multiple": "Spike Multiple"
                        }).style.format({"Actual Volume": "{:,.0f}", f"{10}-Day Avg": "{:,.0f}", "Spike Multiple": "{:.2f}"}),
                        use_container_width=True
                    )
                else:
                    st.info("No volume spikes detected for the selected threshold.")
                
        except Exception as e:
            st.error(f"Error occurred: {e}")

# Stock Screener
elif selected == "Stock Screener":
    st.title("🔍 Stock Screener")
    
    # Predefined list of popular stocks
    popular_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
        'SBIN.NS', 'HINDUNILVR.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'ITC.NS',
        'KOTAKBANK.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
        'SUNPHARMA.NS', 'TATAMOTORS.NS', 'HCLTECH.NS', 'WIPRO.NS', 'ONGC.NS'
    ]
    
    option = st.radio("Select screening option:", 
                     ["Use default list (Nifty 50 stocks)", "Enter custom tickers"])
    
    if option == "Use default list (Nifty 50 stocks)":
        selected_stocks = popular_stocks
    else:
        custom_tickers = st.text_area("Enter tickers (comma-separated):", "RELIANCE.NS, TCS.NS, INFY.NS")
        selected_stocks = [ticker.strip() for ticker in custom_tickers.split(',') if ticker.strip()]
    
    if selected_stocks:
        # Get stock data
        @st.cache_data(ttl=300)
        def get_multiple_stock_data(stocks):
            data = {}
            for ticker in stocks:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    hist = stock.history(period="5d")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        change = current_price - prev_close
                        percent_change = (change / prev_close) * 100
                        
                        data[ticker] = {
                            'Price': current_price,
                            'Change': change,
                            '% Change': percent_change,
                            'Volume': hist['Volume'].iloc[-1],
                            'Market Cap': info.get('marketCap', 'N/A'),
                            'P/E Ratio': info.get('trailingPE', 'N/A')
                        }
                except:
                    continue
            return data
        
        stocks_data = get_multiple_stock_data(selected_stocks)
        
        if stocks_data:
            df = pd.DataFrame.from_dict(stocks_data, orient='index')
            df.index.name = 'Ticker'
            df.reset_index(inplace=True)
            
            # Filters
            st.subheader("Filter Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_price = st.number_input("Min Price", value=0)
                max_price = st.number_input("Max Price", value=10000)
            
            with col2:
                min_change = st.number_input("Min % Change", value=-10.0)
                max_change = st.number_input("Max % Change", value=10.0)
            
            with col3:
                min_volume = st.number_input("Min Volume", value=0)
            
            # Apply filters
            filtered_df = df[
                (df['Price'] >= min_price) & (df['Price'] <= max_price) &
                (df['% Change'] >= min_change) & (df['% Change'] <= max_change) &
                (df['Volume'] >= min_volume)
            ]
            
            # Display results
            st.subheader("Screened Stocks")
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download results
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="stock_screener_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("Could not fetch data for the selected stocks.")
    else:
        st.info("Please select or enter some stock tickers to screen.")

# Predictions
elif selected == "Predictions":
    st.title("📈 Stock Price Predictions")
    
    ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS):", "RELIANCE.NS")
    period = st.selectbox("Select Prediction Period:", ["1 week", "1 month", "3 months"])
    
    if ticker:
        try:
            # Get historical data
            data = yf.download(ticker, period="1y")
            
            if data.empty:
                st.warning("No data available for this ticker.")
            else:
                st.subheader(f"Historical Data for {ticker}")
                st.line_chart(data['Close'])
                
                # Prepare data for prediction
                df = data[['Close']].copy()
                df.reset_index(inplace=True)
                
                # Create features
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['SMA_200'] = df['Close'].rolling(window=200).mean()
                df['EMA_12'] = df['Close'].ewm(span=12).mean()
                df['EMA_26'] = df['Close'].ewm(span=26).mean()
                df['MACD'] = df['EMA_12'] - df['EMA_26']
                df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
                
                # Remove NaN values
                df.dropna(inplace=True)
                
                # Display technical indicators
                st.subheader("Technical Indicators")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.line_chart(df.set_index('Date')[['SMA_50', 'SMA_200', 'Close']])
                    st.write("Moving Averages")
                
                with col2:
                    st.line_chart(df.set_index('Date')['RSI'])
                    st.write("RSI (Relative Strength Index)")
                
                # Simple prediction based on trends
                last_price = df['Close'].iloc[-1]
                
                # Determine trend based on moving averages
                if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
                    trend = "Bullish"
                    trend_strength = (df['SMA_50'].iloc[-1] - df['SMA_200'].iloc[-1]) / df['SMA_200'].iloc[-1] * 100
                else:
                    trend = "Bearish"
                    trend_strength = (df['SMA_200'].iloc[-1] - df['SMA_50'].iloc[-1]) / df['SMA_50'].iloc[-1] * 100
                
                # Generate prediction based on trend
                if trend == "Bullish":
                    if period == "1 week":
                        prediction = last_price * (1 + 0.01)
                    elif period == "1 month":
                        prediction = last_price * (1 + 0.04)
                    else:  # 3 months
                        prediction = last_price * (1 + 0.12)
                else:
                    if period == "1 week":
                        prediction = last_price * (1 - 0.01)
                    elif period == "1 month":
                        prediction = last_price * (1 - 0.04)
                    else:  # 3 months
                        prediction = last_price * (1 - 0.10)
                
                # Display prediction
                st.subheader("Price Prediction")
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Current Price", f"₹{last_price:.2f}")
                col2.metric("Predicted Price", f"₹{prediction:.2f}")
                col3.metric("Expected Change", f"{(prediction/last_price - 1)*100:.2f}%")
                
                st.info(f"Trend: {trend} (Strength: {abs(trend_strength):.2f}%)")
                st.warning("Note: This is a simple prediction based on technical trends and should not be considered as financial advice.")
                
        except Exception as e:
            st.error(f"Error retrieving data: {e}")

# Buy/Sell Predictor
elif selected == "Buy/Sell Predictor":
    st.title("💹 Buy/Sell Predictor")
    
    ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS):", "RELIANCE.NS")
    
    if ticker:
        try:
            # Get historical data
            data = yf.download(ticker, period="6mo")
            
            if data.empty:
                st.warning("No data available for this ticker.")
            else:
                # Calculate indicators
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
                
                # Remove NaN values
                data.dropna(inplace=True)
                
                # Get current values
                current_close = data['Close'].iloc[-1]
                current_sma20 = data['SMA_20'].iloc[-1]
                current_sma50 = data['SMA_50'].iloc[-1]
                current_rsi = data['RSI'].iloc[-1]
                
                # Determine signals
                signals = []
                
                # Moving average crossover
                if current_sma20 > current_sma50:
                    signals.append(("Moving Average Crossover", "BUY", "20-day SMA above 50-day SMA"))
                else:
                    signals.append(("Moving Average Crossover", "SELL", "20-day SMA below 50-day SMA"))
                
                # RSI signal
                if current_rsi < 30:
                    signals.append(("RSI", "BUY", "RSI indicates oversold condition"))
                elif current_rsi > 70:
                    signals.append(("RSI", "SELL", "RSI indicates overbought condition"))
                else:
                    signals.append(("RSI", "HOLD", "RSI in neutral territory"))
                
                # Price vs SMA
                if current_close > current_sma50:
                    signals.append(("Price vs SMA", "BUY", "Price above 50-day SMA"))
                else:
                    signals.append(("Price vs SMA", "SELL", "Price below 50-day SMA"))
                
                # Display current values
                st.subheader("Current Indicators")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Price", f"₹{current_close:.2f}")
                col2.metric("20-day SMA", f"₹{current_sma20:.2f}")
                col3.metric("50-day SMA", f"₹{current_sma50:.2f}")
                col4.metric("RSI", f"{current_rsi:.2f}")
                
                # Display signals
                st.subheader("Trading Signals")
                signals_df = pd.DataFrame(signals, columns=['Indicator', 'Signal', 'Reason'])
                st.dataframe(signals_df, use_container_width=True)
                
                # Overall recommendation
                buy_signals = sum(1 for s in signals if s[1] == 'BUY')
                sell_signals = sum(1 for s in signals if s[1] == 'SELL')
                
                if buy_signals > sell_signals:
                    overall_signal = "BUY"
                    signal_color = "green"
                elif sell_signals > buy_signals:
                    overall_signal = "SELL"
                    signal_color = "red"
                else:
                    overall_signal = "HOLD"
                    signal_color = "orange"
                
                st.markdown(f"<h2 style='text-align: center; color: {signal_color};'>Overall Signal: {overall_signal}</h2>", 
                           unsafe_allow_html=True)
                
                # Charts
                st.subheader("Charts")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='20-day SMA', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='50-day SMA', line=dict(color='green')))
                fig.update_layout(title=f"{ticker} Price with Moving Averages", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')))
                fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig2.update_layout(title="RSI Indicator", yaxis_range=[0, 100], height=300)
                st.plotly_chart(fig2, use_container_width=True)
                
                st.warning("Note: These signals are based on technical indicators only and should not be considered as financial advice. Always do your own research.")
                
        except Exception as e:
            st.error(f"Error retrieving data: {e}")

# News Sentiment
elif selected == "News Sentiment":
    st.title("📊 News Sentiment Analysis")
    
    ticker = st.text_input("Enter Stock Ticker for News Sentiment Analysis:", "AAPL")
    num_articles = st.slider("Number of Articles to Analyze", 5, 20, 10)
    
    if ticker:
        st.info(f"Fetching and analyzing recent news sentiment for {ticker.upper()}...")
        
        articles = get_news(ticker, num_articles)
        
        if articles:
            sentiments = []
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for article in articles:
                title = article["title"]
                description = article.get("description", "")
                text = f"{title}. {description}"
                
                # Analyze sentiment
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                sentiments.append(polarity)
                
                # Classify sentiment
                if polarity > 0.1:
                    sentiment_label = "Positive"
                    positive_count += 1
                elif polarity < -0.1:
                    sentiment_label = "Negative"
                    negative_count += 1
                else:
                    sentiment_label = "Neutral"
                    neutral_count += 1
                
                # Display article with sentiment
                with st.expander(f"{title} - {sentiment_label} ({polarity:.2f})"):
                    st.write(f"**Source:** {article['source']['name']}")
                    st.write(f"**Published:** {article['publishedAt'][:10]}")
                    st.write(description)
                    st.markdown(f"[Read more]({article['url']})")
            
            # Calculate average sentiment
            if sentiments:
                avg_sentiment = round(np.mean(sentiments), 3)
                
                # Display sentiment summary
                st.subheader("Sentiment Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Average Sentiment", f"{avg_sentiment:.3f}")
                col2.metric("Positive Articles", positive_count)
                col3.metric("Negative Articles", negative_count)
                col4.metric("Neutral Articles", neutral_count)
                
                # Sentiment distribution chart
                sentiment_df = pd.DataFrame({
                    'Sentiment': ['Positive', 'Neutral', 'Negative'],
                    'Count': [positive_count, neutral_count, negative_count]
                })
                
                fig = px.pie(sentiment_df, values='Count', names='Sentiment', 
                            title='Sentiment Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
                # Overall sentiment conclusion
                if avg_sentiment > 0.2:
                    st.success("📈 Overall Sentiment: Positive")
                elif avg_sentiment < -0.2:
                    st.error("📉 Overall Sentiment: Negative")
                else:
                    st.info("➖ Overall Sentiment: Neutral")
            else:
                st.warning("No sentiment data to analyze.")
        else:
            st.warning("No articles found for this ticker.")
    else:
        st.info("Please enter a stock ticker to analyze news sentiment.")

# Technical Analysis
elif selected == "Technical Analysis":
    st.title("📊 Technical Analysis")
    
    ticker = st.text_input("Enter Stock Ticker for Technical Analysis:", "RELIANCE.NS")
    period = st.selectbox("Select Time Period:", ["1mo", "3mo", "6mo", "1y", "2y"])
    
    if ticker:
        try:
            data = yf.download(ticker, period=period)
            
            if data.empty:
                st.warning("No data available for this ticker.")
            else:
                st.subheader(f"Technical Analysis for {ticker}")
                
                # Calculate technical indicators
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                data['EMA_12'] = data['Close'].ewm(span=12).mean()
                data['EMA_26'] = data['Close'].ewm(span=26).mean()
                data['MACD'] = data['EMA_12'] - data['EMA_26']
                data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
                data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
                data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
                data['BB_upper'], data['BB_middle'], data['BB_lower'] = ta.volatility.BollingerBands(data['Close']).bollinger_bands()
                
                # Remove NaN values
                data.dropna(inplace=True)
                
                # Display charts
                tab1, tab2, tab3, tab4 = st.tabs(["Price & MA", "MACD", "RSI", "Bollinger Bands"])
                
                with tab1:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=data.index,
                                                open=data['Open'],
                                                high=data['High'],
                                                low=data['Low'],
                                                close=data['Close'],
                                                name='Price'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='20-day SMA', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='50-day SMA', line=dict(color='green')))
                    fig.update_layout(title=f"{ticker} Price with Moving Averages", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')))
                    
                    # Histogram with different colors for positive and negative
                    colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
                    fig.add_trace(go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram', marker_color=colors))
                    
                    fig.update_layout(title="MACD Indicator", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')))
                    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig.add_hline(y=50, line_dash="dash", line_color="gray")
                    fig.update_layout(title="RSI Indicator", yaxis_range=[0, 100], height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=data.index,
                                                open=data['Open'],
                                                high=data['High'],
                                                low=data['Low'],
                                                close=data['Close'],
                                                name='Price'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name='Upper Band', line=dict(color='gray')))
                    fig.add_trace(go.Scatter(x=data.index, y=data['BB_middle'], name='Middle Band', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name='Lower Band', line=dict(color='gray')))
                    fig.update_layout(title="Bollinger Bands", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Current values
                st.subheader("Current Indicator Values")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Price", f"₹{data['Close'].iloc[-1]:.2f}")
                col2.metric("20-day SMA", f"₹{data['SMA_20'].iloc[-1]:.2f}")
                col3.metric("50-day SMA", f"₹{data['SMA_50'].iloc[-1]:.2f}")
                col4.metric("RSI", f"{data['RSI'].iloc[-1]:.2f}")
                
                col5, col6, col7, col8 = st.columns(4)
                col5.metric("MACD", f"{data['MACD'].iloc[-1]:.4f}")
                col6.metric("MACD Signal", f"{data['MACD_Signal'].iloc[-1]:.4f}")
                col7.metric("Upper BB", f"₹{data['BB_upper'].iloc[-1]:.2f}")
                col8.metric("Lower BB", f"₹{data['BB_lower'].iloc[-1]:.2f}")
                
        except Exception as e:
            st.error(f"Error retrieving data: {e}")

# Portfolio Tracker
elif selected == "Portfolio Tracker":
    st.title("💼 Portfolio Tracker")
    
    st.subheader("Add Your Holdings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker = st.text_input("Stock Ticker", "RELIANCE.NS")
    
    with col2:
        quantity = st.number_input("Quantity", min_value=1, value=10)
    
    with col3:
        buy_price = st.number_input("Buy Price (₹)", min_value=0.0, value=2500.0)
    
    if st.button("Add to Portfolio"):
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = []
        
        st.session_state.portfolio.append({
            'ticker': ticker,
            'quantity': quantity,
            'buy_price': buy_price
        })
    
    # Display portfolio
    if 'portfolio' in st.session_state and st.session_state.portfolio:
        st.subheader("Your Portfolio")
        
        portfolio_data = []
        total_investment = 0
        total_current = 0
        
        for holding in st.session_state.portfolio:
            try:
                stock = yf.Ticker(holding['ticker'])
                current_price = stock.info.get('regularMarketPrice', 0)
                
                investment = holding['quantity'] * holding['buy_price']
                current_value = holding['quantity'] * current_price
                pnl = current_value - investment
                pnl_percent = (pnl / investment) * 100
                
                portfolio_data.append({
                    'Ticker': holding['ticker'],
                    'Quantity': holding['quantity'],
                    'Buy Price': holding['buy_price'],
                    'Current Price': current_price,
                    'Investment': investment,
                    'Current Value': current_value,
                    'P&L': pnl,
                    'P&L %': pnl_percent
                })
                
                total_investment += investment
                total_current += current_value
            except:
                continue
        
        if portfolio_data:
            portfolio_df = pd.DataFrame(portfolio_data)
            st.dataframe(portfolio_df, use_container_width=True)
            
            # Portfolio summary
            total_pnl = total_current - total_investment
            total_pnl_percent = (total_pnl / total_investment) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Investment", f"₹{total_investment:,.2f}")
            col2.metric("Current Value", f"₹{total_current:,.2f}")
            col3.metric("Total P&L", f"₹{total_pnl:,.2f}")
            col4.metric("Total P&L %", f"{total_pnl_percent:.2f}%")
            
            # Portfolio allocation chart
            allocation_df = portfolio_df[['Ticker', 'Current Value']].copy()
            allocation_df['Percentage'] = (allocation_df['Current Value'] / total_current) * 100
            
            fig = px.pie(allocation_df, values='Current Value', names='Ticker', 
                        title='Portfolio Allocation')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add stocks to your portfolio to track performance.")
    else:
        st.info("Your portfolio is empty. Add stocks to get started.")

# Economic Calendar
elif selected == "Economic Calendar":
    st.title("📅 Economic Calendar")
    
    # Sample economic events data
    economic_events = [
        {"Date": "2023-08-15", "Country": "US", "Event": "CPI Data", "Impact": "High", "Previous": "3.0%", "Forecast": "3.3%"},
        {"Date": "2023-08-16", "Country": "EU", "Event": "ECB Interest Rate Decision", "Impact": "High", "Previous": "4.25%", "Forecast": "4.50%"},
        {"Date": "2023-08-17", "Country": "US", "Event": "Retail Sales", "Impact": "Medium", "Previous": "0.2%", "Forecast": "0.4%"},
        {"Date": "2023-08-18", "Country": "UK", "Event": "GDP Growth Rate", "Impact": "Medium", "Previous": "0.1%", "Forecast": "0.2%"},
        {"Date": "2023-08-21", "Country": "JP", "Event": "BOJ Monetary Policy Statement", "Impact": "High", "Previous": "-0.10%", "Forecast": "-0.10%"},
        {"Date": "2023-08-22", "Country": "IN", "Event": "RBI Monetary Policy Meeting", "Impact": "High", "Previous": "6.50%", "Forecast": "6.50%"},
    ]
    
    events_df = pd.DataFrame(economic_events)
    
    # Filter options
    st.subheader("Upcoming Economic Events")
    
    col1, col2 = st.columns(2)
    
    with col1:
        country_filter = st.multiselect("Filter by Country", events_df['Country'].unique(), default=events_df['Country'].unique())
    
    with col2:
        impact_filter = st.multiselect("Filter by Impact", events_df['Impact'].unique(), default=events_df['Impact'].unique())
    
    filtered_events = events_df[
        (events_df['Country'].isin(country_filter)) & 
        (events_df['Impact'].isin(impact_filter))
    ]
    
    st.dataframe(filtered_events, use_container_width=True)
    
    # Impact distribution
    st.subheader("Impact Distribution")
    impact_counts = filtered_events['Impact'].value_counts().reset_index()
    impact_counts.columns = ['Impact', 'Count']
    
    fig = px.bar(impact_counts, x='Impact', y='Count', color='Impact',
                title='Number of Events by Impact Level')
    st.plotly_chart(fig, use_container_width=True)
    
    # Country distribution
    st.subheader("Country Distribution")
    country_counts = filtered_events['Country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']
    
    fig = px.pie(country_counts, values='Count', names='Country', 
                title='Events by Country')
    st.plotly_chart(fig, use_container_width=True)
    
    # Add event notification option
    st.subheader("Event Notifications")
    event_notification = st.checkbox("Notify me about high-impact events")
    
    if event_notification:
        st.info("You will receive notifications for high-impact economic events.")
    
    # Economic calendar explanation
    with st.expander("About Economic Calendar"):
        st.markdown("""
        The Economic Calendar shows scheduled economic events that may affect financial markets.
        
        **Impact Levels:**
        - **High:** Likely to cause significant market volatility
        - **Medium:** May cause moderate market movements
        - **Low:** Minimal expected impact on markets
        
        Monitor this calendar to stay informed about important economic releases and central bank decisions.
        """)
