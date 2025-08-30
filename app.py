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
import warnings
import json
from io import StringIO

warnings.filterwarnings('ignore')

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Set page config
st.set_page_config(page_title="MarketMentor", layout="wide", page_icon="📈")

# Custom CSS with white text color
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0A0F1E;
        color: #FFFFFF;
    }

    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc, .css-1outpf7 {
        background-color: #121826 !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #2979FF !important;
    }

    /* Text - Changed to white */
    .stMarkdown, .stText, .stAlert, .stInfo, .stCaption, .stCode {
        color: #FFFFFF !important;
    }

    /* Input widgets */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #1C2230 !important;
        color: #FFFFFF !important;
        border-color: #2C3445;
    }

    /* Buttons */
    .stButton>button {
        background-color: #2979FF !important;
        color: white !important;
        border: none;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #1C68E8 !important;
    }

    /* Dataframes */
    .dataframe {
        background-color: #1C2230 !important;
        color: #FFFFFF !important;
    }

    /* Metric cards */
    .metric-card {
        background-color: #121826;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
        border-left: 4px solid #2979FF;
        color: #FFFFFF;
    }

    /* Section headers */
    .section-header {
        border-bottom: 2px solid #2979FF;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        color: #2979FF;
    }

    /* Positive and negative changes */
    .positive-change {
        color: #00C853 !important;
    }

    .negative-change {
        color: #FF3D00 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #121826;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        border: 1px solid #2C3445;
        color: #FFFFFF;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2979FF !important;
        color: white !important;
    }

    /* Main header */
    .main-header {
        font-size: 2.5rem;
        color: #2979FF;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #2979FF;
    }

    /* Links */
    a {
        color: #2979FF !important;
    }

    a:hover {
        color: #1C68E8 !important;
    }

    /* Plotly chart background */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly div {
        background-color: #121826 !important;
    }

    /* Streamlit sidebar navigation */
    .css-1v3fvcr, .css-1v3fvcr * {
        color: #FFFFFF !important;
    }

    /* Selected menu item */
    .css-1v3fvcr .st-bh, .css-1v3fvcr .st-c2, .css-1v3fvcr .st-c1 {
        background-color: #2979FF !important;
        color: white !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #2C3445;
        color: #9E9E9E;
    }

    /* Additional styling for better contrast */
    .stDataFrame {
        border: 1px solid #2C3445;
    }

    /* Select dropdowns */
    .stSelectbox:first-child>div>div {
        background-color: #1C2230;
        color: #FFFFFF;
    }

    /* Number input */
    input[type="number"] {
        background-color: #1C2230;
        color: #FFFFFF;
    }

    /* Text input */
    input[type="text"] {
        background-color: #1C2230;
        color: #FFFFFF;
    }

    /* Table text color */
    .stTable {
        color: #FFFFFF !important;
    }

    /* Warning and error messages */
    .stWarning, .stError {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    # Replace broken image with text logo
    st.markdown("<h1 style='text-align: center; color: #2979FF; font-size: 24px;'>MarketMentor</h1>",
                unsafe_allow_html=True)
    st.markdown("---")

    selected = option_menu(
        "Navigation",
        ["Home", "Company Overview", "Market Movers", "F&O", "Global Markets",
         "Mutual Funds", "SIP Calculator", "IPO Tracker", "Predictions for Mutual Funds & IPOs",
         "Mutual Fund NAV Viewer", "Sectors", "News", "Learning Materials & Resources", "Volume Spike",
         "Stock Screener", "Predictions", "Buy/Sell Predictor", "News Sentiment"],
        icons=['house', 'building', 'graph-up', 'arrow-left-right', 'globe',
               'bank', 'calculator', 'clipboard-data', 'graph-up-arrow',
               'currency-exchange', 'grid-3x3', 'newspaper', 'book',
               'activity', 'search', 'lightbulb', 'cash-coin', 'chat-quote'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#121826"},
            "icon": {"color": "#2979FF", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#FFFFFF"},
            "nav-link-selected": {"background-color": "#2979FF", "color": "white"},
        }
    )

    # Footer in sidebar
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #FFFFFF;">
        <p>Developed by <strong>Ashwik Bire</strong></p>
        <p>
            <a href="https://www.linkedin.com/in/ashwik-bire" target="_blank" style="color: #2979FF !important;">LinkedIn</a> • 
            <a href="https://github.com/ashwikbire" target="_blank" style="color: #2979FF !important;">GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Home - Market Overview
if selected == "Home":
    st.markdown('<h1 class="main-header">🏠 MarketMentor - Home</h1>', unsafe_allow_html=True)

    # Market Overview
    st.markdown('<h2 class="section-header">📊 Market Overview</h2>', unsafe_allow_html=True)

    indices = {
        "^NSEI": "Nifty 50",
        "^BSESN": "Sensex",
        "^NSEBANK": "Bank Nifty",
        "^NSMIDCP": "Nifty Midcap",
    }

    cols = st.columns(len(indices))
    for idx, (symbol, name) in enumerate(indices.items()):
        with cols[idx]:
            try:
                data = yf.Ticker(symbol).history(period="5d")
                last_close = round(data['Close'].iloc[-1], 2)
                prev_close = round(data['Close'].iloc[-2], 2)
                change = round(last_close - prev_close, 2)
                percent_change = round((change / prev_close) * 100, 2)

                change_class = "positive-change" if change >= 0 else "negative-change"
                change_icon = "📈" if change >= 0 else "📉"

                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<h3>{name}</h3>', unsafe_allow_html=True)
                st.markdown(f'<h2>₹{last_close:,}</h2>', unsafe_allow_html=True)
                st.markdown(f'<p class="{change_class}">{change_icon} {change} ({percent_change}%)</p>',
                            unsafe_allow_html=True)
                st.markdown(f'</div>', unsafe_allow_html=True)
            except:
                st.error(f"Error loading {name} data")

    # New Features Section
    st.markdown('<h2 class="section-header">✨ New Features</h2>', unsafe_allow_html=True)

    feat_cols = st.columns(3)
    with feat_cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3>📊 Advanced Charts</h3>', unsafe_allow_html=True)
        st.markdown('<p>Interactive charts with technical indicators</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with feat_cols[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3>🔔 Price Alerts</h3>', unsafe_allow_html=True)
        st.markdown('<p>Set custom alerts for price movements</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with feat_cols[2]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h3>📱 Mobile App</h3>', unsafe_allow_html=True)
        st.markdown('<p>Access MarketMentor on the go</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Market News
    st.markdown('<h2 class="section-header">📰 Latest Market News</h2>', unsafe_allow_html=True)

    try:
        url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey={NEWS_API_KEY}&language=en&pageSize=5"
        response = requests.get(url)

        if response.status_code == 200:
            articles = response.json().get("articles", [])
            if articles:
                for article in articles:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader(article["title"])
                        st.write(f"*{article['source']['name']} - {article['publishedAt'].split('T')[0]}*")
                        st.write(article.get("description", "No description available."))
                        st.markdown(f"[🔗 Read More]({article['url']})")
                    with col2:
                        if article.get("urlToImage"):
                            st.image(article["urlToImage"], width=150)
                    st.markdown("---")
            else:
                st.warning("No articles found.")
        else:
            st.error("Unable to fetch news articles.")
    except:
        st.error("Error fetching news")

# Company Overview
elif selected == "Company Overview":
    st.markdown('<h1 class="main-header">🏢 Company Overview</h1>', unsafe_allow_html=True)

    ticker = st.text_input("Enter Company Ticker (e.g., RELIANCE.NS, INFY.NS)", "RELIANCE.NS")

    if ticker:
        try:
            company = yf.Ticker(ticker)
            info = company.info

            col1, col2 = st.columns([1, 2])

            with col1:
                if 'longName' in info:
                    st.markdown(f'<h2>{info["longName"]}</h2>', unsafe_allow_html=True)

                if 'sector' in info:
                    st.markdown(f'<p><strong>Sector:</strong> {info["sector"]}</p>', unsafe_allow_html=True)

                if 'industry' in info:
                    st.markdown(f'<p><strong>Industry:</strong> {info["industry"]}</p>', unsafe_allow_html=True)

                if 'website' in info:
                    st.markdown(
                        f'<p><strong>Website:</strong> <a href="{info["website"]}" target="_blank">{info["website"]}</a></p>',
                        unsafe_allow_html=True)

                # Current price data
                hist = company.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = info.get('previousClose', current_price)
                    change = current_price - prev_close
                    percent_change = (change / prev_close) * 100

                    st.markdown(f'<h3>Current Price: ₹{current_price:.2f}</h3>', unsafe_allow_html=True)
                    change_class = "positive-change" if change >= 0 else "negative-change"
                    st.markdown(f'<p class="{change_class}">{change:+.2f} ({percent_change:+.2f}%)</p>',
                                unsafe_allow_html=True)

            with col2:
                # Key metrics
                metrics_data = {}
                if 'marketCap' in info:
                    metrics_data['Market Cap'] = f'₹{info["marketCap"] / 10000000:.2f} Cr'
                if 'trailingPE' in info:
                    metrics_data['P/E Ratio'] = f'{info["trailingPE"]:.2f}'
                if 'priceToBook' in info:
                    metrics_data['P/B Ratio'] = f'{info["priceToBook"]:.2f}'
                if 'dividendYield' in info:
                    metrics_data['Dividend Yield'] = f'{info["dividendYield"] * 100:.2f}%'
                if 'returnOnEquity' in info:
                    metrics_data['ROE'] = f'{info["returnOnEquity"] * 100:.2f}%'

                if metrics_data:
                    st.markdown('<h3>Key Metrics</h3>', unsafe_allow_html=True)
                    metrics_df = pd.DataFrame(list(metrics_data.items()), columns=['Metric', 'Value'])
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # Financial charts
            st.markdown('<h3 class="section-header">Price Chart</h3>', unsafe_allow_html=True)
            period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

            hist_data = company.history(period=period)
            if not hist_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=hist_data.index,
                    open=hist_data['Open'],
                    high=hist_data['High'],
                    low=hist_data['Low'],
                    close=hist_data['Close'],
                    name='Price'
                ))
                fig.update_layout(
                    title=f'{ticker} Price Chart',
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    template='plotly_dark',
                    height=500,
                    plot_bgcolor='#121826',
                    paper_bgcolor='#121826',
                    font=dict(color='#FFFFFF')
                )
                st.plotly_chart(fig, use_container_width=True)

            # Company description
            if 'longBusinessSummary' in info:
                st.markdown('<h3 class="section-header">Company Description</h3>', unsafe_allow_html=True)
                st.write(info['longBusinessSummary'])

            # NEW: Detailed Company Information with JSON export
            st.markdown('<h3 class="section-header">Detailed Company Information</h3>', unsafe_allow_html=True)

            # Create tabs for different info categories
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["Financials", "Holdings", "Analyst Info", "Trading Info", "Raw JSON"])

            with tab1:
                fin_data = {}
                if 'totalRevenue' in info: fin_data[
                    'Total Revenue'] = f'₹{info.get("totalRevenue", 0) / 10000000:.2f} Cr'
                if 'profitMargins' in info: fin_data['Profit Margins'] = f'{info.get("profitMargins", 0) * 100:.2f}%'
                if 'totalDebt' in info: fin_data['Total Debt'] = f'₹{info.get("totalDebt", 0) / 10000000:.2f} Cr'
                if 'totalCash' in info: fin_data['Total Cash'] = f'₹{info.get("totalCash", 0) / 10000000:.2f} Cr'
                if 'freeCashflow' in info: fin_data[
                    'Free Cashflow'] = f'₹{info.get("freeCashflow", 0) / 10000000:.2f} Cr'

                if fin_data:
                    fin_df = pd.DataFrame(list(fin_data.items()), columns=['Metric', 'Value'])
                    st.dataframe(fin_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No financial data available")

            with tab2:
                hold_data = {}
                if 'heldPercentInsiders' in info: hold_data[
                    'Insider Holdings'] = f'{info.get("heldPercentInsiders", 0) * 100:.2f}%'
                if 'heldPercentInstitutions' in info: hold_data[
                    'Institutional Holdings'] = f'{info.get("heldPercentInstitutions", 0) * 100:.2f}%'
                if 'floatShares' in info: hold_data['Float Shares'] = f'{info.get("floatShares", 0):,}'
                if 'sharesOutstanding' in info: hold_data[
                    'Shares Outstanding'] = f'{info.get("sharesOutstanding", 0):,}'

                if hold_data:
                    hold_df = pd.DataFrame(list(hold_data.items()), columns=['Metric', 'Value'])
                    st.dataframe(hold_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No holdings data available")

            with tab3:
                analyst_data = {}
                if 'recommendationKey' in info: analyst_data['Recommendation'] = info.get("recommendationKey",
                                                                                          "N/A").title()
                if 'targetMeanPrice' in info: analyst_data[
                    'Target Mean Price'] = f'₹{info.get("targetMeanPrice", 0):.2f}'
                if 'targetHighPrice' in info: analyst_data[
                    'Target High Price'] = f'₹{info.get("targetHighPrice", 0):.2f}'
                if 'targetLowPrice' in info: analyst_data['Target Low Price'] = f'₹{info.get("targetLowPrice", 0):.2f}'
                if 'numberOfAnalystOpinions' in info: analyst_data['Analyst Opinions'] = info.get(
                    "numberOfAnalystOpinions", "N/A")

                if analyst_data:
                    analyst_df = pd.DataFrame(list(analyst_data.items()), columns=['Metric', 'Value'])
                    st.dataframe(analyst_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No analyst data available")

            with tab4:
                trade_data = {}
                if 'fiftyTwoWeekHigh' in info: trade_data['52 Week High'] = f'₹{info.get("fiftyTwoWeekHigh", 0):.2f}'
                if 'fiftyTwoWeekLow' in info: trade_data['52 Week Low'] = f'₹{info.get("fiftyTwoWeekLow", 0):.2f}'
                if 'fiftyDayAverage' in info: trade_data['50 Day Average'] = f'₹{info.get("fiftyDayAverage", 0):.2f}'
                if 'twoHundredDayAverage' in info: trade_data[
                    '200 Day Average'] = f'₹{info.get("twoHundredDayAverage", 0):.2f}'
                if 'volume' in info: trade_data['Volume'] = f'{info.get("volume", 0):,}'
                if 'averageVolume' in info: trade_data['Avg Volume'] = f'{info.get("averageVolume", 0):,}'
                if 'averageVolume10days' in info: trade_data[
                    'Avg Volume (10D)'] = f'{info.get("averageVolume10days", 0):,}'

                if trade_data:
                    trade_df = pd.DataFrame(list(trade_data.items()), columns=['Metric', 'Value'])
                    st.dataframe(trade_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No trading data available")

            with tab5:
                # Display and allow download of full JSON data
                st.markdown("### Full Company Information (JSON)")
                st.json(info)

                # Create a downloadable JSON file
                json_str = json.dumps(info, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"{ticker}_info.json",
                    mime="application/json"
                )

        except Exception as e:
            st.error(f"Error loading company data: {str(e)}")

# Market Movers - Top Gainers & Losers
elif selected == "Market Movers":
    st.markdown('<h1 class="main-header">📈 Market Movers</h1>', unsafe_allow_html=True)

    # Predefined list of Nifty 50 stocks
    nifty_50 = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'ITC.NS',
        'AXISBANK.NS', 'LT.NS', 'HCLTECH.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS',
        'MARUTI.NS', 'TITAN.NS', 'M&M.NS', 'SUNPHARMA.NS', 'HINDALCO.NS'
    ]


    # Fetch data for all stocks
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_stock_data(tickers):
        data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2d")
                if not hist.empty and len(hist) >= 2:
                    prev_close = hist['Close'].iloc[-2]
                    current_price = hist['Close'].iloc[-1]
                    change = current_price - prev_close
                    percent_change = (change / prev_close) * 100
                    data[ticker] = {
                        'name': stock.info.get('shortName', ticker),
                        'current_price': current_price,
                        'change': change,
                        'percent_change': percent_change
                    }
            except:
                continue
        return data


    stock_data = get_stock_data(nifty_50)

    if stock_data:
        # Create DataFrame
        df = pd.DataFrame.from_dict(stock_data, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'ticker'}, inplace=True)

        # Top Gainers
        gainers = df.nlargest(5, 'percent_change')

        # Top Losers
        losers = df.nsmallest(5, 'percent_change')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<h3 class="section-header">📈 Top Gainers</h3>', unsafe_allow_html=True)
            for _, row in gainers.iterrows():
                st.markdown(f'''
                <div class="metric-card">
                    <h4>{row['name']}</h4>
                    <p>Price: ₹{row['current_price']:.2f}</p>
                    <p class="positive-change">+{row['change']:.2f} (+{row['percent_change']:.2f}%)</p>
                </div>
                ''', unsafe_allow_html=True)

        with col2:
            st.markdown('<h3 class="section-header">📉 Top Losers</h3>', unsafe_allow_html=True)
            for _, row in losers.iterrows():
                st.markdown(f'''
                <div class="metric-card">
                    <h4>{row['name']}</h4>
                    <p>Price: ₹{row['current_price']:.2f}</p>
                    <p class="negative-change">{row['change']:.2f} ({row['percent_change']:.2f}%)</p>
                </div>
                ''', unsafe_allow_html=True)

        # Display full table
        st.markdown('<h3 class="section-header">📊 All Stocks Performance</h3>', unsafe_allow_html=True)
        st.dataframe(df[['name', 'current_price', 'change', 'percent_change']],
                     column_config={
                         "name": "Company Name",
                         "current_price": "Current Price",
                         "change": "Change",
                         "percent_change": "Change %"
                     }, use_container_width=True)
    else:
        st.error("Could not fetch stock data. Please try again later.")

# Add other sections similarly...

else:
    st.info("This section is under development. Check back soon!")
