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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Set page config with dark theme
st.set_page_config(
    page_title="MarketMentor",
    layout="wide",
    page_icon="📈"
)

# Apply custom CSS for dark theme with red accents
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
    }
    .stSelectbox>div>div>select {
        background-color: #262730;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    .stSlider>div>div>div>div {
        background-color: #FF4B4B;
    }
    .stProgress>div>div>div {
        background-color: #FF4B4B;
    }
    .stMetric {
        background-color: #262730;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stAlert {
        background-color: #262730;
    }
    .css-1d391kg {
        background-color: #0E1117;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #FAFAFA;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: #0E1117;
        font-weight: bold;
    }
    .stDataFrame {
        background-color: #262730;
    }
    .stTable {
        background-color: #262730;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    header {
        background-color: #0E1117;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-share.png", width=80)
    st.title("MarketMentor")

    selected = option_menu(
        None,
        ["Home", "Company Overview", "Market Movers", "F&O", "Global Markets", "Mutual Funds", "SIP Calculator",
         "IPO Tracker", "Predictions for Mutual Funds & IPOs", "Mutual Fund NAV Viewer", "Sectors", "News", "Learning",
         "Volume Spike", "Stock Screener", "Predictions", "Buy/Sell Predictor", "News Sentiment"],
        icons=['house', 'building', 'graph-up', 'bar-chart', 'globe', 'bank', 'calculator', 'rocket', 'activity',
               'line-chart', 'grid', 'newspaper', 'book', 'activity', 'search', 'graph-up-arrow', 'currency-exchange',
               'chat'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#0E1117"},
            "icon": {"color": "#FF4B4B", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#FAFAFA"},
            "nav-link-selected": {"background-color": "#FF4B4B", "color": "#0E1117"},
        }
    )

# Home - Market Overview
if selected == "Home":
    st.title("🏠 Home - Market Overview")

    # Add a search bar for quick stock lookup
    col1, col2 = st.columns([3, 1])
    with col1:
        quick_search = st.text_input("🔍 Quick Stock Search", placeholder="Enter ticker (e.g., AAPL, MSFT)")
    with col2:
        st.write("")
        st.write("")
        if st.button("Search"):
            selected = "Company Overview"

    # Market indices performance
    st.subheader("📊 Major Indices Performance")
    indices = {
        "^NSEI": "Nifty 50",
        "^BSESN": "Sensex",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^GSPC": "S&P 500",
    }

    cols = st.columns(len(indices))
    for idx, (symbol, name) in enumerate(indices.items()):
        try:
            data = yf.Ticker(symbol).history(period="1d")
            last_close = round(data['Close'].iloc[-1], 2)
            prev_close = round(data['Close'].iloc[-2] if len(data) > 1 else data['Open'].iloc[-1], 2)
            change = round(last_close - prev_close, 2)
            percent_change = round((change / prev_close) * 100, 2)

            # Determine color based on performance
            delta_color = "normal"
            if percent_change > 0:
                delta_color = "inverse"

            cols[idx].metric(
                label=name,
                value=f"{last_close}",
                delta=f"{percent_change}%",
                delta_color=delta_color
            )
        except:
            cols[idx].metric(label=name, value="N/A", delta="N/A")

    # Add market sentiment indicator
    st.subheader("📈 Market Sentiment")
    sentiment_cols = st.columns(3)

    with sentiment_cols[0]:
        st.markdown("**Overall Sentiment**")
        # Simple sentiment calculation based on index performance
        positive_count = 0
        for symbol in indices.keys():
            try:
                data = yf.Ticker(symbol).history(period="5d")
                if len(data) > 1:
                    current = data['Close'].iloc[-1]
                    previous = data['Close'].iloc[-2]
                    if current > previous:
                        positive_count += 1
            except:
                pass

        sentiment_score = positive_count / len(indices) if indices else 0.5
        sentiment_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sentiment_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Bullish vs Bearish"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#FF4B4B"},
                'steps': [
                    {'range': [0, 35], 'color': "#FF0022"},
                    {'range': [35, 65], 'color': "#888888"},
                    {'range': [65, 100], 'color': "#00AA55"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': sentiment_score * 100
                }
            }
        ))
        sentiment_gauge.update_layout(height=250)
        st.plotly_chart(sentiment_gauge, use_container_width=True)

    with sentiment_cols[1]:
        st.markdown("**Top Gainers**")
        # Example top gainers
        gainers_data = {
            "Stock": ["TSLA", "NVDA", "AMD", "SHOP", "NET"],
            "Change %": [8.7, 6.3, 5.2, 4.8, 4.1]
        }
        gainers_df = pd.DataFrame(gainers_data)
        st.dataframe(gainers_df.style.background_gradient(cmap="Greens"), use_container_width=True)

    with sentiment_cols[2]:
        st.markdown("**Top Losers**")
        # Example top losers
        losers_data = {
            "Stock": ["MRNA", "PTON", "ZM", "ROKU", "DOCU"],
            "Change %": [-6.2, -5.7, -4.9, -4.3, -3.8]
        }
        losers_df = pd.DataFrame(losers_data)
        st.dataframe(losers_df.style.background_gradient(cmap="Reds"), use_container_width=True)

    # Recent news section
    st.subheader("📰 Market News")
    try:
        url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey={NEWS_API_KEY}&pageSize=3"
        response = requests.get(url)

        if response.status_code == 200:
            articles = response.json().get("articles", [])
            news_cols = st.columns(3)

            for idx, article in enumerate(articles[:3]):
                with news_cols[idx]:
                    st.markdown(f"""
                    <div style='background-color: #262730; padding: 15px; border-radius: 10px; height: 220px;'>
                        <h4 style='color: #FF4B4B;'>{article['title'][:50]}...</h4>
                        <p style='font-size: 14px;'>{article['description'][:100]}...</p>
                        <p style='font-size: 12px; color: #888;'>{article['source']['name']} | {article['publishedAt'][:10]}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Business news will appear here. Add a valid News API key to see live news.")
    except:
        st.info("Business news will appear here. Add a valid News API key to see live news.")

# Company Overview - Detailed stock analysis
elif selected == "Company Overview":
    st.title("🏢 Company Overview")

    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, RELIANCE.NS)", "RELIANCE.NS")
    with col2:
        period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

    if ticker:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period=period)

            if hist.empty:
                st.warning("No data available for this ticker.")
            else:
                # Display company info
                st.subheader(f"{info.get('longName', ticker)} ({ticker})")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Open'].iloc[-1]
                    change = current_price - prev_close
                    percent_change = (change / prev_close) * 100

                    st.metric(
                        label="Current Price",
                        value=f"₹{current_price:.2f}" if ticker.endswith('.NS') else f"${current_price:.2f}",
                        delta=f"{percent_change:.2f}%"
                    )

                with col2:
                    st.metric("Open", f"₹{hist['Open'].iloc[-1]:.2f}" if ticker.endswith(
                        '.NS') else f"${hist['Open'].iloc[-1]:.2f}")

                with col3:
                    st.metric("High", f"₹{hist['High'].iloc[-1]:.2f}" if ticker.endswith(
                        '.NS') else f"${hist['High'].iloc[-1]:.2f}")

                with col4:
                    st.metric("Low", f"₹{hist['Low'].iloc[-1]:.2f}" if ticker.endswith(
                        '.NS') else f"${hist['Low'].iloc[-1]:.2f}")

                # Display key metrics
                st.subheader("Key Metrics")
                metrics_cols = st.columns(4)

                with metrics_cols[0]:
                    st.metric("Market Cap", f"₹{info.get('marketCap', 0) / 1e12:.2f}T" if ticker.endswith(
                        '.NS') else f"${info.get('marketCap', 0) / 1e9:.2f}B")

                with metrics_cols[1]:
                    st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")

                with metrics_cols[2]:
                    st.metric("EPS", f"₹{info.get('trailingEps', 0):.2f}" if ticker.endswith(
                        '.NS') else f"${info.get('trailingEps', 0):.2f}")

                with metrics_cols[3]:
                    st.metric("Dividend Yield",
                              f"{info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0:.2f}%")

                # Price chart
                st.subheader("Price Chart")
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
                    title=f"{ticker} Stock Price",
                    yaxis_title="Price",
                    xaxis_title="Date",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                # Volume chart
                st.subheader("Trading Volume")
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=hist.index,
                    y=hist['Volume'],
                    name='Volume',
                    marker_color='#FF4B4B'
                ))
                fig2.update_layout(
                    title=f"{ticker} Trading Volume",
                    yaxis_title="Volume",
                    xaxis_title="Date",
                    template="plotly_dark",
                    height=300
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Financials section
                st.subheader("Financial Performance")
                try:
                    financials = stock.financials
                    if not financials.empty:
                        # Select only the most recent years
                        recent_years = financials.columns[:4]
                        financials_recent = financials[recent_years].T

                        # Select key financial metrics
                        key_metrics = [
                            'Total Revenue', 'Gross Profit', 'Operating Income',
                            'Net Income', 'Total Assets', 'Total LiabilitiesNet Minority Interest'
                        ]

                        # Filter for available metrics
                        available_metrics = [m for m in key_metrics if m in financials.index]
                        financials_filtered = financials.loc[available_metrics, recent_years].T

                        st.dataframe(financials_filtered.style.format("${:,.0f}").background_gradient(cmap="Reds"))
                    else:
                        st.info("Financial data not available for this company.")
                except:
                    st.info("Financial data not available for this company.")

        except Exception as e:
            st.error(f"Error retrieving data: {e}")

# Market Movers - Top Gainers & Losers
elif selected == "Market Movers":
    st.title("📈 Market Movers - Active Stocks, Top Gainers & Losers")

    # Fetch Nifty 50 components
    nifty_url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    try:
        nifty_df = pd.read_csv(nifty_url)
        symbols = nifty_df['Symbol'].tolist()
        symbols = [s + ".NS" for s in symbols]
    except:
        # Fallback if unable to fetch Nifty 50 components
        symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
                   'HINDUNILVR.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS', 'HDFC.NS']

    # Fetch data for all symbols
    data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, symbol in enumerate(symbols):
        status_text.text(f"Loading data... {i + 1}/{len(symbols)}")
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if not hist.empty and len(hist) > 1:
                prev_close = hist['Close'].iloc[-2]
                current_price = hist['Close'].iloc[-1]
                change = current_price - prev_close
                percent_change = (change / prev_close) * 100
                volume = hist['Volume'].iloc[-1]

                data[symbol] = {
                    'Price': current_price,
                    'Change': change,
                    'Change %': percent_change,
                    'Volume': volume
                }
        except:
            pass
        progress_bar.progress((i + 1) / len(symbols))

    status_text.empty()
    progress_bar.empty()

    if data:
        df = pd.DataFrame.from_dict(data, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Symbol'}, inplace=True)

        # Top Gainers
        gainers = df.nlargest(10, 'Change %')

        # Top Losers
        losers = df.nsmallest(10, 'Change %')

        # Most Active (by volume)
        active = df.nlargest(10, 'Volume')

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🚀 Top Gainers")
            gainers_display = gainers[['Symbol', 'Price', 'Change', 'Change %']].copy()
            gainers_display['Change %'] = gainers_display['Change %'].round(2)
            gainers_display['Change'] = gainers_display['Change'].round(2)
            st.dataframe(
                gainers_display.style.format({
                    'Price': '₹{:.2f}',
                    'Change': '₹{:.2f}',
                    'Change %': '{:.2f}%'
                }).background_gradient(subset=['Change %'], cmap='Greens'),
                use_container_width=True
            )

        with col2:
            st.subheader("📉 Top Losers")
            losers_display = losers[['Symbol', 'Price', 'Change', 'Change %']].copy()
            losers_display['Change %'] = losers_display['Change %'].round(2)
            losers_display['Change'] = losers_display['Change'].round(2)
            st.dataframe(
                losers_display.style.format({
                    'Price': '₹{:.2f}',
                    'Change': '₹{:.2f}',
                    'Change %': '{:.2f}%'
                }).background_gradient(subset=['Change %'], cmap='Reds'),
                use_container_width=True
            )

        st.subheader("🔥 Most Active (Volume)")
        active_display = active[['Symbol', 'Price', 'Volume']].copy()
        st.dataframe(
            active_display.style.format({
                'Price': '₹{:.2f}',
                'Volume': '{:,.0f}'
            }).background_gradient(subset=['Volume'], cmap='Oranges'),
            use_container_width=True
        )
    else:
        st.error("Failed to fetch market data. Please try again later.")

# Global Markets - Major Indices
elif selected == "Global Markets":
    st.title("🌍 Global Markets Status")

    global_indices = {
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^GSPC": "S&P 500",
        "^FTSE": "FTSE 100",
        "^N225": "Nikkei 225",
        "^HSI": "Hang Seng",
        "^AXJO": "ASX 200",
        "^GDAXI": "DAX",
        "^FCHI": "CAC 40",
        "^BSESN": "Sensex",
        "^NSEI": "Nifty 50"
    }

    # Fetch data for all indices
    indices_data = {}
    for symbol, name in global_indices.items():
        try:
            data = yf.Ticker(symbol).history(period="2d")
            if len(data) > 1:
                prev_close = data['Close'].iloc[-2]
                current_price = data['Close'].iloc[-1]
                change = current_price - prev_close
                percent_change = (change / prev_close) * 100
                indices_data[name] = {
                    'Price': current_price,
                    'Change': change,
                    'Change %': percent_change
                }
        except:
            indices_data[name] = {
                'Price': 0,
                'Change': 0,
                'Change %': 0
            }

    # Display indices in a grid
    cols = st.columns(4)
    for idx, (name, data) in enumerate(indices_data.items()):
        with cols[idx % 4]:
            # Determine color based on performance
            color = "#FF4B4B" if data['Change %'] < 0 else "#00AA55"
            st.markdown(f"""
            <div style="background-color: #262730; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                <h4 style="margin: 0; color: #FAFAFA;">{name}</h4>
                <h3 style="margin: 5px 0; color: #FAFAFA;">{data['Price']:.2f}</h3>
                <p style="margin: 0; color: {color}; font-weight: bold;">{data['Change']:+.2f} ({data['Change %']:+.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)

    # World map showing market performance
    st.subheader("🌎 Global Market Performance Heatmap")

    # Create sample data for the heatmap
    country_performance = {
        'Country': ['United States', 'United Kingdom', 'Germany', 'France', 'Japan',
                    'China', 'India', 'Australia', 'Brazil', 'Canada'],
        'Code': ['USA', 'GBR', 'DEU', 'FRA', 'JPN', 'CHN', 'IND', 'AUS', 'BRA', 'CAN'],
        'Performance': [2.5, -1.2, 0.8, -0.5, 1.7, -2.1, 3.2, 0.5, -1.8, 1.2]
    }

    heatmap_df = pd.DataFrame(country_performance)

    fig = px.choropleth(
        heatmap_df,
        locations="Code",
        color="Performance",
        hover_name="Country",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        title="Market Performance by Country"
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# Mutual Funds - Insights
elif selected == "Mutual Funds":
    st.title("💼 Mutual Funds Insights")

    # Sample mutual fund data
    mf_categories = {
        "Large Cap": [
            {"Name": "Axis Bluechip Fund", "Returns": "15.0%", "Risk": "Moderate", "Rating": "5★"},
            {"Name": "Mirae Asset Large Cap Fund", "Returns": "13.2%", "Risk": "Moderate", "Rating": "4★"},
            {"Name": "SBI Bluechip Fund", "Returns": "12.8%", "Risk": "Moderate", "Rating": "4★"}
        ],
        "Mid Cap": [
            {"Name": "Axis Midcap Fund", "Returns": "18.5%", "Risk": "High", "Rating": "5★"},
            {"Name": "Kotak Emerging Equity Fund", "Returns": "17.2%", "Risk": "High", "Rating": "4★"},
            {"Name": "SBI Magnum Midcap Fund", "Returns": "16.3%", "Risk": "High", "Rating": "4★"}
        ],
        "Small Cap": [
            {"Name": "Nippon India Small Cap Fund", "Returns": "22.1%", "Risk": "Very High", "Rating": "5★"},
            {"Name": "HDFC Small Cap Fund", "Returns": "20.5%", "Risk": "Very High", "Rating": "4★"},
            {"Name": "SBI Small Cap Fund", "Returns": "19.8%", "Risk": "Very High", "Rating": "4★"}
        ],
        "ELSS": [
            {"Name": "Axis Long Term Equity Fund", "Returns": "16.5%", "Risk": "Moderate", "Rating": "5★"},
            {"Name": "Mirae Asset Tax Saver Fund", "Returns": "15.8%", "Risk": "Moderate", "Rating": "4★"},
            {"Name": "SBI Long Term Equity Fund", "Returns": "14.9%", "Risk": "Moderate", "Rating": "4★"}
        ]
    }

    # Category selector
    category = st.selectbox("Select Fund Category", list(mf_categories.keys()))

    # Display funds in the selected category
    st.subheader(f"{category} Funds")
    funds_df = pd.DataFrame(mf_categories[category])
    st.dataframe(
        funds_df.style.background_gradient(subset=['Returns'], cmap='Greens'),
        use_container_width=True
    )

    # Performance comparison chart
    st.subheader("Category Performance Comparison")
    category_avg = {
        "Large Cap": 13.7,
        "Mid Cap": 17.3,
        "Small Cap": 20.8,
        "ELSS": 15.7
    }

    fig = px.bar(
        x=list(category_avg.keys()),
        y=list(category_avg.values()),
        labels={"x": "Category", "y": "Average Return (%)"},
        color=list(category_avg.values()),
        color_continuous_scale="Greens"
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Fund selector for detailed analysis
    st.subheader("Fund Analyzer")
    selected_fund = st.selectbox("Select a fund for analysis", [f["Name"] for f in mf_categories[category]])

    if selected_fund:
        # Simulate historical NAV data
        dates = pd.date_range(end=datetime.today(), periods=365, freq='D')
        nav_values = np.random.normal(100, 15, 365).cumsum()

        fig = px.line(
            x=dates,
            y=nav_values,
            labels={"x": "Date", "y": "NAV"},
            title=f"{selected_fund} - Historical NAV"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Risk-return metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("1Y Return", "15.2%")
        with col2:
            st.metric("3Y Return", "13.8%")
        with col3:
            st.metric("5Y Return", "12.5%")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Sharpe Ratio", "1.2")
        with col5:
            st.metric("Standard Deviation", "15.8")
        with col6:
            st.metric("Alpha", "2.1")

# Sectors - Sector Performance
elif selected == "Sectors":
    st.title("📊 Sector Wise Performance")

    # Sample sector data
    sector_performance = {
        "Banking": {"Performance": "+1.8%", "Outlook": "Positive", "Top Stock": "HDFC Bank"},
        "IT": {"Performance": "-0.5%", "Outlook": "Neutral", "Top Stock": "TCS"},
        "Energy": {"Performance": "+2.1%", "Outlook": "Positive", "Top Stock": "Reliance"},
        "FMCG": {"Performance": "+0.9%", "Outlook": "Neutral", "Top Stock": "HUL"},
        "Pharma": {"Performance": "-1.2%", "Outlook": "Negative", "Top Stock": "Sun Pharma"},
        "Auto": {"Performance": "+1.0%", "Outlook": "Positive", "Top Stock": "Maruti"},
        "Realty": {"Performance": "+3.2%", "Outlook": "Positive", "Top Stock": "DLF"},
        "Metals": {"Performance": "-2.1%", "Outlook": "Negative", "Top Stock": "Tata Steel"}
    }

    # Create DataFrame for display
    sector_df = pd.DataFrame.from_dict(sector_performance, orient='index')
    sector_df.reset_index(inplace=True)
    sector_df.rename(columns={'index': 'Sector'}, inplace=True)


    # Color code performance
    def color_performance(val):
        color = 'green' if '+' in val else 'red' if '-' in val else 'gray'
        return f'color: {color}'


    st.dataframe(
        sector_df.style.applymap(color_performance, subset=['Performance']),
        use_container_width=True
    )

    # Sector performance chart
    st.subheader("Sector Performance Comparison")
    performance_values = [float(s['Performance'].strip('%')) for s in sector_performance.values()]
    sector_names = list(sector_performance.keys())

    fig = px.bar(
        x=sector_names,
        y=performance_values,
        labels={"x": "Sector", "y": "Performance (%)"},
        color=performance_values,
        color_continuous_scale=px.colors.diverging.RdYlGn
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Sector rotation advice based on performance
    st.subheader("💡 Sector Rotation Advice")

    top_sector = max(sector_performance.items(), key=lambda x: float(x[1]['Performance'].strip('%')))
    bottom_sector = min(sector_performance.items(), key=lambda x: float(x[1]['Performance'].strip('%')))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background-color: #262730; padding: 15px; border-radius: 10px;">
            <h4 style="color: #00AA55;">✅ Consider Adding</h4>
            <p><strong>{top_sector[0]}</strong> is showing the strongest performance at {top_sector[1]['Performance']}.</p>
            <p>Top stock: {top_sector[1]['Top Stock']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background-color: #262730; padding: 15px; border-radius: 10px;">
            <h4 style="color: #FF4B4B;">⚠️ Consider Reducing</h4>
            <p><strong>{bottom_sector[0]}</strong> is showing the weakest performance at {bottom_sector[1]['Performance']}.</p>
            <p>Top stock: {bottom_sector[1]['Top Stock']}</p>
        </div>
        """, unsafe_allow_html=True)

# News - Latest Financial News
elif selected == "News":
    st.title("📰 Latest Financial News")

    news_query = st.text_input("Search Financial News:", "stock market")
    news_category = st.selectbox("Category", ["business", "technology", "general"], index=0)

    if st.button("Fetch News") or news_query:
        try:
            if NEWS_API_KEY:
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
            else:
                # Show sample news if no API key
                sample_news = [
                    {
                        "title": "Stock Markets Reach All-Time High",
                        "source": {"name": "Financial Times"},
                        "publishedAt": "2023-06-15T10:30:00Z",
                        "description": "Global stock markets surged to record levels today as investor confidence grows.",
                        "url": "#"
                    },
                    {
                        "title": "Tech Giants Report Strong Quarterly Earnings",
                        "source": {"name": "Bloomberg"},
                        "publishedAt": "2023-06-14T15:45:00Z",
                        "description": "Major technology companies exceeded analyst expectations in their latest quarterly reports.",
                        "url": "#"
                    },
                    {
                        "title": "Central Bank Holds Interest Rates Steady",
                        "source": {"name": "Reuters"},
                        "publishedAt": "2023-06-13T09:15:00Z",
                        "description": "The central bank decided to maintain current interest rates, citing stable economic conditions.",
                        "url": "#"
                    }
                ]

                for article in sample_news:
                    st.markdown("----")
                    st.subheader(article["title"])
                    st.write(f"*{article['source']['name']} - {article['publishedAt'].split('T')[0]}*")
                    st.write(article.get("description", "No description available."))
                    st.markdown(f"[🔗 Read More]({article['url']})")

        except Exception as e:
            st.error(f"Error fetching news: {e}")

# Learning - Stock Market Resources
elif selected == "Learning":
    st.title("📘 Learn the Stock Market")

    st.markdown("""
    <div style="background-color: #262730; padding: 25px; border-radius: 10px;">
        <h2 style="color: #FF4B4B;">Welcome to the Learning Hub</h2>
        <p>This section is crafted to help <strong>beginners, enthusiasts, and investors</strong> understand how the stock market works — with a strong foundation in both <strong>technical and fundamental analysis</strong>, along with insights from <strong>AI and machine learning</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    # Learning modules
    st.subheader("🎯 Learning Modules")

    modules = st.tabs(["Basics", "Technical Analysis", "Fundamental Analysis", "Trading Strategies"])

    with modules[0]:
        st.markdown("""
        <div style="background-color: #262730; padding: 15px; border-radius: 10px;">
            <h4 style="color: #FF4B4B;">Stock Market Basics</h4>
            <ul>
                <li>What is a stock?</li>
                <li>How stock markets work</li>
                <li>Types of orders (market, limit, stop-loss)</li>
                <li>Understanding market indices</li>
                <li>Risk management principles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with modules[1]:
        st.markdown("""
        <div style="background-color: #262730; padding: 15px; border-radius: 10px;">
            <h4 style="color: #FF4B4B;">Technical Analysis</h4>
            <ul>
                <li>Reading price charts</li>
                <li>Support and resistance levels</li>
                <li>Trend lines and channels</li>
                <li>Technical indicators (RSI, MACD, Moving Averages)</li>
                <li>Chart patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with modules[2]:
        st.markdown("""
        <div style="background-color: #262730; padding: 15px; border-radius: 10px;">
            <h4 style="color: #FF4B4B;">Fundamental Analysis</h4>
            <ul>
                <li>Reading financial statements</li>
                <li>Valuation metrics (P/E, P/B, EV/EBITDA)</li>
                <li>Analyzing company growth prospects</li>
                <li>Industry analysis</li>
                <li>Economic indicators</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with modules[3]:
        st.markdown("""
        <div style="background-color: #262730; padding: 15px; border-radius: 10px;">
            <h4 style="color: #FF4B4B;">Trading Strategies</h4>
            <ul>
                <li>Swing trading</li>
                <li>Day trading</li>
                <li>Position trading</li>
                <li>Options strategies</li>
                <li>Portfolio diversification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Interactive quiz
    st.subheader("🧠 Test Your Knowledge")

    quiz_question = st.selectbox(
        "What does P/E ratio stand for?",
        ["Price-to-Earnings ratio", "Profit-to-Expense ratio", "Portfolio-Equity ratio", "Purchase-Exchange ratio"]
    )

    if quiz_question == "Price-to-Earnings ratio":
        st.success("✅ Correct! The P/E ratio compares a company's stock price to its earnings per share.")
    elif quiz_question:
        st.error("❌ Incorrect. Try again!")

    # Resource links
    st.subheader("📚 Recommended Resources")

    resources = {
        "Books": ["The Intelligent Investor", "A Random Walk Down Wall Street", "Common Stocks and Uncommon Profits"],
        "Websites": ["Investopedia", "Morningstar", "Seeking Alpha"],
        "YouTube Channels": ["Investor's Business Daily", "Rayner Teo", "Andrei Jikh"]
    }

    for category, items in resources.items():
        with st.expander(category):
            for item in items:
                st.write(f"- {item}")

# Volume Spike Detector
elif selected == "Volume Spike":
    st.title("📈 Volume Spike Detector")
    st.markdown("This tool detects unusual volume surges in a stock based on a 10-day rolling average.")

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("🔎 Enter Stock Ticker (e.g., TCS.NS, INFY.NS):", "TCS.NS")
    with col2:
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
                spike_dates = spikes.index.strftime('%Y-%m-%d').tolist()
                spike_volumes = spikes["Volume"].values
                avg_volumes = spikes["Avg_Volume"].values

                spike_data = {
                    "Date": spike_dates,
                    "Volume": spike_volumes,
                    "10-Day Average": avg_volumes,
                    "Ratio": [f"{(v / a):.1f}x" for v, a in zip(spike_volumes, avg_volumes)]
                }

                spike_df = pd.DataFrame(spike_data)
                st.dataframe(spike_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error occurred: {e}")

# News Sentiment - Sentiment Analysis of News
elif selected == "News Sentiment":
    st.title("📊 News Sentiment Analysis")

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Enter Stock Ticker to analyze news sentiment:", "AAPL")
    with col2:
        num_articles = st.slider("Number of Articles to Analyze", 5, 20, 10)

    if ticker:
        st.info(f"Fetching and analyzing recent news sentiment for {ticker.upper()}...")

        # Sample sentiment data if no API key
        if not NEWS_API_KEY:
            sample_articles = [
                {"title": f"{ticker} Reports Strong Quarterly Earnings", "sentiment": 0.8},
                {"title": f"Analysts Upgrade {ticker} to Buy Rating", "sentiment": 0.7},
                {"title": f"{ticker} Faces Regulatory Challenges", "sentiment": -0.6},
                {"title": f"{ticker} Announces New Product Launch", "sentiment": 0.9},
                {"title": f"Competition Intensifies for {ticker}", "sentiment": -0.4},
                {"title": f"{ticker} Expands to New Markets", "sentiment": 0.6},
                {"title": f"{ticker} CEO Resigns Unexpectedly", "sentiment": -0.7},
                {"title": f"{ticker} Partners with Industry Leader", "sentiment": 0.5},
                {"title": f"{ticker} Stock Rated as Overvalued", "sentiment": -0.3},
                {"title": f"{ticker} Announces Stock Buyback Program", "sentiment": 0.4}
            ]

            sentiments = []
            for article in sample_articles[:num_articles]:
                st.write(f"📰 **{article['title']}**")
                st.write(f"🧠 Sentiment Score: {article['sentiment']:.3f}")
                sentiments.append(article['sentiment'])
                st.markdown("---")
        else:
            try:
                url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}&language=en&pageSize={num_articles}"
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
                else:
                    st.error("Failed to fetch news articles.")
                    sentiments = []
            except:
                st.error("Error analyzing news sentiment.")
                sentiments = []

        if sentiments:
            avg_sentiment = round(np.mean(sentiments), 3)

            # Sentiment gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_sentiment,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Average Sentiment for {ticker.upper()}"},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "#FF4B4B"},
                    'steps': [
                        {'range': [-1, -0.5], 'color': "#FF0022"},
                        {'range': [-0.5, 0], 'color': "#FF5566"},
                        {'range': [0, 0.5], 'color': "#55AAFF"},
                        {'range': [0.5, 1], 'color': "#00AA55"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Sentiment interpretation
            if avg_sentiment > 0.2:
                st.success(f"📈 **Overall Sentiment: Positive** (Score: {avg_sentiment})")
                st.info("This stock is receiving generally positive coverage in the news.")
            elif avg_sentiment < -0.2:
                st.error(f"📉 **Overall Sentiment: Negative** (Score: {avg_sentiment})")
                st.info("This stock is receiving generally negative coverage in the news.")
            else:
                st.warning(f"➖ **Overall Sentiment: Neutral** (Score: {avg_sentiment})")
                st.info("This stock is receiving mixed or neutral coverage in the news.")

# Predictions - Stock Price Prediction
elif selected == "Predictions":
    st.title("🔮 Stock Price Predictions")

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Enter Company Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")
    with col2:
        forecast_days = st.slider("Days to Forecast", 7, 90, 30)

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
                st.dataframe(hist.tail().style.format("{:.2f}"))

                # Plot the stock's historical closing price
                st.subheader("📊 Stock Price History")
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
                    title=f"{ticker} Stock Price",
                    yaxis_title="Price",
                    xaxis_title="Date",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                # Calculate moving averages
                sma50 = hist["Close"].rolling(window=50).mean()
                sma200 = hist["Close"].rolling(window=200).mean()

                st.subheader("📉 Moving Averages")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Close Price", line=dict(color='white')))
                fig2.add_trace(go.Scatter(x=hist.index, y=sma50, name="50-Day SMA", line=dict(color='yellow')))
                fig2.add_trace(go.Scatter(x=hist.index, y=sma200, name="200-Day SMA", line=dict(color='red')))
                fig2.update_layout(
                    title="Price vs Moving Averages",
                    yaxis_title="Price",
                    xaxis_title="Date",
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Simple prediction using moving average crossover
                st.subheader("📈 Price Prediction")
                st.info(
                    "This is a simple moving average-based prediction. For more accurate predictions, consider using advanced models.")

                # Generate prediction
                last_50_avg = hist["Close"].tail(50).mean()
                last_200_avg = hist["Close"].tail(200).mean()

                if last_50_avg > last_200_avg:
                    trend = "Bullish"
                    prediction = hist["Close"].iloc[-1] * (1 + 0.001 * forecast_days)  # Small daily increase
                else:
                    trend = "Bearish"
                    prediction = hist["Close"].iloc[-1] * (1 - 0.001 * forecast_days)  # Small daily decrease

                current_price = hist["Close"].iloc[-1]
                percent_change = ((prediction - current_price) / current_price) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price",
                              f"₹{current_price:.2f}" if ticker.endswith('.NS') else f"${current_price:.2f}")
                with col2:
                    st.metric("Predicted Price",
                              f"₹{prediction:.2f}" if ticker.endswith('.NS') else f"${prediction:.2f}")
                with col3:
                    st.metric("Expected Change", f"{percent_change:.2f}%", delta=f"{percent_change:.2f}%")

                st.write(f"**Trend:** {trend}")

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
            hist = stock.history(period="6mo")  # Fetch 6 months of data

            if hist.empty:
                st.warning("No data available for this ticker.")
            else:
                # Calculate technical indicators
                # Moving averages
                hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()

                # RSI
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                hist['RSI'] = 100 - (100 / (1 + rs))

                # MACD
                exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
                exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
                hist['MACD'] = exp12 - exp26
                hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()

                # Current values
                current_price = hist['Close'].iloc[-1]
                sma_20 = hist['SMA_20'].iloc[-1]
                sma_50 = hist['SMA_50'].iloc[-1]
                rsi = hist['RSI'].iloc[-1]
                macd = hist['MACD'].iloc[-1]
                signal = hist['Signal'].iloc[-1]

                # Display current values
                st.subheader("Current Technical Indicators")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Price", f"₹{current_price:.2f}" if ticker.endswith('.NS') else f"${current_price:.2f}")
                with col2:
                    st.metric("RSI", f"{rsi:.2f}")
                with col3:
                    st.metric("MACD", f"{macd:.2f}")
                with col4:
                    st.metric("Signal", f"{signal:.2f}")

                # Generate signals
                signals = []

                # Moving average crossover
                if sma_20 > sma_50:
                    signals.append(("Moving Average Crossover", "BUY", "20-day SMA above 50-day SMA"))
                else:
                    signals.append(("Moving Average Crossover", "SELL", "20-day SMA below 50-day SMA"))

                # RSI signals
                if rsi < 30:
                    signals.append(("RSI", "BUY", "Oversold condition (RSI < 30)"))
                elif rsi > 70:
                    signals.append(("RSI", "SELL", "Overbought condition (RSI > 70)"))
                else:
                    signals.append(("RSI", "HOLD", "RSI in neutral territory"))

                # MACD signals
                if macd > signal:
                    signals.append(("MACD", "BUY", "MACD above signal line"))
                else:
                    signals.append(("MACD", "SELL", "MACD below signal line"))

                # Display signals
                st.subheader("Trading Signals")
                signals_df = pd.DataFrame(signals, columns=['Indicator', 'Signal', 'Reason'])
                st.dataframe(signals_df, use_container_width=True)

                # Overall recommendation
                buy_count = sum(1 for s in signals if s[1] == 'BUY')
                sell_count = sum(1 for s in signals if s[1] == 'SELL')

                if buy_count > sell_count:
                    st.success(
                        f"📈 Overall Recommendation: BUY ({buy_count} out of {len(signals)} indicators suggest BUY)")
                elif sell_count > buy_count:
                    st.error(
                        f"📉 Overall Recommendation: SELL ({sell_count} out of {len(signals)} indicators suggest SELL)")
                else:
                    st.warning(f"⏸️ Overall Recommendation: HOLD (Mixed signals from indicators)")

                # Chart with indicators
                st.subheader("Technical Analysis Chart")
                fig = go.Figure()

                # Price
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='Price'
                ))

                # Moving averages
                fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name='20-Day SMA', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='50-Day SMA', line=dict(color='purple')))

                fig.update_layout(
                    title=f"{ticker} Price with Moving Averages",
                    yaxis_title="Price",
                    xaxis_title="Date",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                # RSI chart
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI', line=dict(color='cyan')))
                fig2.add_hline(y=70, line_dash="dash", line_color="red")
                fig2.add_hline(y=30, line_dash="dash", line_color="green")
                fig2.update_layout(
                    title="RSI Indicator",
                    yaxis_title="RSI",
                    xaxis_title="Date",
                    template="plotly_dark",
                    height=300
                )
                st.plotly_chart(fig2, use_container_width=True)

                # MACD chart
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD', line=dict(color='yellow')))
                fig3.add_trace(go.Scatter(x=hist.index, y=hist['Signal'], name='Signal', line=dict(color='red')))
                fig3.update_layout(
                    title="MACD Indicator",
                    yaxis_title="Value",
                    xaxis_title="Date",
                    template="plotly_dark",
                    height=300
                )
                st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:
            st.error(f"Error retrieving data: {e}")

# Stock Screener - Screen stocks based on criteria
elif selected == "Stock Screener":
    st.title("📊 Stock Screener")

    # Predefined list of 15 companies (Nifty 50 or a custom list of top companies)
    default_companies = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'HINDUNILVR.NS',
        'BAJAJFINSV.NS', 'HDFC.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'AXISBANK.NS', 'MARUTI.NS', 'LT.NS'
    ]

    # Screening criteria
    st.sidebar.header("Screening Criteria")

    min_market_cap = st.sidebar.number_input("Minimum Market Cap (in Cr)", min_value=0, value=10000, step=1000)
    max_pe = st.sidebar.number_input("Maximum P/E Ratio", min_value=0, value=50, step=5)
    min_dividend_yield = st.sidebar.number_input("Minimum Dividend Yield (%)", min_value=0.0, value=0.0, step=0.1)
    sector = st.sidebar.selectbox("Sector", ["Any", "Banking", "IT", "Energy", "FMCG", "Pharma", "Auto"])

    # Ask user whether they want to use the default list or input custom tickers
    choice = st.radio("Choose an option:", ("Use Default List", "Input Custom Tickers"))

    if choice == "Use Default List":
        screened_stocks = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, ticker in enumerate(default_companies):
            status_text.text(f"Screening {ticker}... ({i + 1}/{len(default_companies)})")
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # Apply screening criteria
                market_cap = info.get('marketCap', 0) / 1e7  # Convert to Cr
                pe_ratio = info.get('trailingPE', 0)
                dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0

                if (market_cap >= min_market_cap and
                        pe_ratio <= max_pe and
                        dividend_yield >= min_dividend_yield):

                    # Sector filter
                    if sector == "Any" or sector.lower() in str(info.get('sector', '')).lower():
                        screened_stocks.append({
                            'Ticker': ticker,
                            'Name': info.get('longName', ticker),
                            'Price': info.get('regularMarketPrice', 0),
                            'Market Cap (Cr)': market_cap,
                            'P/E Ratio': pe_ratio,
                            'Dividend Yield (%)': dividend_yield,
                            'Sector': info.get('sector', 'N/A')
                        })
            except:
                pass

            progress_bar.progress((i + 1) / len(default_companies))

        status_text.empty()
        progress_bar.empty()

        if screened_stocks:
            screened_df = pd.DataFrame(screened_stocks)
            st.subheader(f"📋 Screening Results ({len(screened_stocks)} stocks matched your criteria)")
            st.dataframe(
                screened_df.style.format({
                    'Price': '₹{:.2f}',
                    'Market Cap (Cr)': '{:,.0f}',
                    'P/E Ratio': '{:.2f}',
                    'Dividend Yield (%)': '{:.2f}%'
                }).background_gradient(subset=['P/E Ratio'], cmap='Reds_r'),
                use_container_width=True
            )
        else:
            st.warning("No stocks matched your screening criteria. Try loosening your filters.")

    elif choice == "Input Custom Tickers":
        tickers_input = st.text_area("Enter stock tickers (separated by space or comma):", "")

        if tickers_input:
            tickers_list = [ticker.strip() for ticker in tickers_input.replace(',', ' ').split() if ticker.strip()]

            if tickers_list:
                screened_stocks = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, ticker in enumerate(tickers_list):
                    status_text.text(f"Screening {ticker}... ({i + 1}/{len(tickers_list)})")
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info

                        # Apply screening criteria
                        market_cap = info.get('marketCap', 0) / 1e7  # Convert to Cr
                        pe_ratio = info.get('trailingPE', 0)
                        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0

                        if (market_cap >= min_market_cap and
                                pe_ratio <= max_pe and
                                dividend_yield >= min_dividend_yield):

                            # Sector filter
                            if sector == "Any" or sector.lower() in str(info.get('sector', '')).lower():
                                screened_stocks.append({
                                    'Ticker': ticker,
                                    'Name': info.get('longName', ticker),
                                    'Price': info.get('regularMarketPrice', 0),
                                    'Market Cap (Cr)': market_cap,
                                    'P/E Ratio': pe_ratio,
                                    'Dividend Yield (%)': dividend_yield,
                                    'Sector': info.get('sector', 'N/A')
                                })
                    except:
                        pass

                    progress_bar.progress((i + 1) / len(tickers_list))

                status_text.empty()
                progress_bar.empty()

                if screened_stocks:
                    screened_df = pd.DataFrame(screened_stocks)
                    st.subheader(f"📋 Screening Results ({len(screened_stocks)} stocks matched your criteria)")
                    st.dataframe(
                        screened_df.style.format({
                            'Price': '₹{:.2f}',
                            'Market Cap (Cr)': '{:,.0f}',
                            'P/E Ratio': '{:.2f}',
                            'Dividend Yield (%)': '{:.2f}%'
                        }).background_gradient(subset=['P/E Ratio'], cmap='Reds_r'),
                        use_container_width=True
                    )
                else:
                    st.warning("No stocks matched your screening criteria. Try loosening your filters.")
            else:
                st.warning("Please enter valid stock tickers.")

# Add other sections (F&O, SIP Calculator, IPO Tracker, etc.) with similar enhancements
# For brevity, I'll show one more example and you can follow the pattern for the rest

# F&O - Futures and Options
elif selected == "F&O":
    st.title("📊 Futures & Options")

    st.info("F&O data is currently simulated. Integrate with a live API for real-time data.")

    # Simulated F&O data
    fo_data = {
        "Instrument": ["NIFTY", "BANKNIFTY", "RELIANCE", "INFY", "HDFCBANK"],
        "Future Price": [18250.50, 40580.25, 2580.75, 1650.30, 1625.40],
        "OI Change (%)": [2.5, -1.8, 4.2, -0.5, 3.1],
        "PCR": [0.85, 1.25, 0.92, 1.10, 0.78],
        "IV (%)": [15.2, 18.5, 22.1, 16.8, 19.3]
    }

    fo_df = pd.DataFrame(fo_data)
    st.dataframe(
        fo_df.style.background_gradient(subset=['OI Change (%)'], cmap='RdYlGn')
        .background_gradient(subset=['PCR'], cmap='RdYlGn')
        .background_gradient(subset=['IV (%)'], cmap='Reds'),
        use_container_width=True
    )

    # OI analysis
    st.subheader("Open Interest Analysis")
    oi_fig = px.bar(
        fo_df,
        x='Instrument',
        y='OI Change (%)',
        color='OI Change (%)',
        color_continuous_scale='RdYlGn'
    )
    oi_fig.update_layout(template="plotly_dark")
    st.plotly_chart(oi_fig, use_container_width=True)

    # IV analysis
    st.subheader("Implied Volatility Analysis")
    iv_fig = px.bar(
        fo_df,
        x='Instrument',
        y='IV (%)',
        color='IV (%)',
        color_continuous_scale='Reds'
    )
    iv_fig.update_layout(template="plotly_dark")
    st.plotly_chart(iv_fig, use_container_width=True)

# For other sections, follow the same pattern of enhancement
# Add dark theme, improve layouts, add visualizations, etc.

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>MarketMentor - Stock Market Dashboard | Developed with ❤️ using Streamlit</p>
        <p>Disclaimer: This is a simulation for educational purposes only. Not financial advice.</p>
    </div>
    """,
    unsafe_allow_html=True
)
