import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
from newsapi import NewsApiClient
import sqlite3
import hashlib


# ------------------------------- CONFIG ----------------------------------
st.set_page_config(
    page_title="🚀 Ashwik's Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Customize Streamlit Theme
st.markdown(
    """
    <style>
    /* Change background color */
    .stApp {
        background-color: #Black;
    }
    /* Change sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        color: white;
    }
    /* Headings and subheadings */
    h1, h2, h3, h4 {
        color: #0a0a23;
    }
    /* Buttons */
    button {
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 10px;
        padding: 8px 20px;
    }
    /* Input fields */
    input {
        border-radius: 5px;
    }
    /* Metrics (price, day high/low) */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------- NEWS API (Free Key Needed) -----------------

NEWS_API_KEY = '0b08be107dca45d3be30ca7e06544408'

# --------------------------- LOAD DATA FUNCTION --------------------------
@st.cache_resource
def load_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2y")
    return stock, hist

# ---------------------------- DATABASE SETUP ---------------------------
conn = sqlite3.connect("user_data.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)''')
conn.commit()

# ---------------------------- USER AUTHENTICATION ---------------------
def create_user(username, password):
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
    conn.commit()

def check_user(username, password):
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_pw))
    return c.fetchone() is not None

# ---------------------------- SIDEBAR NAVIGATION -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Login", "Sign Up", "📊 Overview", "🔮 Prediction & Forecasting", "📢 News & Events", "💥 Volume Spike Detector", "📚 Learning Material & Purpose"])

# ---------------------------- LOGIN PAGE -----------------------------
if page == "Login":
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_user(username, password):
            st.success("Login Successful!")
            page = "📊 Overview"
        else:
            st.error("Invalid Credentials. Please try again.")

# ---------------------------- SIGN UP PAGE -----------------------------
elif page == "Sign Up":
    st.title("Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if password == confirm_password:
            create_user(username, password)
            st.success("Account Created Successfully!")
        else:
            st.error("Passwords do not match.")


# ---------------------------- OVERVIEW PAGE -------------------------

import streamlit as st
import plotly.graph_objs as go
import time

# --- Premium CSS for Tabs (Glassmorphism + Hover Effects) ---
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: Black;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: Black;
        padding: 8px 20px;
        border-radius: 12px;
        margin-right: 5px;
        color: black;
        font-weight: bold;
        transition: 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: Black;
        color: black;
        transform: scale(1.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: Black;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- Overview Page ---
if page == "📊 Overview":
    st.header("📈 Real-Time Stock Overview Dashboard")

    ticker = st.text_input("🔎 Enter Stock Ticker (e.g., AAPL, TCS.NS):", "AAPL")

    if ticker:
        with st.spinner(f"Fetching data for {ticker}..."):
            time.sleep(1)  # Loading animation
            stock, hist = load_data(ticker)

        # --- Premium Tabs ---
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Price Chart",
            "🏢 Company Info",
            "📊 Technical Indicators",
            "🔍 Volume Analysis",
            "💹 Candlestick View",
            "📅 Dividends & Splits"

        ])

        # --- 📈 Tab 1: Price Chart ---
        with tab1:
            st.subheader(f"📈 {ticker} Stock Price Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name="Close Price", line=dict(color='royalblue')))
            fig.update_layout(
                title=f"{ticker} Closing Price Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- 🏢 Tab 2: Company Info ---
        with tab2:
            st.subheader(f"🏢 {ticker} Company Profile")
            st.json(stock.info)

        # --- 📊 Tab 3: Technical Indicators ---
        with tab3:
            st.subheader(f"📊 {ticker} Technical Indicators")

            # Moving Averages
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            hist['MA200'] = hist['Close'].rolling(window=200).mean()

            ma_fig = go.Figure()
            ma_fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close Price', line=dict(color='blue')))
            ma_fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], name='50-Day MA', line=dict(color='orange')))
            ma_fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], name='200-Day MA', line=dict(color='green')))
            ma_fig.update_layout(
                title="Moving Averages (50 vs 200)",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(ma_fig, use_container_width=True)

            # RSI Indicator
            def calculate_rsi(data, window=14):
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            hist['RSI'] = calculate_rsi(hist)

            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name="RSI", line=dict(color='purple')))
            rsi_fig.update_layout(
                title="Relative Strength Index (RSI)",
                xaxis_title="Date",
                yaxis_title="RSI",
                yaxis=dict(range=[0, 100]),
                template="plotly_white",
                height=400
            )
            st.plotly_chart(rsi_fig, use_container_width=True)

        # --- 🔍 Tab 4: Volume Analysis ---
        with tab4:
            st.subheader(f"🔍 {ticker} Volume & Spike Analysis")

            hist['Volume_MA'] = hist['Volume'].rolling(window=5).mean()
            hist['Spike'] = hist['Volume'] > (2 * hist['Volume_MA'])

            vol_fig = go.Figure()
            vol_fig.add_trace(go.Scatter(x=hist.index, y=hist['Volume'], name="Volume", line=dict(color='gray')))
            vol_fig.add_trace(
                go.Scatter(
                    x=hist[hist['Spike']].index,
                    y=hist[hist['Spike']]['Volume'],
                    mode='markers',
                    name="Volume Spike",
                    marker=dict(size=10, color="red", symbol="star")
                )
            )
            vol_fig.update_layout(
                title="Volume and Spike Detection",
                xaxis_title="Date",
                yaxis_title="Volume",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(vol_fig, use_container_width=True)

            if hist['Spike'].any():
                st.warning("⚡ Volume Spikes Detected! Possible Big Moves Incoming!")

        # --- 💹 Tab 5: Candlestick Chart ---
        with tab5:
            st.subheader(f"💹 {ticker} Candlestick View")
            candle_fig = go.Figure(data=[go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                increasing_line_color='green',
                decreasing_line_color='red'
            )])
            candle_fig.update_layout(
                title="Candlestick Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white",
                height=600
            )
            st.plotly_chart(candle_fig, use_container_width=True)

        # --- 📅 Tab 6: Dividends and Splits ---
        with tab6:
            st.subheader(f"📅 {ticker} Dividends & Splits History")

            if not hist['Dividends'].empty:
                st.write("### 📈 Dividend Payments Over Time")
                div_fig = go.Figure()
                div_fig.add_trace(go.Bar(x=hist.index, y=hist['Dividends'], name="Dividends", marker_color="blue"))
                div_fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Dividend Amount",
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(div_fig, use_container_width=True)
            else:
                st.info("ℹ️ No Dividend Data Found.")

            if not hist['Stock Splits'].empty:
                st.write("### 🔀 Stock Splits Over Time")
                st.dataframe(hist[['Stock Splits']][hist['Stock Splits'] != 0])
            else:
                st.info("ℹ️ No Stock Splits Data Found.")


                # Function to fetch Sector Performance Data
                def fetch_sector_performance():
                    # Example sectors: You can expand this with more
                    sectors = ['XLF', 'XLY', 'XLC', 'XLI', 'XLE', 'XLB', 'XLV', 'XLRE', 'XBI', 'XLA']
                    sector_data = {}
                    for sector in sectors:
                        data = yf.download(sector, period="6mo", interval="1d")  # 6-month data
                        sector_data[sector] = data['Close'].pct_change().cumsum() * 100  # Calculate % growth
                    return sector_data




# ---------------------------- PREDICTION PAGE -------------------------
elif page == "🔮 Prediction & Forecasting":
    st.subheader("🔮 Predict Future Stock Prices")

    ticker = st.text_input("Enter Stock Ticker for Prediction", "AAPL")

    if ticker:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")

        df_train = hist.reset_index()[['Date', 'Close']]
        df_train.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
        df_train['ds'] = pd.to_datetime(df_train['ds']).dt.tz_localize(None)

        model = Prophet(daily_seasonality=True)
        model.fit(df_train)

        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)

        st.subheader("📈 Forecasted Prices")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted Price"))
        fig1.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], name="Actual Price"))
        fig1.update_layout(title=f"{ticker} - 3 Month Forecast", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig1, use_container_width=True)

# ---------------------------- NEWS PAGE -----------------------------
elif page == "📢 News & Events":
    st.subheader("📰 Latest News about Company")

    ticker = st.text_input("Enter Stock Ticker for News", "AAPL")

    if ticker:
        company_name = yf.Ticker(ticker).info.get('shortName', ticker)
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)

        try:
            news = newsapi.get_everything(q=company_name, language='en', sort_by='relevancy', page_size=5)
            for article in news['articles']:
                st.markdown(f"### [{article['title']}]({article['url']})")
                st.write(article['description'])
                st.write("---")
        except Exception as e:
            st.error(f"Error Fetching News: {e}")

# ---------------------------- VOLUME SPIKE DETECTOR ----------------------
elif page == "💥 Volume Spike Detector":
    st.subheader("🚀 Detect Unusual Volume Spikes")

    ticker = st.text_input("Enter Stock Ticker for Volume Spike Detection", "AAPL")

    if ticker:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        hist['Volume_MA'] = hist['Volume'].rolling(window=5).mean()
        hist['Spike'] = hist['Volume'] > (2 * hist['Volume_MA'])

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hist.index, y=hist['Volume'], mode='lines', name="Volume"))
        fig2.add_trace(go.Scatter(x=hist[hist['Spike']].index, y=hist[hist['Spike']]['Volume'], mode='markers', name="Spikes", marker=dict(size=10, color="red")))
        fig2.update_layout(title="Volume & Spikes", xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(fig2, use_container_width=True)

        if hist['Spike'].any():
            st.warning("⚡ Volume Spikes Detected! Possible Big Moves Incoming!")





import pyqrcode
import tempfile
import streamlit as st

# Generate QR codes for the URLs
qr_investopedia = pyqrcode.create("https://www.investopedia.com/terms/s/stockmarket.asp")
qr_coursera = pyqrcode.create("https://www.coursera.org/learn/stock-market")
qr_udemy = pyqrcode.create("https://www.udemy.com/course/stock-market-investing/")
qr_youtube = pyqrcode.create("https://www.youtube.com/c/InvestingAcademy")

# Save QR codes as PNG files in a temporary location
with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_investopedia:
    qr_investopedia.png(tmp_investopedia.name, scale=6)
    investopedia_path = tmp_investopedia.name

with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_coursera:
    qr_coursera.png(tmp_coursera.name, scale=6)
    coursera_path = tmp_coursera.name

with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_udemy:
    qr_udemy.png(tmp_udemy.name, scale=6)
    udemy_path = tmp_udemy.name

with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_youtube:
    qr_youtube.png(tmp_youtube.name, scale=6)
    youtube_path = tmp_youtube.name

# Display the learning material introduction and QR codes using Streamlit
st.title("📚 Stock Market Learning Materials by Ashwik Bire")
st.subheader("Enhance your knowledge of the stock market with these curated resources. Scan the QR codes to access valuable material and start your learning journey.")

st.markdown("""
Here are some excellent resources that will guide you through the intricacies of stock market investing:

1. **[Investopedia Stock Market Basics](https://www.investopedia.com/terms/s/stockmarket.asp)**: An in-depth guide on the stock market.
2. **[Coursera Stock Market Course](https://www.coursera.org/learn/stock-market)**: A comprehensive stock market course from top institutions.
3. **[Udemy Stock Market Investing](https://www.udemy.com/course/stock-market-investing/)**: A practical course focused on stock market investing techniques.
4. **[Investing Academy YouTube Channel](https://www.youtube.com/c/InvestingAcademy)**: A wealth of videos that can help you understand stock market strategies and tips.

You can scan the following QR codes to access these resources quickly:

""")

st.subheader("📱 Scan the QR Codes for Quick Access")
st.image(investopedia_path, caption="Investopedia")
st.image(coursera_path, caption="Coursera")
st.image(udemy_path, caption="Udemy")
st.image(youtube_path, caption="YouTube Channel")

# LinkedIn Profile Section
st.markdown("""
---

Connect with **Ashwik Bire** on LinkedIn for more updates and personal insights into the world of stock market investing:

[LinkedIn Profile](https://www.linkedin.com/in/ashwik-bire-b2a000186)
""")

import streamlit as st

# --- Purpose and Learning Material Section ---

# Title and Purpose of the App
st.title("📈 Stock Market Insights App")
st.header("🎯 Purpose of Our App")

st.write("""
    The **Stock Market Insights App** is designed to provide **real-time stock market analysis**,
    **predict future trends**, and serve as a comprehensive **learning platform** for investors and traders at all levels.
    This app combines **advanced stock prediction models**, **technical analysis tools**, and **educational resources** to help you make smarter, more informed investment decisions.
    Whether you're a **beginner** learning about the stock market or an **advanced trader** using machine learning for stock predictions, this app is for you!
""")

# --- Key Features Section ---
st.header("🔧 Key Features of the App")

features = [
    "📈 **Real-Time Stock Analysis**: Get up-to-date market data and live stock prices.",
    "📊 **Advanced Technical Indicators**: Use tools like RSI, Moving Averages, and Bollinger Bands.",
    "🤖 **Stock Prediction Models**: Predict future stock price movements using machine learning (LSTM, XGBoost).",
    "💬 **Sentiment Analysis**: Understand the sentiment from news and social media to make informed decisions.",
    "📉 **Volume & Trend Analysis**: Detect significant volume spikes and trends.",
    "💼 **Portfolio Management**: Track and manage your investment portfolio.",
    "📚 **Stock Learning Resources**: Access tutorials, courses, and articles to enhance your knowledge.",
    "📊 **Sector Growth Analysis**: Explore high-growth sectors for better investment decisions.",
    "📱 **QR Code Learning Access**: Quick access to top stock market learning platforms like Investopedia and Coursera.",
    "⚙️ **User Customization**: Personalize your watchlist and dashboard for easier tracking."
]

# Displaying the features
for feature in features:
    st.write(feature)

# ----------------------------- FOOTER -------------------------------------
st.markdown("---")
st.markdown("🚀 Built by Ashwik Bire")
