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
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

# News API Key
NEWS_API_KEY = "0b08be107dca45d3be30ca7e06544408"

# Set page config
st.set_page_config(page_title="MarketMentor", layout="wide")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "MarketMentor",
        ["Home", "Company Overview", "Market Movers", "F&O", "Global Markets", "Mutual Funds", "SIP Calculator", "IPO Tracker", "Predictions for Mutual Funds & IPOs", "Mutual Fund NAV Viewer", "Sectors", "News", "Learning", "Volume Spike", "Stock Screener", "Predictions", "Buy/Sell Predictor", "News Sentiment", "Portfolio Tracker"],
        icons=['house', 'graph-up', 'globe', 'bank', 'boxes', 'newspaper', 'building', 'book', 'activity', 'search', 'wallet'],
        menu_icon="cast",
        default_index=0
    )

# Home - Market Overview
if selected == "Home":
    st.title("🏠 Home - Market Overview")
    
    # Add a market overview section
    st.subheader("📊 Market Snapshot")
    
    # Create columns for different market data
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🇮🇳 Indian Indices")
    with col2:
        st.info("🌍 Global Indices")
    with col3:
        st.info("📰 Market News")
    
    # Indian indices with rupee values
    indian_indices = {
        "^NSEI": "Nifty 50",
        "^BSESN": "Sensex",
        "NIFTY_BANK.NS": "Bank Nifty",
    }
    
    # Global indices with dollar values
    global_indices = {
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^GSPC": "S&P 500",
    }
    
    st.subheader("🇮🇳 Indian Indices Performance")
    cols = st.columns(len(indian_indices))
    for idx, (symbol, name) in enumerate(indian_indices.items()):
        data = yf.Ticker(symbol).history(period="1d")
        if not data.empty:
            last_close = round(data['Close'].iloc[-1], 2)
            prev_close = round(data['Close'].iloc[-2], 2) if len(data) > 1 else last_close
            change = round(last_close - prev_close, 2)
            percent_change = round((change / prev_close) * 100, 2)
            
            # Format with rupee symbol and commas
            formatted_value = f"₹{last_close:,.2f}"
            delta_color = "normal"
            if change > 0:
                delta_color = "normal"
                change_symbol = "+"
            else:
                delta_color = "inverse"
                change_symbol = ""
                
            cols[idx].metric(label=name, value=formatted_value, 
                            delta=f"{change_symbol}{change:,.2f} ({change_symbol}{percent_change}%)",
                            delta_color=delta_color)
    
    st.subheader("🌍 Global Indices Performance")
    cols = st.columns(len(global_indices))
    for idx, (symbol, name) in enumerate(global_indices.items()):
        data = yf.Ticker(symbol).history(period="1d")
        if not data.empty:
            last_close = round(data['Close'].iloc[-1], 2)
            prev_close = round(data['Close'].iloc[-2], 2) if len(data) > 1 else last_close
            change = round(last_close - prev_close, 2)
            percent_change = round((change / prev_close) * 100, 2)
            
            # Format with dollar symbol and commas
            formatted_value = f"${last_close:,.2f}"
            delta_color = "normal"
            if change > 0:
                delta_color = "normal"
                change_symbol = "+"
            else:
                delta_color = "inverse"
                change_symbol = ""
                
            cols[idx].metric(label=name, value=formatted_value, 
                            delta=f"{change_symbol}{change:,.2f} ({change_symbol}{percent_change}%)",
                            delta_color=delta_color)
    
    # Add a quick news section
    st.subheader("📰 Latest Market News")
    try:
        url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey={NEWS_API_KEY}&language=en&pageSize=3"
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
    except:
        st.info("News feature temporarily unavailable.")

# ... (rest of your code remains the same until the end)

# Portfolio Tracker - New Feature
elif selected == "Portfolio Tracker":
    st.title("💰 Portfolio Tracker")
    
    # Initialize session state for portfolio
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame(columns=['Stock', 'Quantity', 'Buy Price', 'Current Price', 'Change', 'Value'])
    
    # Add stocks to portfolio
    st.subheader("Add Stock to Portfolio")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stock_symbol = st.text_input("Stock Symbol (e.g., RELIANCE.NS)", "")
    with col2:
        quantity = st.number_input("Quantity", min_value=1, value=1)
    with col3:
        buy_price = st.number_input("Buy Price (₹)", min_value=0.0, value=0.0)
    
    if st.button("Add to Portfolio") and stock_symbol and quantity and buy_price:
        # Get current price
        try:
            stock_data = yf.Ticker(stock_symbol).history(period="1d")
            current_price = stock_data['Close'].iloc[-1] if not stock_data.empty else buy_price
            
            # Add to portfolio
            new_row = {
                'Stock': stock_symbol,
                'Quantity': quantity,
                'Buy Price': buy_price,
                'Current Price': current_price,
                'Change': (current_price - buy_price) / buy_price * 100,
                'Value': quantity * current_price
            }
            
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, pd.DataFrame([new_row])], ignore_index=True)
            st.success(f"Added {stock_symbol} to portfolio!")
        except:
            st.error("Error fetching stock data. Please check the symbol.")
    
    # Display portfolio
    st.subheader("Your Portfolio")
    if not st.session_state.portfolio.empty:
        # Update current prices
        for idx, row in st.session_state.portfolio.iterrows():
            try:
                stock_data = yf.Ticker(row['Stock']).history(period="1d")
                if not stock_data.empty:
                    current_price = stock_data['Close'].iloc[-1]
                    st.session_state.portfolio.at[idx, 'Current Price'] = current_price
                    st.session_state.portfolio.at[idx, 'Change'] = (current_price - row['Buy Price']) / row['Buy Price'] * 100
                    st.session_state.portfolio.at[idx, 'Value'] = row['Quantity'] * current_price
            except:
                pass
        
        # Display portfolio with formatting
        display_df = st.session_state.portfolio.copy()
        display_df['Buy Price'] = display_df['Buy Price'].apply(lambda x: f"₹{x:,.2f}")
        display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"₹{x:,.2f}")
        display_df['Change'] = display_df['Change'].apply(lambda x: f"{x:+.2f}%")
        display_df['Value'] = display_df['Value'].apply(lambda x: f"₹{x:,.2f}")
        
        st.dataframe(display_df)
        
        # Portfolio summary
        total_investment = (st.session_state.portfolio['Quantity'] * st.session_state.portfolio['Buy Price']).sum()
        current_value = st.session_state.portfolio['Value'].sum()
        total_change = (current_value - total_investment) / total_investment * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Investment", f"₹{total_investment:,.2f}")
        col2.metric("Current Value", f"₹{current_value:,.2f}")
        col3.metric("Total Return", f"₹{current_value - total_investment:,.2f}", 
                   f"{total_change:+.2f}%")
        
        # Portfolio pie chart
        st.subheader("Portfolio Allocation")
        fig, ax = plt.subplots()
        ax.pie(st.session_state.portfolio['Value'], 
              labels=st.session_state.portfolio['Stock'], 
              autopct='%1.1f%%')
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.info("Your portfolio is empty. Add some stocks to get started!")
