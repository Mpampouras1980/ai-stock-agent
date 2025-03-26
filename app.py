import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import os
import smtplib
from email.mime.text import MIMEText
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

st.set_page_config(page_title="Stock AI Agent", layout="wide")

# --- Feature engineering functions ---
def add_technical_indicators(df):
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df = df.dropna()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Email alert function ---
def send_email(subject, body):
    sender_email = "gprovopo@googlemail.com"
    receiver_email = "gprovopo@googlemail.com"
    app_password = "knuehkeowscowyhh"
    
    message = MIMEText(body)
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = receiver_email
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.send_message(message)
    except Exception as e:
        st.error(f"Email failed: {e}")

# --- Forecast function ---
def forecast_stock_price(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    if df.empty or len(df) < 60:
        raise ValueError("Not enough data.")
    
    df = add_technical_indicators(df)
    df = df.dropna()
    df['Target'] = df['Close'].shift(-5)  # Forecast horizon: 5 days
    
    df = df.dropna()
    X = df[['Close', 'MA7', 'MA21', 'RSI', 'Momentum']]
    y = df['Target']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = mean_squared_error(y, y_pred, squared=False)

    if rmse / y.mean() > 0.05:
        raise ValueError("Low predictability (RMSE too high)")

    last_row = df.iloc[-1][['Close', 'MA7', 'MA21', 'RSI', 'Momentum']]
    forecast = model.predict([last_row.values.ravel()])[0]
    return forecast

# --- GUI ---
st.title("Stock AI Agent")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
cashout_target = st.slider("Cash-out target value (€)", 1000, 20000, 10000)

min_profit_buy = st.slider("Minimum profit % for Buy", 0.0, 10.0, 1.5)
# max_loss_sell = st.slider("Maximum loss % for Sell", -10.0, 0.0, -1.0)  # Optional

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.lower()

        st.success("Excel file loaded.")

        tickers = df["ticker"].tolist()
        quantities = df["quantity"].tolist()
        total_current_value = 0
        sell_suggestions = []
        buy_suggestions = []
        alert_msg = ""
        
        for i, ticker in enumerate(tickers):
            try:
                forecast = forecast_stock_price(ticker)
                current_price = yf.download(ticker, period="1d")["Close"].iloc[-1]
                quantity = quantities[i]
                total_current_value += current_price * quantity

                if forecast > current_price * (1 + min_profit_buy / 100):
                    buy_suggestions.append({
                        "ticker": ticker,
                        "current": round(current_price, 2),
                        "forecast": round(forecast, 2),
                        "gain %": round((forecast - current_price) / current_price * 100, 2)
                    })

                elif current_price * quantity + 10 > cashout_target:
                    sell_suggestions.append({
                        "ticker": ticker,
                        "current": round(current_price, 2),
                        "forecast": round(forecast, 2),
                        "value": round(current_price * quantity, 2)
                    })

            except Exception as e:
                st.warning(f"{ticker} skipped: {e}")
        
        st.markdown(f"**Current portfolio value:** €{round(total_current_value,2)}")

        if total_current_value > cashout_target:
            alert_msg = f"Cash-out opportunity: Your portfolio value is €{round(total_current_value,2)} exceeding the target €{cashout_target}. Consider selling."

        if alert_msg:
            st.success(alert_msg)
            send_email("Cash-out Alert", alert_msg)

        if buy_suggestions:
            st.subheader("Buy Suggestions")
            st.dataframe(pd.DataFrame(buy_suggestions))

        if sell_suggestions:
            st.subheader("Sell Suggestions (Cash-out)")
            st.dataframe(pd.DataFrame(sell_suggestions))

        if not buy_suggestions and not sell_suggestions:
            st.info("No suggestions today.")

        st.subheader("No action:")
        st.dataframe(df)

    except Exception as e:
        st.error(f"Error processing file: {e}")