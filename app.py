import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

# ---------------------- CONFIGURATION ----------------------

DEFAULT_CASHOUT_TARGET = 10000
EMAIL_RECEIVER = "gprovopo@googlemail.com"
EMAIL_SENDER = "gprovopo@googlemail.com"
EMAIL_APP_PASSWORD = "knuehkeowscowyhh"  # Application-specific password

# ---------------------- UTILITY FUNCTIONS ----------------------

def send_notification(subject, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_APP_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, text)
        server.quit()
    except Exception as e:
        st.warning(f"Notification failed: {e}")

def fetch_stock_data(ticker):
    end = datetime.now()
    start = end - timedelta(days=180)
    data = yf.download(ticker, start=start, end=end)
    return data

def calculate_technical_indicators(df):
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df.dropna(inplace=True)
    return df

def train_model(df):
    df = calculate_technical_indicators(df)
    X = df[['Close', 'MA_5', 'MA_10', 'RSI', 'Momentum']]
    y = df['Close'].shift(-5).dropna()
    X = X.iloc[:-5]
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y)
    return model

def predict_price(df, model):
    df = calculate_technical_indicators(df)
    X_latest = df[['Close', 'MA_5', 'MA_10', 'RSI', 'Momentum']].tail(1)
    return model.predict(X_latest)[0]

# ---------------------- STREAMLIT GUI ----------------------

st.title("Stock AI Agent")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

cashout_target = st.slider("Cash-out target value (€)", min_value=1000, max_value=20000, value=DEFAULT_CASHOUT_TARGET)
min_profit_percent = st.slider("Minimum profit % for Buy", 0.0, 10.0, 1.5)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("Excel file loaded.")

        if not all(col in df.columns for col in ['stock', 'ticker', 'quantity']):
            st.error("Excel must contain columns: stock, ticker, quantity")
        else:
            df['buy_price'] = 0.0
            df['predicted_price'] = 0.0
            df['future_price'] = 0.0
            df['current_price'] = 0.0
            df['action'] = ''
            df['profit'] = 0.0

            total_value = 0.0

            suggestions = []

            for i, row in df.iterrows():
                ticker = row['ticker']
                quantity = row['quantity']
                try:
                    data = fetch_stock_data(ticker)
                    if data is not None and not data.empty:
                        model = train_model(data)
                        pred_price = predict_price(data, model)
                        future_price = pred_price
                        current_price = data['Close'][-1]
                        buy_price = current_price  # live assumption

                        df.at[i, 'buy_price'] = buy_price
                        df.at[i, 'predicted_price'] = pred_price
                        df.at[i, 'future_price'] = future_price
                        df.at[i, 'current_price'] = current_price

                        profit = (pred_price - buy_price) * quantity
                        df.at[i, 'profit'] = profit
                        total_value += current_price * quantity

                        profit_percent = ((pred_price - buy_price) / buy_price) * 100
                        if profit_percent >= min_profit_percent:
                            df.at[i, 'action'] = 'BUY'
                            suggestions.append(row)

                except Exception as e:
                    st.warning(f"{ticker} skipped: {e}")

            st.markdown(f"**Current portfolio value:** €{total_value:.2f}")

            if total_value > cashout_target:
                excess = total_value - cashout_target
                subject = f"[Stock AI Agent] Cash-out Suggestion: Exceeded €{cashout_target}"
                body = f"Your current portfolio value is €{total_value:.2f}, which exceeds your target of €{cashout_target}.\nSuggested cash-out amount: €{excess:.2f}"
                send_notification(subject, body)
                st.info(f"Cash-out recommendation: Sell €{excess:.2f} worth of stocks to match your cash-out target.")

            if suggestions:
                st.subheader("Suggestions:")
                st.dataframe(df[df['action'] == 'BUY'][['stock', 'ticker', 'quantity', 'buy_price', 'predicted_price', 'profit']])
                st.success(f"Total potential profit: €{df['profit'].sum():.2f}")
            else:
                st.info("No suggestions today.")

            st.subheader("No action:")
            st.dataframe(df[df['action'] == ''][['stock', 'ticker', 'quantity']])
    except Exception as e:
        st.error(f"Error: {e}")