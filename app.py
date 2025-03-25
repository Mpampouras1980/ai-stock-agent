import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import os

st.set_page_config(page_title="Stock AI Agent", layout="wide")
st.title("Stock AI Agent")

uploaded_file = st.file_uploader("Upload your stock Excel", type=["xlsx"])

min_profit_threshold = st.slider("Minimum profit % for Buy", min_value=0.0, max_value=10.0, value=1.5, step=0.5)
max_loss_threshold = st.slider("Maximum loss % for Sell", min_value=-10.0, max_value=0.0, value=-1.0, step=0.5)

def predict_next_week_price(ticker_symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    df = yf.download(ticker_symbol, start=start_date, end=end_date)

    if df.empty or len(df) < 30:
        return None

    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    df['Target'] = df['Close'].shift(-5)
    df.dropna(inplace=True)

    X = df[['Close', 'Volume', 'Return']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    latest_data = X.tail(1)
    prediction = model.predict(latest_data)[0]

    return round(float(prediction), 2)

def load_excel_data(file):
    try:
        df = pd.read_excel(file, engine='openpyxl')
        required_columns = {"stock", "ticker", "quantity"}
        if not required_columns.issubset(df.columns.str.lower()):
            return None, "Excel must contain columns: stock, ticker, quantity"
        df.columns = df.columns.str.lower()
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
        return df, None
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def save_updated_excel(df):
    output_path = "my_stocks_minimal_clean.xlsx"
    df.to_excel(output_path, index=False)
    return output_path

if uploaded_file:
    df, error = load_excel_data(uploaded_file)

    if error:
        st.error(error)
    else:
        st.success("Excel file loaded successfully.")

        suggestions = []
        total_profit = 0.0

        for idx, row in df.iterrows():
            ticker = row['ticker']
            quantity = row['quantity']
            current_price = yf.Ticker(ticker).history(period="1d")['Close']
            if current_price.empty:
                continue
            current_price = float(current_price.values[-1])
            predicted_price = predict_next_week_price(ticker)
            if predicted_price is None:
                continue
            buy_price = row.get('buy price', current_price)
            profit_pct = ((predicted_price - buy_price) / buy_price) * 100 if buy_price else 0
            profit_pct = round(float(profit_pct), 2)

            if profit_pct >= min_profit_threshold:
                action = "SELL"
                profit = round((predicted_price - buy_price) * quantity, 2)
                suggestions.append({
                    "stock": row['stock'],
                    "ticker": ticker,
                    "quantity": quantity,
                    "buy price": buy_price,
                    "predicted": predicted_price,
                    "profit %": profit_pct,
                    "action": action,
                    "estimated profit (€)": profit
                })
                total_profit += profit

        if suggestions:
            st.subheader("Suggestions:")
            suggestions_df = pd.DataFrame(suggestions)
            st.dataframe(suggestions_df)

            st.success(f"Total potential profit: €{round(total_profit, 2)}")

            if st.checkbox("✅ Execute suggestion - Save portfolio"):
                for suggestion in suggestions:
                    df.loc[df['ticker'] == suggestion['ticker'], 'quantity'] = 0
                save_updated_excel(df)
                st.success("Portfolio updated and saved successfully.")
        else:
            st.info("No buy or sell suggestions today.")

        st.subheader("No action:")
        st.dataframe(df[['stock', 'ticker', 'quantity']])