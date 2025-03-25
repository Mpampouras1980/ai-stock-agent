import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import os

st.set_page_config(page_title="Stock AI Agent", layout="wide")
st.title("Stock AI Agent")

uploaded_file = st.file_uploader("Upload your stock Excel", type=["xlsx"])

min_profit_threshold = st.slider("Minimum profit % for Buy", min_value=0.0, max_value=10.0, value=1.5, step=0.5)

def compute_indicators(df):
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MOM10"] = df["Close"] - df["Close"].shift(10)
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI14"] = 100 - (100 / (1 + rs))
    return df

def predict_next_week_price(ticker):
    df = yf.download(ticker, period="100d", interval="1d", progress=False)

    if df.empty or len(df) < 60:
        return None

    df = compute_indicators(df)
    df["Target"] = df["Close"].shift(-5)
    df.dropna(inplace=True)

    features = ["Close", "Volume", "MA10", "MA20", "RSI14", "MOM10"]
    X = df[features]
    y = df["Target"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        rmses.append(rmse)

    mean_rmse = np.mean(rmses)
    avg_price = y.mean()

    if mean_rmse > 0.05 * avg_price:
        return None

    model.fit(X, y)
    latest = X.iloc[-1].values.reshape(1, -1)
    prediction = model.predict(latest)[0]

    return round(prediction, 2)

def load_excel(file):
    try:
        df = pd.read_excel(file, engine='openpyxl')
        df.columns = df.columns.str.lower()
        if not {'stock', 'ticker', 'quantity'}.issubset(df.columns):
            return None, "Excel must contain columns: stock, ticker, quantity"
        df['quantity'] = df['quantity'].fillna(0).astype(int)
        return df, None
    except Exception as e:
        return None, f"Error: {e}"

def save_excel(df):
    path = "my_stocks_minimal_clean.xlsx"
    df.to_excel(path, index=False)
    return path

if uploaded_file:
    df, err = load_excel(uploaded_file)

    if err:
        st.error(err)
    else:
        st.success("Excel file loaded.")
        suggestions = []
        total_profit = 0.0

        for _, row in df.iterrows():
            ticker = row["ticker"]
            quantity = row["quantity"]
            try:
                current_price = float(yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1])
                predicted_price = predict_next_week_price(ticker)
                if predicted_price is None:
                    continue
                buy_price = row.get("buy price", current_price)
                profit_pct = ((predicted_price - buy_price) / buy_price) * 100
                if profit_pct >= min_profit_threshold:
                    profit = round((predicted_price - buy_price) * quantity, 2)
                    suggestions.append({
                        "stock": row["stock"],
                        "ticker": ticker,
                        "quantity": quantity,
                        "buy price": round(buy_price, 2),
                        "predicted price": predicted_price,
                        "profit %": round(profit_pct, 2),
                        "estimated profit (€)": profit,
                        "action": "SELL"
                    })
                    total_profit += profit
            except:
                continue

        if suggestions:
            st.subheader("Suggestions:")
            suggestions_df = pd.DataFrame(suggestions)
            st.dataframe(suggestions_df)
            st.success(f"Total potential profit: €{round(total_profit,2)}")

            if st.checkbox("✅ Execute suggestion - Save portfolio"):
                for s in suggestions:
                    df.loc[df["ticker"] == s["ticker"], "quantity"] = 0
                save_excel(df)
                st.success("Portfolio updated and saved.")
        else:
            st.info("No suggestions today.")

        st.subheader("No action:")
        st.dataframe(df[["stock", "ticker", "quantity"]])