import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import joblib
import datetime
import os

st.title("Stock AI Agent")

uploaded_file = st.file_uploader("Upload your stock Excel", type="xlsx")

min_profit = st.slider("Minimum profit % for Buy", min_value=0.0, max_value=10.0, value=1.5, step=0.5)
max_loss = st.slider("Maximum loss % for Sell", min_value=-10.0, max_value=0.0, value=-1.0, step=0.5)

portfolio_file = "latest_portfolio.xlsx"

def load_portfolio(file):
    try:
        return pd.read_excel(file)
    except:
        return pd.DataFrame(columns=["stock", "ticker", "quantity", "buy price"])

def save_portfolio(df):
    df.to_excel(portfolio_file, index=False)

def train_model(ticker):
    df = yf.download(ticker, period="6mo")
    if df.empty or len(df) < 10:
        return None
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df["DateOrdinal"] = df["Date"].map(datetime.datetime.toordinal)
    X = df["DateOrdinal"].values.reshape(-1, 1)
    y = df["Close"].values
    model = LinearRegression()
    model.fit(X, y)
    return model, y[-1]

def make_prediction(model, days=5):
    future_date = datetime.date.today() + datetime.timedelta(days=days)
    ordinal = future_date.toordinal()
    prediction = model.predict([[ordinal]])
    return prediction.item()

if uploaded_file:
    try:
        stocks_df = pd.read_excel(uploaded_file)
        st.success("Excel file loaded successfully.")
        actions = []
        no_action = []
        for index, row in stocks_df.iterrows():
            name = row["stock"]
            ticker = row["ticker"]
            quantity = row["quantity"]

            model_data = train_model(ticker)
            if not model_data:
                st.warning(f"{ticker} skipped: Not enough data")
                no_action.append([name, ticker, quantity, 0, 0, 0, ""])
                continue

            model, current_price = model_data
            predicted_price = make_prediction(model)
            change = float((predicted_price - current_price) / current_price * 100)

            action = ""
            if change >= min_profit:
                action = "BUY"
            elif change <= max_loss:
                action = "SELL"

            if action:
                actions.append([name, ticker, quantity, round(current_price, 2), round(predicted_price, 2), round(change, 2), action])
            else:
                no_action.append([name, ticker, quantity, round(current_price, 2), round(predicted_price, 2), round(change, 2), ""])

        if actions:
            st.subheader("Suggestions:")
            actions_df = pd.DataFrame(actions, columns=["stock", "ticker", "quantity", "buy price", "predicted", "% change", "action"])
            st.dataframe(actions_df)

            total_profit = sum([(row[4] - row[3]) * row[2] for row in actions if row[6] == "BUY"])
            st.success(f"Total potential profit: €{round(total_profit, 2)}")

            execute = st.checkbox("✅ Execute suggestion - Save portfolio")
            if execute:
                portfolio_df = load_portfolio(portfolio_file)

                for row in actions:
                    stock, ticker, qty, buy_price, predicted, change, action = row
                    if action == "BUY":
                        existing = portfolio_df[portfolio_df["ticker"] == ticker]
                        if not existing.empty:
                            idx = existing.index[0]
                            portfolio_df.at[idx, "quantity"] += qty
                        else:
                            portfolio_df = portfolio_df.append({"stock": stock, "ticker": ticker, "quantity": qty, "buy price": buy_price}, ignore_index=True)
                    elif action == "SELL":
                        existing = portfolio_df[portfolio_df["ticker"] == ticker]
                        if not existing.empty:
                            idx = existing.index[0]
                            new_qty = max(0, portfolio_df.at[idx, "quantity"] - qty)
                            portfolio_df.at[idx, "quantity"] = new_qty

                save_portfolio(portfolio_df)
                st.success("✅ Suggestion executed — Portfolio saved.")

        else:
            st.info("No buy or sell suggestions today.")

        if no_action:
            st.subheader("No action:")
            st.dataframe(pd.DataFrame(no_action, columns=["stock", "ticker", "quantity", "buy price", "predicted", "% change", "action"]))

    except Exception as e:
        st.error(f"Error processing file: {e}")