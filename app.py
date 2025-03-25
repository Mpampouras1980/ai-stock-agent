import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import os

st.set_page_config(page_title="Stock AI Agent", layout="centered")
st.title("Stock AI Agent")

uploaded_file = st.file_uploader("Upload your stock Excel", type=["xlsx"])

min_profit = st.slider("Minimum profit % for Buy", 0.0, 10.0, 1.5, 0.5)
max_loss = st.slider("Maximum loss % for Sell", -10.0, 0.0, -1.0, 0.5)

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.success("Excel file loaded successfully.")

        required_columns = ['Stock', 'Ticker', 'Quantity']
        if not all(col in df.columns for col in required_columns):
            st.error("The Excel must have columns: Stock, Ticker, Quantity")
        else:
            df.columns = df.columns.str.lower()
            df.rename(columns={'stock': 'stock', 'ticker': 'ticker', 'quantity': 'quantity'}, inplace=True)

            df['buy price'] = 0.0
            df['predicted price'] = 0.0
            df['% change'] = 0.0
            df['action'] = ""

            total_profit = 0.0

            for index in df.index:
                ticker = df.at[index, 'ticker']
                quantity = df.at[index, 'quantity']

                try:
                    data = yf.download(ticker, period="60d", progress=False)
                    if data.empty or len(data) < 20:
                        continue

                    data['Return'] = data['Close'].pct_change()
                    data.dropna(inplace=True)

                    X = data[['Open', 'High', 'Low', 'Volume']]
                    y = data['Close']

                    model = LinearRegression()
                    model.fit(X, y)

                    latest = data[['Open', 'High', 'Low', 'Volume']].iloc[-1].values.reshape(1, -1)
                    prediction = model.predict(latest)[0]

                    current = data['Close'].iloc[-1]
                    change = (prediction - current) / current * 100
                    profit = (prediction - current) * quantity if change > min_profit or change < max_loss else 0

                    df.at[index, 'buy price'] = round(current, 2)
                    df.at[index, 'predicted price'] = round(prediction, 2)
                    df.at[index, '% change'] = round(change, 2)

                    if change > min_profit:
                        df.at[index, 'action'] = 'BUY'
                        total_profit += profit
                    elif change < max_loss:
                        df.at[index, 'action'] = 'SELL'
                        total_profit += profit

                except Exception as e:
                    st.warning(f"{ticker} skipped: {e}")
                    continue

            suggestions = df[df['action'] != ""].copy()
            no_action = df[df['action'] == ""].copy()

            if not suggestions.empty:
                st.subheader("Suggestions:")
                st.dataframe(suggestions)
            else:
                st.info("No buy or sell suggestions today.")

            if not no_action.empty:
                st.subheader("No action:")
                st.dataframe(no_action[['stock', 'ticker', 'quantity', 'buy price']])

            st.markdown(f"### Total potential profit: **€{total_profit:.2f}**")

            execute = st.checkbox("✅ Execute suggestion - Save portfolio")

            if execute:
                for index in suggestions.index:
                    action = suggestions.at[index, 'action']
                    qty = df.at[index, 'quantity']
                    if action == 'BUY':
                        df.at[index, 'quantity'] = qty + 10
                    elif action == 'SELL':
                        df.at[index, 'quantity'] = max(0, qty - 10)

                df[['stock', 'ticker', 'quantity']].to_excel("my_stocks_minimal_clean.xlsx", index=False)
                st.success("✔️ Portfolio updated and saved!")

    except Exception as e:
        st.error(f"Error processing file: {e}")