import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="Stock AI Agent", layout="centered")
st.title("Stock AI Agent")

# === ΑΝΕΒΑΣΜΑ EXCEL ===
uploaded_file = st.file_uploader("Upload your stock Excel", type=["xlsx"])
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        df.columns = df.columns.str.lower()
        if not {'stock', 'ticker', 'quantity'}.issubset(df.columns):
            st.error("Το αρχείο Excel πρέπει να περιέχει τις στήλες: stock, ticker, quantity")
        else:
            st.success("Excel file loaded successfully.")
            df['quantity'] = df['quantity'].fillna(0).astype(int)
            df['buy price'] = 0.0
            df['predicted price'] = 0.0
            df['% change'] = 0.0
            df['action'] = ""
            df['estimated profit'] = 0.0

            min_profit = st.slider("Minimum profit % for Buy", 0.0, 10.0, 1.5, step=0.5)
            max_loss = st.slider("Maximum loss % for Sell", -10.0, 0.0, -1.0, step=0.5)

            suggestions = []
            total_profit = 0.0

            for i, row in df.iterrows():
                ticker = row['ticker']
                quantity = row['quantity']
                try:
                    data = yf.download(ticker, period="6mo", interval="1d")
                    if data.empty:
                        raise ValueError("No data")

                    data['Date'] = data.index
                    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
                    model = LinearRegression()
                    model.fit(data[['Days']], data['Close'])

                    today_day = data['Days'].max()
                    next_week_day = today_day + 5

                    predicted_today = model.predict([[today_day]])[0]
                    predicted_week = model.predict([[next_week_day]])[0]

                    buy_price = data['Close'].iloc[-1]
                    profit_pct = ((predicted_today - buy_price) / buy_price) * 100

                    if profit_pct > 0:  # μόνο θετικές προτάσεις
                        df.at[i, 'buy price'] = round(buy_price, 2)
                        df.at[i, 'predicted price'] = round(predicted_today, 2)
                        df.at[i, '% change'] = round(profit_pct, 2)
                        df.at[i, 'action'] = "SELL"
                        df.at[i, 'estimated profit'] = round((predicted_today - buy_price) * quantity, 2)
                        total_profit += df.at[i, 'estimated profit']
                        suggestions.append(df.iloc[i])
                except Exception as e:
                    st.warning(f"{ticker} skipped: {str(e)}")

            if suggestions:
                st.subheader("Suggestions:")
                st.dataframe(pd.DataFrame(suggestions)[['stock', 'ticker', 'quantity', 'buy price', 'predicted price', '% change', 'estimated profit']])
                st.success(f"Total potential profit: €{round(total_profit,2)}")

                execute = st.checkbox("✅ Execute suggestion - Save portfolio")
                if execute:
                    for i, row in pd.DataFrame(suggestions).iterrows():
                        idx = df[df['ticker'] == row['ticker']].index
                        df.loc[idx, 'quantity'] = 0
                    df.drop(columns=['buy price', 'predicted price', '% change', 'action', 'estimated profit'], inplace=True)
                    df.to_excel("my_stocks_minimal_clean.xlsx", index=False)
                    st.success("Portfolio updated and saved.")
            else:
                st.info("No buy or sell suggestions today.")
                df.drop(columns=['buy price', 'predicted price', '% change', 'action', 'estimated profit'], inplace=True)

            st.subheader("No action:")
            st.dataframe(df[['stock', 'ticker', 'quantity']])
    except Exception as e:
        st.error(f"Error processing file: {e}")