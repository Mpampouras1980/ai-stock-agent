import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import joblib
import os

st.set_page_config(page_title="Stock AI Agent", layout="centered")

st.title("Stock AI Agent")

uploaded_file = st.file_uploader("Upload your stock Excel", type="xlsx")

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        required_columns = {"Stock", "Ticker", "Quantity"}
        if not required_columns.issubset(set(df.columns)):
            st.error("Excel file must contain the columns: Stock, Ticker, Quantity")
        else:
            # Fetch latest data
            df["Buy Price"] = df["Ticker"].apply(lambda t: yf.download(t, period="1d")["Close"][-1] if not yf.download(t, period="1d").empty else 0)

            # Predict next price using dynamic training
            def predict_price(ticker):
                data = yf.download(ticker, period="1y")
                if data.empty or len(data) < 10:
                    return 0
                data["Days"] = range(len(data))
                X = data["Days"].values.reshape(-1, 1)
                y = data["Close"].values
                model = LinearRegression().fit(X, y)
                next_day = [[len(data)]]
                return round(model.predict(next_day)[0], 2)

            df["Predicted Price"] = df["Ticker"].apply(predict_price)
            df["% Change"] = 100 * (df["Predicted Price"] - df["Buy Price"]) / df["Buy Price"]

            min_profit = st.slider("Minimum profit % for Buy", 0.0, 10.0, 2.0)
            max_loss = st.slider("Maximum loss % for Sell", -10.0, 0.0, -1.0)

            def get_action(change):
                if change >= min_profit:
                    return "BUY"
                elif change <= max_loss:
                    return "SELL"
                else:
                    return "HOLD"

            df["Action"] = df["% Change"].apply(get_action)

            buy_df = df[df["Action"] == "BUY"]
            sell_df = df[df["Action"] == "SELL"]
            hold_df = df[df["Action"] == "HOLD"]

            st.subheader("Suggestions:")
            st.dataframe(pd.concat([buy_df, sell_df], ignore_index=True)[["Stock", "Ticker", "Quantity", "Buy Price", "Predicted Price", "% Change", "Action"]])

            # Show HOLDs separately
            st.subheader("No Action Required:")
            st.dataframe(hold_df[["Stock", "Ticker", "Quantity", "Buy Price"]])

            # Show total expected profit
            df["Profit €"] = (df["Predicted Price"] - df["Buy Price"]) * df["Quantity"]
            total_profit = df.loc[df["Action"] == "BUY", "Profit €"].sum()
            st.success(f"Total potential profit: €{total_profit:.2f}")

            # Save updates
            if st.checkbox("✅ Execute suggestion - Save portfolio"):
                new_df = df.copy()
                new_df.loc[new_df["Action"] == "BUY", "Buy Price"] = new_df["Predicted Price"]
                new_df = new_df[["Stock", "Ticker", "Quantity", "Buy Price"]]
                new_df.to_excel("my_stocks_minimal_clean.xlsx", index=False)
                st.success("Portfolio updated and saved.")

    except Exception as e:
        st.error(f"Error processing file: {e}")