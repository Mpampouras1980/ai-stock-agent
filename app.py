import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import numpy as np
import os

st.set_page_config(page_title="Stock AI Agent", layout="centered")
st.title("Stock AI Agent")

st.markdown("Upload your stock Excel")
uploaded_file = st.file_uploader("Drag and drop file here", type=["xlsx"])

min_profit = st.slider("Minimum profit % for Buy", 0.0, 10.0, 1.5, step=0.5)
max_loss = st.slider("Maximum loss % for Sell", -10.0, 0.0, -1.0, step=0.5)

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.lower()
        st.success("Excel file loaded successfully.")

        if not {"stock", "ticker", "quantity"}.issubset(df.columns):
            st.error("Excel must contain columns: stock, ticker, quantity")
        else:
            predictions = []
            errors = []

            for i, row in df.iterrows():
                try:
                    ticker = row["ticker"]
                    stock = row["stock"]
                    quantity = row["quantity"]

                    hist = yf.download(ticker, period="6mo", interval="1d", progress=False)

                    if hist.empty or len(hist) < 30:
                        errors.append(f"{ticker} skipped: Not enough data.")
                        continue

                    hist = hist.reset_index()
                    hist["Date"] = pd.to_datetime(hist["Date"])
                    hist["Days"] = (hist["Date"] - hist["Date"].min()).dt.days
                    X = hist[["Days"]]
                    y = hist["Close"]

                    model = LinearRegression()
                    model.fit(X, y)

                    future_days = X["Days"].max() + 5
                    predicted_price = model.predict(np.array([[future_days]]))[0]
                    current_price = y.iloc[-1]
                    change_percent = ((predicted_price - current_price) / current_price) * 100
                    gain = (predicted_price - current_price) * quantity

                    if change_percent >= min_profit:
                        predictions.append({
                            "stock": stock,
                            "ticker": ticker,
                            "quantity": quantity,
                            "buy price": round(current_price, 2),
                            "predicted price": round(predicted_price, 2),
                            "% change": f"{round(change_percent, 2)}%",
                            "expected profit (€)": round(gain, 2),
                            "action": "SELL"
                        })

                except Exception as e:
                    errors.append(f"{ticker} skipped: {str(e)}")

            if errors:
                for err in errors:
                    st.warning(err)

            if predictions:
                predictions_df = pd.DataFrame(predictions)
                st.subheader("Suggestions:")
                st.dataframe(predictions_df)

                total_profit = predictions_df["expected profit (€)"].sum()
                st.success(f"Total potential profit: €{round(total_profit, 2)}")

                st.checkbox("✅ Execute suggestion - Save portfolio", value=False)

            else:
                st.info("No buy or sell suggestions today.")

            st.subheader("No action:")
            st.dataframe(df[["stock", "ticker", "quantity"]])

    except Exception as e:
        st.error(f"Error processing file: {e}")