import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import os

st.title("Stock AI Agent")

uploaded_file = st.file_uploader("Upload your stock Excel", type=["xlsx"])
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        df.columns = df.columns.str.strip().str.lower()  # καθαρισμός ονομάτων στηλών

        if not all(col in df.columns for col in ["stock", "ticker", "quantity"]):
            st.error("Excel must include: stock, ticker, quantity")
        else:
            st.success("Excel file loaded successfully.")

            min_profit = st.slider("Minimum profit % for Buy", 0.0, 10.0, 1.5, step=0.1)
            max_loss = st.slider("Maximum loss % for Sell", -10.0, 0.0, -1.0, step=0.1)

            suggestions = pd.DataFrame(columns=["stock", "ticker", "quantity", "buy price", "predicted price", "% change", "action"])

            for index, row in df.iterrows():
                ticker = row["ticker"]
                stock_name = row["stock"]
                quantity = row["quantity"]

                try:
                    data = yf.download(ticker, period="6mo", interval="1d", progress=False)
                    if data.empty:
                        st.warning(f"{ticker} skipped: No data available")
                        continue

                    data.dropna(inplace=True)
                    data["Date"] = data.index
                    data["Days"] = (data["Date"] - data["Date"].min()).dt.days
                    X = data[["Days"]]
                    y = data["Close"]

                    model = LinearRegression()
                    model.fit(X, y)

                    next_day = X["Days"].max() + 1
                    predicted_price = round(float(model.predict([[next_day]])[0]), 2)
                    current_price = round(y.iloc[-1], 2)
                    change_percent = ((predicted_price - current_price) / current_price) * 100

                    if change_percent >= min_profit or change_percent <= max_loss:
                        action = "BUY" if change_percent >= min_profit else "SELL"
                        suggestions = suggestions.append({
                            "stock": stock_name,
                            "ticker": ticker,
                            "quantity": quantity,
                            "buy price": current_price,
                            "predicted price": predicted_price,
                            "% change": round(change_percent, 2),
                            "action": action
                        }, ignore_index=True)

                except Exception as e:
                    st.warning(f"{ticker} skipped: {e}")

            if not suggestions.empty:
                st.subheader("Suggestions:")
                st.dataframe(suggestions)

                total_profit = 0
                for _, row in suggestions.iterrows():
                    if row["action"] == "SELL":
                        profit = (row["predicted price"] - row["buy price"]) * row["quantity"]
                        total_profit += profit
                st.success(f"Total potential profit: €{round(total_profit, 2)}")

                execute = st.checkbox("✅ Execute suggestion - Save portfolio")
                if execute:
                    for _, row in suggestions.iterrows():
                        idx = df[df["ticker"] == row["ticker"]].index
                        if row["action"] == "SELL":
                            df.loc[idx, "quantity"] = max(0, df.loc[idx, "quantity"].values[0] - row["quantity"])
                        elif row["action"] == "BUY":
                            df.loc[idx, "quantity"] = df.loc[idx, "quantity"].values[0] + row["quantity"]

                    df.to_excel("my_stocks_minimal_clean.xlsx", index=False)
                    st.success("Suggestion executed. Portfolio updated.")

            else:
                st.info("No buy or sell suggestions today.")

            st.subheader("No action:")
            st.dataframe(df)

    except Exception as e:
        st.error(f"Error processing file: {e}")