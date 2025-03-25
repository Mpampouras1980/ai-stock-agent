import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import joblib

st.title("Stock AI Agent")

uploaded_file = st.file_uploader("Upload your stock Excel", type=["xlsx"])

min_profit = st.slider("Minimum profit % for Buy", 0.0, 10.0, 1.5)
max_loss = st.slider("Maximum loss % for Sell", -10.0, 0.0, -1.0)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()

        # Προσθήκη κενών στηλών
        df["buy price"] = 0.0
        df["predicted price"] = 0.0
        df["% change"] = 0.0
        df["action"] = ""

        st.success("Excel file loaded successfully.")
        total_profit = 0.0

        for index, row in df.iterrows():
            ticker = row["ticker"]
            quantity = row["quantity"]

            try:
                data = yf.download(ticker, period="6mo", interval="1d")
                if data.empty or "Close" not in data:
                    st.warning(f"{ticker} skipped: No valid historical data.")
                    continue

                data = data.dropna()
                data["Date"] = data.index.map(datetime.toordinal)
                X = data[["Date"]]
                y = data["Close"]

                model = LinearRegression()
                model.fit(X, y)

                future_date = datetime.now() + timedelta(days=5)
                prediction = model.predict([[future_date.toordinal()]])
                predicted_price = prediction.item()
                current_price = y.iloc[-1]
                change = (predicted_price - current_price) / current_price * 100
                profit = (predicted_price - current_price) * quantity

                df.at[index, "buy price"] = round(current_price, 2)
                df.at[index, "predicted price"] = round(predicted_price, 2)
                df.at[index, "% change"] = round(change, 2)

                if change > min_profit:
                    df.at[index, "action"] = "BUY"
                    total_profit += profit
                elif change < max_loss:
                    df.at[index, "action"] = "SELL"
                    total_profit += profit
                else:
                    df.at[index, "action"] = "HOLD"

            except Exception as e:
                st.warning(f"{ticker} skipped: {e}")
                continue

        buy_df = df[df["action"] == "BUY"]
        sell_df = df[df["action"] == "SELL"]
        hold_df = df[df["action"] == "HOLD"]

        if not buy_df.empty or not sell_df.empty:
            st.subheader("Suggestions:")
            st.dataframe(pd.concat([buy_df, sell_df]))
            st.success(f"Total potential profit: €{total_profit:.2f}")
        else:
            st.info("No buy or sell suggestions today.")

        st.subheader("No action:")
        st.dataframe(hold_df)

        if st.checkbox("✅ Execute suggestion - Save portfolio"):
            for index, row in df.iterrows():
                action = row["action"]
                if action == "BUY":
                    df.at[index, "quantity"] += 10
                elif action == "SELL":
                    df.at[index, "quantity"] = max(0, df.at[index, "quantity"] - 10)

            updated_df = df[["stock", "ticker", "quantity"]]
            updated_df.to_excel("my_stocks_minimal_clean.xlsx", index=False)
            st.success("Portfolio updated and saved.")

    except Exception as e:
        st.error(f"Error processing file: {e}")