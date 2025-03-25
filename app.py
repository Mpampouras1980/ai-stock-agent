import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Stock AI Agent")

st.title("Stock AI Agent")
st.subheader("Upload your stock Excel")

uploaded_file = st.file_uploader("Drag and drop file here", type=["xlsx"])

MIN_HISTORY_DAYS = 60

def fetch_stock_history(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=MIN_HISTORY_DAYS)
    df = yf.download(ticker, start=start_date, end=end_date)
    return df["Close"] if not df.empty else None

def predict_price(prices):
    if prices is None or len(prices) < 2:
        return None
    model = LinearRegression()
    X = np.arange(len(prices)).reshape(-1, 1)
    y = np.array(prices)
    model.fit(X, y)
    return round(float(model.predict([[len(prices)]])), 2)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        required_columns = {"Stock", "Ticker", "Quantity"}
        if not required_columns.issubset(df.columns):
            st.error("Excel must contain the following columns: Stock, Ticker, Quantity")
        else:
            min_profit = st.slider("Minimum profit % for Buy", 0.0, 10.0, 1.5, 0.5)
            max_loss = st.slider("Maximum loss % for Sell", -10.0, 0.0, -1.0, 0.5)

            suggestions = []
            total_profit = 0.0

            for _, row in df.iterrows():
                ticker = row["Ticker"]
                quantity = row["Quantity"]
                prices = fetch_stock_history(ticker)
                predicted = predict_price(prices)

                if predicted is None:
                    continue

                current_price = round(prices[-1], 2)
                change_percent = ((predicted - current_price) / current_price) * 100

                action = "HOLD"
                if change_percent >= min_profit:
                    action = "BUY"
                    total_profit += (predicted - current_price) * quantity
                elif change_percent <= max_loss:
                    action = "SELL"
                    total_profit += (predicted - current_price) * quantity

                suggestions.append({
                    "Stock": row["Stock"],
                    "Ticker": ticker,
                    "Quantity": quantity,
                    "Buy Price": current_price,
                    "Predicted Price": predicted,
                    "% Change": f"{change_percent:.2f}%",
                    "Action": action
                })

            suggestions_df = pd.DataFrame(suggestions)

            if not suggestions_df.empty:
                st.subheader("Suggestions:")
                st.dataframe(suggestions_df, use_container_width=True)

                st.success(f"Total potential profit: €{total_profit:.2f}")

                if st.checkbox("✅ Execute suggestion - Save portfolio"):
                    for index, row in suggestions_df.iterrows():
                        action = row["Action"]
                        if action == "BUY":
                            df.loc[df["Ticker"] == row["Ticker"], "Quantity"] += row["Quantity"]
                        elif action == "SELL":
                            df.loc[df["Ticker"] == row["Ticker"], "Quantity"] = max(
                                df.loc[df["Ticker"] == row["Ticker"], "Quantity"].values[0] - row["Quantity"], 0)

                    df.to_excel("my_stocks_minimal_clean.xlsx", index=False)
                    st.success("✔️ Suggestion executed - Portfolio updated and saved!")

            else:
                st.info("No action required based on your criteria.")

    except Exception as e:
        st.error(f"Error processing file: {e}")