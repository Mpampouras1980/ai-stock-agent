import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import datetime
import os

st.set_page_config(page_title="Stock AI Agent", layout="wide")

st.title("Stock AI Agent")

uploaded_file = st.file_uploader("Upload your stock Excel", type=["xlsx"])

min_profit = st.slider("Minimum profit % for Buy", 0.0, 10.0, 1.5, 0.5)
max_loss = st.slider("Maximum loss % for Sell", -10.0, 0.0, -1.0, 0.5)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        df.columns = df.columns.str.lower()

        if not all(col in df.columns for col in ["stock", "ticker", "quantity"]):
            st.error("Excel file must contain the columns: stock, ticker, quantity")
        else:
            st.success("Excel file loaded successfully.")

            today = datetime.date.today()
            end_date = today
            start_date = today - datetime.timedelta(days=90)

            suggestions = []
            updated_portfolio = df.copy()

            for _, row in df.iterrows():
                ticker = row["ticker"]
                quantity = row["quantity"]

                try:
                    data = yf.download(ticker, start=start_date, end=end_date)
                    if data.empty or "Close" not in data.columns:
                        st.warning(f"{ticker} skipped: No data found.")
                        continue

                    data = data.reset_index()
                    data["Days"] = (data["Date"] - data["Date"].min()).dt.days
                    X = data[["Days"]]
                    y = data["Close"]

                    model = LinearRegression()
                    model.fit(X, y)

                    next_week = (data["Days"].max() + 5)
                    predicted_price = model.predict([[next_week]])[0]
                    current_price = y.iloc[-1]
                    buy_price = row.get("buy price", current_price)

                    change_percent = ((predicted_price - buy_price) / buy_price) * 100
                    total_profit = round((predicted_price - buy_price) * quantity, 2)

                    if change_percent > min_profit:
                        suggestions.append({
                            "stock": row["stock"],
                            "ticker": ticker,
                            "quantity": quantity,
                            "buy price": round(buy_price, 2),
                            "predicted price": round(predicted_price, 2),
                            "% change": round(change_percent, 2),
                            "total profit (€)": total_profit,
                            "action": "SELL"
                        })

                except Exception as e:
                    st.warning(f"{ticker} skipped: {e}")

            if suggestions:
                suggestions_df = pd.DataFrame(suggestions)
                st.subheader("Suggestions:")
                st.dataframe(suggestions_df)

                total = suggestions_df["total profit (€)"].sum()
                st.success(f"Total potential profit: €{round(total, 2)}")

                if st.checkbox("✅ Execute suggestion - Save portfolio"):
                    for suggestion in suggestions:
                        idx = updated_portfolio[updated_portfolio["ticker"] == suggestion["ticker"]].index
                        if not idx.empty:
                            updated_portfolio.loc[idx, "quantity"] = 0
                            updated_portfolio.loc[idx, "buy price"] = 0.0

                    updated_file_path = "my_stocks_minimal_clean_fixed_result.xlsx"
                    updated_portfolio.to_excel(updated_file_path, index=False)
                    st.success("Portfolio updated and saved successfully.")
                    with open(updated_file_path, "rb") as f:
                        st.download_button("Download Updated Portfolio", f, file_name=updated_file_path)
            else:
                st.info("No buy or sell suggestions today.")
                st.subheader("No action:")
                st.dataframe(updated_portfolio[["stock", "ticker", "quantity"]])

    except Exception as e:
        st.error(f"Error processing file: {e}")