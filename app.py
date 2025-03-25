import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import joblib
import os

st.set_page_config(page_title="Stock AI Agent", layout="centered")
st.title("Stock AI Agent")

uploaded_file = st.file_uploader("Upload your stock Excel", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()
        required_columns = {'stock', 'ticker', 'quantity'}
        if not required_columns.issubset(set(df.columns)):
            st.error("Το αρχείο Excel πρέπει να περιέχει τις στήλες: Stock, Ticker, Quantity")
            st.stop()
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()

    st.success("Excel file loaded successfully.")

    # Sliders
    min_profit = st.slider("Minimum profit % for Buy", 0.0, 10.0, 1.5, 0.5)
    max_loss = st.slider("Maximum loss % for Sell", -10.0, 0.0, -1.0, 0.5)

    current_prices = {}
    predicted_prices = {}

    st.subheader("Suggestions:")

    for index, row in df.iterrows():
        ticker = row['ticker']
        quantity = row['quantity']

        try:
            data = yf.download(ticker, period="60d", progress=False)
            if data.empty or len(data) < 20:
                continue

            data['Return'] = data['Close'].pct_change()
            data = data.dropna()

            X = data[['Open', 'High', 'Low', 'Volume']]
            y = data['Close']

            model = LinearRegression()
            model.fit(X, y)

            latest = data[['Open', 'High', 'Low', 'Volume']].iloc[-1].values.reshape(1, -1)
            prediction = model.predict(latest)[0]

            current = data['Close'].iloc[-1]
            change = (prediction - current) / current * 100

            current_prices[ticker] = current
            predicted_prices[ticker] = prediction

            df.at[index, 'buy price'] = current
            df.at[index, 'predicted price'] = prediction
            df.at[index, '% change'] = round(change, 2)

            if change > min_profit:
                df.at[index, 'action'] = 'BUY'
            elif change < max_loss:
                df.at[index, 'action'] = 'SELL'
            else:
                df.at[index, 'action'] = ''

        except Exception as e:
            st.warning(f"{ticker} skipped: {e}")
            continue

    if not df['action'].dropna().empty:
        st.dataframe(df[['stock', 'ticker', 'quantity', 'buy price', 'predicted price', '% change', 'action']])

        total_profit = 0.0
        for _, row in df.iterrows():
            if row['action'] in ['BUY', 'SELL']:
                profit = (row['predicted price'] - row['buy price']) * row['quantity']
                if row['action'] == 'SELL':
                    profit *= -1
                total_profit += profit

        st.success(f"Total potential profit: €{total_profit:.2f}")
    else:
        st.info("No BUY or SELL recommendations based on current settings.")

    # Execute suggestion checkbox
    if st.checkbox("✅ Execute suggestion - Save portfolio"):
        updated_df = df.copy()
        for index, row in df.iterrows():
            if row['action'] == 'BUY':
                updated_df.at[index, 'quantity'] += 10  # mock update
            elif row['action'] == 'SELL':
                updated_df.at[index, 'quantity'] = max(0, row['quantity'] - 10)

        updated_df[['stock', 'ticker', 'quantity']].to_excel("my_stocks_minimal_clean.xlsx", index=False)
        st.success("✔️ Portfolio updated and saved.")