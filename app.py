import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import joblib
from datetime import datetime, timedelta
import os

st.title("Stock AI Agent")

uploaded_file = st.file_uploader("Upload your stock Excel", type=["xlsx"])
min_profit = st.slider("Minimum profit % for Buy", 0.0, 10.0, 1.5)
max_loss = st.slider("Maximum loss % for Sell", -10.0, 0.0, -1.0)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("Excel file loaded successfully.")

        suggestions = []
        errors = []
        for index, row in df.iterrows():
            try:
                name = row['stock']
                ticker = row['ticker']
                quantity = row['quantity']

                stock = yf.Ticker(ticker)
                hist = stock.history(period="6mo")

                if hist.empty or len(hist) < 10:
                    errors.append(f"{ticker} skipped: Not enough data.")
                    continue

                hist = hist.reset_index()
                hist['Date'] = pd.to_datetime(hist['Date'])
                hist['Day'] = (hist['Date'] - hist['Date'].min()).dt.days
                X = hist[['Day']]
                y = hist['Close']

                model = LinearRegression()
                model.fit(X, y)

                next_day = X['Day'].max() + 5
                predicted_price = round(float(model.predict([[next_day]])[0]), 2)
                current_price = round(float(y.iloc[-1]), 2)
                change_percent = float((predicted_price - current_price) / current_price * 100)

                if change_percent >= min_profit:
                    suggestions.append({
                        'stock': name,
                        'ticker': ticker,
                        'quantity': quantity,
                        'buy price': current_price,
                        'predicted price': predicted_price,
                        '% change': round(change_percent, 2),
                        'action': 'BUY'
                    })
                elif change_percent <= max_loss:
                    suggestions.append({
                        'stock': name,
                        'ticker': ticker,
                        'quantity': quantity,
                        'buy price': current_price,
                        'predicted price': predicted_price,
                        '% change': round(change_percent, 2),
                        'action': 'SELL'
                    })
            except Exception as e:
                errors.append(f"{ticker} skipped: {str(e)}")

        if errors:
            for err in errors:
                st.warning(err)

        if suggestions:
            suggestions_df = pd.DataFrame(suggestions)
            total_profit = 0.0
            for _, row in suggestions_df.iterrows():
                if row['action'] == 'SELL':
                    profit = (row['predicted price'] - row['buy price']) * row['quantity']
                    total_profit += profit
            st.subheader("Suggestions:")
            st.dataframe(suggestions_df)
            st.success(f"Total potential profit: €{round(total_profit, 2)}")

            execute = st.checkbox("✅ Execute suggestion - Save portfolio")
            if execute:
                new_portfolio = df.copy()
                for _, row in suggestions_df.iterrows():
                    idx = new_portfolio.index[new_portfolio['ticker'] == row['ticker']]
                    if row['action'] == 'BUY':
                        new_portfolio.loc[idx, 'quantity'] += row['quantity']
                    elif row['action'] == 'SELL':
                        new_portfolio.loc[idx, 'quantity'] = max(0, new_portfolio.loc[idx, 'quantity'].values[0] - row['quantity'])

                new_portfolio.to_excel("latest_portfolio.xlsx", index=False)
                st.success("Proposal executed and portfolio saved!")
        else:
            st.info("No buy or sell suggestions today.")
            st.subheader("No action:")
            st.dataframe(df)
            st.success("Total potential profit: €0.00")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")