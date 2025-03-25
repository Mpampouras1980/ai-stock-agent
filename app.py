import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import os
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# === Settings ===
st.set_page_config(page_title="Stock AI Agent", layout="centered")
st.title("Stock AI Agent")

# === Upload Excel ===
uploaded_file = st.file_uploader("Upload your stock Excel", type="xlsx")

if uploaded_file:
    portfolio_df = pd.read_excel(uploaded_file)

    if "Μετοχή" not in portfolio_df.columns or "Ticker" not in portfolio_df.columns or "Ποσότητα" not in portfolio_df.columns:
        st.error("Το αρχείο Excel πρέπει να περιέχει τις στήλες: Μετοχή, Ticker, Ποσότητα")
    else:
        tickers = portfolio_df["Ticker"].tolist()
        quantities = dict(zip(portfolio_df["Ticker"], portfolio_df["Ποσότητα"]))

        # === Fetch historical data and train model dynamically ===
        def train_model(ticker):
            try:
                data = yf.download(ticker, period="6mo", interval="1d")
                data = data.dropna()
                data["Target"] = data["Close"].shift(-5)
                data.dropna(inplace=True)
                X = data[["Close"]]
                y = data["Target"]
                model = LinearRegression()
                model.fit(X, y)
                return model
            except:
                return None

        # === Predict prices using retrained models ===
        predicted_prices = []
        current_prices = []
        models = {}
        for ticker in tickers:
            model = train_model(ticker)
            if model:
                models[ticker] = model
                current_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
                current_prices.append(current_price)
                predicted = model.predict([[current_price]])[0]
                predicted_prices.append(predicted)
            else:
                current_prices.append(0)
                predicted_prices.append(0)

        portfolio_df["Τιμή Αγοράς"] = current_prices
        portfolio_df["Πρόβλεψη Τιμής"] = predicted_prices

        # === Sliders ===
        min_profit = st.slider("Minimum profit % for Buy", 0.0, 10.0, 2.0, step=0.5)
        max_loss = st.slider("Maximum loss % for Sell", -10.0, 0.0, -1.0, step=0.5)

        # === Calculate recommendations ===
        actions = []
        changes = []
        total_profit = 0
        suggestions = []
        no_action = []

        for index, row in portfolio_df.iterrows():
            buy_price = row["Τιμή Αγοράς"]
            predicted_price = row["Πρόβλεψη Τιμής"]
            qty = row["Ποσότητα"]

            if buy_price == 0 or predicted_price == 0:
                action = "Καμία ενέργεια"
                changes.append(0.0)
                no_action.append(row)
            else:
                pct_change = ((predicted_price - buy_price) / buy_price) * 100
                changes.append(pct_change)
                if pct_change >= min_profit:
                    action = "ΑΓΟΡΑ"
                    total_profit += (predicted_price - buy_price) * qty
                    suggestions.append(row)
                elif pct_change <= max_loss:
                    action = "ΠΩΛΗΣΗ"
                    total_profit += (predicted_price - buy_price) * qty
                    suggestions.append(row)
                else:
                    action = "Καμία ενέργεια"
                    no_action.append(row)
            actions.append(action)

        portfolio_df["% Διαφορά"] = [f"{x:.2f}%" for x in changes]
        portfolio_df["Πρόταση"] = actions

        # === Display tables ===
        st.subheader("Προτάσεις:")
        suggestion_df = portfolio_df[portfolio_df["Πρόταση"].isin(["ΑΓΟΡΑ", "ΠΩΛΗΣΗ"])]
        st.dataframe(suggestion_df[["Μετοχή", "Ticker", "Ποσότητα", "Τιμή Αγοράς", "Πρόβλεψη Τιμής", "% Διαφορά", "Πρόταση"]], use_container_width=True)

        st.subheader("Καμία ενέργεια:")
        if no_action:
            no_action_df = pd.DataFrame(no_action)
            st.dataframe(no_action_df[["Μετοχή", "Ticker", "Ποσότητα", "Τιμή Αγοράς"]], use_container_width=True)
        else:
            st.write("Δεν υπάρχουν μετοχές χωρίς πρόταση.")

        st.success(f"Συνολικό πιθανό κέρδος: €{total_profit:.2f}")

        # === Execute action ===
        if st.checkbox("✅ Execute suggestion - Save portfolio"):
            updated_df = portfolio_df.copy()
            for index, row in suggestion_df.iterrows():
                if row["Πρόταση"] == "ΑΓΟΡΑ":
                    updated_df.at[index, "Ποσότητα"] += 10
                elif row["Πρόταση"] == "ΠΩΛΗΣΗ":
                    updated_df.at[index, "Ποσότητα"] = max(0, updated_df.at[index, "Ποσότητα"] - 10)

            updated_df[["Μετοχή", "Ticker", "Ποσότητα"]].to_excel("my_stocks_minimal_sample.xlsx", index=False)
            st.success("Πρόταση εκτελέστηκε - Αποθήκευση χαρτοφυλακίου")