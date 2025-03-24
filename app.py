import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock AI Agent", layout="wide")

st.title("AI Προβλέψεις Χαρτοφυλακίου ΧΑΑ")

uploaded_file = st.file_uploader("Ανέβασε νέο Excel (ή άφησέ το κενό για να φορτωθεί το τελευταίο αποθηκευμένο)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.to_excel("my_stocks_sample.xlsx", index=False)
    st.success("Φορτώθηκε νέο αρχείο.")
else:
    try:
        df = pd.read_excel("my_stocks_sample.xlsx")
    except:
        st.warning("Δεν βρέθηκε αποθηκευμένο αρχείο. Ανέβασε ένα Excel πρώτα.")
        st.stop()

min_profit_pct = st.slider("Ελάχιστο ποσοστό κέρδους για αγορά", min_value=0.0, max_value=10.0, value=6.5, step=0.5)
max_loss_pct = st.slider("Μέγιστο ποσοστό ζημιάς για πώληση", min_value=-10.0, max_value=0.0, value=-1.0, step=0.5)

def predict_future_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        if len(hist) < 5:
            return None
        past_price = hist["Close"][-5]
        current_price = hist["Close"][-1]
        growth_rate = (current_price - past_price) / past_price
        future_price = current_price * (1 + growth_rate)
        return round(future_price, 2)
    except:
        return None

def get_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        return round(stock.history(period="1d")["Close"][-1], 2)
    except:
        return None

predictions = []
no_action = []

for idx, row in df.iterrows():
    name = row["Stock"]
    ticker = row["Ticker"]
    quantity = row["Quantity"]

    current_price = get_current_price(ticker)
    predicted_price = predict_future_price(ticker)

    if current_price is None or predicted_price is None:
        continue

    diff_pct = (predicted_price - current_price) / current_price * 100

    if diff_pct >= min_profit_pct:
        action = "ΑΓΟΡΑ"
    elif diff_pct <= max_loss_pct:
        action = "ΠΩΛΗΣΗ"
    else:
        action = "Καμία ενέργεια"

    entry = {
        "Μετοχή": name,
        "Ticker": ticker,
        "Ποσότητα": quantity,
        "Τιμή Αγοράς": current_price,
        "Πρόβλεψη Τιμής": predicted_price,
        "% Διαφορά": f"{diff_pct:.2f}%",
        "Πρόταση": action
    }

    if action == "Καμία ενέργεια":
        no_action.append(entry)
    else:
        predictions.append(entry)

if predictions:
    st.subheader("Προτάσεις:")
    st.dataframe(pd.DataFrame(predictions))
else:
    st.subheader("Προτάσεις:")
    st.write("empty")

if no_action:
    st.subheader("Καμία ενέργεια:")
    st.dataframe(pd.DataFrame(no_action))

if st.checkbox("✔️ Πρόταση εκτελέστηκε - Αποθήκευση χαρτοφυλακίου"):
    new_df = df.copy()
    for entry in predictions:
        if entry["Πρόταση"] == "ΑΓΟΡΑ":
            new_df.loc[new_df["Ticker"] == entry["Ticker"], "Quantity"] += 10
        elif entry["Πρόταση"] == "ΠΩΛΗΣΗ":
            new_df.loc[new_df["Ticker"] == entry["Ticker"], "Quantity"] = max(
                0, new_df.loc[new_df["Ticker"] == entry["Ticker"], "Quantity"].values[0] - 10
            )
    new_df.to_excel("my_stocks_sample.xlsx", index=False)
    st.success("Το χαρτοφυλάκιο αποθηκεύτηκε με τις νέες ποσότητες.")