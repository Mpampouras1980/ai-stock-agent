import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import os

st.set_page_config(page_title="Stock AI Agent", layout="centered")

st.title("AI Προβλέψεις Χαρτοφυλακίου ΧΑΑ")

DEFAULT_FILE = "last_portfolio.xlsx"
uploaded_file = st.file_uploader("Ανέβασε νέο Excel (ή άφησέ το κενό για να φορτωθεί το τελευταίο αποθηκευμένο)", type=["xlsx"])

# Αν δεν ανέβει νέο αρχείο, φόρτωσε το τελευταίο αποθηκευμένο
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("Φορτώθηκε νέο αρχείο.")
else:
    if os.path.exists(DEFAULT_FILE):
        df = pd.read_excel(DEFAULT_FILE)
        st.info("Φορτώθηκε το προηγούμενο χαρτοφυλάκιο.")
    else:
        st.warning("Δεν βρέθηκε προηγούμενο αρχείο. Ανέβασε αρχείο Excel.")
        st.stop()

min_profit = st.slider("Ελάχιστο ποσοστό κέρδους για αγορά", 0.0, 10.0, 1.0, 0.5)
max_loss = st.slider("Μέγιστο ποσοστό ζημιάς για πώληση", -10.0, 0.0, -1.0, 0.5)

suggestions = []
updated_portfolio = df.copy()

for idx, row in df.iterrows():
    name = row['Μετοχή']
    ticker = row['Ticker']
    try:
        buy_price = float(row['ΤιμήΑγοράς'])
        qty = int(row['Ποσότητα'])
    except:
        continue

    try:
        data = yf.download(ticker, period='6mo', interval='1d')
        data = data.reset_index()[['Date', 'Close']]
        data.columns = ['ds', 'y']

        if len(data) < 30:
            continue

        model = Prophet(daily_seasonality=True)
        model.fit(data)
        future = model.make_future_dataframe(periods=5)
        forecast = model.predict(future)
        predicted_price = forecast['yhat'].iloc[-1]
        diff_percent = ((predicted_price - buy_price) / buy_price) * 100

        if diff_percent >= min_profit:
            action = "ΑΓΟΡΑ"
        elif diff_percent <= max_loss:
            action = "ΠΩΛΗΣΗ"
        else:
            action = "ΚΡΑΤΑ"

        suggestions.append({
            "Μετοχή": name,
            "Ticker": ticker,
            "Τιμή Αγοράς": round(buy_price, 2),
            "Πρόβλεψη Τιμής": round(predicted_price, 2),
            "% Διαφορά": f"{diff_percent:.2f}%",
            "Πρόταση": action
        })

        # Αυτόματη ενημέρωση ποσότητας αν ο χρήστης εκτελέσει την πρόταση
        if action == "ΑΓΟΡΑ":
            updated_portfolio.at[idx, 'Ποσότητα'] += 10
            updated_portfolio.at[idx, 'ΤιμήΑγοράς'] = predicted_price
        elif action == "ΠΩΛΗΣΗ" and updated_portfolio.at[idx, 'Ποσότητα'] >= 10:
            updated_portfolio.at[idx, 'Ποσότητα'] -= 10

    except Exception as e:
        suggestions.append({
            "Μετοχή": name,
            "Ticker": ticker,
            "Πρόβλεψη Τιμής": "-",
            "% Διαφορά": "-",
            "Πρόταση": "Σφάλμα"
        })

results_df = pd.DataFrame(suggestions)
st.subheader("Προτάσεις:")
st.dataframe(results_df)

# Εκτέλεση πρότασης
if st.button("✅ Πρόταση εκτελέστηκε - Αποθήκευση χαρτοφυλακίου"):
    updated_portfolio.to_excel(DEFAULT_FILE, index=False)
    st.success("Το χαρτοφυλάκιο ενημερώθηκε και αποθηκεύτηκε ως default για την επόμενη φορά.")