
import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet

st.set_page_config(page_title="Stock AI Agent", layout="centered")

st.title("AI Προβλέψεις Χαρτοφυλακίου ΧΑΑ")

uploaded_file = st.file_uploader("Ανέβασε το αρχείο Excel με τις μετοχές σου", type=["xlsx"])
min_profit = st.slider("Ελάχιστο ποσοστό κέρδους για αγορά", 0.0, 10.0, 1.0, 0.5)
max_loss = st.slider("Μέγιστο ποσοστό ζημιάς για πώληση", -10.0, 0.0, -1.0, 0.5)

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("Φορτώθηκε το αρχείο με τις μετοχές σου.")

    suggestions = []

    for _, row in df.iterrows():
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
                action = f"ΑΓΟΡΑ"
            elif diff_percent <= max_loss:
                action = f"ΠΩΛΗΣΗ"
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
