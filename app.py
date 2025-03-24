import streamlit as st
import pandas as pd
import yfinance as yf

# Τίτλος εφαρμογής
st.title("Stock AI Agent")

# Ανεβάστε αρχείο Excel
uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("Φορτώθηκε νέο αρχείο.")

    # Επιλογή thresholds
    min_profit = st.slider("Ελάχιστο ποσοστό κέρδους για αγορά", 0.0, 10.0, 6.5, 0.5)
    max_loss = st.slider("Μέγιστο ποσοστό ζημιάς για πώληση", -10.0, 0.0, -1.0, 0.5)

    # Προετοιμασία λίστας προτάσεων και ουδέτερων ενεργειών
    proposals = []
    no_action = []
    total_profit = 0.0

    for index, row in df.iterrows():
        name = row['Μετοχή']
        ticker = row['Ticker']
        qty = row['Ποσότητα']

        try:
            ticker_data = yf.Ticker(ticker)
            todays_data = ticker_data.history(period='1d')
            current_price = todays_data['Close'].iloc[-1]

            # Υπολογισμός μέσης τιμής αγοράς
            if 'Τιμή Αγοράς' in df.columns:
                purchase_price = row['Τιμή Αγοράς']
            else:
                purchase_price = current_price  # fallback για νέα αγορά

            price_diff_pct = ((current_price - purchase_price) / purchase_price) * 100
            price_diff_eur = (current_price - purchase_price) * qty

            if price_diff_pct >= min_profit:
                proposals.append([name, ticker, qty, round(purchase_price, 2), round(current_price, 2), f"{price_diff_pct:.2f}%", "ΑΓΟΡΑ"])
                total_profit += price_diff_eur
            elif price_diff_pct <= max_loss:
                proposals.append([name, ticker, qty, round(purchase_price, 2), round(current_price, 2), f"{price_diff_pct:.2f}%", "ΠΩΛΗΣΗ"])
                total_profit += price_diff_eur
            else:
                no_action.append([name, ticker, qty, round(current_price, 2)])
        except Exception as e:
            st.warning(f"Σφάλμα για {ticker}: {str(e)}")
            continue

    # Εμφάνιση πίνακα προτάσεων
    if proposals:
        st.subheader("Προτάσεις:")
        st.dataframe(pd.DataFrame(proposals, columns=["Μετοχή", "Ticker", "Ποσότητα", "Τιμή Αγοράς", "Πρόβλεψη Τιμής", "↑ % Διαφορά", "Πρόταση"]))
    else:
        st.info("Δεν υπάρχουν προτάσεις για αγορά ή πώληση.")

    # Εμφάνιση πίνακα 'Καμία ενέργεια'
    if no_action:
        st.subheader("Καμία ενέργεια:")
        st.dataframe(pd.DataFrame(no_action, columns=["Μετοχή", "Ticker", "Ποσότητα", "Τιμή Αγοράς"]))

    # Εμφάνιση πιθανού συνολικού κέρδους/ζημιάς
    if proposals:
        if total_profit >= 0:
            st.success(f"Συνολικό πιθανό κέρδος: **{total_profit:.2f} ευρώ**")
        else:
            st.error(f"Συνολική πιθανή ζημιά: **{total_profit:.2f} ευρώ**")

    # Κουμπί αποθήκευσης αν υπάρχουν προτάσεις
    if proposals and st.checkbox("✅ Πρόταση εκτελέστηκε - Αποθήκευση χαρτοφυλακίου"):
        updated_df = df.copy()
        for prop in proposals:
            name, ticker, qty, _, _, _, action = prop
            if action == "ΑΓΟΡΑ":
                updated_df.loc[updated_df['Ticker'] == ticker, 'Ποσότητα'] += qty
            elif action == "ΠΩΛΗΣΗ":
                updated_df.loc[updated_df['Ticker'] == ticker, 'Ποσότητα'] -= qty

        updated_df.to_excel("my_stocks_minimal_sample.xlsx", index=False)
        st.success("Αποθηκεύτηκε το νέο χαρτοφυλάκιο.")