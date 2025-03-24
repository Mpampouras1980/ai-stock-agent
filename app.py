import streamlit as st
import pandas as pd
import yfinance as yf

st.title("Stock AI Agent")

uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("New file loaded.")

    min_profit = st.slider("Minimum profit % for Buy", 0.0, 10.0, 6.5, 0.5)
    max_loss = st.slider("Maximum loss % for Sell", -10.0, 0.0, -1.0, 0.5)

    proposals = []
    no_action = []
    total_profit = 0.0

    for index, row in df.iterrows():
        name = row['Stock']
        ticker = row['Ticker']
        qty = row['Quantity']

        try:
            ticker_data = yf.Ticker(ticker)
            todays_data = ticker_data.history(period='1d')
            current_price = todays_data['Close'].iloc[-1]

            if 'Buy Price' in df.columns:
                purchase_price = row['Buy Price']
            else:
                purchase_price = current_price

            price_diff_pct = ((current_price - purchase_price) / purchase_price) * 100
            price_diff_eur = (current_price - purchase_price) * qty

            if price_diff_pct >= min_profit:
                proposals.append([name, ticker, qty, round(purchase_price, 2), round(current_price, 2), f"{price_diff_pct:.2f}%", "BUY"])
                total_profit += price_diff_eur
            elif price_diff_pct <= max_loss:
                proposals.append([name, ticker, qty, round(purchase_price, 2), round(current_price, 2), f"{price_diff_pct:.2f}%", "SELL"])
                total_profit += price_diff_eur
            else:
                no_action.append([name, ticker, qty, round(current_price, 2)])
        except Exception as e:
            st.warning(f"Error for {ticker}: {str(e)}")
            continue

    if proposals:
        st.subheader("Suggestions:")
        st.dataframe(pd.DataFrame(proposals, columns=["Stock", "Ticker", "Quantity", "Buy Price", "Predicted Price", "% Change", "Action"]))
    else:
        st.info("No buy or sell suggestions.")

    if no_action:
        st.subheader("No Action:")
        st.dataframe(pd.DataFrame(no_action, columns=["Stock", "Ticker", "Quantity", "Buy Price"]))

    if proposals:
        if total_profit >= 0:
            st.success(f"Total potential profit: **€{total_profit:.2f}**")
        else:
            st.error(f"Total potential loss: **€{total_profit:.2f}**")

    if proposals and st.checkbox("✅ Execute suggestion - Save portfolio"):
        updated_df = df.copy()
        for prop in proposals:
            name, ticker, qty, _, _, _, action = prop
            if action == "BUY":
                updated_df.loc[updated_df['Ticker'] == ticker, 'Quantity'] += qty
            elif action == "SELL":
                updated_df.loc[updated_df['Ticker'] == ticker, 'Quantity'] -= qty

        updated_df.to_excel("my_stocks_minimal_sample.xlsx", index=False)
        st.success("Portfolio saved.")