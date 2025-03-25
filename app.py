import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="Stock AI Agent", layout="centered")
st.title("Stock AI Agent")

# === ΑΝΕΒΑΣΜΑ EXCEL ===
uploaded_file = st.file_uploader("Upload your stock Excel", type=["xlsx"])
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        df.columns = df.columns.str.lower()
        if not {'stock', 'ticker', 'quantity'}.issubset(df.columns):
            st.error("Το αρχείο Excel πρέπει να περιέχει τις στήλες: stock, ticker, quantity")
        else:
            st.success("Excel file loaded successfully.")
            df['quantity'] = df['quantity'].fillna(0).astype(int)
            df['buy price'] = 0.0
            df['predicted price'] = 0.0
            df['% change'] = 0.0
            df['action'] = ""
            df['estimated profit'] = 0.0

            min_profit = st.slider("Minimum profit % for Buy", 0.0, 10.0, 1.5, step