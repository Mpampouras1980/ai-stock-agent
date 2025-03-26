import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import io

st.set_page_config(page_title="Stock AI Agent", layout="wide")

def send_email_alert(subject, body):
    sender_email = "gprovopo@googlemail.com"
    receiver_email = "gprovopo@googlemail.com"
    app_password = "knuehkeowscowyhh"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

def fetch_data(ticker):
    df = yf.download(ticker, period="6mo")
    df = df[["Close"]].rename(columns={"Close": "Price"})
    df["MA10"] = df["Price"].rolling(window=10).mean()
    df["RSI"] = compute_rsi(df["Price"], 14)
    df["Momentum"] = df["Price"].diff(4)
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def predict_price(df):
    X = df[["Price", "MA10", "RSI", "Momentum"]]
    y = df["Price"].shift(-5)
    df.dropna(inplace=True)
    X = X[:-5]
    y = y[:-5]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmses.append(rmse)
    avg_rmse = np.mean(rmses)
    model.fit(X, y)
    next_input = df[["Price", "MA10", "RSI", "Momentum"]].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(next_input)[0]
    return round(prediction, 2), round(avg_rmse, 2)

def calculate_portfolio_value(df):
    return (df["quantity"] * df["buy price"]).sum()

st.title("Stock AI Agent")

uploaded_file = st.file_uploader("Upload your stock Excel", type=["xlsx"])
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()
        df["buy price"] = df.get("buy price", 0)
        df["predicted"] = 0.0
        df["rmse"] = 0.0
        suggestions = []

        min_profit = st.slider("Minimum profit % for Buy", 0.0, 10.0, 1.5, step=0.5)
        cashout_target = st.slider("Cash-out target value (€)", 1000, 20000, 10000, step=100)
        portfolio_value = calculate_portfolio_value(df)
        st.markdown(f"**Current portfolio value:** €{portfolio_value:.2f}")

        for i, row in df.iterrows():
            try:
                hist = fetch_data(row["ticker"])
                predicted, rmse = predict_price(hist)
                df.at[i, "predicted"] = predicted
                df.at[i, "rmse"] = rmse
                if row["buy price"] > 0:
                    profit = (predicted - row["buy price"]) / row["buy price"] * 100
                    if profit > min_profit:
                        suggestions.append(("SELL", row["stock"], row["ticker"], row["quantity"], row["buy price"], predicted, profit))
                elif row["buy price"] == 0 and predicted > hist["Price"].iloc[-1] * (1 + min_profit / 100):
                    suggestions.append(("BUY", row["stock"], row["ticker"], "-", hist["Price"].iloc[-1], predicted, (predicted - hist["Price"].iloc[-1]) / hist["Price"].iloc[-1] * 100))
            except Exception as e:
                st.warning(f"{row['ticker']} skipped: {e}")

        if portfolio_value > cashout_target:
            send_email_alert(
                "Ειδοποίηση Είσπραξης",
                f"Η τρέχουσα αξία χαρτοφυλακίου είναι €{portfolio_value:.2f} και υπερβαίνει το όριο είσπραξης €{cashout_target}."
            )

        if suggestions:
            st.subheader("Suggestions:")
            suggestion_df = pd.DataFrame(suggestions, columns=["action", "stock", "ticker", "quantity", "buy price", "predicted", "profit %"])
            st.dataframe(suggestion_df)

            if st.checkbox("✅ Execute suggestion - Save portfolio"):
                for s in suggestions:
                    if s[0] == "BUY":
                        df.loc[df["ticker"] == s[2], "buy price"] = s[5]
                        df.loc[df["ticker"] == s[2], "quantity"] = 100
                    elif s[0] == "SELL":
                        df.loc[df["ticker"] == s[2], "quantity"] = 0
                df[["stock", "ticker", "quantity", "buy price"]].to_excel("my_stocks_minimal_clean_fixed_ready.xlsx", index=False)
                st.success("Portfolio updated and saved.")
        else:
            st.info("No suggestions today.")

        st.subheader("No action:")
        st.dataframe(df[["stock", "ticker", "quantity", "buy price"]])

    except Exception as e:
        st.error(f"Error processing file: {e}")