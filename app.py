import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import yfinance as yf

def get_close_prices(ticker, start_date):
    return yf.download(ticker, start=start_date, auto_adjust=False, progress=False)["Close"].dropna()

def compute_log_returns(price_series):
    return np.log(price_series / price_series.shift(1)).dropna()

def compute_realized_vol(log_returns, window):
    return log_returns.rolling(window).std().iloc[-1] * np.sqrt(252)

def compute_periodic_vol(price_series, period):
    resampled = price_series.resample(period).last()
    lr = compute_log_returns(resampled)
    annualize = np.sqrt(52) if period == "W" else np.sqrt(12)
    return lr.rolling(10).std().iloc[-1] * annualize

def summarize(tickers):
    end_date = datetime.now(timezone.utc)
    start_date = end_date.replace(year=end_date.year - 10)
    windows = {"1Y": 252, "2Y": 504, "5Y": 1260, "10Y": 2520}
    results = {}
    for ticker in tickers:
        prices = get_close_prices(ticker, start_date)
        if prices.empty:
            st.error(f"No price data found for {ticker}. Please check the symbol.")
            continue
        lr = compute_log_returns(prices)
        vols = {label: compute_realized_vol(lr, w) for label, w in windows.items()}
        vols["Weekly"] = compute_periodic_vol(prices, "W")
        vols["Monthly"] = compute_periodic_vol(prices, "ME")
        results[ticker] = vols
    return (pd.DataFrame(results).T * 100).round(2)

st.title("Historical & Periodic Volatility")

ticker_input = st.text_input("").strip().upper()

if st.button("Run"):
    if ticker_input:
        with st.spinner("Downloading data and calculating volatility…"):
            table = summarize([ticker_input])
        if not table.empty:
            st.subheader("Volatility (%)")
            st.dataframe(table)
            st.subheader("1‑Year Volatility Bar Chart")
            st.bar_chart(table["1Y"])
    else:
        st.warning("Please enter a valid ticker.")
else:
    st.info("Type a ticker and click Run.")
