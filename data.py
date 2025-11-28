import numpy as np
import streamlit as st
import yfinance as yf
#Live data from yahoo finance api
@st.cache_resource
def get_live_data(ticker,period,interval):
    df = yf.download(ticker,period=period,interval=interval)
    df.columns = [col[0] for col in df.columns]
    #feature engineering
    df["Price_change"] = df["Close"]-df["Open"]
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_10"] = df["Close"].ewm(span=10).mean()
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["Vol_change"] = df["Volume"].pct_change()

    #Output  --> Next Day's Price
    df["Target"] = df["Close"].shift(-1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    if len(df) < 50:
        st.error("❌ Not enough data to train the model. Try a longer period or different interval.")
        st.stop()
    else:
        return df
