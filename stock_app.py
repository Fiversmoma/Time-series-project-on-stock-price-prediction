import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import datetime
import requests
import json

# App title
st.title("Stock Price Prediction App")
st.markdown("Shown are the stock price data for different companies!\n\n"
            "Built by AMINA TAHIR BALARABE using\n"
            "`Python`, `streamlit`, `statsmodels`, `numpy`, `matplotlib`, `requests`, `json`,  `pandas`, and `datetime`.")

@st.cache
def load_data(ticker):
    data = pd.read_csv(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey=YKDJDK7Y5J8N5AMW&datatype=csv")
    return data

def plot_prediction(ticker, model, start, end, actual, prediction):
    fig, ax = plt.subplots()
    ax.plot(actual, label="actual")
    ax.plot(prediction, label="prediction")
    plt.title(f"{ticker} stock price prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(np.arange(start, end, 30), rotation=90)
    plt.legend()
    st.pyplot(fig)

def get_ticker_data(ticker):
    api_key = "YKDJDK7Y5J8N5AMW"
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey=YKDJDK7Y5J8N5AMW"
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        return data
    else:
        st.write("Error retrieving data. Please check your API key and try again.")
        return None

def sarimax_prediction(data, ticker):
    close_price = data["close"]
    close_price.index = data["timestamp"]
    try:
        model = SARIMAX(close_price, order=(2,1,2), seasonal_order=(2,1,2,30))
        results = model.fit()
    except Exception as e:
        st.write("Error fitting SARIMAX model. Please check the data and try again.")
        return None
    
    prediction = results.predict(start=len(close_price), end=len(close_price)+30, typ='levels').rename(f"{ticker} prediction")
    actual = close_price.tail(30)
    plot_prediction(ticker, model, len(close_price), len(close_price)+30, actual, prediction)
    return prediction

# Sidebar
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.datetime.now().date())


# Ticker symbol input
ticker = st.text_input("Enter ticker symbol:", "ABNB")

# Load data
data = load_data(ticker)

# Get ticker data
ticker_data = get_ticker_data(ticker)

if ticker_data:
    if 'logo_url' in ticker_data:
        string_logo = '<img src=%s>' % ticker_data['logo_url']
        st.markdown(string_logo, unsafe_allow_html=True)
    else:
        st.write("No logo data found for this ticker.")
else:
    st.write("No data found for this ticker.")

# Plotting Time series data 

st.write(data.head())

fig = go.Figure()
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['open'], name='Stock_open'))
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['close'], name='Stock_close'))
fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
st.plotly_chart(fig)



def bollinger_bands(data, window_size=20, num_of_std=2):
    rolling_mean = data['close'].rolling(window_size).mean()
    rolling_std = data['close'].rolling(window_size).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return rolling_mean, upper_band, lower_band

def plot_bollinger_bands(data, window_size=20, num_of_std=2):
    rolling_mean, upper_band, lower_band = bollinger_bands(data, window_size, num_of_std)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['close'], name='Stock_close'))
    fig.add_shape(type='line', x0=data.iloc[0]['timestamp'], x1=data.iloc[-1]['timestamp'], y0=upper_band, y1=upper_band, yref='y', line=dict(color='red'))
    fig.add_shape(type='line', x0=data.iloc[0]['timestamp'], x1=data.iloc[-1]['timestamp'], y0=lower_band, y1=lower_band, yref='y', line=dict(color='green'))
    fig.add_



def plot_bollinger_bands(data, window_size=20, num_of_std=2):
    rolling_mean, upper_band, lower_band = bollinger_bands(data, window_size, num_of_std)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['close'], name='Stock_close'))
    fig.add_shape(type='line', x0=data.iloc[0]['timestamp'], x1=data.iloc[-1]['timestamp'], y0=upper_band, y1=upper_band, yref='y', line=dict(color='red'))
    fig.add_shape(type='line', x0=data.iloc[0]['timestamp'], x1=data.iloc[-1]['timestamp'], y0=lower_band, y1=lower_band, yref='y', line=dict(color='green'))
    fig.add_trace(go.Scatter(x=data['timestamp'], y=rolling_mean, name='rolling_mean', line=dict(color='black')))
    fig.layout.update(title_text="Bollinger Bands", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_bollinger_bands(data)


# Display last 100 data points

st.write("Displaying last 100 data points:")
st.write(data.tail(100))

# Prediction
if st.button("Predict"):
    prediction = sarimax_prediction(data, ticker)
    st.write(prediction)



string_summary = ticker_data
st.info(string_summary)

# Ticker data
st.header('**Ticker data**')
st.write(ticker_data)
