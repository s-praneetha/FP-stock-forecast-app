import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import date, timedelta
import streamlit as st
import wandb
import os
import pandas_datareader.data as web
from alpha_vantage.timeseries import TimeSeries

# ----------------------
# Streamlit UI Setup
# ----------------------
st.set_page_config(page_title="Stock Forecast App", layout="wide")
st.markdown("## 📈 Tata Steel Stock Price Forecasting")

# Sidebar Controls
with st.sidebar:
    st.markdown("### 🧮 Forecasting Controls")
    start_date = st.date_input("📅 Select Start Date", date(2020, 1, 1))
    end_date = date.today() - timedelta(days=1)  # Fixed to today - 1
    st.markdown(f"🛑 **End Date is fixed to:** {end_date}")
    forecast_horizon = st.slider("⏳ Forecast Horizon (Days)", 30, 60, 180)
    ticker = st.text_input("💹 Stock Ticker Symbol", value="TATASTEEL.NS")
    run_forecast = st.button("📊 Run Forecast")

# ----------------------
# Forecast Logic
# ----------------------
if run_forecast:
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    wandb.init(
        project="stock-forecasting-lstm",
        name=f"{ticker}_{date.today()}",
        config={
            "ticker": ticker,
            "start_date": start_date,
            "forecast_horizon": forecast_horizon,
            "model": "LSTM",
            "window_size": 60
        },
        reinit=True
    )

    # Step 1: Download Data (With Fallback)
    # Initialize Alpha Vantage
    alpha_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")  # Set this as env var in Streamlit secrets/render
    ts = TimeSeries(key=alpha_api_key, output_format='pandas')

    try:
        data, meta = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        stocks_df = data[['5. adjusted close']].copy()
        stocks_df.rename(columns={'5. adjusted close': 'Adj_Close'}, inplace=True)
        stocks_df.index = pd.to_datetime(stocks_df.index)
        stocks_df = stocks_df.sort_index()
        stocks_df = stocks_df.loc[start_date:end_date]
    except Exception as e:
        st.error(f"❌ Alpha Vantage fetch failed: {e}")
        stocks_df = pd.DataFrame()

    # Proceed only if data is valid
    if 'Adj Close' in stocks_df.columns:
        stocks_df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)

    if 'Adj_Close' not in stocks_df.columns or stocks_df.empty:
        st.error("'Adj_Close' not found in data or empty dataset.")
    else:
        stocks_df = stocks_df[['Adj_Close']].dropna()
        stocks_df.columns = ['adj_close']

        # Step 2: Preprocessing
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(stocks_df[['adj_close']])

        def create_sequences(data, window_size):
            X = []
            for i in range(window_size, len(data)):
                X.append(data[i - window_size:i, 0])
            return np.array(X)

        window_size = 60
        X_input = create_sequences(scaled_data, window_size)
        X_input = X_input.reshape((X_input.shape[0], X_input.shape[1], 1))

        # Step 3: Load Model
        model = load_model("lstm_model_1.h5", compile=False)

        # Step 4: Forecast
        last_sequence = scaled_data[-window_size:].reshape(1, window_size, 1)
        forecast_scaled = []

        for _ in range(forecast_horizon):
            next_pred = model.predict(last_sequence, verbose=0)[0][0]
            forecast_scaled.append(next_pred)
            last_sequence = np.append(last_sequence[:, 1:, :], [[[next_pred]]], axis=1)

        forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

        # Step 5: Confidence Intervals
        lower_bound = forecast * 0.95
        upper_bound = forecast * 1.05

        # Step 6: Prepare Forecast DataFrame
        forecast_dates = pd.date_range(start=stocks_df.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
        forecast_df = pd.DataFrame({
            'Forecast': forecast,
            'Lower Bound (95%)': lower_bound,
            'Upper Bound (95%)': upper_bound
        }, index=forecast_dates)

        wandb.log({
            "start_price": forecast_df['Forecast'].iloc[0],
            "end_price": forecast_df['Forecast'].iloc[-1],
            "max_price": forecast_df['Forecast'].max(),
            "min_price": forecast_df['Forecast'].min()
        })

        # Step 7: KPI Cards
        first_date = forecast_df.index[0].strftime("%Y-%m-%d")
        last_date = forecast_df.index[-1].strftime("%Y-%m-%d")
        first_value = forecast_df['Forecast'].iloc[0]
        last_value = forecast_df['Forecast'].iloc[-1]

        st.markdown("### 📌 Key Forecast Insights")
        kpi1, spacer, kpi2 = st.columns([2, 0.5, 2])
        with kpi1:
            st.metric(label=f"📅 First Forecast Date\n({first_date})", value=f"{first_value:.2f}")
        with kpi2:
            st.metric(label=f"📅 Last Forecast Date\n({last_date})", value=f"{last_value:.2f}")

        # Step 8: Plot Forecast
        st.markdown(f"### 📆 Forecast for Next **{forecast_horizon}** Days")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(stocks_df.index, stocks_df['adj_close'], label='Historical', color='steelblue')
        ax.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', linestyle='--', color='orange')
        ax.fill_between(forecast_df.index, forecast_df['Lower Bound (95%)'], forecast_df['Upper Bound (95%)'],
                        color='orange', alpha=0.2, label='Confidence Interval (±5%)')
        ax.set_xlabel("Date")
        ax.set_ylabel("Adjusted Close Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Step 9: Forecast Table + CSV
        st.markdown("### 🔢 Forecast Table")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.dataframe(forecast_df.style.format({
                "Forecast": "{:.2f}",
                "Lower Bound (95%)": "{:.2f}",
                "Upper Bound (95%)": "{:.2f}"
            }), use_container_width=True)

            csv = forecast_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name=f"{ticker}_forecast.csv",
                mime='text/csv'
            )

        # Step 10: Visual Insight – Arrows
        st.markdown("### 🔍 Visual Insights on Forecast Fluctuations")

        forecast_df['Change'] = forecast_df['Forecast'].diff()
        colors = ['green' if val >= 0 else 'red' for val in forecast_df['Change'][1:]]
        col1, col2 = st.columns([1, 1])

        with col1:
            fig_left, ax_left = plt.subplots(figsize=(10, 5))
            ax_left.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='black', linewidth=2)
            ax_left.scatter(forecast_df.index[1:], forecast_df['Forecast'][1:], c=colors, s=60)
            ax_left.set_title("Forecast Trend with Daily Markers")
            ax_left.grid(True)
            st.pyplot(fig_left)

        with col2:
            fig_right, ax_right = plt.subplots(figsize=(10, 5))
            ax_right.bar(forecast_df.index[1:], forecast_df['Change'][1:], color=colors)
            ax_right.axhline(0, color='black', linewidth=1.2)
            ax_right.set_title("Day-over-Day Change in Forecast")
            ax_right.grid(True)
            st.pyplot(fig_right)

        wandb.finish()
