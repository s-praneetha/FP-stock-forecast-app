#deployed code
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import date, timedelta
from datetime import datetime
import streamlit as st
import os
import requests_cache
from pandas_datareader import data as pdr
import time
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ----------------------
# Streamlit UI Setup
# ----------------------
st.set_page_config(page_title="Stock Forecast App", layout="wide")
st.markdown("## 📈 Tata Steel Stock Price Forecasting")

# Sidebar Controls
with st.sidebar:
    st.markdown("### 🧮 Forecasting Controls")
    start_date = st.date_input("📅 Select Start Date", date(2023, 1, 1))
    end_date = date.today() - timedelta(days=1)  # Fixed to today - 1
    st.markdown(f"🛑 **End Date is auto-set to yesterday:** {end_date}")
    forecast_horizon = st.slider("⏳ Forecast Horizon (Days)", 60, 90, 360)
    ticker = st.text_input("💹 Stock Ticker Symbol", value="TATASTEEL.NS")
    run_forecast = st.button("📊 Run Forecast")
# Timezone-safe dates (yfinance sometimes expects strings)
                
# ----------------------
# Forecast Logic
# ----------------------
if run_forecast:
    # Secure W&B login using Streamlit secrets
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    # Initialize W&B
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
    reinit=True  # Avoids conflicts in Streamlit reruns 
    )
    # Step 1: Download Data
    stocks_df = yf.download(ticker,
                            start=start_date,
                            end=end_date,
                            interval='1d',
                            auto_adjust=False)
    st.markdown("#### Previous day's Stock Price ")
    last_row = stocks_df.tail(1)  # Get the last row of the dataframe
    styled_table = last_row.style.set_properties(**{'text-align': 'center'})
    st.dataframe(styled_table)
    
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
            X, y = [], []
            for i in range(window_size, len(data)):
                X.append(data[i - window_size:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        window_size = 60
        X, y = create_sequences(scaled_data, window_size)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Train-Test Split (for evaluation)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        
        # Step 3: Build LSTM Model
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        
        
        # Set a custom learning rate
        optimizer = Adam(learning_rate=0.001)
        
        # Compile the model with this optimizer
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        #model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        
        model.save("lstm_model_1.h5")

        y_pred_scaled = model.predict(X_test)
        y_test_scaled = y_test.reshape(-1, 1)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_true = scaler.inverse_transform(y_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        wandb.log({
            "rmse": rmse,
            "mape": mae,
            "r2" : r2
        })
        
        model_artifact = wandb.Artifact('lstm_model', type='model')
        model_artifact.add_file("lstm_model_1.h5")
        #wandb.log({"model_file": model_artifact})
        wandb.log_artifact(model_artifact)


        # Log datasets
        dataset_artifact = wandb.Artifact("Model_Dataset", type="dataset")
        with dataset_artifact.new_file("Model_Dataset.csv") as f:
            stocks_df.to_csv(f)
        wandb.log_artifact(dataset_artifact)

        # Forecasting with the trained model
        forecast_scaled = []
        last_sequence = scaled_data[-window_size:].reshape(1, window_size, 1)
        
        for _ in range(forecast_horizon):
            next_pred = model.predict(last_sequence, verbose=0)[0][0]
            forecast_scaled.append(next_pred)
            last_sequence = np.append(last_sequence[:, 1:, :], [[[next_pred]]], axis=1)

        forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
        
        # Confidence Intervals (±95%)
        lower_bound = forecast * 0.95
        upper_bound = forecast * 1.05

        # Prepare Forecast DataFrame
        forecast_dates = pd.date_range(start=stocks_df.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
        forecast_df = pd.DataFrame({
            'Forecast': forecast,
            'Lower Bound (95%)': lower_bound,
            'Upper Bound (95%)': upper_bound
        }, index=forecast_dates)
        
        # Step 7: KPI Cards – Head1 & Tail1
        first_date = forecast_df.index[0].strftime("%Y-%m-%d")
        last_date = forecast_df.index[-1].strftime("%Y-%m-%d")
        first_value = forecast_df['Forecast'].iloc[0]
        last_value = forecast_df['Forecast'].iloc[-1]

        # Log Forecast Metrics
        wandb.log({
        "start_price": forecast_df['Forecast'].iloc[0],
        "end_price": forecast_df['Forecast'].iloc[-1],
        "max_price": forecast_df['Forecast'].max(),
        "min_price": forecast_df['Forecast'].min()
        })

        st.markdown("### 📌 Key Forecast Insights")
        kpi1, spacer, kpi2 = st.columns([2, 0.5, 2])
        with kpi1:
            st.metric(label=f"📅 First Forecast Date\n({first_date})", value=f"{first_value:.2f}")
        with kpi2:
            st.metric(label=f"📅 Last Forecast Date\n({last_date})", value=f"{last_value:.2f}")

        # Step 8: Plot Forecast with Confidence Interval
        st.markdown(f"### 📆 Forecast for Next **{forecast_horizon}** Days")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(stocks_df.index, stocks_df['adj_close'], label='Historical', color='steelblue')
        ax.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', linestyle='--', color='orange')
        ax.fill_between(forecast_df.index, forecast_df['Lower Bound (95%)'], forecast_df['Upper Bound (95%)'],
                        color='orange', alpha=0.2, label='Confidence Interval (±95%)')
        ax.set_xlabel("Date")
        ax.set_ylabel("Adjusted Close Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Step 9: Forecast Table and Download (Centered)
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

        # Step 10: Visual Insights on Forecasted Data (With Identifiers)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### 🔍 Visual Insights on Forecast Fluctuations")

        forecast_df['Change'] = forecast_df['Forecast'].diff()
        colors = ['green' if val >= 0 else 'red' for val in forecast_df['Change'][1:]]

        col1, col2 = st.columns([1, 1])

        # --------- LEFT PLOT: Forecast Trend with Markers and Labels ---------
        with col1:
            fig_left, ax_left = plt.subplots(figsize=(10, 5))
            ax_left.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='black', linewidth=2)
            ax_left.scatter(forecast_df.index[1:], forecast_df['Forecast'][1:], c=colors, s=60)

            # Annotate start and end
            ax_left.annotate(f"Start: {forecast_df['Forecast'].iloc[0]:.2f}",
                             (forecast_df.index[0], forecast_df['Forecast'].iloc[0]),
                             textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='blue')

            ax_left.annotate(f"End: {forecast_df['Forecast'].iloc[-1]:.2f}",
                             (forecast_df.index[-1], forecast_df['Forecast'].iloc[-1]),
                             textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='blue')

            # Annotate max value
            max_idx = forecast_df['Forecast'].idxmax()
            max_val = forecast_df['Forecast'].max()
            ax_left.annotate(f"Max: {max_val:.2f}",
                             (max_idx, max_val),
                             textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9, color='green')

            ax_left.set_title("Forecast Trend with Daily Markers", fontsize=14)
            ax_left.set_ylabel("Price")
            ax_left.grid(True)
            st.pyplot(fig_left, use_container_width=True)

        # --------- RIGHT PLOT: Daily Change Bars with Highlighted Extremes ---------
        with col2:
            fig_right, ax_right = plt.subplots(figsize=(10, 5))
            ax_right.bar(forecast_df.index[1:], forecast_df['Change'][1:], color=colors)
            ax_right.axhline(0, color='black', linewidth=1.2)

            # Max upward change
            max_up_idx = forecast_df['Change'].idxmax()
            max_up_val = forecast_df['Change'].max()
            ax_right.annotate(f"+{max_up_val:.2f}",
                              (max_up_idx, max_up_val),
                              textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='green')

            # Max downward change
            max_down_idx = forecast_df['Change'].idxmin()
            max_down_val = forecast_df['Change'].min()
            ax_right.annotate(f"{max_down_val:.2f}",
                              (max_down_idx, max_down_val),
                              textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9, color='red')

            ax_right.set_title("Day-over-Day Change in Forecast", fontsize=14)
            ax_right.set_ylabel("Change")
            ax_right.grid(True)
            st.pyplot(fig_right, use_container_width=True)
wandb.finish()
