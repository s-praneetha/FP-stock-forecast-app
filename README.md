# FP-stock-forecast-app

### 📈 Stock Price Forecast App (LSTM + Streamlit)

This app forecasts future stock prices using an LSTM model trained on historical Adjusted Close data.  
Built with **Streamlit**, **TensorFlow**, and **yfinance**.

## Features
- Upload ticker and forecast future price
- Visualize forecast trend + confidence intervals
- Daily fluctuation insights with auto-marked changes
- Download forecasted data as CSV

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
