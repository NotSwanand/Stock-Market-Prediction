import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math
import requests
import warnings
warnings.filterwarnings("ignore")
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model
import joblib
from dotenv import load_dotenv
load_dotenv() 

app = Flask(__name__)

# =================== Load Pretrained Models ===================
lstm_model = load_model("models/lstm_model.h5")
lstm_scaler = joblib.load("models/lstm_scaler.pkl")
lstm_rmse = joblib.load("models/lstm_rmse.pkl")   # validation RMSE from training

xgb_model = joblib.load("models/xgb_model.pkl")
xgb_scaler = joblib.load("models/xgb_scaler.pkl")
xgb_rmse = joblib.load("models/xgb_rmse.pkl")

lr_model = joblib.load("models/lr_model.pkl")
lr_scaler = joblib.load("models/lr_scaler.pkl")
lr_rmse = joblib.load("models/lr_rmse.pkl")


# Warm-up LSTM (avoid first-request delay)
try:
    lstm_model.predict(np.zeros((1, 7, 1)))
except Exception as e:
    print("LSTM warm-up failed:", e)

API_KEY = os.getenv("TWELVE_DATA_KEY")

# =================== Utility Functions ===================
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('index.html')


def get_historical(quote):
    """Fetches historical data from TwelveData"""
    try:
        url = f'https://api.twelvedata.com/time_series?symbol={quote}&interval=1day&outputsize=2000&apikey={API_KEY}'
        response = requests.get(url)
        data = response.json()

        if 'values' not in data:
            print("TwelveData raw response:", data)
            raise ValueError("Invalid TwelveData response")

        df = pd.DataFrame(data['values'])
        df = df.rename(columns={
            'datetime': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['adj close'] = df['close']
        df = df.dropna()

        df = df.tail(600)
        return df

    except Exception as e:
        print(f"TwelveData Error: {e}")
        return pd.DataFrame()


def get_current_price(symbol):
    try:
        url = f'https://api.twelvedata.com/price?symbol={symbol}&apikey={API_KEY}'
        data = requests.get(url).json()
        if 'price' in data:
            return round(float(data['price']), 2)
        return "Unavailable"
    except:
        return "Unavailable"


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# =================== Prediction Functions ===================
def LSTM_ALGO(df):
    """Plot last 400 days with actual vs predicted trace + 7-day forecast"""
    df = df.tail(300).reset_index(drop=True)
    closes = df['close'].values.reshape(-1, 1)
    scaled = lstm_scaler.transform(closes)

    X, y = [], []
    for i in range(7, len(scaled)):
        X.append(scaled[i-7:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    if X.size > 0:
        X = X.reshape((X.shape[0], 7, 1))
        y_pred_scaled = lstm_model.predict(X, verbose=0)
        y_pred = lstm_scaler.inverse_transform(y_pred_scaled).flatten()
        y_true = lstm_scaler.inverse_transform(y.reshape(-1, 1)).flatten()

        # Plot last 100 aligned actual vs predicted
        plt.figure(figsize=(12, 4), dpi=60)
        plt.plot(y_true[-60:], label='Actual Price')
        plt.plot(y_pred[-60:], label='Predicted Price (LSTM)')
        plt.legend()
        plt.savefig('static/LSTM.png')
        plt.close()

    # 7-day forecast
    last_seq = scaled[-7:].reshape((1, 7, 1))
    forecast_prices = []
    current_seq = last_seq.copy()
    for _ in range(7):
        pred_scaled = lstm_model.predict(current_seq, verbose=0)
        pred_price = lstm_scaler.inverse_transform(pred_scaled)[0][0]
        forecast_prices.append(pred_price)

        # Update sequence with new predicted value
        new_seq = np.append(current_seq[0, 1:, 0], pred_scaled[0, 0])
        current_seq = new_seq.reshape((1, 7, 1))

    return forecast_prices[0], lstm_rmse, forecast_prices



def XGBOOST_ALGO(df):
    """Train on 2000 rows, plot last 400 test rows (old OG version, 71 features)."""
    df = df.tail(300).reset_index(drop=True)
    df['Return'] = df['close'].pct_change()

    # Lag features (30 days each → 60 features)
    for lag in range(1, 31):
        df[f'lag_{lag}'] = df['close'].shift(lag)
        df[f'return_lag_{lag}'] = df['Return'].shift(lag)

    # Technical indicators (5 features)
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA21'] = df['close'].rolling(window=21).mean()
    df['EMA'] = df['close'].ewm(span=20, adjust=False).mean()
    df['Volatility'] = df['close'].rolling(window=7).std()
    df['RSI'] = compute_rsi(df['close'], 14)

    # Drop missing rows
    df = df.dropna().reset_index(drop=True)

    # Train/test split
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]

    # Features for test
    X_test = test_df.drop(columns=['date', 'adj close'], errors='ignore').values
    scaled_X_test = xgb_scaler.transform(X_test)  # ✅ matches 71 features
    pred_returns = xgb_model.predict(scaled_X_test)
    last_closes_test = test_df['close'].shift(1).values
    pred_prices = last_closes_test * (1 + pred_returns)

    # Plot last 100
    plt.figure(figsize=(12, 4), dpi=60)
    plt.plot(test_df['close'].values[-60:], label='Actual Price')
    plt.plot(pred_prices[-60:], label='Predicted Price (XGB)')
    plt.legend()
    plt.savefig('static/XGB.png')
    plt.close()

    # Iterative 7-day forecast
    history_prices = df['close'].tolist()
    last_close = history_prices[-1]
    base_row = df.drop(columns=['date', 'adj close'], errors='ignore').iloc[-1:].copy()
    forecast_prices = []

    for _ in range(7):
        scaled_input = xgb_scaler.transform(base_row.values)
        pred_return = xgb_model.predict(scaled_input)[0]
        next_price = last_close * (1 + pred_return)
        forecast_prices.append(float(next_price))

        # Update rolling row
        last_close = next_price
        new_row = base_row.iloc[0].copy()
        new_row['close'] = next_price
        new_row['Return'] = pred_return
        base_row = pd.DataFrame([new_row])

    return df, float(forecast_prices[0]), forecast_prices, float(np.mean(forecast_prices)), xgb_rmse





def ARIMA_ALGO(df):
    """ARIMA: train on 240, test on last 60, forecast 7 ahead"""
    df = df.tail(300).reset_index(drop=True)
    values = df['close'].values

    train_size = 240
    train, test = values[:train_size], values[train_size:]

    history = list(train)
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(6, 1, 0))
        model_fit = model.fit(method_kwargs={"maxiter": 20})
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])

    rmse = math.sqrt(mean_squared_error(test, predictions)) if len(test) > 0 else 0.0

    # Forecast 7 days ahead
    forecast = model_fit.forecast(steps=7)
    arima_pred = forecast[0]

    # Plot last 60 (test set) actual vs predicted
    plt.figure(figsize=(12, 4), dpi=60)
    plt.plot(test, label='Actual Price')
    plt.plot(predictions, label='Predicted Price (ARIMA)')
    plt.legend()
    plt.savefig("static/ARIMA.png")
    plt.close()

    return arima_pred, rmse


def LIN_REG_ALGO(df):
    """Use pretrained Linear Regression for quick predictions."""
    forecast_out = 7
    df = df.tail(300).reset_index(drop=True)

    df["MA7"] = df["close"].rolling(window=7).mean()
    df["MA21"] = df["close"].rolling(window=21).mean()
    df["Return"] = df["close"].pct_change()
    df = df.dropna().reset_index(drop=True)

    features = ["close", "MA7", "MA21", "Return"]
    X = df[features].values

    # forecast next 7 days
    forecast_set = lr_model.predict(lr_scaler.transform(df[features].tail(forecast_out)))
    lr_pred = forecast_set[0][0]
    mean = float(forecast_set.mean())

    # plot last 100 actual vs predicted
    split_index = int(len(X) * 0.8)
    X_test = X[split_index:]
    y_test = df["close"].shift(-forecast_out).dropna().values[split_index:]

    y_pred_test = lr_model.predict(lr_scaler.transform(X_test))

    plt.figure(figsize=(12, 4), dpi=60)
    plt.plot(y_test[-60:], label="Actual Price")
    plt.plot(y_pred_test[-60:], label="Predicted Price (LR)")
    plt.legend()
    plt.savefig("static/LR.png")
    plt.close()

    return df, lr_pred, forecast_set, mean, lr_rmse




def recommending(df, _, today_stock, mean):
    if today_stock.empty:
        return "N/A", "Insufficient Data"
    if today_stock.iloc[-1]['close'] > mean:
        return "FALL", "SELL"
    return "RISE", "BUY"


# =================== Flask Route ===================
@app.route('/insertintotable', methods=['POST'])
def insertintotable():
    quote = request.form.get('nm')
    if not quote:
        return redirect(url_for('index'))

    df = get_historical(quote)
    current_price = get_current_price(quote)

    if df.empty:
        return render_template('index.html', error=True)

    # Only run ARIMA locally, not on Render
    if os.getenv("RENDER"):
        arima_pred, error_arima = 0.0, "N/A"
    else:
        arima_pred, error_arima = ARIMA_ALGO(df.copy())
    
    

    lstm_pred, error_lstm, lstm_forecast = LSTM_ALGO(df.copy())
    _, lr_pred, forecast_set, mean, error_lr = LIN_REG_ALGO(df.copy())
    _, xgb_pred, xgb_forecast, _, error_xgb = XGBOOST_ALGO(df.copy())

    today_stock = df.iloc[-1:].round(2)
    idea, decision = recommending(df, 0, today_stock, mean)

    def format_rmse(val):
        return "N/A" if val == 0.0 else round(val, 2)

    return render_template('results.html',
        quote=quote,
        arima_pred=round(arima_pred, 2) if error_arima != "N/A" else "N/A",
        lstm_pred=round(lstm_pred, 2),
        lr_pred=round(lr_pred, 2),
        xgb_pred=round(xgb_pred, 2),
        open_s=today_stock['open'].to_string(index=False),
        close_s=today_stock['close'].to_string(index=False),
        adj_close=today_stock['adj close'].to_string(index=False),
        high_s=today_stock['high'].to_string(index=False),
        low_s=today_stock['low'].to_string(index=False),
        vol=today_stock['volume'].to_string(index=False),
        current_price=current_price,
        forecast_set=forecast_set.flatten().tolist(),
        xgb_forecast=xgb_forecast,
        forecast_set_lstm=lstm_forecast,
        error_lr=format_rmse(error_lr),
        error_lstm=format_rmse(error_lstm),
        error_arima=error_arima,
        error_xgb=format_rmse(error_xgb),
        idea=idea,
        decision=decision
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
