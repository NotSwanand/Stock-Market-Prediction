# main.py (Render-optimized with debug logs + model toggles)
import os
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
import math, requests, warnings, joblib
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Detect if running on Render
ON_RENDER = bool(os.getenv("RENDER"))

# Import ARIMA only locally
ARIMA = None
if not ON_RENDER:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as e:
        print("Could not import ARIMA:", e)

# =================== TOGGLES ===================
ENABLE_LSTM = True
ENABLE_XGB = True
ENABLE_LR = True
ENABLE_ARIMA = not ON_RENDER   # skip ARIMA on Render
# ===============================================
def ARIMA_ALGO(df):
    """Stub if ARIMA disabled on Render."""
    return 0.0, "N/A"
# Load Pretrained Models
if ENABLE_LSTM:
    lstm_model = load_model("models/lstm_model.h5")
    lstm_scaler = joblib.load("models/lstm_scaler.pkl")
    lstm_rmse = joblib.load("models/lstm_rmse.pkl")
    try:
        lstm_model.predict(np.zeros((1, 7, 1)))  # warm-up
    except Exception as e:
        print("LSTM warm-up failed:", e)

if ENABLE_XGB:
    xgb_model = joblib.load("models/xgb_model.pkl")
    xgb_scaler = joblib.load("models/xgb_scaler.pkl")
    xgb_rmse = joblib.load("models/xgb_rmse.pkl")

if ENABLE_LR:
    lr_model = joblib.load("models/lr_model.pkl")
    lr_scaler = joblib.load("models/lr_scaler.pkl")
    lr_rmse = joblib.load("models/lr_rmse.pkl")

API_KEY = os.getenv("TWELVE_DATA_KEY")

# ---------- Utility ----------
def save_plot(path):
    try:
        if not ON_RENDER:
            plt.savefig(path)
    finally:
        plt.close()

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
    try:
        url = f'https://api.twelvedata.com/time_series?symbol={quote}&interval=1day&outputsize=2000&apikey={API_KEY}'
        response = requests.get(url, timeout=15)
        data = response.json()
        if 'values' not in data:
            print("TwelveData raw response:", data)
            raise ValueError("Invalid TwelveData response")

        df = pd.DataFrame(data['values'])
        df['date'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('date')
        for col in ['open','high','low','close','volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['adj close'] = df['close']
        df = df.dropna()
        return df.tail(600).reset_index(drop=True)
    except Exception as e:
        print("TwelveData Error:", e)
        return pd.DataFrame()

def get_current_price(symbol):
    try:
        url = f'https://api.twelvedata.com/price?symbol={symbol}&apikey={API_KEY}'
        data = requests.get(url, timeout=10).json()
        return round(float(data.get("price", "nan")), 2)
    except Exception:
        return "Unavailable"

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ---------- Models ----------
def LSTM_ALGO(df):
    print("➡️ Running LSTM...")
    df = df.tail(300).reset_index(drop=True)
    closes = df['close'].values.reshape(-1, 1)
    scaled = lstm_scaler.transform(closes)

    X, y = [], []
    for i in range(7, len(scaled)):
        X.append(scaled[i-7:i, 0]); y.append(scaled[i, 0])
    if not X: return 0.0, lstm_rmse, []

    X = np.array(X).reshape((-1, 7, 1))
    y_pred = lstm_scaler.inverse_transform(lstm_model.predict(X, verbose=0)).flatten()
    y_true = lstm_scaler.inverse_transform(np.array(y).reshape(-1, 1)).flatten()

    plt.figure(figsize=(10, 3), dpi=60)
    plt.plot(y_true[-60:], label="Actual")
    plt.plot(y_pred[-60:], label="Predicted")
    plt.legend()
    save_plot("static/LSTM.png")

    # forecast
    last_seq = scaled[-7:].reshape((1, 7, 1))
    forecast = []
    for _ in range(7):
        pred_scaled = lstm_model.predict(last_seq, verbose=0)
        pred_price = lstm_scaler.inverse_transform(pred_scaled)[0][0]
        forecast.append(pred_price)
        last_seq = np.append(last_seq[0, 1:, 0], pred_scaled[0, 0]).reshape((1, 7, 1))
    print("✅ LSTM done.")
    return forecast[0], lstm_rmse, forecast

def XGBOOST_ALGO(df):
    print("➡️ Running XGBoost...")
    df = df.tail(300).reset_index(drop=True)
    df['Return'] = df['close'].pct_change()
    for lag in range(1, 31):
        df[f'lag_{lag}'] = df['close'].shift(lag)
        df[f'return_lag_{lag}'] = df['Return'].shift(lag)
    df['MA7'] = df['close'].rolling(7).mean()
    df['MA21'] = df['close'].rolling(21).mean()
    df['EMA'] = df['close'].ewm(span=20).mean()
    df['Volatility'] = df['close'].rolling(7).std()
    df['RSI'] = compute_rsi(df['close'], 14)
    df = df.dropna().reset_index(drop=True)

    if len(df) < 50: return df, 0.0, [], 0.0, xgb_rmse
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]
    X_test = test_df.drop(columns=['date','adj close'], errors='ignore').values
    pred_returns = xgb_model.predict(xgb_scaler.transform(X_test))
    pred_prices = test_df['close'].shift(1).values * (1 + pred_returns)

    plt.figure(figsize=(10, 3), dpi=60)
    plt.plot(test_df['close'].values[-60:], label="Actual")
    plt.plot(pred_prices[-60:], label="Predicted")
    plt.legend()
    save_plot("static/XGB.png")

    # forecast
    forecast, last_close = [], df['close'].iloc[-1]
    base_row = df.drop(columns=['date','adj close'], errors='ignore').iloc[-1:].copy()
    for _ in range(7):
        pred_return = xgb_model.predict(xgb_scaler.transform(base_row.values))[0]
        next_price = last_close * (1 + pred_return)
        forecast.append(float(next_price))
        last_close = next_price
        base_row.iloc[0]['close'], base_row.iloc[0]['Return'] = next_price, pred_return
    print("✅ XGBoost done.")
    return df, forecast[0], forecast, float(np.mean(forecast)), xgb_rmse

def LIN_REG_ALGO(df):
    print("➡️ Running Linear Regression...")
    df = df.tail(300).reset_index(drop=True)
    df["MA7"] = df["close"].rolling(7).mean()
    df["MA21"] = df["close"].rolling(21).mean()
    df["Return"] = df["close"].pct_change()
    df = df.dropna().reset_index(drop=True)

    forecast_set = lr_model.predict(lr_scaler.transform(df[["close","MA7","MA21","Return"]].tail(7)))
    lr_pred = forecast_set[0][0]; mean = float(forecast_set.mean())

    split_idx = int(len(df) * 0.8)
    X_test = df[["close","MA7","MA21","Return"]].values[split_idx:]
    y_test = df["close"].shift(-7).dropna().values[split_idx:]
    if len(y_test) > 0:
        y_pred_test = lr_model.predict(lr_scaler.transform(X_test))
        plt.figure(figsize=(10, 3), dpi=60)
        plt.plot(y_test[-60:], label="Actual")
        plt.plot(y_pred_test[-60:], label="Predicted")
        plt.legend()
        save_plot("static/LR.png")
    print("✅ Linear Regression done.")
    return df, lr_pred, forecast_set, mean, lr_rmse

# ---------- Flask route ----------
@app.route('/insertintotable', methods=['POST'])
def insertintotable():
    quote = request.form.get('nm')
    if not quote: return redirect(url_for('index'))

    df = get_historical(quote)
    if df.empty: return render_template("index.html", error=True)
    current_price = get_current_price(quote)

    if ENABLE_ARIMA: arima_pred, error_arima = ARIMA_ALGO(df.copy())
    else: arima_pred, error_arima = 0.0, "N/A"

    lstm_pred, error_lstm, lstm_forecast = (LSTM_ALGO(df.copy()) if ENABLE_LSTM else (0.0,"N/A",[]))
    _, lr_pred, forecast_set, mean, error_lr = (LIN_REG_ALGO(df.copy()) if ENABLE_LR else (df,0.0,[],0.0,"N/A"))
    _, xgb_pred, xgb_forecast, _, error_xgb = (XGBOOST_ALGO(df.copy()) if ENABLE_XGB else (df,0.0,[],0.0,"N/A"))

    today_stock = df.iloc[-1:].round(2)
    idea, decision = ("N/A","Insufficient Data") if today_stock.empty else (
        ("FALL","SELL") if today_stock.iloc[-1]['close']>mean else ("RISE","BUY"))

    return render_template("results.html",
        quote=quote, arima_pred=arima_pred, lstm_pred=lstm_pred, lr_pred=lr_pred, xgb_pred=xgb_pred,
        open_s=today_stock['open'].to_string(index=False),
        close_s=today_stock['close'].to_string(index=False),
        adj_close=today_stock['adj close'].to_string(index=False),
        high_s=today_stock['high'].to_string(index=False),
        low_s=today_stock['low'].to_string(index=False),
        vol=today_stock['volume'].to_string(index=False),
        current_price=current_price,
        forecast_set=list(forecast_set.flatten()) if hasattr(forecast_set,"flatten") else list(forecast_set),
        xgb_forecast=xgb_forecast, forecast_set_lstm=lstm_forecast,
        error_lr=error_lr, error_lstm=error_lstm, error_arima=error_arima, error_xgb=error_xgb,
        idea=idea, decision=decision)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
