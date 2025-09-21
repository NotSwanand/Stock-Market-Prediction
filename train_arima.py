import requests
import pandas as pd
import numpy as np
import joblib, math
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima

# ================== Config ==================
API_KEY = "cda2e65c28ee4d44a12f6d37d3e4667b"   # <- put your key here
SYMBOL = "AAPL"                               # default stock to train on

# ================== Fetch Data ==================
def get_historical(symbol, api_key=API_KEY):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=2000&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        raise ValueError(f"Twelve Data Error: {data}")

    df = pd.DataFrame(data["values"])
    df = df.rename(columns={
        "datetime": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna()
    return df

df = get_historical(SYMBOL)

# ================== ARIMA Training ==================
# Use last 2000 rows → 1600 train + 400 test
df = df.tail(2000).reset_index(drop=True)
values = df["close"].values

train_size = 1600
train, test = values[:train_size], values[train_size:]

print(f"Training ARIMA on {train_size} points, testing on {len(test)} points...")

# Auto ARIMA to select best (p,d,q)
print("Finding best ARIMA order...")
auto_model = auto_arima(
    train,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    max_order=10,       # search space
    max_p=6, max_q=6,   # reasonable limits
    d=1,                # first differencing
)
print("Best ARIMA order:", auto_model.order)

# Fit ARIMA with chosen order
model = ARIMA(train, order=auto_model.order)
model_fit = model.fit(method_kwargs={"maxiter": 20})

# Predictions on test set
predictions = model_fit.forecast(steps=len(test))

# RMSE
rmse = math.sqrt(mean_squared_error(test, predictions))
print(f"ARIMA RMSE: {rmse}")

# ================== Plot ==================
plt.figure(figsize=(12,6))
plt.plot(test, label="Actual")
plt.plot(predictions, label=f"ARIMA Predicted {auto_model.order}")
plt.legend()
plt.title(f"ARIMA Model Performance on {SYMBOL} (400 test points)")
plt.savefig("static/ARIMA.png")
plt.close()

# ================== Save Model & Metrics ==================
joblib.dump(model_fit, "models/arima_model.pkl")
joblib.dump(rmse, "models/arima_rmse.pkl")
joblib.dump(auto_model.order, "models/arima_order.pkl")

print("ARIMA model, order, and RMSE saved successfully.")
