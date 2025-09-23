import requests
import pandas as pd
import numpy as np
import joblib, math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ================== Config ==================
API_KEY = "cda2e65c28ee4d44a12f6d37d3e4667b"   # replace with your real key
SYMBOL = "AAPL"                       # stock used for training (baseline model)

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

# ================== Feature Engineering ==================
forecast_out = 7
df["MA7"] = df["close"].rolling(window=7).mean()
df["MA21"] = df["close"].rolling(window=21).mean()
df["Return"] = df["close"].pct_change()
df = df.dropna().reset_index(drop=True)

df["target"] = df["close"].shift(-forecast_out)
df = df.dropna().reset_index(drop=True)

features = ["close", "MA7", "MA21", "Return"]
X, y = df[features].values, df["target"].values.reshape(-1, 1)

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ================== Scale + Train ==================
scaler = StandardScaler()
X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

model = LinearRegression().fit(X_train_scaled, y_train)

# ================== Evaluate ==================
y_pred = model.predict(X_test_scaled)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print(f"Linear Regression RMSE: {rmse}")

# ================== Save Model & Scaler ==================
joblib.dump(model, "models/lr_model.pkl")
joblib.dump(scaler, "models/lr_scaler.pkl")
joblib.dump(rmse, "models/lr_rmse.pkl")

print("✅ Linear Regression model, scaler, and RMSE saved successfully.")
