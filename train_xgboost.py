import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import joblib, math

# ================== Config ==================
API_KEY = "cda2e65c28ee4d44a12f6d37d3e4667b"   # <- put your key here
SYMBOL = "AAPL"                   # default stock to train on

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
# Daily returns
df["Return"] = df["close"].pct_change()

# Lag features (price and returns)
for lag in range(1, 31):  # last 30 days
    df[f"lag_{lag}"] = df["close"].shift(lag)
    df[f"return_lag_{lag}"] = df["Return"].shift(lag)

# Moving averages & EMA
df["MA7"] = df["close"].rolling(window=7).mean()
df["MA21"] = df["close"].rolling(window=21).mean()
df["EMA"] = df["close"].ewm(span=20, adjust=False).mean()

# Volatility
df["Volatility"] = df["close"].rolling(window=7).std()

# RSI
delta = df["close"].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
ema_up = up.ewm(com=13, adjust=False).mean()
ema_down = down.ewm(com=13, adjust=False).mean()
rs = ema_up / ema_down
df["RSI"] = 100 - (100 / (1 + rs))

df = df.dropna()

# ================== Target ==================
# Predict next day return instead of price
df["target"] = df["Return"].shift(-1)
df = df.dropna()

# ================== Dataset ==================
features = [col for col in df.columns if col not in ["date", "target"]]
X, y = df[features].values, df["target"].values

# Train-test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Scale features
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# ================== Model & Tuning ==================
param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

xgb = XGBRegressor(objective="reg:squarederror", random_state=42)

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=10,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=1,
    n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
print("Best XGBoost Params:", search.best_params_)

# ================== Evaluation ==================
y_pred = best_model.predict(X_test)

# Convert return predictions into actual price predictions
# (compare with actual close prices)
df_test = df.iloc[split:].copy()
df_test["predicted_return"] = y_pred
df_test["predicted_price"] = df_test["close"].shift(1) * (1 + df_test["predicted_return"])

# Align actual vs predicted
actual_prices = df_test["close"].iloc[1:]     # drop first NaN
predicted_prices = df_test["predicted_price"].iloc[1:]

rmse = math.sqrt(mean_squared_error(actual_prices, predicted_prices))
print(f"Improved XGBoost RMSE ($): {rmse}")

# ================== Save Model & Scaler ==================
joblib.dump(best_model, "models/xgb_model.pkl")
joblib.dump(scaler, "models/xgb_scaler.pkl")
joblib.dump(rmse, "models/xgb_rmse.pkl")
