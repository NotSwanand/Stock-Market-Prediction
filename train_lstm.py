import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import joblib, math

# Load historical stock data
df = pd.read_csv("AAPL.csv")   # replace with your stock file
data = df['close'].values.reshape(-1, 1)

# Scale to 0–1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Train-test split
split = int(len(scaled_data) * 0.8)
train, test = scaled_data[:split], scaled_data[split:]

# Create sequences (7 days → next day)
X_train, y_train = [], []
for i in range(7, len(train)):
    X_train.append(train[i-7:i, 0])
    y_train.append(train[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(7, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Prepare test set
X_test, y_test = [], []
for i in range(7, len(test)):
    X_test.append(test[i-7:i, 0])
    y_test.append(test[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Predictions
predicted_scaled = model.predict(X_test)

# Inverse transform to dollars
predicted_real = scaler.inverse_transform(predicted_scaled)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

# Compute RMSE in dollars
rmse = math.sqrt(mean_squared_error(y_test_real, predicted_real))
print("LSTM RMSE ($):", rmse)

# Save model, scaler, and RMSE
model.save("models/lstm_model.h5")
joblib.dump(scaler, "models/lstm_scaler.pkl")
joblib.dump(rmse, "models/lstm_rmse.pkl")
