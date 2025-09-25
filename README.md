# 📈 Stock Market Prediction Dashboard

A web-based stock prediction dashboard built with **Flask** that integrates multiple ML/DL models for forecasting stock prices.  
The app provides interactive visualizations, forecasts, and buy/sell recommendations using live stock data.

---

## 🚀 Features

- Fetches **real-time & historical stock data** using [Twelve Data API](https://twelvedata.com/).  
- Multiple prediction models:
  - **Linear Regression**
  - **XGBoost**
  - **LSTM (Deep Learning)**
  - **ARIMA (Statistical Model)**
- 7-day price forecasting with visualization.  
- Automatic **Buy / Sell signal** recommendation.  
- Clean, responsive dashboard UI.  

---

## 🖥️ Deployment Notes

- **Railway Deployment** → All models (LSTM, XGBoost, ARIMA, Linear Regression) run smoothly.  
- **Local Development** → All models also run fully.  
- **Render Deployment** → Due to CPU constraints, some heavy models may be **disabled/reduced** for performance.  

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)  
- **Frontend:** HTML, CSS (Bootstrap + custom styling)  
- **ML/DL Libraries:** TensorFlow/Keras, XGBoost, scikit-learn, statsmodels  
- **Data Source:** Twelve Data API (with Alpha Vantage as fallback)  
- **Deployment:** Railway / Render  

---

## 📂 Project Structure

├── main2.py # Flask backend with ML models
├── templates/
│ ├── index.html # Input form & landing page
│ ├── results.html # Dashboard with results
├── static/
│ ├── index.css # Styles for landing page
│ ├── results.css # Styles for dashboard
│ ├── LR.png, XGB.png… # Generated prediction plots
├── models/
│ ├── lstm_model.h5 # Pretrained LSTM model
│ ├── xgb_model.pkl # Pretrained XGBoost model
│ ├── scalers + rmse.pkl # Supporting files


---

## ⚡ Setup & Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/stock-prediction-dashboard.git
   cd stock-prediction-dashboard

2. API Keys

Replace the apikey in main.py with your own Twelve Data API key

3. Create virtual environment & install dependencies
  ```bash
  python -m venv venv
  source venv/bin/activate   # On Linux/Mac
  venv\Scripts\activate      # On Windows
  pip install -r requirements.txt
```
4. Run Locally
   ```bash
   python main2.py

---

## 📊 Models Used

Linear Regression → Captures trend using moving averages and returns.

XGBoost → Gradient boosting with lag features & technical indicators.

LSTM → Sequential deep learning model for time series forecasting.

## 📌 Future Improvements

Dockerize for smoother deployment.

Add caching layer to reduce API calls.

Improve LSTM architecture for longer forecasts.
ARIMA → Classical statistical model for autoregressive forecasting.

📜 License

MIT License © 2025
