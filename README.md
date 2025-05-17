# 🔮 FastAPI Invoice Item Forecasting API

A FastAPI-based application that forecasts item-level demand using ARIMA time series modeling. Designed to work with sales data containing fields like date, customer ID, item ID, quantity, unit price, and discount.

---

## 📦 Features

- Demand forecasting using ARIMA
- Bulk and individual item forecasting
- JSON API with FastAPI
- Automatically generated interactive API docs

---

## 🛠️ Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/your-username/fastapi-forecast.git
cd fastapi-forecast

### Install Required Dependencies

pip install -r requirements.txt

### 🚀 Run the App Locally

uvicorn main:app --reload

