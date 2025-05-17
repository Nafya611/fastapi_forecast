from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

router = APIRouter()

class ForecastResult(BaseModel):
    average_quantity: float
    rmse: float

@router.get("/forecast/{item_id}", response_model=ForecastResult)
def forecast(item_id: str):
    df = pd.read_csv("data/invoice_customer_data.csv")
    df["date"] = pd.to_datetime(df["date"])

    df_item = df[df["item_id"] == item_id]

    if df_item.empty:
        return ForecastResult(average_quantity=0.0, rmse=0.0)

    daily_sales = df_item.groupby("date")["quantity"].sum().asfreq("D").fillna(0)

    if len(daily_sales) < 40:
        return ForecastResult(average_quantity=0.0, rmse=0.0)

    train = daily_sales[:-30]
    test = daily_sales[-30:]

    model_eval = ARIMA(train, order=(5, 1, 2)).fit()
    pred_test = model_eval.forecast(steps=30)
    rmse = float(np.sqrt(mean_squared_error(test, pred_test)))

    model_final = ARIMA(daily_sales, order=(5, 1, 2)).fit()
    forecast = model_final.forecast(steps=30).clip(lower=0)
    avg_quantity = float(forecast.mean())

    return ForecastResult(average_quantity=avg_quantity,rmse=rmse)
