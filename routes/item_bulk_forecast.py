from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

router = APIRouter()

class ItemForecast(BaseModel):
    item_id: str
    predicted_average_quantity: float
    rmse: float

@router.get("/bulk-item-forecast", response_model=list[ItemForecast])
def bulk_item_forecast():
    df = pd.read_csv("data/invoice_customer_data.csv")
    df["date"] = pd.to_datetime(df["date"])

    all_items = df["item_id"].unique()
    results = []

    for item_id in all_items:
        df_item = df[df["item_id"] == item_id]
        daily_sales = df_item.groupby("date")["quantity"].sum().asfreq("D").fillna(0)

        if len(daily_sales) < 40 or daily_sales.sum() == 0:
            continue

        try:
            # Split into training and test for RMSE
            train = daily_sales[:-30]
            test = daily_sales[-30:]

            model_eval = ARIMA(train, order=(5, 1, 2)).fit()
            pred_test = model_eval.forecast(steps=30)
            rmse = float(np.sqrt(((pred_test - test) ** 2).mean()))

            # Final model with all data
            model_final = ARIMA(daily_sales, order=(5, 1, 2)).fit()
            forecast = model_final.forecast(steps=30).clip(lower=0)
            avg_quantity = float(forecast.mean())

            results.append({
                "item_id": item_id,
                "predicted_average_quantity": round(avg_quantity, 2),
                "rmse": round(rmse, 2)
            })
        except:
            continue

    return results
