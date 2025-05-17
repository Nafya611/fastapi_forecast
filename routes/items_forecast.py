from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

router = APIRouter()

class ItemForecast(BaseModel):
    itemId: str
    itemName: str
    predictedAverageQuantity: float
    rmse: float

@router.get("/Items-forecast", response_model=list[ItemForecast])
def bulk_item_forecast():
    df = pd.read_csv("data/invoice_customer_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    all_items = df["ItemId"].unique()
    results = []

    for ItemId in all_items:
        df_item = df[df["ItemId"] == ItemId]
        daily_sales = df_item.groupby("Date")["Quantity"].sum().asfreq("D").fillna(0)

        if len(daily_sales) < 40 or daily_sales.sum() == 0:
            continue

        try:
            # Split into training and test for RMSE
            train = daily_sales[:-30]
            test = daily_sales[-30:]

            model_eval = ARIMA(train, order=(1, 1,1)).fit()
            pred_test = model_eval.forecast(steps=30)
            rmse = float(np.sqrt(((pred_test - test) ** 2).mean()))

            # Final model with all data
            model_final = ARIMA(daily_sales, order=(1, 1, 1)).fit()
            forecast = model_final.forecast(steps=30).clip(lower=0)
            avg_Quantity = float(forecast.mean())

            # Get item name (assumes it is consistent per item_id)
            ItemName = df_item["ItemName"].iloc[0] if "ItemName" in df_item.columns else "Unknown"

            results.append({
                "itemId": ItemId,
                "itemName": ItemName,
                "predictedAverageQuantity": round(avg_Quantity, 2),
                "rmse": round(rmse, 2)
            })
        except:
            continue

    results.sort(key=lambda x: x["predictedAverageQuantity"], reverse=True)
    return results
