from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

router = APIRouter()

class Prediction(BaseModel):
    itemId: str
    itemName: str
    predicted_average_Quantity: float

def predict_items_per_customer(CustomerId, forecast_days=30, file_path="data/invoice_customer_data.csv"):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])

    customer_df = df[df["CustomerId"] == CustomerId]
    if customer_df.empty:
        return []

    predictions = []
    grouped = customer_df.groupby("ItemId")

    for itemId, group in grouped:
        daily = group.groupby("Date")["Quantity"].sum().asfreq("D").fillna(0)

        if len(daily) < 40 or daily.sum() == 0:
            continue

        try:
            model = ARIMA(daily, order=(2, 1, 2)).fit()
            forecast = model.forecast(steps=forecast_days).clip(lower=0)
            avg_Quantity = forecast.mean()

            itemName = group["ItemName"].iloc[0] if "ItemName" in group.columns else ""
            predictions.append({
                "itemId": itemId,
                "itemName": itemName,
                "predicted_average_Quantity": round(avg_Quantity, 2)
            })
        except:
            continue

    predictions.sort(key=lambda x: x["predicted_average_Quantity"], reverse=True)
    return predictions

@router.get("/forecast/{CustomerId}", response_model=list[Prediction])
def forecast_customer_items(CustomerId: str):
    predictions = predict_items_per_customer(CustomerId)

    if not predictions:
        raise HTTPException(status_code=404, detail="No data found for this customer or not enough history.")

    return predictions
