from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

router = APIRouter()

class Prediction(BaseModel):
    item_id: str
    predicted_average_quantity: float

def predict_items_per_customer(customer_id, forecast_days=30, file_path="data/invoice_customer_data.csv"):
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])

    customer_df = df[df["customer_id"] == customer_id]
    if customer_df.empty:
        return []

    predictions = []
    grouped = customer_df.groupby("item_id")

    for item_id, group in grouped:
        daily = group.groupby("date")["quantity"].sum().asfreq("D").fillna(0)

        if len(daily) < 40 or daily.sum() == 0:
            continue

        try:
            model = ARIMA(daily, order=(2, 1, 2)).fit()
            forecast = model.forecast(steps=forecast_days).clip(lower=0)
            avg_quantity = forecast.mean()

            predictions.append({
                "item_id": item_id,
                "predicted_average_quantity": round(avg_quantity, 2)
            })
        except:
            continue

    predictions.sort(key=lambda x: x["predicted_average_quantity"], reverse=True)
    return predictions

@router.get("/forecast/{customer_id}", response_model=list[Prediction])
def forecast_customer_items(customer_id: str):
    predictions = predict_items_per_customer(customer_id)

    if not predictions:
        raise HTTPException(status_code=404, detail="No data found for this customer or not enough history.")

    return predictions
