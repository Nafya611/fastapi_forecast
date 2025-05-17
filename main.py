from fastapi import FastAPI
from routes import Customer_forecast,items_forecast

app = FastAPI()

app.include_router(Customer_forecast.router, prefix="/customer", tags=["customer"])
app.include_router(items_forecast.router, prefix="/items", tags=["items-forecast"])

