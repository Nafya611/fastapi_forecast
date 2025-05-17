from fastapi import FastAPI
from routes import Customer_forecast,Item_forecast,item_bulk_forecast

app = FastAPI()

app.include_router(Customer_forecast.router, prefix="/customer", tags=["customer"])
app.include_router(Item_forecast.router, prefix="/item", tags=["item"])
app.include_router(item_bulk_forecast.router, prefix="/items", tags=["bulk-item-forecast"])

