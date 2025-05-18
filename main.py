from fastapi import FastAPI
from routes import Customer_forecast,items_forecast
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.include_router(Customer_forecast.router, prefix="/customer", tags=["customer"])
app.include_router(items_forecast.router, prefix="/items", tags=["items-forecast"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)