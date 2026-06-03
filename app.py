from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained model
model = joblib.load(
    "models/best_model.pkl"
)

# Create FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predict whether a customer is high risk",
    version="1.0"
)

# Input schema
class CustomerData(BaseModel):
    transaction_count: int
    total_amount: float
    avg_amount: float
    std_amount: float
    max_amount: float
    min_amount: float
    avg_hour: float
    avg_day: float
    avg_month: float
    fraud_count: int
    recency_days: int


@app.get("/")
def home():
    return {
        "message": "Credit Risk API is running"
    }


@app.post("/predict")
def predict(data: CustomerData):

    input_df = pd.DataFrame(
        [{
            "transaction_count": data.transaction_count,
            "total_amount": data.total_amount,
            "avg_amount": data.avg_amount,
            "std_amount": data.std_amount,
            "max_amount": data.max_amount,
            "min_amount": data.min_amount,
            "avg_hour": data.avg_hour,
            "avg_day": data.avg_day,
            "avg_month": data.avg_month,
            "fraud_count": data.fraud_count,
            "recency_days": data.recency_days
        }]
    )

    prediction = model.predict(
        input_df
    )[0]

    probability = model.predict_proba(
        input_df
    )[0][1]

    return {
        "prediction": int(prediction),
        "risk_probability": round(
            float(probability),
            4
        ),
        "risk_label": (
            "High Risk"
            if prediction == 1
            else "Low Risk"
        )
    }