from pydantic import BaseModel


class CustomerData(BaseModel):
    transaction_count: float
    total_amount: float
    avg_amount: float
    std_amount: float
    max_amount: float
    min_amount: float
    avg_hour: float
    avg_day: float
    avg_month: float
    fraud_count: float
    recency_days: float


class PredictionResponse(BaseModel):
    prediction: int
    risk_probability: float
    risk_label: str