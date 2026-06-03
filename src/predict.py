import joblib
import pandas as pd

model = joblib.load(
    "models/best_model.pkl"
)

sample = pd.DataFrame(
    [{
        "transaction_count": 12,
        "total_amount": 5000,
        "avg_amount": 400,
        "std_amount": 100,
        "max_amount": 700,
        "min_amount": 50,
        "avg_hour": 12,
        "avg_day": 15,
        "avg_month": 6,
        "fraud_count": 0,
        "recency_days": 5
    }]
)

prediction = model.predict(sample)

print(prediction)