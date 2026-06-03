from fastapi import FastAPI
from src.api.pydantic_models import (
    CustomerData,
    PredictionResponse
)

import pandas as pd
import joblib


app = FastAPI(
    title="Credit Risk API",
    version="1.0.0"
)


model = joblib.load(
    "models/best_model.pkl"
)


@app.get("/")
def home():
    return {
        "message": "Credit Risk API Running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    data: CustomerData
):

    input_df = pd.DataFrame(
        [data.model_dump()]
    )

    prediction = int(
        model.predict(input_df)[0]
    )

    probability = float(
        model.predict_proba(
            input_df
        )[0][1]
    )

    return PredictionResponse(
        prediction=prediction,
        risk_probability=round(
            probability,
            4
        ),
        risk_label=(
            "High Risk"
            if prediction == 1
            else "Low Risk"
        )
    )