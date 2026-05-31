import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregate transaction-level data into customer-level features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Convert datetime
        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"]
        )

        # Time features
        df["hour"] = df["TransactionStartTime"].dt.hour
        df["day"] = df["TransactionStartTime"].dt.day
        df["month"] = df["TransactionStartTime"].dt.month
        df["year"] = df["TransactionStartTime"].dt.year

        # Snapshot date for Recency calculation
        snapshot_date = df["TransactionStartTime"].max()

        # Customer-level aggregation
        customer_df = (
            df.groupby("CustomerId")
            .agg(
                transaction_count=("TransactionId", "count"),
                total_amount=("Amount", "sum"),
                avg_amount=("Amount", "mean"),
                std_amount=("Amount", "std"),
                max_amount=("Amount", "max"),
                min_amount=("Amount", "min"),
                avg_hour=("hour", "mean"),
                avg_day=("day", "mean"),
                avg_month=("month", "mean"),
                fraud_count=("FraudResult", "sum"),
                last_transaction=("TransactionStartTime", "max"),
            )
            .reset_index()
        )

        # Recency Feature
        customer_df["recency_days"] = (
            snapshot_date - customer_df["last_transaction"]
        ).dt.days

        # Remove helper column
        customer_df.drop(
            columns=["last_transaction"],
            inplace=True
        )

        # Handle customers with only one transaction
        customer_df["std_amount"] = (
            customer_df["std_amount"]
            .fillna(0)
        )

        return customer_df


# Numerical Features
numerical_features = [
    "transaction_count",
    "total_amount",
    "avg_amount",
    "std_amount",
    "max_amount",
    "min_amount",
    "avg_hour",
    "avg_day",
    "avg_month",
    "fraud_count",
    "recency_days",
]


# Numerical Pipeline
numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numerical_features
        )
    ],
    remainder="drop"
)


# Full Pipeline
full_pipeline = Pipeline(
    steps=[
        (
            "customer_aggregation",
            CustomerAggregator()
        ),
        (
            "preprocessing",
            preprocessor
        )
    ]
)


def load_data(filepath):
    """
    Load raw transaction data.
    """
    return pd.read_csv(filepath)


def save_processed_data(data, filepath):
    """
    Save processed dataset.
    """
    pd.DataFrame(data).to_csv(
        filepath,
        index=False
    )


if __name__ == "__main__":

    df = load_data(
        "data/raw/data.csv"
    )

    processed_data = (
        full_pipeline.fit_transform(df)
    )

    print(
        f"Processed dataset shape: {processed_data.shape}"
    )
    processed_df = pd.DataFrame(
    processed_data,
    columns=numerical_features
)
# Save processed dataset
processed_df.to_csv(
    "data/processed/processed_data.csv",
    index=False
)

print(
    "Processed data saved to data/processed/processed_data.csv"
)