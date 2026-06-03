import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregate transaction-level data into customer-level features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        # Validate required columns
        required_columns = [
            "CustomerId",
            "TransactionId",
            "Amount",
            "TransactionStartTime",
            "FraudResult"
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # Convert datetime
        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"]
        )

        # Extract time features
        df["hour"] = df["TransactionStartTime"].dt.hour
        df["day"] = df["TransactionStartTime"].dt.day
        df["month"] = df["TransactionStartTime"].dt.month
        df["year"] = df["TransactionStartTime"].dt.year

        snapshot_date = (
            df["TransactionStartTime"].max()
        )

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
            snapshot_date
            - customer_df["last_transaction"]
        ).dt.days

        customer_df.drop(
            columns=["last_transaction"],
            inplace=True
        )

        # Customers with one transaction have NaN std
        customer_df["std_amount"] = (
            customer_df["std_amount"]
            .fillna(0)
        )

        return customer_df


def load_data(filepath):
    """
    Load raw transaction data safely.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"File not found: {filepath}"
        )

    return pd.read_csv(filepath)


def create_rfm_table(df):
    """
    Create Recency, Frequency, Monetary metrics.
    """

    data = df.copy()

    data["TransactionStartTime"] = pd.to_datetime(
        data["TransactionStartTime"]
    )

    snapshot_date = (
        data["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        data.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    snapshot_date - x.max()
                ).days
            ),
            Frequency=(
                "TransactionId",
                "count"
            ),
            Monetary=(
                "Amount",
                "sum"
            )
        )
        .reset_index()
    )

    return rfm


def cluster_customers(rfm):
    """
    Cluster customers using KMeans.
    """

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["cluster"] = (
        kmeans.fit_predict(rfm_scaled)
    )

    return rfm


def create_proxy_target(rfm):
    """
    Create proxy high-risk target.
    """

    cluster_summary = (
        rfm.groupby("cluster")
        [["Recency", "Frequency", "Monetary"]]
        .mean()
    )

    print("\nCluster Summary")
    print(cluster_summary)

    # Least engaged customers:
    # lowest frequency and lowest monetary value
    high_risk_cluster = (
        cluster_summary
        .sort_values(
            by=["Frequency", "Monetary"],
            ascending=True
        )
        .index[0]
    )

    print(
        f"\nHigh Risk Cluster: {high_risk_cluster}"
    )

    rfm["is_high_risk"] = np.where(
        rfm["cluster"] == high_risk_cluster,
        1,
        0
    )

    return rfm


if __name__ == "__main__":

    print("\nLoading raw data...")

    df = load_data(
        "data/raw/data.csv"
    )

    print(
        f"Raw Dataset Shape: {df.shape}"
    )

    # Customer-level features
    aggregated_df = (
        CustomerAggregator()
        .fit_transform(df)
    )

    print(
        f"\nAggregated Shape: {aggregated_df.shape}"
    )

    # RFM Table
    rfm = create_rfm_table(df)

    # Clustering
    rfm = cluster_customers(rfm)

    # Target Engineering
    rfm = create_proxy_target(rfm)

    # Merge target back
    final_df = aggregated_df.merge(
        rfm[
            [
                "CustomerId",
                "is_high_risk"
            ]
        ],
        on="CustomerId",
        how="left"
    )

    print(
        f"\nFinal Dataset Shape: {final_df.shape}"
    )

    print(
        "\nTarget Distribution"
    )

    print(
        final_df["is_high_risk"]
        .value_counts()
    )

    # Data quality validation
    print(
        "\nMissing Values Check"
    )

    print(
        final_df.isnull().sum()
    )

    if final_df.isnull().sum().sum() > 0:
        raise ValueError(
            "Dataset still contains missing values."
        )

    # Save processed dataset
    output_path = (
        "data/processed/processed_data.csv"
    )

    final_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nProcessed dataset saved successfully."
    )

    print(
        f"Location: {output_path}"
    )