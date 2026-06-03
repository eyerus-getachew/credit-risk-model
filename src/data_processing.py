import pandas as pd
import numpy as np

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

        # Convert datetime
        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"]
        )

        # Time features
        df["hour"] = df["TransactionStartTime"].dt.hour
        df["day"] = df["TransactionStartTime"].dt.day
        df["month"] = df["TransactionStartTime"].dt.month
        df["year"] = df["TransactionStartTime"].dt.year

        # Snapshot date
        snapshot_date = df["TransactionStartTime"].max()

        # Aggregate customer features
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

        # Recency feature
        customer_df["recency_days"] = (
            snapshot_date
            - customer_df["last_transaction"]
        ).dt.days

        customer_df.drop(
            columns=["last_transaction"],
            inplace=True
        )

        customer_df["std_amount"] = (
            customer_df["std_amount"]
            .fillna(0)
        )

        return customer_df


def load_data(filepath):
    """
    Load raw data.
    """
    return pd.read_csv(filepath)


def create_rfm_table(df):
    """
    Create RFM metrics.
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
    Create high-risk proxy label.
    """

    cluster_summary = (
        rfm.groupby("cluster")
        [
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
        .mean()
    )

    print("\nCluster Summary")
    print(cluster_summary)

    high_risk_cluster = (
        cluster_summary["Frequency"]
        .idxmin()
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

    # Load raw data
    df = load_data(
        "data/raw/data.csv"
    )

    # Customer-level features
    aggregated_df = (
        CustomerAggregator()
        .fit_transform(df)
    )

    print(
        f"Aggregated Shape: {aggregated_df.shape}"
    )

    # Create RFM table
    rfm = create_rfm_table(df)

    # Cluster customers
    rfm = cluster_customers(rfm)

    # Create proxy target
    rfm = create_proxy_target(rfm)

    # Merge target
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

    # Save final dataset
    final_df.to_csv(
        "data/processed/processed_data.csv",
        index=False
    )

    print(
        "\nProcessed dataset saved to:"
    )

    print(
        "data/processed/processed_data.csv"
    )