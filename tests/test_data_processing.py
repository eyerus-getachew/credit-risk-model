import pandas as pd

from src.data_processing import (
    create_rfm_table,
    create_proxy_target
)


def test_create_rfm_table_returns_expected_columns():

    sample_df = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "TransactionId": [101, 102, 103],
        "Amount": [100, 200, 300],
        "TransactionStartTime": [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03"
        ]
    })

    rfm = create_rfm_table(sample_df)

    expected_columns = [
        "CustomerId",
        "Recency",
        "Frequency",
        "Monetary"
    ]

    assert list(rfm.columns) == expected_columns


def test_create_proxy_target_creates_binary_label():

    sample_rfm = pd.DataFrame({
        "CustomerId": [1, 2, 3],
        "Recency": [100, 10, 50],
        "Frequency": [1, 20, 5],
        "Monetary": [50, 1000, 300],
        "cluster": [0, 1, 2]
    })

    result = create_proxy_target(sample_rfm)

    assert "is_high_risk" in result.columns

    assert set(
        result["is_high_risk"].unique()
    ).issubset({0, 1})