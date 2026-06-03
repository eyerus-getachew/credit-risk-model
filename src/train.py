import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/processed_data.csv"
)

print(
    f"Dataset Shape: {df.shape}"
)

# --------------------------------------------------
# Validation
# --------------------------------------------------

if df.empty:
    raise ValueError(
        "Dataset is empty."
    )

if "is_high_risk" not in df.columns:
    raise ValueError(
        "Target column is missing."
    )

if df.isnull().sum().sum() > 0:
    raise ValueError(
        "Dataset contains missing values."
    )

# --------------------------------------------------
# Features & Target
# --------------------------------------------------

X = df.drop(
    columns=[
        "CustomerId",
        "is_high_risk"
    ]
)

y = df["is_high_risk"]

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"\nTraining Shape: {X_train.shape}"
)

print(
    f"Testing Shape: {X_test.shape}"
)

# --------------------------------------------------
# MLflow Setup
# --------------------------------------------------

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

results = []

# ==================================================
# MODEL 1 : LOGISTIC REGRESSION
# ==================================================

print(
    "\nTraining Logistic Regression..."
)

with mlflow.start_run(
    run_name="Logistic_Regression"
):

    logistic_pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    random_state=42,
                    max_iter=5000
                )
            )
        ]
    )

    param_grid = {
        "model__C": [
            0.01,
            0.1,
            1,
            10
        ]
    }

    grid_search = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    best_model = (
        grid_search.best_estimator_
    )

    y_pred = best_model.predict(
        X_test
    )

    y_prob = best_model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    mlflow.log_params(
        grid_search.best_params_
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )

    mlflow.sklearn.log_model(
        best_model,
        "model"
    )

    results.append(
        {
            "model":
            "Logistic Regression",
            "f1":
            f1,
            "model_obj":
            best_model
        }
    )

    print(
        "\nLogistic Regression Results"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

# ==================================================
# MODEL 2 : RANDOM FOREST
# ==================================================

print(
    "\nTraining Random Forest..."
)

with mlflow.start_run(
    run_name="Random_Forest"
):

    rf_model = RandomForestClassifier(
        random_state=42
    )

    rf_params = {
        "n_estimators":
        [100, 200, 300],

        "max_depth":
        [5, 10, 15, None],

        "min_samples_split":
        [2, 5, 10]
    }

    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=rf_params,
        n_iter=10,
        cv=5,
        scoring="f1",
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(
        X_train,
        y_train
    )

    best_rf = (
        random_search.best_estimator_
    )

    y_pred = best_rf.predict(
        X_test
    )

    y_prob = best_rf.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    mlflow.log_params(
        random_search.best_params_
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )

    mlflow.sklearn.log_model(
        best_rf,
        "model"
    )

    results.append(
        {
            "model":
            "Random Forest",
            "f1":
            f1,
            "model_obj":
            best_rf
        }
    )

    print(
        "\nRandom Forest Results"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

# --------------------------------------------------
# Best Model
# --------------------------------------------------

best_result = max(
    results,
    key=lambda x: x["f1"]
)

print(
    "\n================================="
)

print(
    f"Best Model: {best_result['model']}"
)

print(
    f"Best F1 Score: {best_result['f1']:.4f}"
)

print(
    "================================="
)


os.makedirs("models", exist_ok=True)

joblib.dump(
    best_result["model_obj"],
    "models/best_model.pkl"
)
# Register model in MLflow Model Registry

client = MlflowClient()

try:
    client.create_registered_model(
        "CreditRiskModel"
    )
except Exception:
    pass

mlflow.sklearn.log_model(
    best_result["model_obj"],
    artifact_path="best_model",
    registered_model_name="CreditRiskModel"
)

print(
    "\nModel registered as: CreditRiskModel"
)

print("\nBest model saved!")