# Task 5 — Model Training, Experiment Tracking, and Deployment

## Objective

The objective of this task is to develop a structured machine learning workflow for credit risk prediction. This includes model training, hyperparameter tuning, experiment tracking using MLflow, model evaluation, and deployment through a FastAPI application.

---

## Data Preparation

### Dataset

The processed customer-level dataset generated in Task 4 is used as input for model training.

**Input File**

```text
data/processed/processed_data.csv
```

### Features

The following engineered customer-level features are used for training:

- transaction_count
- total_amount
- avg_amount
- std_amount
- max_amount
- min_amount
- avg_hour
- avg_day
- avg_month
- fraud_count
- recency_days

### Target Variable

```text
is_high_risk
```

- 1 → High Risk Customer
- 0 → Low Risk Customer

### Train-Test Split

The dataset is split into training and testing subsets:

- Training Set: 80%
- Testing Set: 20%

A fixed random seed is used to ensure reproducibility.

```python
random_state=42
```

Stratified sampling is applied to preserve class distribution.

---

## Model Selection

Two machine learning algorithms were trained and compared:

### 1. Logistic Regression

A machine learning pipeline was created consisting of:

- StandardScaler
- LogisticRegression

### 2. Random Forest Classifier

An ensemble tree-based classifier was trained using randomized hyperparameter search.

---

## Hyperparameter Tuning

### Logistic Regression

Grid Search Cross Validation was used to identify the optimal regularization parameter.

Parameters explored:

```python
{
    "model__C": [0.01, 0.1, 1, 10]
}
```

### Random Forest

Randomized Search Cross Validation was used to explore the parameter space efficiently.

Parameters explored:

```python
{
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10]
}
```

Cross-validation:

```python
cv=5
```

Scoring metric:

```python
f1
```

---

## Experiment Tracking with MLflow

MLflow was integrated to track all experiments.

### Logged Information

For every model run:

#### Parameters

- Hyperparameters
- Search results

#### Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

#### Artifacts

- Trained model
- Model metadata
- Environment configuration

### Experiment Name

```text
Credit_Risk_Modeling
```

---

## Model Evaluation

The following evaluation metrics were used:

### Accuracy

Measures the proportion of correctly classified observations.

### Precision

Measures how many predicted positive cases are actually positive.

### Recall

Measures how many actual positive cases are correctly identified.

### F1 Score

Harmonic mean of Precision and Recall.

### ROC-AUC

Measures the model's ability to distinguish between classes.

---

## Results

### Logistic Regression

| Metric | Score |
|----------|----------|
| Accuracy | 0.9987 |
| Precision | 0.9965 |
| Recall | 1.0000 |
| F1 Score | 0.9982 |
| ROC-AUC | 0.9999 |

### Random Forest

| Metric | Score |
|----------|----------|
| Accuracy | 0.9933 |
| Precision | 0.9895 |
| Recall | 0.9930 |
| F1 Score | 0.9912 |
| ROC-AUC | 0.9999 |

---

## Best Model

Based on the F1 Score comparison:

```text
Logistic Regression
```

was selected as the best-performing model.

### Best F1 Score

```text
0.9982
```

---

## Model Persistence

The selected model is serialized and stored for inference.

Example:

```text
models/best_model.pkl
```

---

## FastAPI Deployment

A REST API was developed using FastAPI to serve predictions.

### API Endpoint

```http
POST /predict
```

### Sample Request

```json
{
  "transaction_count": 10,
  "total_amount": 50000,
  "avg_amount": 5000,
  "std_amount": 1200,
  "max_amount": 8000,
  "min_amount": 2000,
  "avg_hour": 14,
  "avg_day": 15,
  "avg_month": 6,
  "fraud_count": 0,
  "recency_days": 5
}
```

### Sample Response

```json
{
  "prediction": 0,
  "risk_probability": 0,
  "risk_label": "Low Risk"
}
```

---

## Running the Training Pipeline

```bash
python src/train.py
```

---

## Running the FastAPI Application

```bash
python -m uvicorn app:app --reload
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Project Deliverables

- Customer-level feature engineering
- Train-test data split
- Logistic Regression model
- Random Forest model
- Hyperparameter tuning
- MLflow experiment tracking
- Model artifact logging
- FastAPI prediction service
- Interactive Swagger documentation
- Model evaluation and comparison

---

## Conclusion

A complete machine learning workflow was implemented for credit risk prediction. Multiple models were trained and evaluated using cross-validation and hyperparameter tuning. MLflow was used for experiment tracking, while FastAPI was used to expose the best-performing model through a production-ready prediction API.