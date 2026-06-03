# Credit Risk Probability Model for Alternative Data

## Project Overview

Bati Bank is partnering with an eCommerce platform to introduce a Buy Now, Pay Later (BNPL) service. To support responsible lending decisions, the bank requires a credit risk scoring system capable of estimating the likelihood that a customer will default on future credit obligations.

The challenge is that the available dataset contains transaction histories but does not include a direct loan default label. Therefore, this project develops an end-to-end credit risk modeling pipeline that uses alternative behavioral data to construct a proxy target variable and predict customer risk.

The final solution will include:

- Customer behavioral feature engineering
- Proxy target variable creation using RFM analysis
- Credit risk model development and evaluation
- Model tracking with MLflow
- REST API deployment using FastAPI
- Containerization with Docker
- CI/CD automation using GitHub Actions

---

# Business Problem

Traditional credit scoring models rely on historical loan repayment information. However, many customers interacting with digital financial services may not have formal credit histories.

This project leverages alternative transaction data from the Xente eCommerce platform to estimate creditworthiness and support Buy Now, Pay Later lending decisions.

The objective is to identify customers who are likely to represent higher credit risk and generate a risk score that can be integrated into Bati Bank's lending workflow.

---

# Dataset Description

The dataset contains transaction-level records collected from the Xente platform.

Each record contains information about:

- Customer identifiers
- Transaction timestamps
- Transaction amounts
- Product categories
- Providers
- Payment channels
- Fraud indicators

The data is recorded at the transaction level and therefore requires aggregation into customer-level behavioral features before modeling.

---

# Credit Scoring Business Understanding

## 1. Basel II and Model Interpretability

The Basel II Capital Accord requires financial institutions to maintain transparent, explainable, and well-documented risk measurement systems. Since credit scores directly influence lending decisions, institutions must be able to justify how risk estimates are produced.

Basel II influences this project in several ways:

- All feature engineering steps are documented and reproducible.
- Customer behavior is transformed into measurable risk indicators.
- Model development decisions are traceable and auditable.
- Results can be monitored and validated over time.
- Model outputs can be explained to both regulators and internal stakeholders.

Because of these requirements, model interpretability is an important consideration alongside predictive performance.

---

## 2. Why a Proxy Target Variable is Necessary

The provided dataset contains transaction behavior but does not contain a direct default label.

Supervised machine learning requires a target variable. Therefore, a proxy target must be created from observable customer behavior.

This project uses RFM analysis:

- Recency
- Frequency
- Monetary Value

Customers are segmented according to their engagement patterns using K-Means clustering.

The least-engaged customer segment is assumed to represent higher credit risk and is assigned:

```text
is_high_risk = 1
```

All remaining customers are assigned:

```text
is_high_risk = 0
```

This proxy target enables model training despite the absence of historical default information.

### Risks of Proxy-Based Prediction

While useful, proxy targets introduce several limitations:

- Behavioral inactivity may not indicate actual default risk.
- Some customers may be incorrectly classified.
- The proxy may contain hidden biases.
- Good performance against the proxy does not guarantee prediction of true future defaults.

These risks should be considered when deploying the model in production.

---

## 3. Interpretable vs High-Performance Models

Credit risk modeling often involves balancing predictive performance and regulatory transparency.

### Logistic Regression with WoE

Advantages:

- Highly interpretable
- Easy to explain
- Common in traditional credit scoring
- Supports scorecard development

Disadvantages:

- Limited ability to model complex relationships
- Lower predictive power in some scenarios

### Gradient Boosting Models

Advantages:

- Strong predictive performance
- Captures nonlinear patterns
- Handles complex interactions

Disadvantages:

- Reduced interpretability
- More difficult to explain individual decisions
- Requires additional explainability techniques

### Project Approach

Both interpretable and high-performance models will be evaluated.

Final model selection will consider:

- Predictive performance
- Interpretability
- Regulatory compliance
- Operational maintainability

rather than accuracy alone.

---

# Project Structure

```text
credit-risk-model/

├── .github/workflows/
│   └── ci.yml

├── data/
│   ├── raw/
│   └── processed/

├── notebooks/
│   └── eda.ipynb

├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py

├── tests/
│   └── test_data_processing.py

├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Tasks Completed

## Task 1 — Credit Risk Business Understanding

### Objectives

- Understand credit risk fundamentals.
- Study Basel II regulatory requirements.
- Understand alternative credit scoring approaches.
- Evaluate interpretability versus performance trade-offs.

### Deliverables

- Project repository initialized.
- Repository structure created.
- Business understanding documented.

---

## Task 2 — Exploratory Data Analysis (EDA)

### Objectives

- Explore transaction-level data.
- Identify quality issues.
- Understand customer behavior.
- Generate feature engineering hypotheses.

### Analysis Performed

- Dataset overview
- Data type inspection
- Summary statistics
- Missing value analysis
- Numerical feature distributions
- Categorical feature distributions
- Correlation analysis
- Outlier detection

### Key Insights

#### 1. Customer-Level Aggregation is Essential

Credit decisions are made at the customer level while the dataset is recorded at the transaction level.

#### 2. Customer Behavior is Highly Heterogeneous

Transaction frequency and spending patterns vary substantially across customers.

#### 3. Transaction Amounts are Strongly Skewed

Several features contain extreme values and require robust aggregation techniques.

#### 4. Temporal Features Provide Valuable Signals

Transaction timestamps contain useful behavioral information.

#### 5. RFM Segmentation Appears Promising

Customer engagement patterns support the creation of a behavioral risk proxy.

---

## Task 3 — Feature Engineering

### Objectives

Transform raw transaction data into a customer-level analytical dataset.

### Features Engineered

#### Aggregate Features

- Transaction Count
- Total Transaction Amount
- Average Transaction Amount
- Standard Deviation of Transaction Amount
- Maximum Transaction Amount
- Minimum Transaction Amount

#### Temporal Features

- Average Transaction Hour
- Average Transaction Day
- Average Transaction Month

#### Risk Features

- Fraud Count
- Recency Days

### Data Processing Pipeline

Implemented using:

- Scikit-Learn Pipelines
- Custom Transformers
- StandardScaler
- Missing Value Handling

### Output

Customer-level processed dataset ready for modeling.

---

## Task 4 — Proxy Target Variable Engineering

### Objective

The dataset does not contain a direct loan default label, which is required for supervised credit risk modeling. Therefore, a proxy target variable was engineered using customer behavioral data. The goal is to identify customers who exhibit low engagement patterns that may indicate a higher likelihood of future credit risk.

---

### Why a Proxy Target is Needed

Traditional credit scoring models are trained using historical repayment outcomes, such as whether a borrower defaulted on a loan. Since the Xente transaction dataset contains no loan performance information, a proxy target must be created from observable customer behavior.

This approach is commonly used in alternative credit scoring systems where traditional credit bureau data or repayment history is unavailable.

---

### Methodology

### Step 1: Calculate RFM Metrics

Customer behavior was summarized using the RFM framework:

| Metric | Description |
|----------|-------------|
| Recency | Number of days since the customer's most recent transaction |
| Frequency | Total number of transactions completed by the customer |
| Monetary | Total transaction value generated by the customer |

A snapshot date was defined as one day after the latest transaction date in the dataset to ensure consistent recency calculations across all customers.

---

### Step 2: Customer Segmentation

Customers were segmented into behavioral groups using the K-Means clustering algorithm.

#### Preprocessing

Before clustering:

- RFM features were standardized using `StandardScaler`
- The number of clusters was set to **3**
- `random_state=42` was used to ensure reproducibility

#### Clustering Process

K-Means was applied to the standardized RFM features to identify groups of customers with similar transaction behavior.

The resulting clusters represented distinct customer engagement profiles.

---

### Step 3: High-Risk Cluster Identification

Cluster profiles were analyzed by examining their average:

- Recency
- Frequency
- Monetary Value

The cluster characterized by:

- Higher recency (less recent activity)
- Lower transaction frequency
- Lower transaction value

was identified as the least-engaged customer segment.

These customers demonstrate lower platform engagement and are therefore considered more likely to represent elevated credit risk.

---

### Step 4: Proxy Target Creation

A binary target variable named `is_high_risk` was created.

| Value | Meaning |
|---------|---------|
| 1 | High-risk customer |
| 0 | Low-risk customer |

Customers belonging to the least-engaged cluster were assigned:

```text
is_high_risk = 1
```

All remaining customers were assigned:

```text
is_high_risk = 0
```

---

### Results

The clustering process produced a meaningful separation between higher-risk and lower-risk customer groups.

Target distribution:

| Label | Count |
|---------|---------|
| Low Risk (0) | 2316 |
| High Risk (1) | 1426 |

This distribution provides a sufficiently balanced target variable for downstream model training and evaluation.

---

### Business Justification

The proxy target is based on the assumption that customers who:

- Transact infrequently
- Generate lower monetary value
- Remain inactive for longer periods

are more likely to exhibit repayment risk than highly active customers.

Although customer inactivity is not equivalent to actual loan default, it provides a practical and data-driven approximation of credit risk when repayment history is unavailable.

---

### Limitations

Several limitations should be acknowledged:

- Customer inactivity does not necessarily imply future default.
- Some customers may transact infrequently while remaining financially reliable.
- The target is derived from behavioral assumptions rather than observed repayment outcomes.

Therefore, the resulting model should be interpreted as predicting behavioral credit risk rather than true default probability.

Future model iterations should replace the proxy target with actual loan repayment outcomes once such data becomes available.

---

### Output

The final processed dataset contains:

- Customer-level aggregated features
- RFM-derived behavioral indicators
- Proxy target variable (`is_high_risk`)

The dataset is automatically generated by:

```text
src/data_processing.py
```

and stored in:

```text
data/processed/processed_data.csv
```
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
# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Jupyter Notebook
- Git
- GitHub

---

# Future Work

- Weight of Evidence (WoE) transformation
- Information Value (IV) analysis
- Model training and comparison
- Hyperparameter tuning
- MLflow experiment tracking
- FastAPI deployment
- Docker containerization
- CI/CD automation
- Model monitoring

---
