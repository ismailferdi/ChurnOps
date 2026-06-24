# ChurnOps: End-to-End ML Pipeline with Deployment

Customer churn prediction system built with scikit-learn, MLflow, FastAPI, Docker, SQLite logging, and PSI-based drift monitoring.

---

## Overview

ChurnOps is an end-to-end machine learning project that takes a customer churn prediction model from experimentation to deployment.

The project includes:

* Data preprocessing with scikit-learn ColumnTransformer
* Random Forest and Logistic Regression model training
* MLflow experiment tracking
* FastAPI inference service
* SQLite prediction logging
* Population Stability Index (PSI) drift monitoring
* Docker containerization
* Automated testing with pytest

The model is trained on the Telco Customer Churn dataset and exposes a REST API for real-time predictions.

---

## Features

* Trained churn prediction model
* Serialized preprocessing pipeline
* MLflow experiment tracking
* REST API with FastAPI
* Prediction probability output
* SQLite prediction logging
* PSI drift monitoring
* Docker deployment
* Unit and integration tests

---

## Project Structure

```text
churnops/
├── api/
├── core/
├── data/
├── docker/
├── mlruns/
├── model/
├── monitoring/
├── notebooks/
├── tests/
├── requirements.txt
└── README.md
```

---

## Prerequisites

* Python 3.11+
* Docker Desktop
* Git
* Kaggle account (for dataset download)

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd churnops
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a local environment file:

```bash
cp .env.example .env
```

Update the values inside `.env`:

```env
MODEL_PATH=model/artifacts/model.pkl
PREPROCESSOR_PATH=model/artifacts/preprocessor.pkl
REFERENCE_DISTRIBUTIONS_PATH=model/artifacts/reference_distributions.pkl
PREDICTION_DB_PATH=predictions.db
MLFLOW_TRACKING_URI=mlruns
CORS_ALLOWED_ORIGINS=["http://localhost:3000"]
```

---

## Dataset

Download the Telco Customer Churn dataset from Kaggle and place it in:

```text
data/raw/telco_churn.csv
```

Dataset:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## Model Training

Train all models:

```bash
python -m model.train
```

This will:

* Fit the preprocessing pipeline
* Train Random Forest
* Train Logistic Regression
* Save artifacts
* Log metrics to MLflow

Artifacts are saved to:

```text
model/artifacts/
├── model.pkl
├── preprocessor.pkl
└── reference_distributions.pkl
```

---

## MLflow Tracking

Start MLflow UI:

```bash
mlflow ui --backend-store-uri mlruns/
```

Open:

```text
http://localhost:5000
```

Verify:

* Random Forest run exists
* Logistic Regression run exists
* Metrics are logged correctly
* Random Forest outperforms Logistic Regression

---

## Evaluation

Evaluate the saved model:

```bash
python -m model.evaluate
```

Outputs:

* Classification Report
* ROC-AUC Score

---

## Running the API

Start FastAPI:

```bash
uvicorn api.main:app --reload
```

API available at:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Example Response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

---

### Predict Churn

```http
POST /predict
```

Example Request:

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.9,
  "TotalCharges": 1078.8
}
```

Example Response:

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8421,
  "prediction_label": "Will Churn"
}
```

---

### Drift Metrics

```http
GET /metrics
```

Example Response:

```json
{
  "total_predictions": 500,
  "window_size": 500,
  "drift_metrics": [
    {
      "feature": "MonthlyCharges",
      "psi": 0.07,
      "status": "OK"
    }
  ]
}
```

---

## Architecture

```text
POST /predict
CustomerFeatures JSON
      │
      ▼
┌─────────────┐     ┌───────────────────┐     ┌──────────┐
│  FastAPI    │────▶│  Preprocessor.pkl │────▶│ Model.pkl│
│ /predict    │     │ ColumnTransformer │     │   (RF)   │
└─────────────┘     └───────────────────┘     └──────────┘
      │                                          │
      │      Prediction + Probability            │
      ◀──────────────────────────────────────────┘
      │
      ▼
┌──────────────────┐
│ Prediction Logger│ → predictions.db
└──────────────────┘
                                │
GET /metrics                    ▼
      │      ┌──────────────────────────┐
      └─────▶│      Drift Detector      │
             │     PSI per Feature      │
             └──────────────────────────┘
```

---

## PSI Thresholds

| PSI         | Status  |
| ----------- | ------- |
| < 0.10      | OK      |
| 0.10 - 0.20 | WARNING |
| > 0.20      | ALERT   |

Interpretation:

* OK → Stable distribution
* WARNING → Moderate shift detected
* ALERT → Significant drift detected

---

## Testing

Run all tests:

```bash
pytest tests/ -v
```

---

## Docker

Build containers:

```bash
docker compose -f docker/docker-compose.yml build
```

Start services:

```bash
docker compose -f docker/docker-compose.yml up
```

Services:

* FastAPI → http://localhost:8000
* MLflow → http://localhost:5000

---

## Notes

* Predictions are logged to SQLite for monitoring purposes.
* SQLite is suitable for development and demonstration environments.
* Production deployments should use PostgreSQL or another managed database.
* Model artifacts are mounted as a read-only volume in Docker, allowing model updates without rebuilding the image.
* Docker Compose commands must be executed from the project root directory.

---

## Future Improvements

* Streamlit frontend
* Batch prediction endpoint
* Model versioning
* Cloud deployment
* CI/CD with GitHub Actions
* Slack alerting for drift detection
* Categorical drift monitoring

---

## License

This project is provided for educational and portfolio purposes.