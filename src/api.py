import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time
import os

MODEL_DIR = "models"

app = FastAPI(
    title="ChurnGuard API",
    description="Customer churn prediction service",
    version="1.0.0"
)

model         = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

REQUEST_COUNT = Counter(
    "churnguard_requests_total",
    "Total prediction requests",
    ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "churnguard_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)
CHURN_PREDICTED = Counter(
    "churnguard_churn_predicted_total",
    "Total customers predicted as churned"
)


class CustomerFeatures(BaseModel):
    gender: int                = Field(..., ge=0, le=1,    description="0=Female, 1=Male")
    SeniorCitizen: int         = Field(..., ge=0, le=1)
    Partner: int               = Field(..., ge=0, le=1)
    Dependents: int            = Field(..., ge=0, le=1)
    tenure: int                = Field(..., ge=0,          description="Months with company")
    PhoneService: int          = Field(..., ge=0, le=1)
    MultipleLines: int         = Field(..., ge=0, le=2)
    InternetService: int       = Field(..., ge=0, le=2)
    OnlineSecurity: int        = Field(..., ge=0, le=2)
    OnlineBackup: int          = Field(..., ge=0, le=2)
    DeviceProtection: int      = Field(..., ge=0, le=2)
    TechSupport: int           = Field(..., ge=0, le=2)
    StreamingTV: int           = Field(..., ge=0, le=2)
    StreamingMovies: int       = Field(..., ge=0, le=2)
    Contract: int              = Field(..., ge=0, le=2,    description="0=M2M, 1=1yr, 2=2yr")
    PaperlessBilling: int      = Field(..., ge=0, le=1)
    PaymentMethod: int         = Field(..., ge=0, le=3)
    MonthlyCharges: float      = Field(..., gt=0)
    TotalCharges: float        = Field(..., ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str


@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForestClassifier", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    start = time.time()
    try:
        data = pd.DataFrame([customer.model_dump()])[feature_names]
        data_scaled = scaler.transform(data)

        proba = model.predict_proba(data_scaled)[0][1]
        prediction = bool(proba >= 0.5)

        if proba >= 0.7:
            risk = "high"
        elif proba >= 0.4:
            risk = "medium"
        else:
            risk = "low"

        if prediction:
            CHURN_PREDICTED.inc()

        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

        return PredictionResponse(
            churn_probability=round(float(proba), 4),
            churn_prediction=prediction,
            risk_level=risk
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)