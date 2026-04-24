import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.api import app

client = TestClient(app)

VALID_CUSTOMER = {
    "gender": 0,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 2,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 1,
    "OnlineSecurity": 0,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "Contract": 0,
    "PaperlessBilling": 1,
    "PaymentMethod": 2,
    "MonthlyCharges": 70.35,
    "TotalCharges": 150.30,
}

LOW_RISK_CUSTOMER = {
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 1,
    "tenure": 60,
    "PhoneService": 1,
    "MultipleLines": 1,
    "InternetService": 1,
    "OnlineSecurity": 2,
    "OnlineBackup": 2,
    "DeviceProtection": 2,
    "TechSupport": 2,
    "StreamingTV": 1,
    "StreamingMovies": 1,
    "Contract": 2,
    "PaperlessBilling": 0,
    "PaymentMethod": 0,
    "MonthlyCharges": 45.00,
    "TotalCharges": 2700.00,
}


class TestHealth:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model" in data
        assert "version" in data

    def test_health_status_is_ok(self):
        response = client.get("/health")
        assert response.json()["status"] == "ok"


class TestPredict:
    def test_predict_returns_200(self):
        response = client.post("/predict", json=VALID_CUSTOMER)
        assert response.status_code == 200

    def test_predict_response_structure(self):
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "risk_level" in data

    def test_churn_probability_is_between_0_and_1(self):
        response = client.post("/predict", json=VALID_CUSTOMER)
        prob = response.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_churn_prediction_is_boolean(self):
        response = client.post("/predict", json=VALID_CUSTOMER)
        pred = response.json()["churn_prediction"]
        assert isinstance(pred, bool)

    def test_risk_level_is_valid(self):
        response = client.post("/predict", json=VALID_CUSTOMER)
        risk = response.json()["risk_level"]
        assert risk in ["low", "medium", "high"]

    def test_low_risk_customer_scores_lower_than_high_risk(self):
        high_risk = client.post("/predict", json=VALID_CUSTOMER).json()
        low_risk = client.post("/predict", json=LOW_RISK_CUSTOMER).json()
        assert low_risk["churn_probability"] < high_risk["churn_probability"]

    def test_prediction_consistent_with_probability(self):
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        if data["churn_probability"] >= 0.5:
            assert data["churn_prediction"] is True
        else:
            assert data["churn_prediction"] is False

    def test_risk_level_consistent_with_probability(self):
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        prob = data["churn_probability"]
        risk = data["risk_level"]
        if prob >= 0.7:
            assert risk == "high"
        elif prob >= 0.4:
            assert risk == "medium"
        else:
            assert risk == "low"


class TestValidation:
    def test_missing_field_returns_422(self):
        incomplete = VALID_CUSTOMER.copy()
        del incomplete["tenure"]
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_invalid_gender_value_returns_422(self):
        bad_payload = VALID_CUSTOMER.copy()
        bad_payload["gender"] = 5
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_negative_monthly_charges_returns_422(self):
        bad_payload = VALID_CUSTOMER.copy()
        bad_payload["MonthlyCharges"] = -10.0
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_string_instead_of_int_returns_422(self):
        bad_payload = VALID_CUSTOMER.copy()
        bad_payload["tenure"] = "twelve"
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422


class TestMetrics:
    def test_metrics_endpoint_returns_200(self):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_prometheus_data(self):
        response = client.get("/metrics")
        assert b"churnguard_requests_total" in response.content

    def test_metrics_content_type(self):
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]
