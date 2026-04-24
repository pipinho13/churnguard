import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
import mlflow
import mlflow.sklearn
import joblib
import os

MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME    = "churnguard"
MODEL_DIR          = "models"
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)

def load_and_preprocess(url: str):
    df = pd.read_csv(url)

    df.drop(columns=["customerID"], inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"
    ]
    for col in binary_cols:
        df[col] = (df[col] == "Yes").astype(int)

    df["gender"] = (df["gender"] == "Male").astype(int)

    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaymentMethod"
    ]
    le = LabelEncoder()
    for col in multi_cols:
        df[col] = le.fit_transform(df[col])

    return df

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_and_preprocess(DATA_URL)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    params = {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": 42,
    }

    with mlflow.start_run():
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train_sc, y_train)

        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc",  auc)
        mlflow.sklearn.log_model(model, "model")

        print(f"\nAccuracy : {acc:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

    joblib.dump(model,  os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.pkl"))
    print(f"\nArtifacts saved to {MODEL_DIR}/")

if __name__ == "__main__":
    train()