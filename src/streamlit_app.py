import streamlit as st
import requests
import plotly.graph_objects as go
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

# API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ChurnGuard",
    page_icon="🛡️",
    layout="wide"
)

st.title("ChurnGuard")
st.caption("Customer churn risk prediction")

with st.sidebar:
    st.header("Customer profile")

    st.subheader("Demographics")
    gender         = st.selectbox("Gender",          ["Female", "Male"])
    senior         = st.selectbox("Senior citizen",  ["No", "Yes"])
    partner        = st.selectbox("Has partner",     ["No", "Yes"])
    dependents     = st.selectbox("Has dependents",  ["No", "Yes"])
    tenure         = st.slider("Tenure (months)", 0, 72, 12)

    st.subheader("Services")
    phone          = st.selectbox("Phone service",    ["No", "Yes"])
    multi_lines    = st.selectbox("Multiple lines",   ["No phone service", "No", "Yes"])
    internet       = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    security       = st.selectbox("Online security",  ["No internet service", "No", "Yes"])
    backup         = st.selectbox("Online backup",    ["No internet service", "No", "Yes"])
    protection     = st.selectbox("Device protection",["No internet service", "No", "Yes"])
    tech_support   = st.selectbox("Tech support",     ["No internet service", "No", "Yes"])
    streaming_tv   = st.selectbox("Streaming TV",     ["No internet service", "No", "Yes"])
    streaming_mov  = st.selectbox("Streaming movies", ["No internet service", "No", "Yes"])

    st.subheader("Billing")
    contract       = st.selectbox("Contract type",    ["Month-to-month", "One year", "Two year"])
    paperless      = st.selectbox("Paperless billing",["No", "Yes"])
    payment        = st.selectbox("Payment method",   [
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                        "Electronic check",
                        "Mailed check"
                     ])
    monthly        = st.number_input("Monthly charges ($)", 0.0, 200.0, 65.0, step=0.5)
    total          = st.number_input("Total charges ($)",   0.0, 10000.0, float(tenure * monthly), step=1.0)

    predict_btn    = st.button("Predict churn risk", type="primary", use_container_width=True)


def encode(mapping, value):
    return mapping[value]

BINARY      = {"No": 0, "Yes": 1}
MULTI3      = {"No phone service": 0, "No": 0, "Yes": 1,
               "No internet service": 0, "DSL": 1, "Fiber optic": 2}
CONTRACT    = {"Month-to-month": 0, "One year": 1, "Two year": 2}
PAYMENT     = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)":   1,
    "Electronic check":          2,
    "Mailed check":              3
}

def build_payload():
    return {
        "gender":           1 if gender == "Male" else 0,
        "SeniorCitizen":    BINARY[senior],
        "Partner":          BINARY[partner],
        "Dependents":       BINARY[dependents],
        "tenure":           tenure,
        "PhoneService":     BINARY[phone],
        "MultipleLines":    MULTI3[multi_lines],
        "InternetService":  MULTI3[internet],
        "OnlineSecurity":   MULTI3[security],
        "OnlineBackup":     MULTI3[backup],
        "DeviceProtection": MULTI3[protection],
        "TechSupport":      MULTI3[tech_support],
        "StreamingTV":      MULTI3[streaming_tv],
        "StreamingMovies":  MULTI3[streaming_mov],
        "Contract":         CONTRACT[contract],
        "PaperlessBilling": BINARY[paperless],
        "PaymentMethod":    PAYMENT[payment],
        "MonthlyCharges":   monthly,
        "TotalCharges":     total,
    }

def risk_color(risk):
    return {"low": "#1D9E75", "medium": "#EF9F27", "high": "#E24B4A"}[risk]

def make_gauge(probability, risk):
    color = risk_color(risk)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 48}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0,  40],  "color": "#E1F5EE"},
                {"range": [40, 70],  "color": "#FAEEDA"},
                {"range": [70, 100], "color": "#FCEBEB"},
            ],
            "threshold": {
                "line":  {"color": color, "width": 3},
                "thickness": 0.8,
                "value": probability * 100
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=20, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#444441"
    )
    return fig


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("How to use")
    st.markdown("""
    1. Fill in the customer profile in the sidebar
    2. Click **Predict churn risk**
    3. Review the risk score and recommendations below
    """)

    st.subheader("API status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.success("API is online")
        else:
            st.error("API returned an error")
    except Exception:
        st.error("Cannot reach the API — make sure uvicorn is running on port 8000")

with col2:
    st.subheader("Quick stats")
    st.metric("Tenure",          f"{tenure} months")
    st.metric("Monthly charges", f"${monthly:.2f}")
    st.metric("Total charges",   f"${total:.2f}")


if predict_btn:
    payload = build_payload()
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        prob  = result["churn_probability"]
        pred  = result["churn_prediction"]
        risk  = result["risk_level"]
        color = risk_color(risk)

        st.divider()
        st.subheader("Prediction result")

        r1, r2, r3 = st.columns(3)
        r1.metric("Churn probability", f"{prob * 100:.1f}%")
        r2.metric("Prediction",        "Will churn" if pred else "Will stay")
        r3.metric("Risk level",        risk.upper())

        st.plotly_chart(make_gauge(prob, risk), use_container_width=True)

        st.subheader("Recommendations")
        if risk == "high":
            st.error("High churn risk detected")
            st.markdown("""
            - Offer a discounted long-term contract upgrade
            - Assign a dedicated account manager
            - Proactively reach out within 48 hours
            """)
        elif risk == "medium":
            st.warning("Medium churn risk")
            st.markdown("""
            - Send a satisfaction survey
            - Highlight underused services they are paying for
            - Consider a loyalty discount on next billing cycle
            """)
        else:
            st.success("Low churn risk — customer appears satisfied")
            st.markdown("""
            - Good candidate for an upsell conversation
            - Consider enrolling in a referral program
            """)

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Is uvicorn running?")
    except Exception as e:
        st.error(f"Prediction failed: {e}")