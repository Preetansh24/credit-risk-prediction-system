import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("saved_models/best_model_deployment.pkl")

model_data = load_model()
model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["feature_names"]
encoders = model_data.get("label_encoders", {})

st.title("💳 Credit Risk Prediction System")
st.success(f"Model Loaded: {model_data['model_name']}")

st.header("Applicant Details")

income = st.number_input("Annual Income", min_value=1000, value=75000)
employment = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
loan_amount = st.number_input("Loan Amount", min_value=1000, value=35000)
loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])
credit_util = st.slider("Credit Utilization", 0.0, 1.0, 0.45)
prev_defaults = st.number_input("Previous Defaults", min_value=0, value=0)
age = st.number_input("Age", min_value=18, value=35)
marital = st.selectbox("Marital Status", ["Single", "Married"])
dependents = st.number_input("Number of Dependents", min_value=0, value=2)

def preprocess_input():
    df = pd.DataFrame([{
        "Income_annual": income,
        "Employment_Status": employment,
        "Loan_Amount": loan_amount,
        "Loan_Term_Months": loan_term,
        "Credit_Utilization": credit_util,
        "Previous_Defaults_Count": prev_defaults,
        "Age": age,
        "Marital_Status": marital,
        "Num_Dependents": dependents
    }])

    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))

    df["Debt_to_Income_Ratio"] = df["Loan_Amount"] / (df["Income_annual"] + 1)
    df["Monthly_Payment_Burden"] = (df["Loan_Amount"] / df["Loan_Term_Months"]) / (df["Income_annual"]/12 + 1)
    df["Credit_Risk_Score"] = df["Credit_Utilization"] * 0.6 + df["Previous_Defaults_Count"] * 0.4
    df["Income_per_Dependent"] = df["Income_annual"] / (df["Num_Dependents"] + 1)
    df["Loan_to_Income"] = df["Loan_Amount"] / df["Income_annual"]

    for f in features:
        if f not in df.columns:
            df[f] = 0

    df = df[features]
    return scaler.transform(df)

if st.button("🔍 Predict Credit Risk"):
    X = preprocess_input()
    risk = model.predict_proba(X)[0][1]

    st.subheader("Prediction Result")

    if risk >= 0.6:
        st.error(f"🚨 HIGH RISK — Default Probability: {risk:.2%}")
    elif risk >= 0.4:
        st.warning(f"⚠️ MEDIUM RISK — Default Probability: {risk:.2%}")
    else:
        st.success(f"✅ LOW RISK — Default Probability: {risk:.2%}")
