import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)

# --------------------------------------------------
# Load Model Safely (Cloud + Local)
# --------------------------------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "best_model_deployment.pkl")
    return joblib.load(MODEL_PATH)

model_data = load_model()
model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["feature_names"]
encoders = model_data.get("label_encoders", {})

# --------------------------------------------------
# Header UI
# --------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">💳 Credit Risk Prediction System</h1>
    <p style="text-align:center;color:gray;">
    AI-powered loan default risk assessment
    </p>
    """,
    unsafe_allow_html=True
)

st.success(f"✅ Model Loaded: **{model_data['model_name']}**")

# --------------------------------------------------
# Input Form
# --------------------------------------------------
st.markdown("### 📋 Applicant Details")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income", min_value=1000, value=75000)
    employment = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
    loan_amount = st.number_input("Loan Amount", min_value=1000, value=35000)
    loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])

with col2:
    credit_util = st.slider("Credit Utilization", 0.0, 1.0, 0.45)
    prev_defaults = st.number_input("Previous Defaults", min_value=0, value=0)
    age = st.number_input("Age", min_value=18, value=35)
    dependents = st.number_input("Dependents", min_value=0, value=2)

marital = st.selectbox("Marital Status", ["Single", "Married"])

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
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

    # Encode categorical
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))

    # Feature engineering (same as training)
    df["Debt_to_Income_Ratio"] = df["Loan_Amount"] / (df["Income_annual"] + 1)
    df["Monthly_Payment_Burden"] = (df["Loan_Amount"] / df["Loan_Term_Months"]) / (df["Income_annual"]/12 + 1)
    df["Credit_Risk_Score"] = df["Credit_Utilization"] * 0.6 + df["Previous_Defaults_Count"] * 0.4
    df["Income_per_Dependent"] = df["Income_annual"] / (df["Num_Dependents"] + 1)
    df["Loan_to_Income"] = df["Loan_Amount"] / df["Income_annual"]

    # Ensure feature order
    for f in features:
        if f not in df.columns:
            df[f] = 0

    df = df[features]
    return scaler.transform(df), df

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.markdown("---")
if st.button("🔍 Predict Credit Risk", use_container_width=True):
    X_scaled, df_final = preprocess_input()
    prob_default = model.predict_proba(X_scaled)[0][1]

    st.markdown("## 📊 Prediction Result")

    if prob_default >= 0.6:
        st.error(f"🚨 **HIGH RISK**\n\nDefault Probability: **{prob_default:.2%}**")
        decision = "REJECT"
    elif prob_default >= 0.4:
        st.warning(f"⚠️ **MEDIUM RISK**\n\nDefault Probability: **{prob_default:.2%}**")
        decision = "MANUAL REVIEW"
    else:
        st.success(f"✅ **LOW RISK**\n\nDefault Probability: **{prob_default:.2%}**")
        decision = "APPROVE"

    st.info(f"📌 **Recommendation:** {decision}")

    # --------------------------------------------------
    # Optional Explanation Section
    # --------------------------------------------------
    with st.expander("📈 Why this decision? (Feature Importance)"):
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False).head(10)

            st.bar_chart(importance.set_index("Feature"))
        else:
            st.write("Feature importance not available for this model type.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center;color:gray;font-size:13px;">
    Built with ❤️ using Streamlit & Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

