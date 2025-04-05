import streamlit as st
import pandas as np
import numpy as np
import joblib

# Load models and scaler
try:
    rf_model = joblib.load("models/random_forest.pkl")
    xgb_model = joblib.load("models/xgboost.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.title("💳 AI-Based Credit Risk Assessment")

# Model Selection moved above
st.markdown("### 🔍 **Select a Model for Prediction**")
model_choice = st.selectbox("Choose a model:", ["Random Forest", "XGBoost"])

st.markdown("### 🧾 Enter Applicant Details:")

# Numeric inputs
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income", min_value=1000, value=50000)
person_emp_length = st.slider("Employment Length (in years)", 0, 50, 5)
loan_amnt = st.number_input("Loan Amount", min_value=100, value=5000)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)
loan_percent_income = st.number_input("Loan Percent of Income", 0.0, 1.0, 0.2)
cb_person_cred_hist_length = st.slider("Credit History Length (in years)", 0, 50, 5)

# Categorical inputs
home_ownership = st.selectbox("Home Ownership", ["OTHER", "OWN", "RENT"])
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
loan_grade = st.selectbox("Loan Grade", ["B", "C", "D", "E", "F", "G"])
default_history = st.selectbox("Default on File", ["No", "Yes"])

# Manual encoding based on training features
features = [
    person_age,
    person_income,
    person_emp_length,
    loan_amnt,
    loan_int_rate,
    loan_percent_income,
    cb_person_cred_hist_length,

    1 if home_ownership == "OTHER" else 0,
    1 if home_ownership == "OWN" else 0,
    1 if home_ownership == "RENT" else 0,

    1 if loan_intent == "EDUCATION" else 0,
    1 if loan_intent == "HOMEIMPROVEMENT" else 0,
    1 if loan_intent == "MEDICAL" else 0,
    1 if loan_intent == "PERSONAL" else 0,
    1 if loan_intent == "VENTURE" else 0,

    1 if loan_grade == "A" else 0,
    1 if loan_grade == "B" else 0,
    1 if loan_grade == "C" else 0,
    1 if loan_grade == "D" else 0,
    1 if loan_grade == "E" else 0,
    1 if loan_grade == "F" else 0,
    1 if loan_grade == "G" else 0,

    1 if default_history == "Yes" else 0
]

# Convert and scale
input_array = np.array([features])
try:
    input_scaled = scaler.transform(input_array)
except Exception as e:
    st.error(f"Error during input scaling: {e}")
    st.stop()

# Predict using selected model
if st.button("Predict Credit Risk"):
    try:
        if model_choice == "Random Forest":
            prediction = rf_model.predict(input_scaled)[0]
        else:
            prediction = xgb_model.predict(input_scaled)[0]

        if prediction == 0:
            st.success(f"✅ Approved: Low Risk (Model: {model_choice})")
        else:
            st.error(f"❌ Rejected: High Risk (Model: {model_choice})")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
