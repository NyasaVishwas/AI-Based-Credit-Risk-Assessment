import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Load models and scaler
try:
    rf_model = joblib.load("models/random_forest.pkl")
    xgb_model = joblib.load("models/xgboost.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()
    
# Session state for storing prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

st.title("💳 AI-Based Credit Risk Assessment")

# Select model
st.markdown("### 🔍 Select Prediction Model:")
model_option = st.selectbox("Model", ["Random Forest", "XGBoost"])
model = rf_model if model_option == "Random Forest" else xgb_model

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
        if model_option == "Random Forest":
            prediction = rf_model.predict(input_scaled)[0]
            probability = rf_model.predict_proba(input_scaled)[0][1]
        else:
            prediction = xgb_model.predict(input_scaled)[0]
            probability = xgb_model.predict_proba(input_scaled)[0][1]

        if prediction == 0:
            st.success(f"✅ Approved: Low Risk (Model: {model_option})")
        else:
            st.error(f"❌ Rejected: High Risk (Model: {model_option})")

        # Show pie chart of risk probability
        fig, ax = plt.subplots()
        ax.pie([probability, 1 - probability],
               labels=['High Risk', 'Low Risk'],
               colors=['red', 'green'],
               autopct='%1.1f%%',
               startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        # Save to session history
        st.session_state.prediction_history.append({
            "Model": model_option,
            "Age": person_age,
            "Income": person_income,
            "Loan": loan_amnt,
            "Risk (%)": round(probability * 100, 2),
            "Result": "High Risk" if prediction == 1 else "Low Risk"
        })

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        
# Show prediction history chart
if st.session_state.prediction_history:
    hist_df = pd.DataFrame(st.session_state.prediction_history)
    st.markdown("### 📊 Prediction History")
    st.dataframe(hist_df)

    fig2, ax2 = plt.subplots()
    ax2.plot(hist_df['Risk (%)'], marker='o', linestyle='-', color='purple')
    ax2.set_title("Risk Probability Trend")
    ax2.set_ylabel("Risk (%)")
    ax2.set_xlabel("Prediction Index")
    ax2.grid(True)
    st.pyplot(fig2)