!pip install streamlit
import streamlit as st
import pandas as pd
import pickle

with open("Loanapproval.pkl", "rb") as f:
    model = pickle.load(f)
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")
st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details to check loan approval status")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=1000, value=50000)
home_ownership = st.selectbox(
    "Home Ownership", ["MORTGAGE", "OTHER", "OWN", "RENT"]
)
loan_amount = st.number_input("Loan Amount", min_value=500, value=10000)
loan_intent = st.selectbox(
    "Loan Intent",
    ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
input_data = pd.DataFrame([{
    "person_age": age,
    "person_income": income,
    "person_home_ownership": home_ownership,
    "loan_amnt": loan_amount,
    "loan_intent": loan_intent,
    "credit_score": credit_score
}])
input_encoded = pd.get_dummies(
    input_data,
    columns=["person_home_ownership", "loan_intent"],
    drop_first=True
)
model_features = model.feature_names_in_
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)
if st.button("Predict Loan Status"):
    prediction = model.predict(input_encoded)
    probability = model.predict_proba(input_encoded)

    if prediction[0] == 1:
        st.success(f"✅ Loan Approved (Probability: {probability[0][1]*100:.2f}%)")
    else:

        st.error(f"❌ Loan Not Approved (Probability: {probability[0][1]*100:.2f}%)")
