import streamlit as st
import pandas as pd
from data_loader import load_data, clean_data
from utils import encode_features
from model import train_model

# Title
st.title("Loan Eligibility Predictor")

# Load and prepare data
df = load_data("credit.csv")
df = clean_data(df)
df = encode_features(df)
model, acc = train_model(df)

st.success(f"Model trained with accuracy: {acc}")

st.subheader("Enter Applicant Details")

# Form for user input
gender = st.selectbox("Gender", ['Male', 'Female'])
married = st.selectbox("Married", ['Yes', 'No'])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
applicant_income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)

# Encode manually (based on training logic)
def manual_encode(data_dict):
    encoded = {
        'Gender': 1 if data_dict['Gender'] == 'Male' else 0,
        'Married': 1 if data_dict['Married'] == 'Yes' else 0,
        'Education': 1 if data_dict['Education'] == 'Graduate' else 0,
        'Self_Employed': 1 if data_dict['Self_Employed'] == 'Yes' else 0,
        'Dependents': int(data_dict['Dependents']),
        'Credit_History': data_dict['Credit_History'],
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0}[data_dict['Property_Area']],
        'ApplicantIncome': data_dict['ApplicantIncome'],
        'LoanAmount': data_dict['LoanAmount']
    }
    return pd.DataFrame([encoded])

# Predict
if st.button("Check Loan Eligibility"):
    input_data = {
        'Gender': gender,
        'Married': married,
        'Education': education,
        'Self_Employed': self_employed,
        'Dependents': dependents,
        'Credit_History': credit_history,
        'Property_Area': property_area,
        'ApplicantIncome': applicant_income,
        'LoanAmount': loan_amount
    }

    input_df = manual_encode(input_data)
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Denied.")
