# app.py
import streamlit as st
import pandas as pd
from model import load_model
from utils import preprocess_input, make_prediction

st.title("🏦 Loan Eligibility Prediction App")
st.markdown("Fill out the form below to check if you're eligible for a loan.")

# User input form
def user_input():
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Married = st.selectbox('Married', ['Yes', 'No'])
    Dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    Education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox('Self Employed', ['Yes', 'No'])
    ApplicantIncome = st.number_input('Applicant Income', min_value=0)
    CoapplicantIncome = st.number_input('Coapplicant Income', min_value=0)
    LoanAmount = st.number_input('Loan Amount (in 1000s)', min_value=0)
    Loan_Amount_Term = st.number_input('Loan Amount Term (in days)', min_value=0)
    Credit_History = st.selectbox('Credit History', [1.0, 0.0])
    Property_Area = st.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])

    data = {
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area
    }
    return pd.DataFrame([data])

input_df = user_input()

# Load model and make prediction
if st.button("Check Eligibility"):
    model = load_model()
    processed_input = preprocess_input(input_df)
    prediction = make_prediction(model, processed_input)
    result = "✅ Eligible for Loan" if prediction == 1 else "❌ Not Eligible"
    st.subheader("Prediction Result:")
    st.success(result)
