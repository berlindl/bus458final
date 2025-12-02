# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn # This is needed for the pickle file to load!

# Load the trained model
# IMPORTANT: Update this path and file name if you saved it differently
with open("458finalcomp.pkl", "rb") as file:
    model = pickle.load(file)

# Title for the app
st.markdown(
    "<h1 style='text-align: center; background-color: #008080; padding: 10px; color: #ffffff;'><b>Loan Approval Likelihood Predictor</b></h1>",
    unsafe_allow_html=True
)

# Numeric and Categorical inputs
st.header("Enter Loan Applicant's Details")

# --- Numerical Inputs (Raw values needed for log transformation) ---
fico = st.slider("FICO Score", min_value=450, max_value=850, value=700, step=1)
# Note: req_loan is used for the requested amount slider
req_loan = st.slider("Requested Loan Amount ($)", min_value=5000.0, max_value=150000.0, value=30000.0, step=1000.0)
mgi = st.slider("Monthly Gross Income ($)", min_value=1000.0, max_value=15000.0, value=5000.0, step=100.0)
mhp = st.slider("Monthly Housing Payment ($)", min_value=500.0, max_value=4000.0, value=1200.0, step=50.0)
# 'applications' was a column in your data, assuming 1 for a single request
applications = 1

# --- Binary and Categorical Inputs ---
bankrupt = st.selectbox("Ever Bankrupt or Foreclosed", [0, 1], format_func=lambda x: 'Yes (1)' if x == 1 else 'No (0)')

reason = st.selectbox("Reason for Loan", [
    "credit_card_refinancing", "debt_conslidation", "home_improvement",
    "major_purchase", "cover_an_unexpected_cost", "other"
])

# NOTE: Using a simplified list for the app's sector dropdown, including 'Missing'
employment_status = st.selectbox("Employment Status", ["full_time", "part_time", "unemployed"])

employment_sector = st.selectbox("Employment Sector", [
    "Missing", "financials", "health_care", "information_technology", "industrials",
    "real_estate", "utilities", "consumer_discretionary", "communication_services",
    "consumer_staples", "materials", "energy"
])

lender = st.selectbox("Lender Partner", ["A", "B", "C"])


# --- Create the input data as a DataFrame (Base Data) ---
input_data = pd.DataFrame({
    "applications": [applications],
    "Requested_Loan_Amount": [req_loan], # Using req_loan slider variable
    "FICO_score": [fico],
    "Ever_Bankrupt_or_Foreclose": [bankrupt],
    
    # Raw variables needed for log transformation (MUST be included here)
    # The 'Granted_Loan_Amount' in your training code was replaced by 'Requested_Loan_Amount'
    "Monthly_Gross_Income": [mgi],
    "Monthly_Housing_Payment": [mhp],
    "Requested_Loan_Amount_for_log": [req_loan], # Temp column for log transform
    
    # Categorical inputs
    "Reason": [reason],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Lender": [lender]
})


# --- Prepare Data for Prediction (EXACTLY MATCHING TRAINING PIPELINE) ---

# 1. Apply Log Transformations using np.log1p (log(1+x))
# Note: Renamed log-transformed variable to match your training code names
input_data['ln_Monthly_Gross_Income'] = np.log1p(input_data['Monthly_Gross_Income'])
input_data['ln_Monthly_Housing_Payment'] = np.log1p(input_data['Monthly_Housing_Payment'])
input_data['lon_Granted_Loan_Amount'] = np.log1p(input_data['Requested_Loan_Amount_for_log'])


# 2. Drop the raw and excluded columns (MUST MATCH YOUR X DEFINITION)
columns_to_drop = [
    'Monthly_Gross_Income',
    'Monthly_Housing_Payment',
    'Requested_Loan_Amount_for_log' # Drop the raw column used for log transform
    # 'Granted_Loan_Amount' and 'Fico_Score_group' were dropped from the training DataFrame,
    # but since they were not created in the app, we don't drop them here.
]
input_data = input_data.drop(columns=columns_to_drop)


# 3. One-hot encode the user's input.
categorical_cols = ['Reason', 'Employment_Status', 'Employment_Sector', 'Lender']
input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols, dtype=float)


# 4. Align columns with the trained model's feature set
model_columns = model.feature_names_in_
final_data = pd.DataFrame(columns=model_columns)
final_data = pd.concat([final_data, input_data_encoded], ignore_index=True)
final_data = final_data.fillna(0) # Fill NaN from missing dummy columns with 0
input_data_aligned = final_data[model_columns]


# Predict button
if st.button("Predict Approval"):
    # Predict using the loaded model
    # Model returns 1 (Approved) or 0 (Denied)
    prediction = model.predict(input_data_aligned)[0]

    st.markdown("---")
    st.subheader("Prediction Result:")

    if prediction == 1:
        st.success(f"✅ Prediction: Approved (1)")
        st.write("The model predicts this application is likely to be **APPROVED**.")
    else:
        st.error(f"❌ Prediction: Denied (0)")
        st.write("The model predicts this application is likely to be **DENIED**.")
