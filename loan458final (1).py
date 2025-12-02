# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn # This is needed for the pickle file to load!

# --- 1. CONFIGURATION AND MODEL LOADING ---

# Set a wide page configuration for better aesthetics
st.set_page_config(layout="wide")

# Load the trained model
# IMPORTANT: Update this path and file name if you saved it differently
try:
    with open("458finalcomp.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Model file '458finalcomp.pkl' not found. Please ensure it's in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- 2. TITLE AND HEADER (The 'Cool' Part) ---

# Custom Title for "Lou & Petes Loan Approval Predicta"
st.markdown(
    """
    <div style='background-color: #2F4F4F; padding: 15px; border-radius: 10px; text-align: center;'>
        <h1 style='color: #FFD700; margin: 0;'>
            💰 Lou & Pete's Loan Approval Predicta 💰
        </h1>
        <p style='color: #f0f0f0; margin: 5px 0 0;'>
            Powered by Data Analytics
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("") # Add a little space

st.subheader("Applicant Profile Builder")

# Use columns for a cool, two-column layout
col1, col2 = st.columns(2)


# --- 3. INPUT WIDGETS (Grouped for uniqueness) ---

# --- Column 1: Core Financials ---
with col1:
    with st.container(border=True):
        st.subheader("Credit & Income Metrics")

        # FICO Score using number_input for a clean look, often better for specific numbers
        fico = st.number_input(
            "FICO Score (550 - 850)", 
            min_value=450, 
            max_value=850, 
            value=700, 
            step=1,
            help="The primary measure of credit risk."
        )

        # Monthly Gross Income slider
        mgi = st.slider(
            "Monthly Gross Income ($)", 
            min_value=1000.0, 
            max_value=15000.0, 
            value=5000.0, 
            step=100.0
        )

        # Monthly Housing Payment slider
        mhp = st.slider(
            "Monthly Housing Payment ($)", 
            min_value=500.0, 
            max_value=4000.0, 
            value=1200.0, 
            step=50.0
        )
        
        # Binary input (Bankruptcy)
        bankrupt = st.selectbox(
            "Ever Bankrupt or Foreclosed", 
            [0, 1], 
            format_func=lambda x: 'Yes (1)' if x == 1 else 'No (0)'
        )
        # 'applications' was a column in your data, assuming 1 for a single request
        applications = 1


# --- Column 2: Loan & Employment Details ---
with col2:
    with st.container(border=True):
        st.subheader("Loan & Employment Details")

        # Requested Loan Amount slider (req_loan)
        req_loan = st.slider(
            "Requested Loan Amount ($)", 
            min_value=5000.0, 
            max_value=150000.0, 
            value=30000.0, 
            step=1000.0
        )
        
        # Categorical Selectboxes
        reason = st.selectbox("Reason for Loan", [
            "credit_card_refinancing", "debt_conslidation", "home_improvement",
            "major_purchase", "cover_an_unexpected_cost", "other"
        ])

        employment_status = st.selectbox("Employment Status", ["full_time", "part_time", "unemployed"])

        employment_sector = st.selectbox("Employment Sector", [
            "Missing", "financials", "health_care", "information_technology", "industrials",
            "real_estate", "utilities", "consumer_discretionary", "communication_services",
            "consumer_staples", "materials", "energy"
        ])

        lender = st.selectbox("Lender Partner", ["A", "B", "C"])


# --- 4. DATA PREPARATION (LOGIC MUST BE UNCHANGED) ---

# --- Create the input data as a DataFrame (Base Data) ---
input_data = pd.DataFrame({
    "applications": [applications],
    "Requested_Loan_Amount": [req_loan],
    "FICO_score": [fico],
    "Ever_Bankrupt_or_Foreclose": [bankrupt],
    
    # Raw variables needed for log transformation
    "Monthly_Gross_Income": [mgi],
    "Monthly_Housing_Payment": [mhp],
    "Requested_Loan_Amount_for_log": [req_loan], # Temp column for log transform
    
    # Categorical inputs
    "Reason": [reason],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Lender": [lender]
})

# 1. Apply Log Transformations using np.log1p (log(1+x))
input_data['ln_Monthly_Gross_Income'] = np.log1p(input_data['Monthly_Gross_Income'])
input_data['ln_Monthly_Housing_Payment'] = np.log1p(input_data['Monthly_Housing_Payment'])
input_data['lon_Granted_Loan_Amount'] = np.log1p(input_data['Requested_Loan_Amount_for_log'])


# 2. Drop the raw and excluded columns (MUST MATCH YOUR X DEFINITION)
columns_to_drop = [
    'Monthly_Gross_Income',
    'Monthly_Housing_Payment',
    'Requested_Loan_Amount_for_log'
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


# --- 5. PREDICTION AND OUTPUT ---

st.markdown("---")
# Predict button placed in the center
if st.button("Predict Loan Approval Likelihood", type="primary", use_container_width=True):
    # Predict using the loaded model
    prediction = model.predict(input_data_aligned)[0]

    st.subheader("Prediction Result:")

    if prediction == 1:
        st.success(f"🎉 Prediction: APPROVED (1)")
        st.balloons() # Add some celebration!
        st.markdown("**This application is highly likely to be APPROVED.** The customer's profile meets the minimum lending criteria.")
    else:
        st.error(f"⚠️ Prediction: DENIED (0)")
        st.markdown("**This application is likely to be DENIED.** The customer's profile suggests a higher risk.")
