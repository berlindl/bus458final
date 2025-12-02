# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn # This is needed for the pickle file to load!

# --- 1. CONFIGURATION, STATE, AND MODEL LOADING ---

# Set a wide page configuration for better aesthetics
st.set_page_config(layout="wide")

# Initialize session state variables
if 'results_data' not in st.session_state:
    st.session_state['results_data'] = None
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Load the trained model
try:
    # Use the relative path you corrected
    with open("458finalcomp.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Model file '458finalcomp.pkl' not found. Please ensure it's in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- 2. TITLE AND HEADER ---

# Custom Title for "Lou & Petes Loan Approval Predicta"
st.markdown(
    """
    <div style='background-color: #556B2F; padding: 20px; border-radius: 15px; text-align: center; border: 3px solid #FFD700;'>
        <h1 style='color: #FFD700; margin: 0; font-size: 3em;'>
            ⭐ Lou & Pete's Loan Approval Predicta ⭐
        </h1>
        <p style='color: #f0f0f0; margin: 5px 0 0; font-size: 1.2em;'>
            Predicting Approval Likelihood and Optimizing Lender Matching
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("") # Add a little space

st.subheader("Applicant Profile Builder")

# Use columns for a cool, two-column layout
col1, col2 = st.columns(2)


# --- 3. INPUT WIDGETS ---
# Variables initialized here will persist across reruns
with col1:
    with st.container(border=True):
        st.subheader("💳 Credit & Income Metrics")

        fico = st.number_input("FICO Score (450 - 850)", min_value=450, max_value=850, value=700, step=1, help="The primary measure of credit risk.")
        req_loan = st.slider("Requested Loan Amount ($)", min_value=5000.0, max_value=150000.0, value=30000.0, step=1000.0)
        mgi = st.slider("Monthly Gross Income ($)", min_value=1000.0, max_value=15000.0, value=5000.0, step=100.0)
        mhp = st.slider("Monthly Housing Payment ($)", min_value=500.0, max_value=4000.0, value=1200.0, step=50.0)
        bankrupt = st.selectbox("Ever Bankrupt or Foreclosed", [0, 1], format_func=lambda x: 'Yes (1)' if x == 1 else 'No (0)')
        applications = 1

with col2:
    with st.container(border=True):
        st.subheader("💼 Loan & Employment Details")

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


# --- 4. DATA PREPARATION FUNCTIONS (UNCHANGED) ---

def preprocess_data(input_df, model_columns):
    """Applies log transforms, drops raw columns, and aligns dummies for prediction."""
    df = input_df.copy()
    
    # 1. Apply Log Transformations
    df['ln_Monthly_Gross_Income'] = np.log1p(df['Monthly_Gross_Income'])
    df['ln_Monthly_Housing_Payment'] = np.log1p(df['Monthly_Housing_Payment'])
    df['lon_Granted_Loan_Amount'] = np.log1p(df['Requested_Loan_Amount_for_log'])

    # 2. Drop the raw and excluded columns
    columns_to_drop = [
        'Monthly_Gross_Income',
        'Monthly_Housing_Payment',
        'Requested_Loan_Amount_for_log'
    ]
    df = df.drop(columns=columns_to_drop)

    # 3. One-hot encode the user's input.
    categorical_cols = ['Reason', 'Employment_Status', 'Employment_Sector', 'Lender']
    input_data_encoded = pd.get_dummies(df, columns=categorical_cols, dtype=float)

    # 4. Align columns
    final_data = pd.DataFrame(columns=model_columns)
    final_data = pd.concat([final_data, input_data_encoded], ignore_index=True)
    final_data = final_data.fillna(0)
    return final_data[model_columns]


# --- 5. PREDICTION LOGIC (MOVED TO A FUNCTION) ---

def run_prediction():
    """Calculates probabilities and expected payouts for all lenders and stores them in session state."""
    
    # Define Payouts (Business Secret)
    payouts = {"A": 250, "B": 350, "C": 150} 
    lenders = ["A", "B", "C"]
    
    # Base Dataframe Creation
    base_input_data = pd.DataFrame({
        "applications": [applications],
        "Requested_Loan_Amount": [req_loan],
        "FICO_score": [fico],
        "Ever_Bankrupt_or_Foreclose": [bankrupt],
        "Monthly_Gross_Income": [mgi],
        "Monthly_Housing_Payment": [mhp],
        "Requested_Loan_Amount_for_log": [req_loan],
        "Reason": [reason],
        "Employment_Status": [employment_status],
        "Employment_Sector": [employment_sector],
        "Lender": ["A"] # Placeholder
    })
    
    # Initialize trackers
    max_expected_payout = -1
    max_prob = -1
    optimal_revenue_lender = None
    most_likely_lender = None
    
    # Store all results here
    full_results = {}
    
    # Predict for all three lenders and track best matches
    for lender_name in lenders:
        current_data = base_input_data.copy()
        current_data["Lender"] = lender_name
        
        input_aligned = preprocess_data(current_data, model.feature_names_in_)
        prob = model.predict_proba(input_aligned)[0][1]
        expected_payout = prob * payouts[lender_name]
        
        full_results[lender_name] = {'prob': prob, 'payout': payouts[lender_name], 'expected_payout': expected_payout}
        
        # Track Max Probability (for customer insight)
        if prob > max_prob:
            max_prob = prob
            most_likely_lender = lender_name
            
        # Track Max Expected Payout (for business insight)
        if expected_payout > max_expected_payout:
            max_expected_payout = expected_payout
            optimal_revenue_lender = lender_name
            
    # Save the necessary data to session state
    st.session_state.results_data = {
        'full_results': full_results,
        'max_prob': max_prob,
        'most_likely_lender': most_likely_lender,
        'max_expected_payout': max_expected_payout,
        'optimal_revenue_lender': optimal_revenue_lender
    }
    
    st.success("Prediction complete! Results are shown below.")


# --- 6. BUTTON AND RESULT DISPLAY ---

st.markdown("---")
# Call the prediction function when the button is clicked
st.button("PREDICT APPROVAL LIKELIHOOD", type="primary", use_container_width=True, on_click=run_prediction)

# Only show results if they exist in session state
if st.session_state.results_data:
    
    data = st.session_state.results_data
    full_results = data['full_results']
    max_prob = data['max_prob']
    most_likely_lender = data['most_likely_lender']
    max_expected_payout = data['max_expected_payout']
    optimal_revenue_lender = data['optimal_revenue_lender']
    lenders = list(full_results.keys())
    payouts = {l: full_results[l]['payout'] for l in lenders}


    # --- Prediction Summary ---
    st.subheader("🎯 Prediction Summary")
    
    if max_prob >= 0.5:
        st.success(f"🎉 **APPROVED!** The customer is best matched with **Lender {most_likely_lender}**, yielding a maximum approval likelihood of **{max_prob:.2%}**.")
    else:
        st.warning(f"⚠️ **DENIED.** Even with the optimal match (Lender {most_likely_lender}), the maximum approval likelihood is **{max_prob:.2%}**.")

    st.markdown("---")
    
    # --- Lender Comparison Output ---
    st.subheader("📊 Lender Approval Likelihood Comparison")
    st.markdown("This comparison shows the customer's chance of approval with each potential partner.")

    col_a, col_b, col_c = st.columns(3)
    cols = [col_a, col_b, col_c]
    
    for i, lender_name in enumerate(lenders):
        prob = full_results[lender_name]['prob']
        
        label_text = f"Lender {lender_name} (Payout: ${payouts[lender_name]})"
        
        with cols[i]:
            st.metric(
                label=label_text, 
                value=f"{prob:.1%}",
                # Highlight based on max probability
                delta="Highest Approval Chance" if lender_name == most_likely_lender else None,
                delta_color="normal" if lender_name == most_likely_lender else "off"
            )

    st.markdown("---")
    
    # --- NEW: Password Protection for Business Payout Analysis ---
    st.subheader("🔒 Business Payout Analysis (Password="wayne")")
    
    # Define a password check function
    def check_password_and_set_state():
        if st.session_state.password_input == "wayne":
            st.session_state.logged_in = True
        else:
            st.session_state.logged_in = False
            
    # Use key for text_input to link it to session state
    st.text_input("Enter password to view Business Insights", type="password", key="password_input", on_change=check_password_and_set_state)

    if st.session_state.logged_in:
        st.success("Access Granted! Displaying business insights.")
        
        st.markdown("This section calculates the expected revenue for the platform by routing the customer to the optimal lending partner.")
        
        business_col1, business_col2, business_col3 = st.columns(3)
        
        with business_col1:
            st.metric(
                label="Optimal Revenue Lender",
                value=f"Lender {optimal_revenue_lender}",
                help="The lender that offers the highest Expected Payout (Probability * Payout)."
            )

        with business_col2:
            st.metric(
                label="Lender Payout (if Approved)",
                value=f"${payouts[optimal_revenue_lender]}",
                help=f"The cash reward the platform receives if Lender {optimal_revenue_lender} approves the loan."
            )
            
        with business_col3:
            st.metric(
                label="MAX Expected Payout (Revenue)",
                value=f"${max_expected_payout:.2f}",
                delta_color="off",
                help="The predicted revenue for the platform: Max Approval Probability * Payout."
            )
    elif st.session_state.password_input:
        st.error("Access Denied: Incorrect password.")
