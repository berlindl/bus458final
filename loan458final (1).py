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


# --- 2. TITLE AND HEADER (The 'Cool' Part) ---

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

# --- Column 1: Core Financials ---
with col1:
    with st.container(border=True):
        st.subheader("💳 Credit & Income Metrics")

        # FICO Score using number_input for a clean look
        fico = st.number_input(
            "FICO Score (450 - 850)",
            min_value=450,
            max_value=850,
            value=700,
            step=1,
            help="The primary measure of credit risk."
        )

        # Requested Loan Amount slider (req_loan)
        req_loan = st.slider(
            "Requested Loan Amount ($)",
            min_value=5000.0,
            max_value=150000.0,
            value=30000.0,
            step=1000.0
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
        applications = 1


# --- Column 2: Loan & Employment Details ---
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
        
        # Moved the 'Lender' input into a toggle/comparison box, so it's removed here

# --- 4. DATA PREPARATION FUNCTIONS (Needed for Reusability) ---

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

# --- 5. PREDICTION AND OUTPUT ---

st.markdown("---")
if st.button("PREDICT APPROVAL LIKELIHOOD", type="primary", use_container_width=True):
    
    # Base Dataframe Creation (used for all prediction scenarios)
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
        # The 'Lender' column will be added/swapped in the loop below
        "Lender": ["A"] 
    })
    
    lenders = ["A", "B", "C"]
    results = {}
    
    # --- Predict for all three lenders ---
    for lender_name in lenders:
        # Clone base data and set the current lender
        current_data = base_input_data.copy()
        current_data["Lender"] = lender_name
        
        # Preprocess and align
        input_aligned = preprocess_data(current_data, model.feature_names_in_)
        
        # Predict probability for class 1 (Approved)
        prob = model.predict_proba(input_aligned)[0][1]
        results[lender_name] = prob

    # --- Find Best Lender (Highest Probability) ---
    best_lender = max(results, key=results.get)
    best_prob = results[best_lender]
    
    # --- 6. DISPLAY RESULTS ---
    
    st.subheader("🎯 Prediction Summary")
    
    # Determine the binary outcome for the highest probability
    if best_prob >= 0.5:
        st.success(f"🎉 APPROVED! The application has a maximum approval likelihood of {best_prob:.2%} with Lender {best_lender}.")
    else:
        st.warning(f"⚠️ DENIED. The application's maximum approval likelihood is only {best_prob:.2%} (with Lender {best_lender}).")

    st.markdown("---")
    
    # --- Lender Comparison Output ---
    st.subheader("📊 Lender Approval Likelihood Comparison")
    st.markdown("Toggle through the lenders below to see the precise probability of approval for this specific customer profile.")

    comparison_data = pd.DataFrame({
        "Lender": lenders,
        "Approval Likelihood": [f"{results[l]} ({'Approved' if results[l] >= 0.5 else 'Denied'})" for l in lenders]
    })
    comparison_data = comparison_data.set_index("Lender")
    
    # Display the comparison in an appealing metric/table format
    col_a, col_b, col_c = st.columns(3)
    
    # Optional: Display as metrics with a highlight on the best lender
    cols = [col_a, col_b, col_c]
    payouts = {"A": 250, "B": 350, "C": 150} # Payouts for added business context
    
    for i, lender_name in enumerate(lenders):
        prob = results[lender_name]
        is_approved = prob >= 0.5
        
        label_text = f"Lender {lender_name} (Payout: ${payouts[lender_name]})"
        
        # Use success/error icons based on threshold
        icon = "✅" if is_approved else "❌"
        
        with cols[i]:
            st.metric(
                label=label_text, 
                value=f"{prob:.1%}",
                delta="Optimal Match" if lender_name == best_lender else None,
                delta_color="normal"
            )
