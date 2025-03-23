import streamlit as st  # Import first
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Set Streamlit page config (must be the first Streamlit command)
st.set_page_config(page_title="Parkinson's Predictor", page_icon="🧠", layout="wide")

# Load the trained model
MODEL_PATH = "models/random_forest.pkl"
rf_model = joblib.load(MODEL_PATH)

# Feature names
FEATURE_NAMES = [
    "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking", 
    "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality", 
    "FamilyHistoryParkinsons", "TraumaticBrainInjury", "Hypertension", "Diabetes", 
    "Depression", "Stroke", "SystolicBP", "DiastolicBP", "CholesterolTotal", 
    "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "UPDRS", 
    "MoCA", "FunctionalAssessment", "Tremor", "Rigidity", "Bradykinesia", 
    "PosturalInstability", "SpeechProblems", "SleepDisorders", "Constipation"
]

# Custom CSS
st.markdown("""
    <style>
        .main { background-color: #f4f4f4; }
        .stButton>button { background-color: #ff4b4b; color: white; font-size: 18px; }
        .stButton>button:hover { background-color: #ff1a1a; }
        .title-text { font-size: 36px; font-weight: bold; text-align: center; color: #2C3E50; }
        .info-text { font-size: 18px; text-align: center; color: #7F8C8D; }
        .prediction-container { padding: 20px; border-radius: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/7/7b/Parkinson%27s_disease.svg", width=120)
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Feature Importance", "About"])

# Header
st.markdown("<p class='title-text'>🧠 Parkinson's Disease Prediction</p>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Enter patient details to predict Parkinson’s disease likelihood.</p>", unsafe_allow_html=True)

# ---- Home Page ----
if page == "Home":
    with st.form("prediction_form"):
        st.subheader("🗃 Patient Details")
        
        # Input fields using columns
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 20, 100, 50, 1)
            bmi = st.number_input("BMI", 10.0, 50.0, 22.5, 0.1)
            systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
            diastolic_bp = st.number_input("Diastolic BP", 50, 130, 80)
            cholesterol_total = st.number_input("Total Cholesterol", 100, 300, 200)
            cholesterol_ldl = st.number_input("LDL Cholesterol", 50, 200, 100)
            updrs = st.number_input("UPDRS Score", 0.0, 200.0, 40.0, 0.1)
            moca = st.number_input("MoCA Score", 0, 30, 20)

        with col2:
            gender = st.radio("Gender", ["Male", "Female"])
            smoking = st.radio("Smoking", ["No", "Yes"])
            alcohol = st.radio("Alcohol Consumption", ["No", "Yes"])
            physical_activity = st.radio("Physical Activity", ["No", "Yes"])
            diet = st.radio("Diet Quality", ["Poor", "Good"])
            sleep = st.radio("Sleep Quality", ["Poor", "Good"])
            family_history = st.radio("Family History", ["No", "Yes"])
            brain_injury = st.radio("Brain Injury", ["No", "Yes"])

        expand = st.expander("Show Additional Health Factors")
        with expand:
            col3, col4 = st.columns(2)
            with col3:
                hypertension = st.radio("Hypertension", ["No", "Yes"])
                diabetes = st.radio("Diabetes", ["No", "Yes"])
                depression = st.radio("Depression", ["No", "Yes"])
                stroke = st.radio("Stroke", ["No", "Yes"])
            with col4:
                rigidity = st.radio("Rigidity", ["No", "Yes"])
                tremor = st.radio("Tremor", ["No", "Yes"])
                bradykinesia = st.radio("Bradykinesia", ["No", "Yes"])
                speech_problems = st.radio("Speech Problems", ["No", "Yes"])

        submit = st.form_submit_button("Predict")

    # Prediction logic
    if submit:
        input_data = pd.DataFrame([[
            age, 1 if gender == "Male" else 0, 0, 0, bmi, 1 if smoking == "Yes" else 0, 
            1 if alcohol == "Yes" else 0, 1 if physical_activity == "Yes" else 0, 1 if diet == "Good" else 0,
            1 if sleep == "Good" else 0, 1 if family_history == "Yes" else 0, 1 if brain_injury == "Yes" else 0,
            1 if hypertension == "Yes" else 0, 1 if diabetes == "Yes" else 0, 1 if depression == "Yes" else 0,
            1 if stroke == "Yes" else 0, systolic_bp, diastolic_bp, cholesterol_total, cholesterol_ldl, 100, 150,
            updrs, moca, 1, 1 if tremor == "Yes" else 0, 1 if rigidity == "Yes" else 0, 
            1 if bradykinesia == "Yes" else 0, 1, 1 if speech_problems == "Yes" else 0, 0, 0
        ]], columns=FEATURE_NAMES)

        prediction = rf_model.predict(input_data)[0]
        confidence = rf_model.predict_proba(input_data)[0][prediction] * 100

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ The model predicts the patient HAS Parkinson's Disease with {confidence:.2f}% confidence.")
        else:
            st.success(f"✅ The model predicts the patient DOES NOT HAVE Parkinson's Disease with {confidence:.2f}% confidence.")

# ---- Feature Importance Page ----
elif page == "Feature Importance":
    st.subheader("Key Features Affecting Prediction")
    feature_importance = rf_model.feature_importances_
    st.bar_chart(pd.Series(feature_importance, index=FEATURE_NAMES).nlargest(10))

# ---- About Page ----
elif page == "About":
    st.markdown("### About This App\n🔹 Uses **Random Forest Model** for prediction\n🔹 Developed using **Streamlit & Python**\n🔹 Helps in **early detection of Parkinson’s disease**")
