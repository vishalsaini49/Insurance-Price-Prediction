import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model and scaler
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title("Insurance Premium Price Prediction")

# Sidebar: BMI Calculator
st.sidebar.title("Know your BMI")

height_sidebar = st.sidebar.number_input("Height (cm)", min_value=145, max_value=188, value=170)
weight_sidebar = st.sidebar.number_input("Weight (kg)", min_value=51, max_value=132, value=70)

# Predict Button
if st.sidebar.button("Calculate BMI"):
    bmi = weight_sidebar / ((height_sidebar / 100) ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        category = "Normal weight"
    elif 25 <= bmi < 29.9:
        category = "Overweight"
    else:
        category = "Obese"

    st.sidebar.write(f"**BMI Index:** {bmi:.2f}")
    st.sidebar.write(f"You are **{category}**")

# Main page: Demographic Details
st.subheader("Demographic Details")

age = st.slider("Age", 18, 66, 30)
# age = st.number_input("Age (yrs)", min_value=18, max_value=66, value=30)

col1, col2 = st.columns(2)

height = col1.number_input("Height", min_value=145, max_value=188, value=170)
weight = col2.number_input("Weight", min_value=51, max_value=132, value=70)

# Main page: Medical History
st.subheader("Medical History")

col3, col4 = st.columns(2)

with col3:
    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
    bp_problems = st.selectbox("Blood Pressure Problems", [0, 1], format_func=lambda x: "Yes" if x else "No")
    transplants = st.selectbox("Any Transplants", [0, 1], format_func=lambda x: "Yes" if x else "No")
    cancer_history = st.selectbox("Family History of Cancer", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col4:
    chronic_diseases = st.selectbox("Any Chronic Diseases", [0, 1], format_func=lambda x: "Yes" if x else "No")
    allergies = st.selectbox("Known Allergies", [0, 1], format_func=lambda x: "Yes" if x else "No")
    num_surgeries = st.selectbox("Number of Major Surgeries", [0, 1, 2, 3])

# Create input DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "Diabetes": [diabetes],
    "BloodPressureProblems": [bp_problems],
    "AnyTransplants": [transplants],
    "AnyChronicDiseases": [chronic_diseases],
    "Height": [height],
    "Weight": [weight],
    "KnownAllergies": [allergies],
    "HistoryOfCancerInFamily": [cancer_history],
    "NumberOfMajorSurgeries": [num_surgeries]
})

# Standard scaling for numerical features
num_features = ["Age", "Height", "Weight"]
input_data[num_features] = scaler.transform(input_data[num_features])

# Predict Button
if st.button("Predict Premium Price"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Premium Price: ₹{prediction[0]:,.2f}")