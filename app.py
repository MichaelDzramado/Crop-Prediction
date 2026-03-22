
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("final_rf_pipeline.pkl")

st.title(" Crop Prediction App")

# Inputs
col1, col2 = st.columns(2)

with col1:
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    soil_ec = st.number_input("Soil EC", 0.0, 10.0, 1.0)
    phosphorus = st.number_input("Phosphorus", 0, 100, 40)
    potassium = st.number_input("Potassium", 0, 100, 50)

with col2:
    urea = st.number_input("Urea", 0, 100, 30)
    tsp = st.number_input("T.S.P", 0, 100, 20)
    mop = st.number_input("M.O.P", 0, 100, 25)
    moisture = st.number_input("Moisture", 0, 100, 60)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        'pH': ph,
        'Soil EC': soil_ec,
        'Phosphorus': phosphorus,
        'Potassium': potassium,
        'Urea': urea,
        'T.S.P': tsp,
        'M.O.P': mop,
        'Moisture': moisture,
        'Temperature': temperature
    }])

    st.write("Input Data:", input_data)  # Debug line


    prediction = model.predict(input_data)

    st.success(f"Recommended Crop: {prediction[0]}") 

    st.write("This app predicts the most suitable crop based on soil nutrientsand environmental conditions using a Machine Learning model.")	

  

    st.markdown("---")
    st.write("Developed by Michael Dzramado")
