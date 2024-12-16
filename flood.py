# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the trained model
with open('xgb_model.sav', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title("Flood Path Predictor")
st.write("Enter the required features for prediction:")

# Input fields for features
Latitude = st.number_input("Latitude (decimal degrees):", min_value=-90.0, max_value=90.0, value=0.0)
Longitude = st.number_input("Longitude (decimal degrees):", min_value=-180.0, max_value=180.0, value=0.0)
temperature_2m_mean = st.number_input("Temperature 2m Mean (°C):", min_value=-50.0, max_value=50.0, value=25.0)
precipitation_sum = st.number_input("Precipitation Sum (mm):", min_value=0.0, max_value=500.0, value=50.0)
wind_speed_10m_max = st.number_input("Wind Speed 10m Max (m/s):", min_value=0.0, max_value=100.0, value=10.0)
Duration = st.number_input("Duration (hours):", min_value=1, max_value=240, value=24)  # Ensure lowercase 'd'

# Predict button
if st.button("Predict Flood Path"):
    # Combine inputs into a DataFrame
    input_data = pd.DataFrame({
        'Latitude': [Latitude],
        'Longitude': [Longitude],
        'temperature_2m_mean': [temperature_2m_mean],
        'precipitation_sum': [precipitation_sum],
        'wind_speed_10m_max': [wind_speed_10m_max],
        'Duration': [Duration]  # Ensure lowercase 'd'
    })

    # Preprocess the input data using the scaler
    numerical_features = ['temperature_2m_mean', 'precipitation_sum', 'wind_speed_10m_max']

    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    # Make predictions
    prediction = model.predict(input_data)

    # Display the prediction
    if prediction[0] == 1:
        st.write("Prediction: High Risk of Flooding")
    else:
        st.write("Prediction: No Flooding Expected")
