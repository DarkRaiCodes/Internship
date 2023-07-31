import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load the scaler and model
scaler = pickle.load(open('SavedModels/Transformer.pkl', 'rb'))
model = load_model('SavedModels/AirQulaity.h5')

def rangeAssign(predictedVal):
    predictedVal = int(predictedVal)
    if predictedVal >= 0 and predictedVal < 50:
        return "Good"
    elif predictedVal >= 51 and predictedVal < 100:
        return "Moderate"
    elif predictedVal >= 101 and predictedVal < 150:
        return "Unhealthy for Sensitive Groups"
    elif predictedVal >= 151 and predictedVal < 200:
        return "Unhealthy"
    elif predictedVal >= 201 and predictedVal < 300:
        return "Very Unhealthy"
    elif predictedVal >= 301:
        return "Hazardous"

def predictor(singleEntry):
    singleEntry = scaler.transform(singleEntry.reshape(-1, 9))
    return model.predict(singleEntry)[0][0]

# Streamlit app code
st.title("Air Quality Prediction")

# User input form
st.subheader("Enter Air Quality Parameters")
co = st.number_input("CO")
pm25 = st.number_input("PM2.5")
no2 = st.number_input("NO2")
pm10 = st.number_input("PM10")
so2 = st.number_input("SO2")
nox = st.number_input("NOx")
no = st.number_input("NO")
toluene = st.number_input("Toluene")
o3 = st.number_input("O3")

# Predict air quality
singleEntry = np.array([co, pm25, no2, pm10, so2, nox, no, toluene, o3])
predictedVal = predictor(singleEntry)
prediction_range = rangeAssign(predictedVal)

# Display prediction result
st.subheader("Air Quality Prediction Result")
st.write("Predicted AQI Value:", predictedVal)
st.write("AQI Bucket:", prediction_range)
