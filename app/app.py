import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error
import os

BASE_DIR = os.path.dirname(__file__) 
MODEL_PATH = os.path.join(BASE_DIR, "power_consumption_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title(" Power Consumption Predictor")

num_cols = [
    'Global_reactive_power','Voltage','Global_intensity',
    'Sub_metering_1','Sub_metering_2','Sub_metering_3',
    'lag_1','lag_2','hour','day_of_week','month'
]

col1, col2 = st.columns(2)

 
with col1:
    st.header("Single Prediction")

    inputs = {
        'Global_reactive_power': st.number_input("Global Reactive Power", 0.0, 1.0),
        'Voltage': st.number_input("Voltage", 200.0, 260.0),
        'Global_intensity': st.number_input("Global Intensity", 0.0, 50.0),
        'Sub_metering_1': st.number_input("Sub_metering_1", 0.0, 50.0),
        'Sub_metering_2': st.number_input("Sub_metering_2", 0.0, 50.0),
        'Sub_metering_3': st.number_input("Sub_metering_3", 0.0, 50.0),
        'lag_1': st.number_input("Lag 1", 0.0, 10.0),
        'lag_2': st.number_input("Lag 2", 0.0, 10.0),
        'hour': st.slider("Hour", 0, 23),
        'day_of_week': st.slider("Day of Week (0=Mon)", 0, 6),
        'month': st.slider("Month", 1, 12)
    }

    if st.button("Predict"):
        sample = pd.DataFrame([inputs])
        pred = model.predict(sample)[0]
        st.success(f"Predicted Power: {pred:.3f} kW")
 
with col2:
    st.header("Batch Prediction")

    file = st.file_uploader("Upload CSV", type="csv")
    if st.button("Predict Batch", key="batch_predict"):

        if file:
            df = pd.read_csv(file)

            # Ensure Datetime exists
            if {'Date', 'Time'}.issubset(df.columns):
                df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                df = df.sort_values('Datetime')

                # Time features
                df['hour'] = df['Datetime'].dt.hour
                df['day_of_week'] = df['Datetime'].dt.dayofweek
                df['month'] = df['Datetime'].dt.month
            else:
                st.error("CSV must contain Date and Time columns")
                st.stop()
                
            if df.empty:
                st.error("No valid rows left after creating lag features.")
                st.stop()

            # Create lag features (SAME AS TRAINING)
            if 'Global_active_power' in df.columns:
                df['lag_1'] = df['Global_active_power'].shift(1)
                df['lag_2'] = df['Global_active_power'].shift(2)
            else:
                st.error("Global_active_power required to create lag features")
                st.stop()

            # Drop rows with missing lags
            df = df.dropna(subset=['lag_1', 'lag_2'])
     
            # Predict
            preds = model.predict(df)
            df["Predicted_Power"] = preds

            # Metrics if actual exists
            if 'Global_active_power' in df.columns:
                mae = mean_absolute_error(df['Global_active_power'], preds)
                st.metric("MAE", f"{mae:.4f}")

            st.dataframe(df)
     
