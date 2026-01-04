# Power Consumption Prediction App

This project predicts household power consumption using historical electrical measurements and time-based features.  
It includes a trained machine learning model, a Streamlit web application, and a Docker setup for easy deployment.

---

## Overview

The goal of this project is to predict **Global Active Power (kW)** using:
- Electrical sensor readings
- Past power usage (lag features)
- Time information such as hour, day of week, and month

The model is trained on the *Individual Household Electric Power Consumption* dataset.

---

## Features Used

- Global reactive power  
- Voltage  
- Global intensity  
- Sub-metering (1, 2, 3)  
- Lag features (`lag_1`, `lag_2`)  
- Hour of day  
- Day of week  
- Month  

Lag features help the model capture time-based patterns in electricity usage.

---

## Model Details

- **Algorithm:** Random Forest Regressor  
- **Preprocessing:**
  - Median imputation for missing values
  - Feature selection using Mutual Information
- **Train/Test Split:** Time-based split to avoid data leakage

---

## Model Performance

Performance on unseen test data:

- **MAE:** ~0.0177  
- **RMSE:** ~0.0329  
- **R² Score:** ~0.9987  

---

## Streamlit Application

The app provides:
- Single power consumption prediction
- Batch prediction using CSV upload
- Automatic lag feature creation
- Error metric display (MAE) when actual values are available

---

## Running Locally
```bash
Install dependencies:
pip install -r requirements.txt


Run the app:
streamlit run app/app.py

Running with Docker

Build the image:
docker build -t power-consumption-app .


Run the container:
docker run -p 5000:8501 power-consumption-app


Open:
http://localhost:5000



