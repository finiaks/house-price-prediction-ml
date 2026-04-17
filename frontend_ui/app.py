import streamlit as st
import pandas as pd
import joblib as job

#Load Model and Scaler
model = job.load("../model/house_pricing_pred_model.pkl")
scaler = job.load("../model/scaler.pkl")

#Title
st.title("House Price Prediction App")

st.write("Enter House details to predict price")

#Inputs
area = st.number_input("Enter Area", min_value = 500, max_value = 5000, step = 100)
bedrooms = st.number_input("Enter Number of Bedrooms", min_value = 1, max_value = 10, step = 1)
age = st.number_input("Enter Age", min_value = 0, max_value = 100, step = 1)

#Button 
if st.button("Predict Price"):
    new_data = pd.DataFrame([[area,bedrooms,age]], columns = ["Area","Bedrooms","Age"])

    new_data_scaled = scaler.transform(new_data)

    pred = model.predict(new_data_scaled)

    st.success(f"Predicted Price: ₹  {int(pred.flatten()[0])}")