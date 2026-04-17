import streamlit as st
import pandas as pd
import joblib as job

#Load Model and Scaler
model = job.load("../model/house_pricing_pred_model.pkl")
scaler = job.load("../model/scaler.pkl")

#Title
st.set_page_config(page_title = "House Price Predictor", page_icon = "🏠")

st.markdown("""
<h1 style='text-align: center; color: #00C9A7;'>
🏠 House Price Prediction App
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center; font-size:1.1rem; padding-bottom:1rem'>
Predict house prices instantly using Machine Learning
</p>
""", unsafe_allow_html=True)

#Inputs

col1,col2 = st.columns(2)

with col1:
    area = st.number_input("Enter Area", min_value = 500, max_value = 5000, step = 100)

with col2:    
    bedrooms = st.number_input("Enter Number of Bedrooms", min_value = 1, max_value = 10, step = 1)

age = st.number_input("Enter Age", min_value = 0, max_value = 100, step = 1)

#Button 
if st.button("Predict Price"):
    new_data = pd.DataFrame([[area,bedrooms,age]], columns = ["Area","Bedrooms","Age"])

    new_data_scaled = scaler.transform(new_data)

    pred = model.predict(new_data_scaled)

    st.success(f"Estimated Price: ₹  {int(pred.flatten()[0])}")

#Footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    background-color: #0E1117;
    color: white;
    font-size: 1rem;
}
</style>

<div class="footer">
    Developed by Akshay Prakash | Powered by Machine Learning
</div>
""", unsafe_allow_html=True)