import pandas as pd
import joblib as job

#Load Model
model = job.load("../model/house_pricing_pred_model.pkl")
scaler = job.load("../model/scaler.pkl")

#User Input
area = float(input("Enter the Area:"))
bedrooms = int(input("Enter the Number of Bedrooms:"))
age = float(input("Enter the Age of House:"))

#Predict
df = pd.DataFrame([[area,bedrooms,age]], columns = ["Area","Bedrooms","Age"])
scaled_df = scaler.transform(df)
pred = model.predict(scaled_df)

print("Predicted house pricing:",int(pred.flatten()[0]))