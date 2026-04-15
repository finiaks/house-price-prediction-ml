import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib as job

#Data
df = pd.read_csv("../data/house.csv")

#Features
x = df[["Area","Bedrooms","Age"]]
y = df[["Price"]]

#Scaling Data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Create Model
model = LinearRegression()

#Train Model
model.fit(x_scaled,y)

#Predict
new_df = pd.DataFrame([[1800,3,5]], columns = ["Area","Bedrooms","Age"])
scaled_df = scaler.transform(new_df)
pred = model.predict(scaled_df)

print("Predicted Pricing for House:",int(pred.flatten()[0]))

#Save the Model
job.dump(model,"../model/house_pricing_pred_model.pkl")

#Save the Scaler
job.dump(scaler,"../model/scaler.pkl")