import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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

#Split Data
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y, test_size = 0.3, random_state = 42)

#Train Model
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

#Metrics
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("MAE:",mae)
print("MSE:",mse)
print("R2 Score:",r2)

#Predict
new_df = pd.DataFrame([[1800,3,5]], columns = ["Area","Bedrooms","Age"])
scaled_df = scaler.transform(new_df)
pred = model.predict(scaled_df)

print("Predicted Pricing for House:",int(pred.flatten()[0]))

#Save the Model
job.dump(model,"../model/house_pricing_pred_model.pkl")

#Save the Scaler
job.dump(scaler,"../model/scaler.pkl")