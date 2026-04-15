# House Price Prediction using Linear Regression (Machine Learning)

## 📌 Project Overview

This project predicts **house prices** based on multiple features such as **area, number of bedrooms, and age of the house** using a **Linear Regression model**. It demonstrates how machine learning can handle real-world problems with multiple inputs.

---

## 🚀 Features

- Multi-feature Linear Regression model
- Feature scaling using StandardScaler
- Prediction system for new user input
- Model and scaler saving using joblib
- Clean and structured project architecture

---

## 🧠 Concepts Used

- Supervised Learning
- Linear Regression (Multiple Variables)
- Feature Scaling
- Model Training & Prediction
- Data Preprocessing

---

## 📂 Project Structure

```
house-price-prediction/
│
├── data/
│   └── house.csv
│
├── model/
│   ├── house_pricing_pred_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── main.py        # Training and saving model
│   └── predict_price.py     # User input prediction
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The dataset contains:

- **Area** → Size of the house
- **Bedrooms** → Number of bedrooms
- **Age** → Age of the house
- **Price** → Target value

Example:

```
Area,Bedrooms,Age,Price
1000,2,10,30000
1500,3,5,50000
2000,4,3,65000
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/finiaks/house-price-prediction-ml.git
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Train the model

```bash
python src/main.py
```

---

### 4️⃣ Run prediction

```bash
python src/predict.py
```

---

### 5️⃣ Enter input

```
Enter Area: 2000
Enter Bedrooms: 4
Enter Age: 3
```

---

### ✅ Output

```
Predicted Price: 64153
```

---

## ⚠️ Important Notes

- Input data is scaled before prediction
- Model performance depends on dataset quality
- Works best within dataset range

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Joblib

---

## 📈 Future Improvements

- Add more features (location, floor, amenities)
- Use larger real-world dataset
- Build web app using Streamlit
- Deploy model online

---

## 💡 Conclusion

This project shows how machine learning can be used to predict house prices using multiple features and proper data preprocessing techniques.

---

## 👨‍💻 Author

Akshay Prakash
