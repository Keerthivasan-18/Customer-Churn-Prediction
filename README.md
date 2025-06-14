# 🔄 Customer Churn Prediction using Machine Learning

This project focuses on building a machine learning model to predict whether a customer will churn (leave a service) or not. It uses a cleaned telecom dataset and applies Exploratory Data Analysis (EDA), preprocessing, and a Random Forest classifier to make predictions.

---

## 📁 Dataset Overview

- **Source**: Telco Customer Churn Dataset (commonly available on Kaggle)
- **Records**: ~7,000
- **Target**: `Churn` (Yes / No)

### 🔢 Key Features:
- Demographics: Gender, SeniorCitizen, Partner, Dependents
- Service usage: InternetService, StreamingTV, TechSupport
- Account info: Contract, MonthlyCharges, TotalCharges, Tenure

---

## 🎯 Objective

To develop a classification model that accurately identifies customers likely to churn, enabling companies to improve retention strategies.

---

## 🛠️ Tools & Technologies

- Python 🐍
- Google Colab
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn (Random Forest, Label Encoding, StandardScaler)

---

## 📊 EDA Highlights

- Countplots of Churn vs Gender, Contract, InternetService
- Boxplots of Tenure and MonthlyCharges against Churn
- Heatmap of feature correlation
- Detected class imbalance

---

## 🧠 Machine Learning Workflow

1. **Data Cleaning**:
   - Removed `customerID`
   - Converted `TotalCharges` to numeric
   - Dropped any missing values

2. **Preprocessing**:
   - Label encoding for categorical columns
   - Feature scaling using `StandardScaler`

3. **Model**:
   - `RandomForestClassifier` with 100 trees
   - Trained on 80% of data

4. **Evaluation**:
   - Accuracy Score
   - Confusion Matrix
   - Precision, Recall, F1-Score

---

## 📈 Model Performance

| Metric      | Score   |
|-------------|---------|
| Accuracy    | ~83%    |
| Precision   | ~0.84   |
| Recall      | ~0.80   |
| F1-Score    | ~0.82   |

*(You can update these with your actual results)*

---

## 🧪 Example Code Snippets

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
