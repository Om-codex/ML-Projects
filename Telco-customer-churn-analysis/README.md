# 📉 Customer Churn Prediction – Machine Learning Project

> Predicting telecom customer churn using machine learning to help businesses retain valuable customers.

---

## 🧭 Project Overview

Customer churn is a critical business challenge faced by subscription-based companies, especially in telecom.  
This project aims to **predict whether a customer will leave (churn)** based on demographic, account, and service details.

By identifying potential churners early, companies can **proactively offer retention incentives**, thereby reducing losses and improving customer satisfaction.

---

## 🧠 Problem Statement

> Given customer data such as contract type, monthly charges, and tenure, can we predict whether the customer will churn?

The goal is to build a **binary classification model** that predicts churn (`Yes` or `No`) and provide **actionable insights** to reduce churn rate.

---

## 📊 Dataset Information

**Dataset:** [Telco Customer Churn Dataset (IBM)](https://www.kaggle.com/blastchar/telco-customer-churn)

- **Rows:** ~7,000 customers  
- **Columns:** 21 features  
- **Target Variable:** `Churn` (Yes / No)

| Feature Type | Example Features |
|---------------|------------------|
| Customer Info | gender, SeniorCitizen, Partner, Dependents |
| Account Info  | tenure, Contract, PaperlessBilling, PaymentMethod |
| Service Info  | InternetService, OnlineSecurity, StreamingTV |
| Charges Info  | MonthlyCharges, TotalCharges |

---

## ⚙️ Technologies Used

| Category | Libraries / Tools |
|-----------|------------------|
| Data Analysis | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Modeling | `scikit-learn`, `LogisticRegression` |
| Environment | `Jupyter Notebook`, `Python 3.10+` |

---

## 🧩 Project Workflow

### 1️⃣ Data Preprocessing
- Converted `TotalCharges` to numeric type.
- Encoded categorical variables using `LabelEncoder` and `OneHotEncoder`.
- Scaled numerical features using `StandardScaler`.
- Removed irrelevant features like `customerID`.
- Handled missing values and class imbalance.

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualized churn distribution and feature correlations.
- Discovered key churn indicators:
  - Customers on **month-to-month contracts** churned significantly more.
  - **Shorter tenure** and **higher monthly charges** increased churn likelihood.
  - Customers without **online security** or **tech support** were more likely to churn.

### 3️⃣ Model Building
- Tested multiple models:
  - Logistic Regression  
  - Random Forest Classifier  
  - XGBoost Classifier  
- Evaluated with:
  - Accuracy, Precision, Recall, F1-Score  
  - ROC-AUC Curve  
  - Confusion Matrix

### 4️⃣ Model Evaluation

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|-----------|------------|---------|-----------|
| Gradient Boosting Classifier | 0.817 | 0.695 | 0.552 | 0.616 |
| Logistic Regression | 0.807 | 0.656 | 0.572 | 0.611 |
| Voting Classifier | 0.817 | 0.718 | 0.507 | 0.595 |
| Random Forest Classifier | 0.801 | 0.660 | 0.516 | 0.579 |
| XGBoost | 0.789 | 0.619 | 0.529 | 0.570 |
| AdaBoost Classifier | 0.801 | 0.667 | 0.497 | 0.569 |
| Support Vector Classifier (SVC) | 0.798 | 0.660 | 0.490 | 0.563 |
| K-Nearest Neighbors (KNN) | 0.767 | 0.565 | 0.533 | 0.548 |
| Decision Tree Classifier | 0.731 | 0.494 | 0.501 | 0.497 |

### 4️⃣ Model Performance after Hyperparameter Tuning

| Model                     | Best Parameters                                                                                  | Recall Score (CV) |
|----------------------------|--------------------------------------------------------------------------------------------------|-------------------|
| SVC                        | {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}                                                | 0.840             |
| Logistic Regression         | {'C': 0.01, 'max_iter': 300, 'penalty': 'l1', 'solver': 'liblinear'}                           | 0.827             |
| Decision Tree               | {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}        | 0.789             |
| Random Forest               | {'max_depth': 10, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 100}          | 0.755             |
| K-Nearest Neighbors (KNN)   | {'n_neighbors': 7, 'p': 1, 'weights': 'uniform'}                                               | 0.539             |
| Gradient Boosting           | {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}                 | 0.516             |
| AdaBoost                    | {'learning_rate': 1.0, 'n_estimators': 300}                                                    | 0.511             |


✅ **Final Model:** Logistic Regression with High Recall Value & Better Metrics than SVC.
| Metric                | Score  |
|------------------------|--------|
| **Accuracy**           | 0.7189 |
| **Precision**          | 0.482  |
| **Recall**             | 0.8009 |
| **F1-Score**           | 0.6018 |

**Classification Report**

| Class | Precision | Recall | F1-Score | Support |
|--------|------------|---------|-----------|----------|
| 0 | 0.91 | 0.69 | 0.78 | 1294 |
| 1 | 0.48 | 0.80 | 0.60 | 467 |
| **Accuracy** |  |  | **0.72** | **1761** |
| **Macro Avg** | 0.69 | 0.75 | 0.69 | 1761 |
| **Weighted Avg** | 0.79 | 0.72 | 0.73 | 1761 |



💡 **Business Insight:**  
Customers with **month-to-month contracts**, **short tenure**, and **high charges** are at highest churn risk.  
These customers should be **targeted for loyalty discounts** or **contract upgrades**.

---


## 💡 Results Summary

- Achieved **71% accuracy** and **High Recall Value of 0.80** using Logistic Regression.
- Identified key churn indicators leading to **actionable business strategies**.

---

## 🚀 Future Improvements

- Deploy model using **Streamlit / Gradio** for live churn prediction.  
- Integrate **AutoML / Hyperparameter tuning (Optuna)**.  
- Experiment with **Deep Learning models (ANNs)** for tabular data.  
- Develop a **customer retention dashboard** showing churn probability per segment.

---

## 📂 Project Structure

Telco-Customer-Churn/
│
├── data/
│ └── customer-churn.csv
│
├── notebooks/
│ └── Telco-customer-churn-prediction.ipynb
│
├── README.md
├── requirements.txt
└── final_log_reg.pkl


---

## 🧾 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Om-codex/ML-Projects/Telco-customer-churn-analysis
cd telco-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook Telco-customer-churn-prediction.ipynb
