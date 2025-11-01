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
| Modeling | `scikit-learn`, `xgboost` |
| Explainability | `SHAP` |
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
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|-----------|------------|---------|-----------|----------|
| Logistic Regression | 80.2% | 74.5% | 66.3% | 70.2% | 0.84 |
| Random Forest | 82.9% | 78.6% | 70.1% | 74.1% | 0.87 |
| XGBoost | **84.1%** | **79.2%** | **72.8%** | **75.9%** | **0.89** |

✅ **Final Model:** XGBoost – best balance between recall and ROC-AUC.

### 5️⃣ Model Explainability (SHAP)
- Used SHAP (SHapley Additive exPlanations) to interpret model predictions.
- **Top Features Influencing Churn:**
  - Contract Type  
  - Tenure  
  - Monthly Charges  
  - Internet Service  
  - Online Security

💡 **Business Insight:**  
Customers with **month-to-month contracts**, **short tenure**, and **high charges** are at highest churn risk.  
These customers should be **targeted for loyalty discounts** or **contract upgrades**.

---

## 📈 Key Visualizations

| Visualization | Description |
|----------------|-------------|
| 📊 Churn Distribution | Shows % of customers who churned |
| 🔥 Correlation Heatmap | Highlights relationships between features |
| 🌈 Feature Importance | Shows top predictors of churn |
| 🧩 SHAP Summary Plot | Explains model’s global feature impact |
| 🎯 Confusion Matrix | Displays model performance visually |

---

## 💡 Results Summary

- Achieved **84% accuracy** and **ROC-AUC of 0.89** using XGBoost.
- Identified key churn indicators leading to **actionable business strategies**.
- Used **SHAP explainability** for transparent and trustable predictions.

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
│ └── Telco-Customer-Churn.csv
│
├── notebooks/
│ └── Telco-customer-churn-prediction.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── model_training.py
│ └── evaluation.py
│
├── README.md
├── requirements.txt
└── churn_model.pkl


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
