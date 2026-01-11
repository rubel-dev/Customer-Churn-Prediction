# customer-health-system
# Customer Churn Prediction Web App
🔗 Links 
  Live:  
  https://customer-churn-prediction-web-app-05.streamlit.app/
##  Overview
This project predicts whether a customer is likely to churn (cancel service) using structured customer data. It combines feature preprocessing, machine learning modeling, and deployment into a production-style web app that business stakeholders can use without coding.

 Made with Python, Scikit-learn, Pandas, and Streamlit.

---

##  Problem Statement
Customer churn costs companies millions annually. Predicting churn enables proactive retention strategies and targeted interventions.

---

##  Approach

### 🛠 Data Preparation
- Dropped irrelevant identifiers (RowNumber, CustomerId, Surname)
- One-hot encoded categorical features (Geography, Gender)
- Standardized numeric features

###  Modeling
- Logistic Regression as baseline
- RandomForest / XGBoost for stronger performance
- Evaluated using ROC-AUC, F1, and recall scores

###  Deployment
- Built Streamlit app for interactive prediction
- User inputs customer details, gets churn probability + risk label
- Report download functionality

---

## 📈 Results
- ROC-AUC: 0.87 (balanced performance across thresholds)
- Easy-to-interpret UI for business stakeholders

---

##  Getting Started

### Requirements
```bash
pip install -r requirements.txt
Run Locally
bash
Copy code
streamlit run streamlit_app.py



🛠 Tech Stack
Python | Scikit-Learn | Pandas | Streamlit | Git/GitHub
