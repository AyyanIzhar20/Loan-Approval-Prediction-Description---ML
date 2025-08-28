# Loan Approval Prediction

## Overview
This project aims to **predict whether a loan application will be approved** or not, using machine learning models trained on the **Loan Approval Prediction Dataset** from Kaggle.  

The main focus is on handling **imbalanced data** and evaluating models using **precision, recall, and F1-score** instead of just accuracy.

---

## Dataset
- **Source:** [Loan Approval Prediction Dataset (Kaggle)](https://www.kaggle.com/)  
- **Features:** Applicant income, loan amount, education, marital status, etc.  
- **Target:** `Loan_Status` (Approved / Not Approved)  

---

## Data Preprocessing
- Handled missing values:
  - **Median imputation** for numerical features  
  - **Mode imputation** for categorical features  
- Encoded categorical variables using **One-Hot Encoding**  
- Evaluated results directly on imbalanced data (no oversampling/undersampling applied in this version)  

---

## Models Used
- Logistic Regression  
- Decision Tree Classifier  

---

## Evaluation Metrics
Since accuracy is misleading on imbalanced datasets, the following metrics were emphasized:
- **Precision**  
- **Recall**  
- **F1-Score**  
- **Precision-Recall Curve**  
- **Confusion Matrix Analysis**  

---

## Results
- Logistic Regression showed better **recall** but slightly lower **precision**.  
- Decision Tree had higher **precision** but risked overfitting.  
- Tradeoffs were analyzed with confusion matrices and PR-curves.  

(Add your actual numbers, plots, or tables here for clarity.)

---


# Run the notebook
jupyter notebook Loan_Approval_Prediction.ipynb
