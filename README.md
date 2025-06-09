# Python Projects by Syeda Sania Bokhari

This repository contains two distinct data-science projects, each housed in its own folder. Below is an overview of each project, its purpose, and how to run it.

---

## 1. Telecom Customer Churn Analysis  
**Folder:** `Telecom_customer_churn_analysis/`

### Project Overview  
Telecom companies lose significant revenue when customers cancel their subscriptions (churn). In this project, we explore a telecom customer dataset to identify patterns and risk factors that lead to churn, then build predictive models to flag likely churners. By combining exploratory data analysis, feature engineering, and machine-learning, the goal is to help the business target retention efforts more effectively.

### Key Steps & Contents  
1. **Data Ingestion & Cleaning**  
   - Load raw customer data (demographics, billing information, service usage).  
   - Handle missing values, correct data types, and remove duplicate records.  

2. **Exploratory Data Analysis (EDA)**  
   - Visualize distribution of categorical features (e.g., contract type, payment method).  
   - Examine relationships between tenure, monthly charges, and churn rates.  
   - Identify key risk factors (e.g., high monthly charges, short tenure, certain services).

3. **Feature Engineering**  
   - Create new features (e.g., tenure buckets, aggregated service counts).  
   - Encode categorical variables using one-hot encoding or label encoding.  
   - Scale numerical features where appropriate.

4. **Modeling & Evaluation**  
   - Split data into training and testing sets (stratified on churn label).  
   - Train multiple classifiers:  
     - Logistic Regression  
     - Decision Tree  
     - Random Forest  
     - XGBoost  
     - Support Vector Machine (SVM)  
     - K-Nearest Neighbors (KNN)  
     - Bagging and AdaBoost ensembles  
     - Neural Network (MLP)  
   - Use grid search (or randomized search) to tune hyperparameters for each model.  
   - Evaluate on the test set using metrics: accuracy, precision, recall, F₁-score, ROC-AUC.  
   - Plot confusion matrices and feature importances for tree-based models.

5. **Results & Insights**  
   - Identify the top model (e.g., XGBoost or Random Forest) that best balances recall/precision on the “churn” class.  
   - Highlight key predictors (e.g., monthly charges, contract length, internet service).  
   - Summarize business recommendations (e.g., offer discounts to high-risk customers, bundle services to increase tenure).


