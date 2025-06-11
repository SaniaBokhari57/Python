# Python Projects by Syeda Sania Bokhari

This repository contains two distinct data-science projects, each housed in its own folder. Below is an overview of each project, its purpose, and business value. 

---

# Project 1: Telecom Customer Churn Analysis & Clustering
# Project Overview
This project analyzes customer churn behavior using supervised machine learning models and unsupervised clustering techniques. The dataset is from a telecom company, and the goal is to: 
- Predict churn using models such as Random Forest, XGBoost, and K-Nearest Neighbors (KNN).
- Segment customers into clusters using K-Means and Hierarchical Clustering to uncover business insights.
- Recommend actions based on churn risk profiles.

# Objectives
- Perform data preprocessing and handle missing values.
- Use GridSearchCV for hyperparameter tuning.
- Compare model performance using accuracy scores.
- Identify customer clusters and interpret churn risk using profiling.
- Derive business recommendations based on analytical findings.

# Target Variable: churn (1 = churned, 0 = stayed)

# Tools & Libraries Used
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- XGBoost
- Jupyter Notebook / Google Colab

# Workflow Summary
1. Preprocessing
One-hot encoding for categorical variables.
Missing value imputation using SimpleImputer.
Feature scaling via StandardScaler.

2. Modeling
Three supervised ML models were trained using GridSearchCV:
✅ Random Forest
✅ XGBoost
✅ K-Nearest Neighbors (KNN)

Each model was evaluated using:
Cross-validation accuracy
Test set accuracy
Classification report

3. Clustering & Customer Segmentation
Used unsupervised models to group customers:

4. K-Means Clustering (Silhouette and Elbow Method for k selection)

5. Hierarchical Clustering (Dendrogram + Agglomerative)

Clusters were interpreted based on average churn rate and key features like tenure, senior citizen status, internet service, and contract type.

# Model Comparison
Model	Test Accuracy
- Random Forest	76.86%
- XGBoost	77.99%
- KNN	74.92%

✅ XGBoost performed best in terms of test set accuracy.

# Key Cluster Profiles & Insights
Cluster 0 – “Mid‑Tenure, Internet Users”
Churn: ~29%
Tenure: 33 months
Strategy: Offer loyalty-based contracts and bundled perks.

Cluster 1 – “Senior Customers, High‑Risk”
Churn: ~42%
Senior: 100%
Strategy: Create senior-friendly, auto-renewing long-term plans and VIP support.

Cluster 2 – “No‑Internet, Low Churn”
Churn: ~7%
Internet: 0%
Strategy: Upsell internet & smart home bundles to a stable segment.

# Business Value
- Targeted Retention: Focus efforts on high-churn clusters (e.g. seniors).
- Strategic Upsell: Low-churn families can be approached for premium services.
- Smart Resource Allocation: Optimize retention budget based on cluster needs.

# Project 2: Heart Attack Risk Prediction & Patient Segmentation
# Project Overview:
This project aims to predict the likelihood of heart attacks in patients using a variety of machine learning algorithms and segment the population into risk-based clusters using unsupervised learning techniques. The analysis is based on clinical and lifestyle attributes such as age, BMI, cholesterol levels, blood pressure, smoking habits, and diet.

# Objective
- Predict patients at risk of heart attacks using classification models.
- Segment patients into risk groups using clustering techniques.
- Identify the most influential health and lifestyle variables.

# Data Overview
The dataset consists of anonymized patient data with the following features:
- Demographics: Age, Gender
- Clinical Factors: Blood Pressure (Systolic & Diastolic), Cholesterol, BMI
- Lifestyle: Smoking habits, Diet type (low-fat/high-fat), Physical activity

Target: Heart attack occurrence (0 = No, 1 = Yes)

# # Workflow Summary
Preprocessing:
- Null/missing value handling
- One-hot encoding of categorical variables
- Feature scaling with StandardScaler
- Addressed class imbalance using SMOTE
- Model Training
- Implemented various supervised classifiers
- Applied GridSearchCV for hyperparameter tuning
- Evaluated models with cross-validation (F1 as primary metric)
- Clustering
- Scaled full dataset
- Applied K-Means and Hierarchical Clustering
- Used Elbow & Silhouette methods for optimal k

# Modeling & Evaluation
Supervised Learning Models Used:
Model	Purpose / Highlights
- Logistic Regression	Baseline interpretability
- Gaussian Naive Bayes	Highest recall and F1-score on at-risk class
- Random Forest	Robust performance, feature importance
- Support Vector Machine	SVM with RBF kernel for non-linearity
- XGBoost	Tuned with GridSearchCV, good precision-recall
- K-Nearest Neighbors	Distance-based, tuned k and distance metric

# All models evaluated on accuracy, precision, recall, F1-score, and confusion matrix.

# Best Performing Model: Gaussian Naive Bayes
Recall (at-risk class): 0.49
F1-score (at-risk class): 0.41

Outperformed others in identifying true positives.

# Unsupervised Clustering
K-Means Clustering:
- Elbow + Silhouette analysis → Optimal k = 2
- Cluster 0: Majority with lower cholesterol/BMI, non-smokers, healthier profiles.
- Cluster 1: Higher BMI, blood pressure, cholesterol, and smoking incidence.

# Hierarchical Clustering:
Confirmed the two-group structure via Ward linkage + dendrogram
Aligned closely with K-Means segmentation.

# Key Insights
✅ Supervised Learning:
Naive Bayes is most effective for identifying at-risk patients.
Top predictors: BMI, Cholesterol, Systolic/Diastolic BP.
Lifestyle variables (smoking, diet) played significant secondary roles.

✅ Clustering:
Patient population naturally separates into low-risk and high-risk groups.
Enables preemptive monitoring and tailored intervention for cluster 1 individuals.

# Technical Stack
Language: Python
Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost
Modeling Tools: SMOTE, GridSearchCV, StandardScaler
Visualization: Confusion matrix heatmaps, silhouette plots, dendrograms

# Practical Takeaways
- Use clustering for risk stratification of patients.
- Deploy Naive Bayes in hospital triage systems for quick risk screening.
- Prioritize BMI, cholesterol, and BP as red-flag indicators in patient assessments.

-- END --
