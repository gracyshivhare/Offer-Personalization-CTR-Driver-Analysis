# Offer-Personalization-CTR-Driver-Analysis

## Overview
This repository contains the submission for the **American Express Decision Science Track – Campus Challenge 2025**.  
The challenge focused on designing and implementing a scalable machine learning solution capable of generating accurate predictions on the provided dataset.  
The predictive performance was evaluated through a public leaderboard, simulating a real-world decision science problem faced by a global financial services firm.

---

## Problem Statement
The 2025 American Express Campus Challenge presented a data-driven problem centered around **decision science applications** in a simulated environment such as a "Campus Super Bowl" or cricket match performance scenario used as an analogy for customer analytics.

Participants were required to:
- Build a **predictive modeling solution** using the provided dataset.
- Create **derived variables** from existing data without modifying identifier fields.
- Maintain the **original data structure**.
- Ensure the solution is **scalable and reproducible** in real-world settings.

---

## Objective
To develop a machine learning model that accurately predicts outcomes based on provided features while maintaining interpretability, scalability, and adherence to the competition guidelines.

---

## Approach

### 1. Data Understanding
The dataset contained both numerical and categorical variables representing simulated performance metrics.  
Initial analysis involved:
- Identifying variable distributions and correlations.
- Handling missing values and outliers.
- Removing identifier or non-informative attributes.

### 2. Feature Engineering
New variables were engineered to improve model accuracy:
- Aggregations (mean, max, ratio-based features)
- Interaction terms between key variables
- Normalization and log transformations for scaling

### 3. Model Development
Multiple gradient boosting models were trained and optimized:
- **XGBoost Ranker** (`xgb_ranker_map7.pkl`)
- **XGBoost Probability Model** (`xgb_prob_map7.pkl`)
- **LightGBM Classifier** (`lgb_model.pkl`)

A blending approach combining outputs from these models produced the final submission.

### 4. Evaluation
Model performance was assessed using:
- Cross-validation on training data
- Public leaderboard results from the competition
- Consistency between validation and leaderboard metrics

---

## Tools and Technologies
- **Python** (v3.9+)
- **Core Libraries:**
  - `pandas`, `numpy` – Data preprocessing and analysis
  - `scikit-learn` – Evaluation metrics and preprocessing utilities
  - `xgboost`, `lightgbm` – Predictive modeling
  - `pickle` – Model storage and loading
- **Environment:** Jupyter Notebook / Python Script Execution
