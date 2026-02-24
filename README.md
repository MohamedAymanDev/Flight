✈️ Flight Delay & Status Prediction System
📌 Project Overview

This project aims to build a Machine Learning system for predicting flight status based on historical flight and operational data.
The model analyzes temporal, operational, and route-based features to determine whether a flight will be delayed or on time.

The system evaluates multiple advanced ensemble and classical machine learning algorithms to select the most robust and generalizable model.

📊 Dataset Description

The dataset contains flight-level operational records including:

Temporal Features:

Quarter

Month

Day of Month

Day of Week

Flight Date

Airline & Route Features:

Airline

Origin City

Destination City

Operational Metrics:

Departure Delay (DepDelay)

Arrival Delay (ArrDelay)

Air Time

Distance

Target Variable:

Flight_Status (Delayed / On-Time)

🤖 Models Implemented

The project compares multiple models:

AdaBoost

CatBoost

XGBoost

LightGBM

Hist Gradient Boosting

Gradient Boosting

Random Forest

Extra Trees

Logistic Regression

Naive Bayes

Decision Tree

📈 Model Performance & Insights

AdaBoost achieved the highest test accuracy (91.4%), showing strong generalization capability.

Boosting models (CatBoost, XGBoost, LightGBM) showed stable performance around ~90% test accuracy.

Random Forest and Extra Trees showed extreme overfitting (≈99% training accuracy but lower test accuracy).

Decision Tree significantly overfitted the training data.

Balanced Accuracy confirms the models handle class imbalance reasonably well.

🔎 Conclusion:
Ensemble boosting methods provide the best trade-off between bias and variance for this classification task.

🛠️ Technical Workflow

Data Cleaning & Preprocessing

Feature Engineering

Encoding Categorical Variables

Train/Test Split

Model Training

Performance Evaluation using:

Accuracy

Precision

Recall

F1-Score

Balanced Accuracy

Model Comparison & Selection

🚀 Business Impact

Helps airlines predict delays in advance

Improves operational planning

Enhances passenger experience

Reduces cost associated with unexpected delays
