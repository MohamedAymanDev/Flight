# ✈️ Flight Delay & Status Prediction System  

## 📌 Project Overview  

This project focuses on building a Machine Learning classification system to predict flight status (Delayed vs On-Time) using historical flight operational data.

The system analyzes temporal, airline, and route-based features to determine whether a flight will experience delays. Multiple machine learning algorithms were implemented and compared to identify the most robust and generalizable model.

---

## 📊 Dataset Description  

The dataset contains flight-level operational records including:

### 🔹 Temporal Features
- Quarter  
- Month  
- Day of Month  
- Day of Week  
- Flight Date  

### 🔹 Airline & Route Features
- Airlines  
- Origin City Name  
- Destination City Name  

### 🔹 Operational Metrics
- Departure Delay (DepDelay)  
- Arrival Delay (ArrDelay)  
- Air Time  
- Distance  

### 🎯 Target Variable
- Flight_Status (Delayed / On-Time)

---

## 🤖 Models Implemented  

The following machine learning models were trained and evaluated:

- AdaBoost  
- CatBoost  
- XGBoost  
- LightGBM  
- Hist Gradient Boosting  
- Gradient Boosting  
- Random Forest  
- Extra Trees  
- Logistic Regression  
- Naive Bayes  
- Decision Tree  

---

## 📈 Model Performance Summary  

- AdaBoost achieved the highest Test Accuracy (91.4%) with strong generalization.
- Boosting models (CatBoost, XGBoost, LightGBM) delivered stable performance around 90% accuracy.
- Random Forest and Extra Trees showed clear overfitting (≈99% training accuracy but lower test accuracy).
- Decision Tree significantly overfitted the training data.
- Balanced Accuracy confirms reasonable handling of class imbalance.

### ✅ Final Model Selection:
AdaBoost was selected as the optimal model due to its superior balance between bias and variance.

---

## 🛠️ Machine Learning Pipeline  

1. Data Cleaning & Preprocessing  
2. Feature Engineering  
3. Encoding Categorical Variables  
4. Train/Test Split  
5. Model Training  
6. Performance Evaluation using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Balanced Accuracy  
7. Model Comparison & Overfitting Analysis  

---

## 🚀 Business Impact  

- Enables airlines to predict potential delays in advance  
- Improves operational planning and scheduling  
- Enhances passenger satisfaction  
- Reduces unexpected operational costs  

---

## 💻 Technologies Used  

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- LightGBM  
- CatBoost  
- Matplotlib / Seaborn  

---

## 📌 Conclusion  

Ensemble boosting algorithms demonstrated superior performance in predicting flight status.  
The project highlights the importance of model comparison, overfitting detection, and balanced evaluation metrics in building reliable machine learning systems.
