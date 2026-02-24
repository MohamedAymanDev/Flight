# pipeline_module.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler

# ========================
# Custom Preprocessing Transformer
# ========================
class FlightPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols, numerical_cols, onehot, scaler_std, scaler_robust):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.onehot = onehot
        self.scaler_std = scaler_std
        self.scaler_robust = scaler_robust
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Categorical
        X_cat = X[self.categorical_cols]
        X_cat_enc = self.onehot.transform(X_cat)
        
        # Numerical
        X_num = X[self.numerical_cols]
        X_num_std = self.scaler_std.transform(X_num)
        X_num_robust = self.scaler_robust.transform(X_num)
        
        # Combine
        X_final = np.concatenate([X_num_std, X_num_robust, X_cat_enc], axis=1)
        return X_final

# ========================
# Full Pipeline Wrapper
# ========================
class FullPipeline:
    def __init__(self, model, preprocessor, target_encoder):
        self.model = model
        self.preprocessor = preprocessor
        self.target_encoder = target_encoder
    
    def predict(self, df):
        X = self.preprocessor.transform(df)
        y_pred = self.model.predict(X)
        y_pred_labels = self.target_encoder.inverse_transform(y_pred)
        df['Flight_Status_Pred'] = y_pred_labels
        return df