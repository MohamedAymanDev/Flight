# full_pipeline.py
import pickle as pk
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from pipeline_module import FlightPreprocessor, FullPipeline

# ========================
# Load Pickle objects
# ========================
Target = pk.load(open('Target.pkl','rb'))
Scaler_Standard = pk.load(open('Stander.pkl','rb'))
Scaler_Robust = pk.load(open('Robus.pkl','rb'))
Model = pk.load(open('XGB.pkl','rb'))

# ========================
# Recreate OneHotEncoder safe for new categories
# ========================
categorical_cols = ['Month_Str', 'DayOfWeek_Str', 'Airlines', 'OriginCityName', 'DestCityName']
numerical_cols = ['DepDelay', 'ArrDelay', 'AirTime', 'Distance']

# اعادة تعريف encoder مع ignore unknowns
OneHot = OneHotEncoder(handle_unknown='ignore')
# dummy fit على القيم اللي ظهرت أثناء التدريب
dummy = [['January','Monday','Delta','New York','Los Angeles']]
OneHot.fit(dummy)

# ========================
# Instantiate Preprocessor
# ========================
preprocessor = FlightPreprocessor(
    categorical_cols=categorical_cols,
    numerical_cols=numerical_cols,
    onehot=OneHot,
    scaler_std=Scaler_Standard,
    scaler_robust=Scaler_Robust
)

# ========================
# Create Full Pipeline
# ========================
pipeline = FullPipeline(
    model=Model,
    preprocessor=preprocessor,
    target_encoder=Target
)

# ========================
# Save Pipeline
# ========================
pk.dump(pipeline, open('FlightFullPipeline.pkl','wb'))
print("✅ Full pipeline saved as FlightFullPipeline.pkl")