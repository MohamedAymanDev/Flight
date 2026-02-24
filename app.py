# app.py
import streamlit as st
import pandas as pd
import pickle as pk
from pipeline_module import FlightPreprocessor, FullPipeline
import time

# =========================
# Load pipeline
# =========================
@st.cache_resource
def load_pipeline():
    with open("FlightFullPipeline.pkl", "rb") as f:
        return pk.load(f)

pipeline = load_pipeline()

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="✈️ Flight Status Predictor",
    page_icon="✈️",
    layout="wide"
)

# =========================
# Background Animation
# =========================
st.markdown(
"""
<style>
body {
    background-image: url("https://media.giphy.com/media/3oKIPwoeGErMmaI43C/giphy.gif");
    background-size: cover;
    background-attachment: fixed;
}
.big-font { font-size:32px !important; font-weight:bold; color:#003366; }
.sub-font { font-size:18px !important; color:#ffffff; text-shadow: 1px 1px 2px #000;}
</style>
""", unsafe_allow_html=True
)

# =========================
# Title & intro
# =========================
st.markdown('<p class="big-font">✈️ Flight Status Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">Predict whether a flight is likely to be On-Time or Delayed</p>', unsafe_allow_html=True)
st.divider()

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("Enter Flight Details")
month = st.sidebar.selectbox("Month", ["January","February","March","April","May","June",
                                       "July","August","September","October","November","December"])
day_of_week = st.sidebar.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday",
                                                   "Friday","Saturday","Sunday"])
airline = st.sidebar.text_input("Airline", value="Delta")
origin = st.sidebar.text_input("Origin City", value="New York")
destination = st.sidebar.text_input("Destination City", value="Los Angeles")
dep_delay = st.sidebar.number_input("Departure Delay (minutes)", value=0)
arr_delay = st.sidebar.number_input("Arrival Delay (minutes)", value=0)
air_time = st.sidebar.number_input("Air Time (minutes)", value=60)
distance = st.sidebar.number_input("Distance (miles)", value=500)
st.sidebar.markdown("---")
st.sidebar.markdown("**Made with ❤️ by Mohamed Ayman**")

# =========================
# Prediction
# =========================
if st.button("Predict Flight Status"):
    with st.spinner("Predicting... ✈️"):
        time.sleep(1)

        input_df = pd.DataFrame({
            "Month_Str": [month],
            "DayOfWeek_Str": [day_of_week],
            "Airlines": [airline],
            "OriginCityName": [origin],
            "DestCityName": [destination],
            "DepDelay": [dep_delay],
            "ArrDelay": [arr_delay],
            "AirTime": [air_time],
            "Distance": [distance]
        })

        result_df = pipeline.predict(input_df)
        prediction = result_df["Flight_Status_Pred"].values[0]

        st.divider()
        st.markdown("### Prediction Result")

        if prediction.lower() == "delayed":
            st.error("⚠️ Flight is likely to be DELAYED!")
            st.balloons()
        else:
            st.success("✅ Flight is likely to be ON-TIME!")
            st.snow()