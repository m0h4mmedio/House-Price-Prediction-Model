import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# ---------------------------------------------------------
# Load Model and Pipeline
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    pipeline = joblib.load("pipeline.pkl")

    return model, pipeline


model, pipeline = load_model()


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.title("🏠 House Price Prediction")
st.write("Enter the details of the house below to predict its median value.")


# ---------------------------------------------------------
# Input Form
# ---------------------------------------------------------
with st.form("house_prediction_form"):

    st.subheader("House Details")

    longitude = st.number_input(
        "Longitude",
        value=-122.23,
        format="%.4f"
    )

    latitude = st.number_input(
        "Latitude",
        value=37.88,
        format="%.4f"
    )

    housing_median_age = st.number_input(
        "Housing Median Age",
        min_value=1.0,
        max_value=100.0,
        value=25.0
    )

    total_rooms = st.number_input(
        "Total Rooms",
        min_value=1.0,
        value=2000.0
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        min_value=1.0,
        value=400.0
    )

    population = st.number_input(
        "Population",
        min_value=1.0,
        value=1000.0
    )

    households = st.number_input(
        "Households",
        min_value=1.0,
        value=350.0
    )

    median_income = st.number_input(
        "Median Income",
        min_value=0.0,
        value=4.0
    )

    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        [
            "<1H OCEAN",
            "INLAND",
            "NEAR OCEAN",
            "NEAR BAY",
            "ISLAND"
        ]
    )

    predict_button = st.form_submit_button(
        "🔮 Predict House Price"
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if predict_button:

    input_data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

    try:
        # Transform the input using the same pipeline
        transformed_data = pipeline.transform(input_data)

        # Make prediction
        prediction = model.predict(transformed_data)[0]

        st.success("Prediction completed successfully! 🎉")

        st.metric(
            label="Predicted House Value",
            value=f"${prediction:,.2f}"
        )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
