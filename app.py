"""Interactive house-value prediction app.

Run with: streamlit run app.py
Keep model.pkl and pipeline.pkl in this same directory.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="HomeValue Studio", page_icon="🏠", layout="wide")


@st.cache_resource(show_spinner="Loading the trained prediction model…")
def load_model():
    """Load the fitted transformer and model once per Streamlit session."""
    app_folder = Path(__file__).parent
    model_path = app_folder / "model.pkl"
    pipeline_path = app_folder / "pipeline.pkl"

    if not model_path.exists() or not pipeline_path.exists():
        missing = [
            name
            for name, path in (("model.pkl", model_path), ("pipeline.pkl", pipeline_path))
            if not path.exists()
        ]
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing)}")

    return joblib.load(model_path), joblib.load(pipeline_path)


# These are real observations from the California Housing dataset. Values are in USD.
EXAMPLES = {
    "Bay view starter home — $452,600": {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY",
        "actual_price": 452600.0,
    },
    "Bay Area family home — $358,500": {
        "longitude": -122.22,
        "latitude": 37.86,
        "housing_median_age": 21.0,
        "total_rooms": 7099.0,
        "total_bedrooms": 1106.0,
        "population": 2401.0,
        "households": 1138.0,
        "median_income": 8.3014,
        "ocean_proximity": "NEAR BAY",
        "actual_price": 358500.0,
    },
    "Established Bay home — $352,100": {
        "longitude": -122.24,
        "latitude": 37.85,
        "housing_median_age": 52.0,
        "total_rooms": 1467.0,
        "total_bedrooms": 190.0,
        "population": 496.0,
        "households": 177.0,
        "median_income": 7.2574,
        "ocean_proximity": "NEAR BAY",
        "actual_price": 352100.0,
    },
}

DEFAULT_HOME = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 25.0,
    "total_rooms": 2000.0,
    "total_bedrooms": 400.0,
    "population": 1000.0,
    "households": 350.0,
    "median_income": 4.0,
    "ocean_proximity": "<1H OCEAN",
}


def set_home_values(home: dict) -> None:
    """Copy an example into the widgets, excluding its target price."""
    for field, value in home.items():
        if field != "actual_price":
            st.session_state[field] = value
    st.session_state["active_example"] = home.get("actual_price")


for field, value in DEFAULT_HOME.items():
    st.session_state.setdefault(field, value)
st.session_state.setdefault("active_example", None)


st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(135deg, #f6fbf7 0%, #edf4ff 100%); }
      .hero { padding: 1.4rem 1.7rem; border-radius: 20px; color: white;
              background: linear-gradient(110deg, #0d5c4d, #1678a5); margin-bottom: 1.25rem; }
      .hero h1 { margin: 0; font-size: 2.2rem; }
      .hero p { margin: .35rem 0 0; font-size: 1.05rem; opacity: .92; }
      .eyebrow { font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 700; opacity: .78; }
      .watermark { position: fixed; bottom: 14px; right: 24px; z-index: 999;
                   color: rgba(13, 92, 77, .34); font-weight: 800; letter-spacing: .1em;
                   font-size: .78rem; pointer-events: none; }
      div[data-testid="stMetric"] { background: rgba(255,255,255,.72); border-radius: 14px; padding: .8rem; }
    </style>
    <div class="watermark">m0h4mmedio</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1564013799919-ab600027ffc6?auto=format&fit=crop&w=900&q=80")
    st.header("Welcome to HomeValue Studio")
    st.write(
        "A hands-on first machine-learning project that turns housing details into an estimated median home value."
    )
    st.divider()
    st.subheader("How to use it")
    st.markdown(
        """
        1. Choose a real example or enter your own home details.
        2. Click **Predict house value**.
        3. Review the estimate and, for examples, compare it with the known sale value.

        *Tip: Median income is reported in the California Housing dataset's original units, not as a dollar salary.*
        """
    )
    st.divider()
    st.caption("Created by m0h4mmedio · First ML project")


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">California housing intelligence</div>
      <h1>🏠 HomeValue Studio</h1>
      <p>Explore a home, make a prediction, and understand the number behind it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

example_choice = st.selectbox(
    "Try a real California Housing example",
    options=["Custom home"] + list(EXAMPLES.keys()),
    help="Examples include a known historical median value so you can assess the model's estimate.",
)

if example_choice != "Custom home":
    left, right = st.columns([3, 1])
    with left:
        st.info("This is a real dataset observation. Load it, then predict to compare the model with its recorded value.")
    with right:
        if st.button("Load example", use_container_width=True):
            set_home_values(EXAMPLES[example_choice])
            st.rerun()
else:
    st.session_state["active_example"] = None

st.subheader("Tell us about the home")
with st.form("house_prediction_form", border=False):
    location_col, home_col, community_col = st.columns(3)
    with location_col:
        st.caption("LOCATION")
        longitude = st.number_input("Longitude", format="%.4f", key="longitude")
        latitude = st.number_input("Latitude", format="%.4f", key="latitude")
        ocean_proximity = st.selectbox(
            "Ocean proximity",
            ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"],
            key="ocean_proximity",
        )
    with home_col:
        st.caption("HOME PROFILE")
        housing_median_age = st.number_input("Median age (years)", min_value=1.0, max_value=100.0, key="housing_median_age")
        total_rooms = st.number_input("Total rooms", min_value=1.0, key="total_rooms")
        total_bedrooms = st.number_input("Total bedrooms", min_value=1.0, key="total_bedrooms")
    with community_col:
        st.caption("NEIGHBOURHOOD")
        population = st.number_input("Population", min_value=1.0, key="population")
        households = st.number_input("Households", min_value=1.0, key="households")
        median_income = st.number_input("Median income (dataset units)", min_value=0.0, format="%.4f", key="median_income")

    predict_button = st.form_submit_button("🔮 Predict house value", use_container_width=True, type="primary")


def predict_value(data: pd.DataFrame) -> float:
    model, pipeline = load_model()
    transformed_data = pipeline.transform(data)
    return float(model.predict(transformed_data)[0])


def matching_actual_price(data: pd.DataFrame) -> float | None:
    """Return a recorded target only when the submitted inputs are an intact example."""
    submitted = data.iloc[0].to_dict()
    for example in EXAMPLES.values():
        numeric_match = all(
            np.isclose(float(submitted[field]), float(example[field]))
            for field in DEFAULT_HOME
            if field != "ocean_proximity"
        )
        if numeric_match and submitted["ocean_proximity"] == example["ocean_proximity"]:
            return float(example["actual_price"])
    return None


if predict_button:
    input_data = pd.DataFrame(
        {
            "longitude": [longitude], "latitude": [latitude],
            "housing_median_age": [housing_median_age], "total_rooms": [total_rooms],
            "total_bedrooms": [total_bedrooms], "population": [population],
            "households": [households], "median_income": [median_income],
            "ocean_proximity": [ocean_proximity],
        }
    )
    try:
        prediction = predict_value(input_data)
        st.session_state["last_prediction"] = prediction
        st.session_state["last_input"] = input_data
    except Exception as error:
        st.error(f"I couldn't make a prediction: {error}")
        st.caption("Check that model.pkl and pipeline.pkl are trained on these same nine feature names.")


if "last_prediction" in st.session_state:
    prediction = st.session_state["last_prediction"]
    actual_price = matching_actual_price(st.session_state["last_input"])
    st.divider()
    st.subheader("Your valuation")

    if actual_price is not None:
        difference = prediction - actual_price
        error_percent = abs(difference) / actual_price * 100
        estimate_col, actual_col, accuracy_col = st.columns(3)
        estimate_col.metric("Model estimate", f"${prediction:,.0f}")
        actual_col.metric("Recorded value", f"${actual_price:,.0f}")
        accuracy_col.metric("Difference", f"${difference:,.0f}", f"{error_percent:.1f}% from recorded value")
        st.bar_chart(pd.DataFrame({"Value (USD)": [prediction, actual_price]}, index=["Model estimate", "Recorded value"]))
        if error_percent <= 10:
            st.success("Strong example result: the estimate is within 10% of the recorded value.")
        else:
            st.warning("A useful reminder: a model estimates patterns; it cannot know every feature of an individual property.")
    else:
        low, main, high = st.columns(3)
        low.metric("Context range", f"${prediction * 0.88:,.0f}")
        main.metric("Estimated median value", f"${prediction:,.0f}")
        high.metric("Context range", f"${prediction * 1.12:,.0f}")
        st.caption("The range is a simple ±12% discussion aid, not a calibrated confidence interval or an appraisal.")
        st.success("Prediction complete. Try loading an example above to see a transparent actual-value comparison.")


with st.expander("🧠 Meet the creator & see how this project works", expanded=False):
    st.markdown(
        """
        ### Hi, I'm m0h4mmedio

        Welcome to my first machine-learning web application. I built HomeValue Studio to make a model feel less like a black box: you can change a home's details, see an estimate immediately, and test it against real examples from the California Housing dataset.

        ### Behind the prediction

        I trained a regression model on California neighbourhood data. The app collects nine inputs: geographic position, home age, room and bedroom counts, population, households, median income, and proximity to the ocean. A saved **pipeline** prepares the numeric and categorical values exactly as it did during training—for example, encoding ocean proximity—and the saved **model** converts those prepared features into a predicted median value.

        ### Tools I used

        - **Python** for the application logic
        - **Pandas** and **NumPy** for structured data
        - **scikit-learn** for the preprocessing pipeline and regression model
        - **Joblib** to save and reload the trained artifacts
        - **Streamlit** to turn the model into this interactive web experience

        This is an educational price estimate, not a professional property appraisal. Housing markets move, and features such as condition, renovations, schools, and exact street location can matter greatly.
        """
    )
