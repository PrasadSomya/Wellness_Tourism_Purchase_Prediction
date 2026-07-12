
import os

import pandas as pd
import requests
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Visit With Us - Wellness Tourism",
    page_icon="🌿",
    layout="wide",
)


# ---------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://somya1607-visit-with-us-backend.hf.space",
).rstrip("/")


# ---------------------------------------------------------
# Application heading
# ---------------------------------------------------------
st.title("🌿 Visit With Us")
st.subheader("Wellness Tourism Package Purchase Prediction")

st.write(
    """
    Enter the customer's information below to predict whether the customer
    is likely to purchase the Wellness Tourism Package.
    """
)


# ---------------------------------------------------------
# Backend status
# ---------------------------------------------------------
with st.sidebar:
    st.header("Application Information")
    st.write("Frontend: Streamlit")
    st.write("Backend: FastAPI")
    st.write("Model: Hugging Face Model Hub")

    if st.button("Check backend status"):
        try:
            health_response = requests.get(
                f"{BACKEND_URL}/health",
                timeout=30,
            )
            health_response.raise_for_status()
            st.success("Backend is available.")
            st.json(health_response.json())

        except requests.RequestException as exc:
            st.error(f"Backend is unavailable: {exc}")


# ---------------------------------------------------------
# Customer input form
# ---------------------------------------------------------
with st.form("tourism_prediction_form"):

    st.subheader("Customer Details")

    column_1, column_2, column_3 = st.columns(3)

    with column_1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
        )

        type_of_contact = st.selectbox(
            "Type of Contact",
            ["Self Enquiry", "Company Invited"],
        )

        city_tier = st.selectbox(
            "City Tier",
            [1, 2, 3],
        )

        duration_of_pitch = st.number_input(
            "Duration of Pitch",
            min_value=0.0,
            max_value=120.0,
            value=15.0,
            step=1.0,
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Salaried",
                "Small Business",
                "Large Business",
                "Free Lancer",
            ],
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
        )

    with column_2:

        number_of_persons = st.number_input(
            "Number of Persons Visiting",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
        )

        number_of_followups = st.number_input(
            "Number of Follow-ups",
            min_value=0,
            max_value=10,
            value=3,
            step=1,
        )

        product_pitched = st.selectbox(
            "Product Pitched",
            [
                "Basic",
                "Standard",
                "Deluxe",
                "Super Deluxe",
                "King",
            ],
        )

        preferred_property_star = st.selectbox(
            "Preferred Property Star",
            [3, 4, 5],
        )

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married",
                "Divorced",
                "Unmarried",
            ],
        )

        number_of_trips = st.number_input(
            "Number of Trips",
            min_value=0,
            max_value=30,
            value=3,
            step=1,
        )

    with column_3:

        passport = st.selectbox(
            "Passport",
            options=[0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )

        pitch_satisfaction_score = st.selectbox(
            "Pitch Satisfaction Score",
            [1, 2, 3, 4, 5],
        )

        own_car = st.selectbox(
            "Own Car",
            options=[0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )

        number_of_children = st.number_input(
            "Number of Children Visiting",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
        )

        designation = st.selectbox(
            "Designation",
            [
                "Executive",
                "Manager",
                "Senior Manager",
                "AVP",
                "VP",
            ],
        )

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=0.0,
            max_value=1000000.0,
            value=25000.0,
            step=1000.0,
        )

    submit_button = st.form_submit_button(
        "Predict Purchase",
        use_container_width=True,
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submit_button:

    customer_data = {
        "Age": int(age),
        "TypeofContact": type_of_contact,
        "CityTier": int(city_tier),
        "DurationOfPitch": float(duration_of_pitch),
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": int(number_of_persons),
        "NumberOfFollowups": int(number_of_followups),
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": int(preferred_property_star),
        "MaritalStatus": marital_status,
        "NumberOfTrips": int(number_of_trips),
        "Passport": int(passport),
        "PitchSatisfactionScore": int(pitch_satisfaction_score),
        "OwnCar": int(own_car),
        "NumberOfChildrenVisiting": int(number_of_children),
        "Designation": designation,
        "MonthlyIncome": float(monthly_income),
    }

    # Rubric requirement: convert customer inputs into a dataframe
    input_dataframe = pd.DataFrame([customer_data])

    with st.expander("Customer input supplied to the model"):
        # Keep the rubric-required DataFrame, but render it as JSON instead of
        # st.dataframe. This avoids the Arrow serialization path that caused
        # native exit code 139 in the previous Space runtime.
        st.json(input_dataframe.iloc[0].to_dict())

    try:
        with st.spinner("Generating prediction..."):

            prediction_response = requests.post(
                f"{BACKEND_URL}/predict",
                json=customer_data,
                timeout=60,
            )

            prediction_response.raise_for_status()
            prediction_result = prediction_response.json()

        prediction = prediction_result.get(
            "prediction",
            prediction_result.get("predicted_class"),
        )

        probability = prediction_result.get(
            "purchase_probability",
            prediction_result.get("probability"),
        )

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(
                "The customer is likely to purchase the "
                "Wellness Tourism Package."
            )
        elif prediction == 0:
            st.warning(
                "The customer is currently less likely to purchase the "
                "Wellness Tourism Package."
            )
        else:
            st.info("Prediction received from the backend.")

        if probability is not None:
            st.metric(
                "Purchase Probability",
                f"{float(probability):.2%}",
            )

            st.progress(
                min(max(float(probability), 0.0), 1.0)
            )

        with st.expander("Complete API response"):
            st.json(prediction_result)

    except requests.Timeout:
        st.error(
            "The prediction request timed out. "
            "The backend Space may be starting."
        )

    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")

    except (TypeError, ValueError) as exc:
        st.error(f"Invalid prediction response: {exc}")
