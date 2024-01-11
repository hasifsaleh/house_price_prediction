import streamlit as st
import pandas as pd

import pickle

import joblib


# Manual encodings (same as used in the model training)
property_type_encoding = {
    "Service Residence": 1,
    "Apartment": 2,
    "Flat": 3,
    "Studio": 4,
    "Condominium": 5,
    "Others": 6,
    "Townhouse Condo": 7,
    "Duplex": 8,
}

state_encoding = {
    "Kuala Lumpur": 1,
    "Melaka": 2,
    "Selangor": 3,
    "Penang": 4,
    "Johor": 5,
    "Sarawak": 6,
    "Putrajaya": 7,
    "Perak": 8,
    "Negeri Sembilan": 9,
    "Sabah": 10,
    "Pahang": 11,
    "Terengganu": 12,
    "Kelantan": 13,
    "Labuan": 14,
    "Kedah": 15,
}


# Load your trained model (ensure it's in the same directory or provide the correct path)
with open("random_forest_model.pkl", "rb") as file:
    model_pipeline = pickle.load(file)

# Loading the model using joblib
# model_pipeline = joblib.load("random_forest_model.joblib")


# Function to encode user inputs and make predictions
def make_prediction(bathroom, bedroom, state, property_type):
    # Apply the manual encoding to the inputs
    encoded_state = state_encoding.get(state, 0)
    encoded_property_type = property_type_encoding.get(property_type, 0)

    # Create a DataFrame with the input values
    input_df = pd.DataFrame(
        [[bathroom, bedroom, encoded_state, encoded_property_type]],
        columns=["Bathroom", "Bedroom", "state", "Property Type"],
    )

    # Make a prediction
    prediction = model_pipeline.predict(input_df)
    return prediction


# Streamlit UI
def main():
    st.title("House Price Prediction")

    # User inputs
    bathroom = st.number_input("Bathroom", min_value=1, max_value=10, step=1, value=2)
    bedroom = st.number_input("Bedroom", min_value=1, max_value=10, step=1, value=3)
    state = st.selectbox("State", options=list(state_encoding.keys()))
    property_type = st.selectbox(
        "Property Type", options=list(property_type_encoding.keys())
    )

    # Predict button
    if st.button("Predict"):
        # Get prediction
        prediction = make_prediction(bathroom, bedroom, state, property_type)
        st.success(f"The predicted house price is: RM {round(prediction[0],2)}")


if __name__ == "__main__":
    main()
