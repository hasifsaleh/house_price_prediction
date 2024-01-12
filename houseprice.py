import streamlit as st
import pandas as pd
import pickle


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


# Load trained model
with open("random_forest_model.pkl", "rb") as file:
    model_pipeline = pickle.load(file)


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
# def main():
#     st.title("House Price Prediction")

#     # User inputs
#     bathroom = st.number_input("Bathroom", min_value=1, max_value=10, step=1, value=2)
#     bedroom = st.number_input("Bedroom", min_value=1, max_value=10, step=1, value=3)
#     state = st.selectbox("State", options=list(state_encoding.keys()))
#     property_type = st.selectbox(
#         "Property Type", options=list(property_type_encoding.keys())
#     )

#     # Predict button
#     if st.button("Predict"):
#         # Get prediction
#         prediction = make_prediction(bathroom, bedroom, state, property_type)
#         st.success(f"The predicted house price is: RM {round(prediction[0],2)}")


def main():
    st.title(":cityscape: House Price Prediction")

    # Using columns for a better layout
    col1, col2 = st.columns(2)

    with col1:
        bathroom = st.number_input(
            "Bathroom", min_value=1, max_value=10, step=1, value=2
        )

    with col2:
        bedroom = st.number_input("Bedroom", min_value=1, max_value=10, step=1, value=3)

    # Dropdowns for state and property type
    state = st.selectbox("State", options=list(state_encoding.keys()), index=0)
    property_type = st.selectbox(
        "Property Type", options=list(property_type_encoding.keys()), index=0
    )

    # Predict button with better styling
    if st.button("Predict", key="predict"):
        prediction = make_prediction(bathroom, bedroom, state, property_type)
        st.success(f"The predicted house price is: RM {round(prediction[0], 2)}")

    # Adding a footer for aesthetic balance
    st.markdown("---")
    st.markdown("House Price Prediction App | Developed by Group 10")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
