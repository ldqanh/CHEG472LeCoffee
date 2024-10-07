import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as model_file:
        return pickle.load(model_file)

# Load the label encoder
@st.cache_resource
def load_label_encoder():
    with open('label_encoder.pkl', 'rb') as label_encoder_file:
        return pickle.load(label_encoder_file)

model = load_model()
label_encoder = load_label_encoder()

# Title of the app
st.title("Coffee Type Prediction")

# Sidebar inputs for user preferences
st.sidebar.header("User Preferences")
time_of_day = st.sidebar.selectbox("Time of Day", ['morning', 'afternoon', 'evening'])
coffee_strength = st.sidebar.selectbox("Coffee Strength", ['mild', 'regular', 'strong'])
sweetness_level = st.sidebar.selectbox("Sweetness Level", ['unsweetened', 'lightly sweetened', 'sweet'])
milk_type = st.sidebar.selectbox("Milk Type", ['none', 'regular', 'skim', 'almond'])
coffee_temperature = st.sidebar.selectbox("Coffee Temperature", ['hot', 'iced', 'cold brew'])
flavored_coffee = st.sidebar.selectbox("Flavored Coffee", ['yes', 'no'])
caffeine_tolerance = st.sidebar.selectbox("Caffeine Tolerance", ['low', 'medium', 'high'])
coffee_bean = st.sidebar.selectbox("Coffee Bean", ['Arabica', 'Robusta', 'blend'])
coffee_size = st.sidebar.selectbox("Coffee Size", ['small', 'medium', 'large'])
dietary_preferences = st.sidebar.selectbox("Dietary Preferences", ['none', 'vegan', 'lactose-intolerant'])

# Button to trigger prediction
if st.button('Predict Coffee Type'):
    # Create a DataFrame with the user inputs
    input_data = pd.DataFrame({
        'Token_0': [time_of_day],
        'Token_1': [coffee_strength],
        'Token_2': [sweetness_level],
        'Token_3': [milk_type],
        'Token_4': [coffee_temperature],
        'Token_5': [flavored_coffee],
        'Token_6': [caffeine_tolerance],
        'Token_7': [coffee_bean],
        'Token_8': [coffee_size],
        'Token_9': [dietary_preferences]
    })

    # One-hot encode the input data
    input_encoded = pd.get_dummies(input_data, columns=[f'Token_{i}' for i in range(10)])

    # Get the feature names from the model
    feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else input_encoded.columns

    # Ensure all columns from training are present in the input
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder columns to match the training data
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

    # Check if input_encoded is not empty before scaling
    if not input_encoded.empty:
        # Create a new StandardScaler and fit it to the input data
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_encoded)

        # Make the prediction
        prediction = model.predict(input_scaled)[0]

        # Reverse the label encoding
        coffee_type = label_encoder.inverse_transform([prediction])[0]

        # Display the prediction
        st.subheader(f"Recommended Coffee: {coffee_type}")
    else:
        st.error("Unable to make a prediction. Please ensure all inputs are selected.")
else:
    st.write("Please select your preferences and click 'Predict Coffee Type' to get a recommendation.")
