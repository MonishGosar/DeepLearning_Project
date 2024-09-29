import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained ANN model
model_file = 'ann.h5'  # update this path if necessary

# Load the ANN model
ann_model = load_model(model_file)

# Define the mappings for categorical columns
mappings = {
    'Gender': {'Male': 0, 'Female': 1},
    'Customer Type': {'Loyal Customer': 0, 'disloyal Customer': 1},
    'Type of Travel': {'Personal Travel': 0, 'Business travel': 1},
    'Class': {'Eco Plus': 0, 'Business': 1, 'Eco': 2},
    'satisfaction': {'neutral or dissatisfied': 0, 'satisfied': 1}
}

# Define the columns required for the model
columns = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
           'Flight Distance', 'Departure/Arrival time convenient', 'Seat comfort', 
           'On-board service', 'Leg room service', 'Baggage handling', 
           'Checkin service', 'Inflight service', 'Cleanliness', 
           'Departure Delay in Minutes', 'Arrival Delay in Minutes']

# Streamlit App Title
st.title('Customer Satisfaction Prediction App')

# Create columns to organize the inputs better
col1, col2 = st.columns(2)

# Inputs in first column
with col1:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    customer_type = st.selectbox('Customer Type', ['Loyal Customer', 'disloyal Customer'])
    age = st.slider('Age', min_value=18, max_value=100, value=30)
    type_of_travel = st.selectbox('Type of Travel', ['Personal Travel', 'Business travel'])
    travel_class = st.selectbox('Class', ['Eco Plus', 'Business', 'Eco'])
    flight_distance = st.number_input('Flight Distance', min_value=0, value=500)

# Inputs in second column
with col2:
    departure_arrival_time_convenient = st.slider('Departure/Arrival time convenient', 0, 5, 3)
    seat_comfort = st.slider('Seat comfort', 0, 5, 3)
    on_board_service = st.slider('On-board service', 0, 5, 3)
    leg_room_service = st.slider('Leg room service', 0, 5, 3)
    baggage_handling = st.slider('Baggage handling', 0, 5, 3)
    checkin_service = st.slider('Checkin service', 0, 5, 3)

# Create another row for the remaining inputs
col3, col4 = st.columns(2)

# Inputs in third column
with col3:
    inflight_service = st.slider('Inflight service', 0, 5, 3)
    cleanliness = st.slider('Cleanliness', 0, 5, 3)

# Inputs in fourth column
with col4:
    departure_delay_minutes = st.number_input('Departure Delay in Minutes', min_value=0, value=0)
    arrival_delay_minutes = st.number_input('Arrival Delay in Minutes', min_value=0, value=0)

# Convert the inputs to the required format
input_data = pd.DataFrame({
    'Gender': [mappings['Gender'][gender]],
    'Customer Type': [mappings['Customer Type'][customer_type]],
    'Age': [age],
    'Type of Travel': [mappings['Type of Travel'][type_of_travel]],
    'Class': [mappings['Class'][travel_class]],
    'Flight Distance': [flight_distance],
    'Departure/Arrival time convenient': [departure_arrival_time_convenient],
    'Seat comfort': [seat_comfort],
    'On-board service': [on_board_service],
    'Leg room service': [leg_room_service],
    'Baggage handling': [baggage_handling],
    'Checkin service': [checkin_service],
    'Inflight service': [inflight_service],
    'Cleanliness': [cleanliness],
    'Departure Delay in Minutes': [departure_delay_minutes],
    'Arrival Delay in Minutes': [arrival_delay_minutes]
})

# Make a prediction
if st.button('Predict Satisfaction'):
    # Ensure the input data is scaled appropriately if your model requires it
    prediction = ann_model.predict(input_data)
    prediction = (prediction > 0.5).astype(int)[0][0]

    # # Process the output if necessary, e.g., using argmax for categorical outputs
    # predicted_class = np.argmax(prediction, axis=1)
    satisfaction_mapping = {0: 'Neutral or Dissatisfied', 1: 'Satisfied'}
    st.write(f'The customer is likely to be: **{satisfaction_mapping[prediction]}**')

