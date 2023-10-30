import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the model and other necessary components
loaded_model = joblib.load('best_model.sav')
scaler = joblib.load('scaler.sav')  # Assuming you've saved the scaler
label_encoders = joblib.load('label_encoders.sav')  # Assuming you've saved the label encoders

# Streamlit app title
st.title('PNC Health Prediction App')

# User input fields
# Replace the options in selectboxes with your actual categorical options
woman_id = st.number_input('Woman ID', min_value=0)
age = st.number_input('Age', min_value=0)
education = st.selectbox('Education Level', ['Primary', 'Secondary', 'Higher'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Widowed'])
economic_status = st.selectbox('Economic Status', ['Low', 'Medium', 'High'])
health_status = st.selectbox('Health Status', ['Good', 'Average', 'Poor'])
danger_signs = st.number_input('Danger Signs', min_value=0)
number_of_ANC_visits = st.number_input('Number of ANC Visits', min_value=0)
received_IRS = st.selectbox('Received IRS', ['Yes', 'No'])
PNC_48_hours = st.selectbox('PNC 48 hours', ['Yes', 'No'])
PNC_7_days = st.selectbox('PNC 7 days', ['Yes', 'No'])
PNC_6_weeks = st.selectbox('PNC 6 weeks', ['Yes', 'No'])
distance_to_health_facility = st.number_input('Distance to Health Facility', min_value=0.0)
transportation_type = st.selectbox('Transportation Type', ['Type1', 'Type2', 'Type3'])  # Replace with actual types
transportation_cost = st.number_input('Transportation Cost', min_value=0)
household_income = st.number_input('Household Income', min_value=0)
chw_id = st.number_input('CHW ID', min_value=0)
chw_age = st.number_input('CHW Age', min_value=0)
chw_sex = st.selectbox('CHW Sex', ['Male', 'Female'])
contraception_type = st.selectbox('Contraception Type', ['Type1', 'Type2', 'Type3'])  # Replace with actual types
multiples = st.selectbox('Multiples', ['Yes', 'No'])
mother_weight = st.number_input('Mother Weight', min_value=0.0)
chronic_conditions = st.selectbox('Chronic Conditions', ['Yes', 'No'])
season = st.selectbox('Season', ['Rainy', 'Dry'])  # Replace with actual seasons
ANC_timing = st.selectbox('ANC Timing', ['On-time', 'Late', 'None'])
days_to_delivery = st.number_input('Days to Delivery', min_value=0)
high_risk_PNC = st.selectbox('High Risk PNC', ['Yes', 'No'])

# When the user presses the submit button
if st.button('Predict'):
    # Create a data frame from the user inputs
    input_data = pd.DataFrame([[woman_id, age, education, marital_status, economic_status, health_status, 
                                danger_signs, number_of_ANC_visits, received_IRS, PNC_48_hours, PNC_7_days, 
                                PNC_6_weeks, distance_to_health_facility, transportation_type, 
                                transportation_cost, household_income, chw_id, chw_age, chw_sex, 
                                contraception_type, multiples, mother_weight, chronic_conditions, 
                                season, ANC_timing, days_to_delivery, high_risk_PNC]],
                              columns=['woman_id', 'age', 'education', 'marital_status', 'economic_status', 
                                       'health_status', 'danger_signs', 'number_of_ANC_visits', 'received_IRS', 
                                       'PNC_48_hours', 'PNC_7_days', 'PNC_6_weeks', 'distance_to_health_facility', 
                                       'transportation_type', 'transportation_cost', 'household_income', 
                                       'chw_id', 'chw_age', 'chw_sex', 'contraception_type', 'multiples', 
                                       'mother_weight', 'chronic_conditions', 'season', 'ANC_timing', 
                                       'days_to_delivery', 'high_risk_PNC'])

    # Encode categorical variables
    for column in label_encoders:
        if column in input_data:
            input_data[column] = label_encoders[column].transform(input_data[column])

    # Scale the features
    input_scaled = scaler.transform(input_data)

    # Make a prediction
    prediction = loaded_model.predict(input_scaled)

    # Display the prediction
    st.write(f'The predicted probability of low PNC is: {prediction[0]}')


