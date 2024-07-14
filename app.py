import pickle
import pandas as pd
import numpy as np
import streamlit as st

# Load the model and data
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('cartransformed data.csv')

# Streamlit app
st.title('Car Price Predictor')

# Sidebar for user input
st.sidebar.header('User Input Features')

# Company
companies = sorted(car['company'].unique())
companies.insert(0, 'Select Company')
company = st.sidebar.selectbox('Company', companies)

# Car Model
car_models = sorted(car['name'].unique())
car_model = st.sidebar.selectbox('Car Model', car_models)

# Year
year = sorted(car['year'].unique(), reverse=True)
selected_year = st.sidebar.selectbox('Year', year)

# Fuel Type
fuel_type = car['fuel_type'].unique()
selected_fuel_type = st.sidebar.selectbox('Fuel Type', fuel_type)

# Kilometers Driven
kms_driven = st.sidebar.number_input('Kilometers Driven', min_value=0, value=0, step=1000)

# Prediction button
if st.sidebar.button('Predict'):
    # Prepare the input data
    input_data = pd.DataFrame(
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
        data=np.array([car_model, company, selected_year, kms_driven, selected_fuel_type]).reshape(1, 5)
    )

    # Make prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f'The predicted price of the car is ₹{np.round(prediction[0], 2)}')

