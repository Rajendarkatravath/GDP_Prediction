import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model and scaler
with open('lasso_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
st.title('GDP per Capita Prediction')

# Input fields for independent variables
year = st.number_input('Year')
cpi = st.number_input('Consumer price index (2010 = 100)')
gdp_lcu = st.number_input('GDP (current LCU)')
inflation = st.number_input('Inflation, GDP deflator (annual %)')
exchange_rate = st.number_input('Official exchange rate (LCU per US$, period average)')
reserves = st.number_input('Total reserves (includes gold, current US$)')
population_total = st.number_input('Population, total')
population_age_15_64 = st.number_input('Population ages 15-64 (% of total population)')
money_supply_m3 = st.number_input('Money Supply M3')
base_money = st.number_input('Base Money')
currency_in_circulation = st.number_input('Currency in Circulation')
bank_reserves = st.number_input('Bank Reserves')
currency_outside_banks = st.number_input('Currency Outside Banks')
quasi_money = st.number_input('Quasi Money')
other_assets_net = st.number_input('Other Assets Net')
cbn_bills = st.number_input('CBN Bills')
special_intervention_reserves = st.number_input('Special Intervention Reserves')
gdp_billions = st.number_input('GDPBillions of US $')
per_capita_usd = st.number_input('Per CapitaUS $')
petrol_price = st.number_input('Petrol Price (Naira)')

# Create input data
input_data = np.array([[year, cpi, gdp_lcu, inflation, exchange_rate, reserves,
                        population_total, population_age_15_64, money_supply_m3,
                        base_money, currency_in_circulation, bank_reserves,
                        currency_outside_banks, quasi_money, other_assets_net,
                        cbn_bills, special_intervention_reserves, gdp_billions,
                        per_capita_usd, petrol_price]])

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Predict
if st.button('Predict GDP per Capita'):
    prediction = model.predict(input_data_scaled)
    st.write(f'Predicted GDP per Capita: ${prediction[0]:,.2f}')
