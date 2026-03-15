# uac-care-load-forecasting
# UAC Care Load Forecasting Dashboard

## Project Overview

This project builds a predictive analytics system to forecast future care load for Unaccompanied Alien Children (UAC) using time-series analysis and machine learning.

The goal is to transform historical reporting data into predictive intelligence that helps policymakers anticipate future care capacity needs.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Streamlit

## Dataset Description

The dataset includes daily records such as:

Date – Reporting date

Children apprehended and placed in CBP custody – Daily intake volume

Children in CBP custody – Active CBP care load

Children transferred out of CBP custody – Flow into HHS system

Children in HHS Care – Active HHS care load

Children discharged from HHS Care – Successful sponsor placements

## Methodology

1. Time-series preparation and cleaning
2. Feature engineering
3. Machine learning model training
4. Forecast generation
5. Visualization using Streamlit dashboard

## Forecasting Model

The project uses Random Forest Regressor to predict future care load.

## Evaluation Metrics

MAE – Mean Absolute Error

RMSE – Root Mean Squared Error

MAPE – Mean Absolute Percentage Error

## Dashboard Features

- Interactive forecast visualization
- Adjustable forecast horizon
- Dataset preview
- Machine learning predictions

## How to Run the Project

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

## Deployment

The project can be deployed using Streamlit Community Cloud.

## Author

Muskan Pandey  
Third Year Student  
AI / Machine Learning Enthusiast
