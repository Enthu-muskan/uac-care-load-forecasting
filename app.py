import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="UAC Care Forecast Dashboard", layout="wide")

st.title("UAC Care Load Forecasting Dashboard")

file_name = "HHS_Unaccompanied_Alien_Children_Program.csv"

# Check dataset
if not os.path.exists(file_name):
    st.error("Dataset file not found in repository.")
else:

    data = pd.read_csv(file_name)

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")

    target = "Children in HHS Care"

    # Clean numeric values
    data[target] = data[target].astype(str).str.replace(",", "")
    data[target] = pd.to_numeric(data[target], errors="coerce")

    data = data.dropna(subset=[target])

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # -------------------------
    # KPI CARDS
    # -------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(data))
    col2.metric("Average Care Load", int(data[target].mean()))
    col3.metric("Maximum Care Load", int(data[target].max()))

    # -------------------------
    # EDA GRAPH
    # -------------------------

    st.subheader("Historical Care Load Trend")

    fig1, ax1 = plt.subplots()
    ax1.plot(data["Date"], data[target])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Children in HHS Care")
    ax1.set_title("Care Load Over Time")

    st.pyplot(fig1)

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------

    data["Day_Index"] = np.arange(len(data))

    X = data[["Day_Index"]]
    y = data[target]

    # -------------------------
    # RANDOM FOREST MODEL
    # -------------------------

    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X, y)

    rf_pred = rf_model.predict(X)

    rf_mae = mean_absolute_error(y, rf_pred)

    # -------------------------
    # ARIMA MODEL
    # -------------------------

    arima_model = ARIMA(y, order=(1,1,1))
    arima_fit = arima_model.fit()

    arima_pred = arima_fit.predict(start=0, end=len(y)-1)

    arima_mae = mean_absolute_error(y, arima_pred)

    # -------------------------
    # MODEL COMPARISON
    # -------------------------

    st.subheader("Model Performance Comparison")

    comparison_df = pd.DataFrame({
        "Model": ["Random Forest", "ARIMA"],
        "MAE": [rf_mae, arima_mae]
    })

    st.dataframe(comparison_df)

    # -------------------------
    # FORECAST SETTINGS
    # -------------------------

    st.sidebar.header("Forecast Settings")

    forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 1, 30, 7)

    # Random Forest Forecast
    future_index = np.arange(len(data), len(data)+forecast_days).reshape(-1,1)

    rf_forecast = rf_model.predict(future_index)

    # ARIMA Forecast
    arima_forecast = arima_fit.forecast(steps=forecast_days)

    # -------------------------
    # FORECAST RESULTS
    # -------------------------

    st.subheader("Forecast Results")

    forecast_df = pd.DataFrame({
        "Day": range(1, forecast_days+1),
        "Random Forest Forecast": rf_forecast,
        "ARIMA Forecast": arima_forecast
    })

    st.dataframe(forecast_df)

    # -------------------------
    # FORECAST GRAPH
    # -------------------------

    st.subheader("Forecast Visualization")

    fig2, ax2 = plt.subplots()

    ax2.plot(data["Day_Index"], y, label="Historical Data")

    ax2.plot(range(len(data), len(data)+forecast_days),
             rf_forecast, label="Random Forest Forecast")

    ax2.plot(range(len(data), len(data)+forecast_days),
             arima_forecast, label="ARIMA Forecast")

    ax2.set_xlabel("Time Index")
    ax2.set_ylabel("Children in HHS Care")

    ax2.legend()

    st.pyplot(fig2)

    st.success("Forecast generated successfully using ARIMA and Random Forest models.")
