import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.title("UAC Care Load Forecast Dashboard")

file_name = "HHS_Unaccompanied_Alien_Children_Program.csv"

# Check dataset
if not os.path.exists(file_name):
    st.error("Dataset file not found in repository.")
else:

    data = pd.read_csv(file_name)

    st.subheader("Original Dataset Preview")
    st.write(data.head())

    # Convert Date column
    data["Date"] = pd.to_datetime(data["Date"])

    # Sort by date
    data = data.sort_values("Date")

    # Remove rows with missing values
    data = data.dropna()

    # Create numeric index
    data["Day_Index"] = np.arange(len(data))

    target = "Children in HHS Care"

    X = data[["Day_Index"]]
    y = data[target]

    # Train model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    days = st.slider("Forecast Days", 1, 30, 7)

    future = np.arange(len(data), len(data)+days).reshape(-1,1)

    forecast = model.predict(future)

    st.subheader("Forecast Results")

    forecast_df = pd.DataFrame({
        "Day": range(1, days+1),
        "Predicted Care Load": forecast
    })

    st.write(forecast_df)

    # Plot
    fig, ax = plt.subplots()

    ax.plot(data["Day_Index"], y, label="Historical Data")
    ax.plot(range(len(data), len(data)+days), forecast, label="Forecast")

    ax.set_xlabel("Time")
    ax.set_ylabel("Children in HHS Care")
    ax.legend()

    st.pyplot(fig)
