import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.title("UAC Care Load Forecast Dashboard")

# show files in repo
st.write("Files inside repository:")
st.write(os.listdir())

file_name = "HHS_Unaccompanied_Alien_Children_Program.csv"

if file_name not in os.listdir():
    st.error("Dataset file not found in GitHub repository.")
else:

    data = pd.read_csv(file_name)

    st.subheader("Dataset Preview")
    st.write(data.head())

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")

    data["Day_Index"] = np.arange(len(data))

    target = "Children in HHS Care"

    X = data[["Day_Index"]]
    y = data[target]

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

    fig, ax = plt.subplots()

    ax.plot(data["Day_Index"], y, label="Historical Data")
    ax.plot(range(len(data), len(data)+days), forecast, label="Forecast")

    ax.set_xlabel("Time")
    ax.set_ylabel("Children in HHS Care")

    ax.legend()

    st.pyplot(fig)
