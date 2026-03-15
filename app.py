import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.title("UAC Care Load Forecasting Dashboard")

# Load dataset
data = pd.read_csv("./dataset.csv")

data["Date"] = pd.to_datetime(data["Date"])

st.subheader("Dataset Preview")
st.write(data)

# Feature engineering
data["Day_Index"] = np.arange(len(data))

X = data[["Day_Index"]]
y = data["Children_HHS_Care"]

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Forecast horizon
days = st.slider("Select Forecast Days", 1, 30, 7)

future = np.arange(len(data), len(data) + days).reshape(-1,1)

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
