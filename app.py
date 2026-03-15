import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="UAC Care Load Forecast Dashboard")

st.title("UAC Care Load Forecasting Dashboard")

st.write("Predictive analytics dashboard for forecasting children care load using machine learning.")

# Load dataset
data = pd.read_csv("dataset.csv")

# Convert date column
data["Date"] = pd.to_datetime(data["Date"])

st.subheader("Dataset Preview")
st.dataframe(data)

# Feature Engineering
data["Day_Index"] = np.arange(len(data))

X = data[["Day_Index"]]
y = data["Children_HHS_Care"]

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Forecast horizon
st.sidebar.header("Forecast Settings")
days = st.sidebar.slider("Select Forecast Days", 1, 30, 7)

future_index = np.arange(len(data), len(data) + days).reshape(-1,1)

forecast = model.predict(future_index)

st.subheader("Forecast Results")

forecast_df = pd.DataFrame({
    "Day": range(1, days + 1),
    "Predicted HHS Care Load": forecast
})

st.write(forecast_df)

# Plot results
fig, ax = plt.subplots()

ax.plot(data["Day_Index"], y, label="Historical Data")
ax.plot(range(len(data), len(data)+days), forecast, label="Forecast")

ax.set_xlabel("Time")
ax.set_ylabel("Children in HHS Care")
ax.set_title("Future Care Load Forecast")
ax.legend()

st.pyplot(fig)

st.success("Forecast generated successfully using Random Forest Model")
