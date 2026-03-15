import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.title("UAC Care Load Forecasting Dashboard")

st.write("Forecasting future children care load using Machine Learning")

# Load the uploaded dataset
data = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")

# Convert date column
data["Date"] = pd.to_datetime(data["Date"])

# Sort by date
data = data.sort_values("Date")

st.subheader("Dataset Preview")
st.dataframe(data)

# Select the target column
target_column = "Children in HHS Care"

# Feature Engineering
data["Day_Index"] = np.arange(len(data))

X = data[["Day_Index"]]
y = data[target_column]

# Train Machine Learning Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Forecast Horizon
st.sidebar.header("Forecast Settings")
days = st.sidebar.slider("Select Forecast Days", 1, 30, 7)

future_days = np.arange(len(data), len(data) + days).reshape(-1,1)

forecast = model.predict(future_days)

st.subheader("Forecast Results")

forecast_df = pd.DataFrame({
    "Day": range(1, days + 1),
    "Predicted Children in HHS Care": forecast
})

st.write(forecast_df)

# Visualization
fig, ax = plt.subplots()

ax.plot(data["Day_Index"], y, label="Historical Data")
ax.plot(range(len(data), len(data) + days), forecast, label="Forecast")

ax.set_xlabel("Time")
ax.set_ylabel("Children in HHS Care")
ax.set_title("Future Care Load Forecast")

ax.legend()

st.pyplot(fig)

st.success("Forecast generated successfully using Random Forest Model")
