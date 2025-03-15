import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample temperature data (in Celsius) over 10 days
days = np.arange(1, 11).reshape(-1, 1)
temperatures = np.array([30, 32, 31, 29, 35, 36, 37, 34, 33, 38])

# --- Algorithmic Computation (C/Δ): Moving Average ---
def moving_average(data, window=3):
    return np.convolve(data, np.ones(window)/window, mode='valid')

ma_temps = moving_average(temperatures)

# --- Data-Driven Computation (Δ/C): Linear Regression ---
model = LinearRegression()
model.fit(days, temperatures)
future_days = np.arange(1, 13).reshape(-1, 1)
predicted_temps = model.predict(future_days)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(days, temperatures, marker='o', linestyle='-', label='Actual Temperature')
plt.plot(np.arange(2, 10), ma_temps, marker='s', linestyle='--', label='Moving Average')
plt.plot(future_days, predicted_temps, linestyle='dashed', label='Linear Regression Prediction')
plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.title('Algorithmic vs. Data-Driven Temperature Prediction')
plt.show()
