import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.linear_model import LinearRegression  # For linear regression model

# User input for temperature data
temperatures = []  # Empty list to store user-input temperatures

print("Enter temperature readings (type 'done' to finish):")
while True:
    temp = input(f"Day {len(temperatures) + 1}: ")  # Take user input
    if temp.lower() == 'done':
        break  # Stop taking input when user enters 'done'
    try:
        temperatures.append(float(temp))  # Convert input to float and store
    except ValueError:
        print("Invalid input. Please enter a number or 'done' to finish.")

num_days = len(temperatures)  # Number of days is determined by the number of inputs
days = np.array(range(1, num_days + 1)).reshape(-1, 1)  # Create day numbers

# Convert temperature list to numpy array
temperatures = np.array(temperatures)

# --- Algorithmic Computation: Moving Average ---
def moving_average(data, window=3):
    """Calculates the moving average with a given window size."""
    return np.convolve(data, np.ones(window) / window, mode='valid')  # Smooths data using a rolling average

ma_temps = moving_average(temperatures) if len(temperatures) >= 3 else temperatures  # Compute moving average if enough data

# --- Data-Driven Computation: Linear Regression ---
model = LinearRegression()  # Initialize the linear regression model
model.fit(days, temperatures)  # Train the model using the given data
future_days = np.array(range(1, num_days + 3)).reshape(-1, 1)  # Extend days for future predictions
predicted_temps = model.predict(future_days)  # Predict future temperatures

# Plot results
plt.figure(figsize=(10, 5))  # Set figure size
plt.plot(days, temperatures, 'o-', label='Actual Temperature')  # Plot actual temperature data
if len(temperatures) >= 3:
    plt.plot(np.arange(2, len(ma_temps) + 2), ma_temps, 's--', label='Moving Average')  # Plot moving average trend
plt.plot(future_days, predicted_temps, 'd--', label='Linear Regression Prediction')  # Plot regression predictions
plt.xlabel('Days')  # Label x-axis
plt.ylabel('Temperature (°C)')  # Label y-axis
plt.legend()  # Add legend to distinguish plots
plt.title('Algorithmic vs. Data-Driven Temperature Prediction')  # Set plot title
plt.show()  # Display the graph
