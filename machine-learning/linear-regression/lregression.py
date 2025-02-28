import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
reviews = pd.read_csv("../machine-learning/temp-dataset/temptrain.csv", names=["x", "y"])
df = pd.DataFrame(reviews)

# Convert to numeric, checking for NaNs
df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")

x_array = df["x"].to_numpy()
y_array = df["y"].to_numpy()

# Hypothesis function
def hypothesis(param_1, param_2, input_value):
    return param_1 + param_2 * input_value

def mean_squared_error(param_1, param_2, x_array, y_array):
    m = len(y_array)
    predictions = hypothesis(param_1, param_2, x_array)
    errors = predictions - y_array
    cost = (1 / (2 * m)) * np.sum(np.square(errors))
    return cost

# Gradient Descent
def gradient_descent(iterations, l_rate, x_array, y_array):
    m = len(y_array)
    param_1, param_2 = 0, 0  
    for i in range(iterations):
        predictions = hypothesis(param_1, param_2, x_array)
        errors = predictions - y_array

        gradient_param1 = (1/m) * np.sum(errors)
        gradient_param2 = (1/m) * np.sum(errors * x_array)

        param_1 -= l_rate * gradient_param1
        param_2 -= l_rate * gradient_param2

    return param_1, param_2

# Parameters for gradient descent
learning_rate = 0.0001
n_iterations = 1000

param_1, param_2 = gradient_descent(n_iterations, learning_rate, x_array, y_array)

# Print final parameters
print(f"Final Parameters: param_1 = {param_1}, param_2 = {param_2}")

y_pred = hypothesis(param_1, param_2, x_array)

plt.scatter(x_array, y_array, color="blue", label="Actual Data")
plt.plot(x_array, y_pred, color="red", label="Linear Regression")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()
