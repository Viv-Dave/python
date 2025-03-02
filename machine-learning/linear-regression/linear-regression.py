import numpy as np
import pandas as pd


property_df = pd.read_csv("../linear-regression/dataset/train.csv")
df_filtered = property_df.drop(columns=["size_units", "lot_size_units"])
df_filtered.dropna(inplace=True)

df = df_filtered.apply(pd.to_numeric, errors='coerce')

df_array = df.to_numpy()

X = df_array[:, :-1]  
y = df_array[:, -1]   

# Initialize parameters
learning_rate = 0.01
n_iterations = 10000
params = np.zeros(X.shape[1])
print(params)
# def hypothesis(X, params):
#     return np.dot(X, params)

# def mean_squared_error(y_true, y_pred):
#     return np.mean((y_true - y_pred) ** 2)

# # Define gradient descent function
# def gradient_descent(X, y, params, learning_rate, iterations):
#     m = len(y)
    
#     for _ in range(iterations):
#         predictions = hypothesis(X, params)
#         errors = predictions - y

#         if np.isnan(errors).any() or np.isinf(errors).any():
#             print("Error: NaN or Inf encountered in errors!")
#             exit()

#         gradients = (1/m) * np.dot(X.T, errors)
#         params -= learning_rate * gradients

#         if np.isnan(params).any() or np.isinf(params).any():
#             print("Error: NaN or Inf encountered in params!")
#             exit()
    
    
#     return params


theta = np.linalg.inv(X.T @ X) @ X.T @ y
print("Optimized Parameters:", theta)
