import numpy as np
import pandas as pd


property_df = pd.read_csv("../linear-regression/dataset/train.csv")
df_filtered = property_df.drop(columns=["size_units", "lot_size_units"])
df_filtered.fillna(df_filtered.mean(), inplace=True) 

df = df_filtered.apply(pd.to_numeric, errors='coerce')

df_array = df.to_numpy()

X = df_array[:, :-1]  
y = df_array[:, -1]   

# Initialize parameters
learning_rate = 0.01
n_iterations = 10000
params = np.zeros(X.shape[1])
print(params)
def hypothesis(X, params):
    return np.dot(X, params)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

theta = np.linalg.inv(X.T @ X) @ X.T @ y
print("Optimized Parameters:", theta)
y_predicted = hypothesis(X, theta)
mean_squared_error = mean_squared_error(y, y_predicted)
print(mean_squared_error)