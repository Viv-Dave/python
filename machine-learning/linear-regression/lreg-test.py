import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Update the file path to an absolute path
test_df = pd.read_csv("C:/Users/vivek/OneDrive/desktop/python/machine-learning/linear-regression/temp-dataset/test.csv", names=["x", "y"])
df = pd.DataFrame(test_df)

df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")

df.dropna(inplace=True)

x_array = df["x"].to_numpy()
y_array = df["y"].to_numpy()

def hypothesis(param_1, param_2, input_value):
    return param_1 + param_2 * input_value

def mean_squared_error(param_1, param_2, x_array, y_array):
    m = len(y_array)
    predictions = hypothesis(param_1, param_2, x_array)
    errors = predictions - y_array
    cost = (1 / (2 * m)) * np.sum(np.square(errors))
    return cost
param_1 = 0.011937938999784087
param_2 = 0.9988705055011785
mse_test = mean_squared_error(param_1, param_2, x_array, y_array)
print(f"Mean Squared Error/Cost Function : {mse_test}")
# transpose_X = np.transpose(df[["x"]])
# inverse = np.multiply(transpose_X, df[["x"]])
# inverse = np.linalg.inv(inverse)
# print(np.size(inverse))
# new_parameter = np.multiply(inverse, transpose_X)*df[["y"]]
# print(new_parameter)