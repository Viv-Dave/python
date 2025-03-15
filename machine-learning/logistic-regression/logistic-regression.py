import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
fields = ["radius_mean","texture_mean","perimeter_mean","area_mean","concavity_mean", "radius_worst", "perimeter_worst","area_worst", "concavity_worst","texture_worst"]
X = pd.read_csv("../logistic-regression/dataset/data.csv", usecols=fields)
y = pd.read_csv("../logistic-regression/dataset/data.csv", usecols=["diagnosis"])
y["output_conditional"] = np.where(y['diagnosis'] == "B", 0,1)
y = y["output_conditional"].values 
X.insert(0, 'bias', 1)
X = X.values
# params = np.zeros(X.shape[1])
# def hypothesis(params, X):
#     return np.dot(X, params)
# def sigmoid(z):
#     sigmoid = 1/(1+np.exp(z))
#     return sigmoid
# def mean_squared_error():
#     return None
# def gradient_ascent(y,iterations, X, learning_rate, params):
#     m = len(y)
#     for _ in range(iterations):
#         predictions = sigmoid(np.dot(X, params))
#         errors = predictions - y

#         gradients = (1/m)*np.dot(X.T, errors)
#         params -= learning_rate*gradients

#     return params
# learning_rate = 0.01
# iterations = 2000
# updated_params = gradient_ascent(y,iterations,X, learning_rate, params)
# print(updated_params)

from sklearn import linear_model
model = linear_model.LogisticRegression(solver='saga', max_iter=5000)
model.fit(X, y)
print(model.coef_)





# updated = [7.44666960e+00 ,1.30056604e+02, 1.60896170e+02, 8.59204830e+02,
#  7.28726334e+03 ,1.19759607e+00, 1.57408285e+02, 2.18341210e+02
#  ,1.05290869e+03, 1.05940048e+04 ,3.35640285e+00]
# positive = [  0.5006545,    3.85060401,   3.36126413,  21.40255745,   7.4405349,
#   -0.1100135,    4.07821012,   3.97949385,  20.81689013, -11.62038037,
#   -0.25208665]
