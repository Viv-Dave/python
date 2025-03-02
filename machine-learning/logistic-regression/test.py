import numpy as np
import pandas as pd

def sigmoid(z):
    sigmoid = 1/(1+np.exp(z))
    return sigmoid
def predict(X, params, threshold=0.5):
    probabilities = sigmoid(np.dot(X, params))  
    return (probabilities >= threshold).astype(int)
 
fields = ["radius_mean","texture_mean","perimeter_mean","area_mean","concavity_mean", "radius_worst", "perimeter_worst","area_worst", "concavity_worst","texture_worst"]
X = pd.read_csv("../logistic-regression/dataset/data.csv", usecols=fields)
y = pd.read_csv("../logistic-regression/dataset/data.csv", usecols=["diagnosis"])
y_test = np.where(y['diagnosis'] == "B", 0, 1)
X.insert(0, 'bias', 1)
positive_params = np.array([ 0.5006545,    3.85060401,   3.36126413,  21.40255745,   7.4405349,
   -0.1100135,    4.07821012,   3.97949385,  20.81689013, -11.62038037,
   -0.25208665])
X_test = X.values 
y_pred = predict(X_test, positive_params)
accuracy = np.mean(y_pred == y_test) * 100
print(f"\n \n \n Test Accuracy for Bening or Malignant: {accuracy:.2f}%")