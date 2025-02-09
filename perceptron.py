import numpy as np

confidence = 0
input = np.array([[0,0],[0,1],[1,0],[1,1]])
and_weights = np.array([[0.3],[0.4]])
or_weights = np.array([[0.5],[0.5]])
bias = 0;
output = np.dot(input, and_weights) + bias
for i in range(len(output)):
    if output[i] <0.5:
        output[i] = 0
    else:
        output[i] = 1
print(output)