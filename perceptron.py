import numpy as np

confidence = 0
input = np.array([[1,1,1],[0,0,0],[0,0,0]])
input_2 = np.array([[1,0,0],[1,0,0], [1,0,0]])
error_input = np.array([[1,0,0], [0,1,0], [0,0,1]])
weights = np.array([1 ,1, 1])
print(weights)
print(f"INPUT_1:{np.dot(input, weights)}")
print(f"INPUT_2: {np.dot(input_2, weights)}")
print(f"INPUT_3: {np.dot(error_input, weights)}")
# and_weights = np.array([[0.3],[0.4]])
# or_weights = np.array([[0.5],[0.5]])
# bias = 0;
# output = np.dot(input, and_weights) + bias
# for i in range(len(output)):
#     if output[i] <0.5:
#         output[i] = 0
#     else:
#         output[i] = 1
# print(output)