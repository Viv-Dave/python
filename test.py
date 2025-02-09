import numpy as np

# Input vector (3 features)
input_data = np.array([1.0, 2.0, 3.0])

# Weight matrix (3 inputs -> 2 outputs)
weights = np.array([[0.2, 0.8], 
                    [0.5, 0.3], 
                    [0.9, 0.1]])

# Bias vector (1 bias per output neuron)
bias = np.array([0.1, 0.2])

# Compute output: (input * weights) + bias
output = np.dot(input_data, weights) + bias
print(output)  # Resulting vector with 2 outputs
