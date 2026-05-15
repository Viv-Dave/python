import numpy as np
array = np.reshape(np.arange(0,12), shape=(3,4))
copy = array
answer = np.concatenate((array, copy), axis=0)
array1d = np.arange(0,10)
print(array1d)
print(np.split(array1d,10))
print(answer.shape)
np.sort(array, axis=0)
print(array)