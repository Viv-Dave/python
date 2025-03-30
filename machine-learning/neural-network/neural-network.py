# import numpy as np
# import pandas as pd
# import torch
# from PIL import Image
# import torchvision.transforms as transforms

# image = Image.open("../neural-network/moon.jpg")
# transform = transforms.ToTensor()
# image_tensor = transform(image)
# flattened = torch.flatten(image_tensor)
# x = flattened #input tensor
# print(flattened.shape)
# def initialize_weight(weights):
#     new_weights = []
#     for i in range(len(weights)):
#         m,p = weights[i].shape
#         temp_weights = np.random.normal(0, np.sqrt(0.2), (m,p))
#         new_weights.append(temp_weights)
#     return new_weights
# weights_1 = np.zeros((3,2))
# weights_2 = np.zeros((2,1))
# weights_3 = np.zeros((1,1))
# output_layer = np.array([1,0])
# weights_array = [weights_1, weights_2, weights_3]
# value = initialize_weight(weights_array)
# print(value[0])
