name = "Jessie"
import string
import unicodedata
import torch
# We can use "_" to represent an out-of-vocabulary character, that is, any character we are not handling in our model
allowed_characters = string.ascii_letters + " .,;'" + "_"
n_letters = len(allowed_characters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )
def letterToIndex(letter):
    # return our out-of-vocabulary character if we encounter a letter unknown to our model
    if letter not in allowed_characters:
        return allowed_characters.find("_")
    else:
        return allowed_characters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
test_name = name
tensorised = lineToTensor(test_name)
print(tensorised.flatten().shape)
# shape = [58,32]
# b = torch.randn(shape)
# print(b.shape)
# # print (f"The letter 'a' becomes {lineToTensor('a')}") 
# # print (f"The name {test_name} becomes {tensorised}") 
# print(tensorised.shape)
# answer = tensorised @ b
# print(answer.shape)
torch.concat