from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = 'D:/python/deep-learning/numpy-basics/image.jpg'

image = Image.open(IMAGE_PATH).convert('L')
A = np.array(image, dtype=np.float64)
U, S, Vt = np.linalg.svd(A)
print(S[:30])
print(S[-30:])
plt.semilogy(S)
plt.xlabel("Index")
plt.ylabel("Singular value (log scale)")
plt.title("Singular Value Spectrum")
plt.show()

# k_max = min(A.shape)
# print(f"Choose from {k_max} values")
# k = int(input(""))
# def reconstruct(A, k):
#     U, S, Vt = np.linalg.svd(A)
#     Uk = U[:, :k]
#     Sk = np.diag(S[:k])
#     Vk = Vt[:k, :]
#     return Uk @ Sk @ Vk

# def relative_fro_error(A, A_k):
#     return np.linalg.norm(A - A_k, 'fro') / np.linalg.norm(A, 'fro')
# A_k = reconstruct(A, k)
# print(relative_fro_error(A, A_k))
# plt.subplot(2,1,1)
# plt.imshow(A_k)
# plt.title(f"Reconstructed for {k}")
# plt.subplot(2,1,2)
# plt.imshow(A)
# plt.title(f"Original")
# plt.show()