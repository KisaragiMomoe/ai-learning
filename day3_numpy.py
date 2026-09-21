import numpy as np
# v = np.array([1, 2, 3])
# print("v = ", v)
# print("v的形状: ", v.shape)

# M = np.array([
    # [1, 2, 3],
    # [4, 5, 6],
# ])

# print("\nM = ")
# print(M)
# print("M的形状: ", M.shape)

# print("\nM的转置: ")
# print(M.T)
# print("M的形状: ", M.T.shape)

# A = np.array([
#     [1, 2],
#     [3, 4],
# ])
# B = np.array([
#     [5, 6],
#     [7, 8],
# ])
# C = A @ B
# print(C)

def softmax(x):
    x_max = np.max(x, axis = -1, keepdims = True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis = -1, keepdims = True)

# scores = np.array([2.0, 1.0, 0.1])
# print(softmax(scores))

# scores_2 = np.array([
#     [1.0, 2.0, 3.0],
#     [1.0, 2.0, 3.0],
# ])
# print(softmax(scores_2))

np.random.seed(42)

Q = np.random.randn(3, 4)
K = np.random.randn(3, 4)

d_k = 4

scores = Q @ K.T / np.sqrt(d_k)

print(scores.shape)

attn = softmax(scores)
print(attn)
print(attn.sum(axis = -1))
