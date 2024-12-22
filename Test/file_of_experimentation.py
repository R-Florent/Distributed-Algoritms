import numpy as np

A=[1,2,3,4,5]
b = np.zeros(len(A))
print(b)
n = len(A)
for i in range(n):
    norm = np.linalg.norm(A[i,:])
    A[i,:] = A[i,:] / norm
    b[i] = b[i] / norm

print(A)