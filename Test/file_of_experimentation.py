import numpy as np
max_iter = 100
A=[1,2,3,4,5]
b = np.zeros(len(A))
print(b)
n = len(A)
#for i in range(n):
#    norm = np.linalg.norm(A[i,:])
#    A[i,:] = A[i,:] / norm
#    b[i] = b[i] / norm

print(A)

eqn_err1 = np.zeros((max_iter+1,n))

print(eqn_err1)