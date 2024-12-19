import numpy as np
from numpy import linalg as LA

# Matrix A and vector b
A = np.array([[4, 1, 2],
              [3, 5, 1],
              [1, 1, 3]])

b = np.array([4, 7, 3])

n = len(b)
#W = (1/n) * np.ones((n,n))

W = np.zeros((n,n))
for i in range(n):
    W[i,i] = 1/3
for i in range(n-1):
    W[i,i+1] = 1/3
    W[i+1,i] = 1/3
W[0,9] = 1/3
W[9,0] = 1/3


def local_error(A,b,X,W):
    err = np.zeros(n)
    for i in range(n):
        err[i] = abs(np.dot(A[i,:],X[:,i])-b[i])
    disagree = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if W[i,j] > 0.0:
                dis = LA.norm(X[:,i]-X[:,j])/LA.norm(X[:,i])
                if dis > disagree[i]:
                    disagree[i] = dis
    return err, disagree

def global_error(A,b,X):
    sol = LA.solve(A,b)
    err = np.zeros(n)
    for i in range(n):
        err[i] = LA.norm(X[:,i]-sol)/LA.norm(sol)
    return err

def projection(a,b,x):
    return x-np.dot(np.outer(a,a),x)+b*a



max_iter = 15000
tol = 1e-5
eps = 1e-9

eqn_err2 = np.zeros((max_iter+1,n))
cons_err2 = np.zeros((max_iter+1,n))
glob_err2 = np.zeros((max_iter+1,n))

def Inertial_Projected_Consensus_Algorithm(A, b):
    n=len(b)
    X_i = 2.0 * np.random.rand(n, n) - np.ones((n, n))
    iter2 = 0
    Y = np.zeros((n, n))
    step = np.zeros(n)
    for i in range(n):
        step[i] = float('inf')
    theta = np.zeros(n)
    while (max(eqn_err2[iter2, :]) > tol or max(cons_err2[iter2, :]) > tol) and (iter2 < max_iter):
        iter2 += 1
        for i in range(n):
            X_i[:, i] = projection(A[i, :].T, b[i], X_i[:, i])
        X_i = np.dot(X_i, W.T)
        for i in range(n):
            Y[:, i] = projection(A[i, :].T, b[i], X_i[:, i])
        Y = np.dot(Y, W.T)
        for i in range(n):
            if abs(np.dot(A[i, :], Y[:, i] - X_i[:, i])) < eps:
                theta[i] = float('inf')
            else:
                theta[i] = (np.dot(A[i, :], X_i[:, i]) - b[i]) / np.dot(A[i, :], X_i[:, i] - Y[:, i])
        for i in range(n):
            for j in range(n):
                if W[i, j] > 0.0 and theta[j] > eps and theta[j] < step[i]:
                    step[i] = theta[j]
            if step[i] < float('inf'):
                step[i] = 1.5 * step[i]
                X_i[:, i] = X_i[:, i] + step[i] * (Y[:, i] - X_i[:, i])
        X_i = np.dot(X_i, W.T)
        eqn_err2[iter2, :], cons_err2[iter2, :] = local_error(A, b, X_i, W)
        glob_err2[iter2, :] = global_error(A, b, X_i)
        return iter2, X_i

solution =Inertial_Projected_Consensus_Algorithm(A, b)

print(solution)