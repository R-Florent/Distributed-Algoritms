                      # -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:48:39 2023

@author: norikazu
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
#from numpy.linalg import solve

n = 10
A = 2.0*np.random.rand(n,n)-np.ones((n,n))
b = 2.0*np.random.rand(n)-np.ones(n)

# normalization of each row of A
for i in range(n):
    norm = np.linalg.norm(A[i,:])
    A[i,:] = A[i,:] / norm
    b[i] = b[i] / norm

print("Condition Number of A:",LA.cond(A))

# finding the true solution
#solution = LA.solve(A,b)
#print("Solution:",solution)


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

eqn_err1 = np.zeros((max_iter+1,n))
cons_err1 = np.zeros((max_iter+1,n))
glob_err1 = np.zeros((max_iter+1,n))

eqn_err2 = np.zeros((max_iter+1,n))
cons_err2 = np.zeros((max_iter+1,n))
glob_err2 = np.zeros((max_iter+1,n))

eqn_err3 = np.zeros((max_iter+1,n))
cons_err3 = np.zeros((max_iter+1,n))
glob_err3 = np.zeros((max_iter+1,n))

eqn_err4 = np.zeros((max_iter+1,n))
cons_err4 = np.zeros((max_iter+1,n))
glob_err4 = np.zeros((max_iter+1,n))

X_init = 2.0*np.random.rand(n,n)-np.ones((n,n))
#X_init = np.dot(X_init,W.T)


eqn_err1[0,:], cons_err1[0,:] = local_error(A,b,X_init,W)
glob_err1[0,:] = global_error(A,b,X_init)

eqn_err2[0,:] = eqn_err1[0,:]
cons_err2[0,:] = cons_err1[0,:]
glob_err2[0,:] = glob_err1[0,:]

eqn_err3[0,:] = eqn_err1[0,:]
cons_err3[0,:] = cons_err1[0,:]
glob_err3[0,:] = glob_err1[0,:]

eqn_err4[0,:] = eqn_err1[0,:]
cons_err4[0,:] = cons_err1[0,:]
glob_err4[0,:] = glob_err1[0,:]

# Simple Projected Consensus Algorithm
iter1 = 0
X_s = X_init
while (max(eqn_err1[iter1,:]) > tol or max(cons_err1[iter1,:]) > tol) and (iter1 < max_iter):
    iter1 += 1
    for i in range(n):
        X_s[:,i] = projection(A[i,:].T,b[i],X_s[:,i])
    X_s = np.dot(X_s,W.T)
    eqn_err1[iter1,:], cons_err1[iter1,:] = local_error(A,b,X_s,W)
    glob_err1[iter1,:] = global_error(A,b,X_s)

fig = plt.figure()
plt.xlabel("number of iterations")
plt.ylabel("global error")
plt.yscale('log')
for i in range(n):
    plt.plot(glob_err1[0:iter1,i],label='agent '+str(i+1))
plt.legend()
plt.show()

# Inertial Projected Consensus Algorithm
iter2 = 0
X_i = X_init
Y = np.zeros((n,n))
step = np.zeros(n)
for i in range(n):
    step[i] = float('inf')
theta = np.zeros(n)
while (max(eqn_err2[iter2,:]) > tol or max(cons_err2[iter2,:]) > tol) and (iter2 < max_iter):
    iter2 += 1
    for i in range(n):
        X_i[:,i] = projection(A[i,:].T,b[i],X_i[:,i])
    X_i = np.dot(X_i,W.T)
    for i in range(n):
        Y[:,i] = projection(A[i,:].T,b[i],X_i[:,i])
    Y = np.dot(Y,W.T)
    for i in range(n):
        if abs(np.dot(A[i,:],Y[:,i]-X_i[:,i])) < eps:
            theta[i] = float('inf')
        else:
            theta[i] = (np.dot(A[i,:],X_i[:,i])-b[i])/np.dot(A[i,:],X_i[:,i]-Y[:,i])
    for i in range(n):
        for j in range(n):
            if W[i,j] > 0.0 and theta[j] > eps and theta[j] < step[i]:
                step[i] = theta[j]
        if step[i] < float('inf'):
            step[i] = 1.5 * step[i]
            X_i[:,i] = X_i[:,i]+step[i]*(Y[:,i]-X_i[:,i])
    X_i = np.dot(X_i,W.T)
    eqn_err2[iter2,:], cons_err2[iter2,:] = local_error(A,b,X_i,W)
    glob_err2[iter2,:] = global_error(A,b,X_i)

fig = plt.figure()
plt.xlabel("number of iterations")
plt.ylabel("global error")
plt.yscale('log')
for i in range(n):
    plt.plot(glob_err2[0:iter2,i],label='agent '+str(i+1))
plt.legend()
plt.show()

# Gradient Consensus Algorithm
iter3 = 0
X_g = X_init
#X_g = np.dot(X_g,W.T)
Y = np.zeros((n,n))
step = np.zeros(n)
while (max(eqn_err3[iter3,:]) > tol or max(cons_err3[iter3,:]) > tol) and (iter3 < max_iter):
    iter3 += 1
    for i in range(n):
        Y[:,i] = (b[i]-np.dot(A[i,:],X_g[:,i]))*A[i,:].T
    Y = np.dot(Y,W.T)
    for i in range(n):
        theta[i] = (b[i]-np.dot(A[i,:],X_g[:,i]))/np.dot(A[i,:],Y[:,i])
        if theta[i] < eps:
            theta[i] = float('inf')
    for i in range(n):
        step[i] = float('inf')
        for j in range(n):
            if W[i,j] > 0.0 and theta[j] > eps and theta[j] < step[i]:
                step[i] = theta[j]
        if step[i] < float('inf'):
            step[i] = 1.5 * step[i]
            X_g[:,i] = X_g[:,i]+step[i]*Y[:,i]
    X_g = np.dot(X_g,W.T)
    eqn_err3[iter3,:], cons_err3[iter3,:] = local_error(A,b,X_g,W)
    glob_err3[iter3,:] = global_error(A,b,X_g)

# # Gradient Consensus Algorithm improved
#
# step_init = 0.50  # Pas initial
# k = 0.01         # Facteur de régulation
# iter4 = 0
#
# while (max(eqn_err4[iter4,:]) > tol or max(cons_err4[iter4,:]) > tol) and (iter4 < max_iter):
#     iter4 += 1
#     for i in range(n):
#         Y[:,i] = (b[i]-np.dot(A[i,:],X_g[:,i]))*A[i,:].T
#     Y = np.dot(Y,W.T)
#     for i in range(n):
#         theta[i] = (b[i]-np.dot(A[i,:],X_g[:,i]))/np.dot(A[i,:],Y[:,i])
#         if theta[i] < eps:
#             theta[i] = float('inf')
#     for i in range(n):
#         step[i] = step_init / (np.linalg.norm(Y[:, i]) + k)  # Pas décroissant régulé
#         X_g[:, i] = X_g[:, i] + step[i] * Y[:, i]
#     X_g = np.dot(X_g,W.T)
#     eqn_err4[iter4,:], cons_err4[iter4,:] = local_error(A,b,X_g,W)
#     glob_err4[iter4,:] = global_error(A,b,X_g)
#
#
#
# fig = plt.figure()
# plt.xlabel("number of iterations")
# plt.ylabel("global error")
# plt.yscale('log')
# for i in range(n):
#     plt.plot(glob_err4[0:iter3,i],label='agent '+str(i+1))
# plt.legend()
# plt.show()

fig = plt.figure()
plt.xlabel("number of iterations")
plt.ylabel("global error")
#plt.yscale('log')
plt.plot(glob_err1[0:iter1,0],label="conventional")
plt.plot(glob_err2[0:iter2,0],label="inertial")
plt.plot(glob_err3[0:iter3,0],label="gradient")
# plt.plot(glob_err4[0:iter4,0],label="gradient improved")
plt.legend()
plt.show()
