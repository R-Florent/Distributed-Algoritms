import json
import numpy as np
import time

from matplotlib import pyplot as plt
from numpy import linalg as LA

tolerance = 1e-6

# Charger les données des systèmes
with open("../../ressource/System_of_linear_equations/systems_data_3x3_to_10x10.json", "r") as f:
    systems_data = json.load(f)

n = 10
A = 2.0*np.random.rand(n,n)-np.ones((n,n))
b = 2.0*np.random.rand(n)-np.ones(n)
X_init = 2.0*np.random.rand(n,n)-np.ones((n,n))

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

def Inertial_Projected_Consensus_Algorithm(A, b, max_iterations=10000, tol=1e-5):
    iter2 = 0
    X_i = X_init
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

    fig = plt.figure()
    plt.xlabel("number of iterations")
    plt.ylabel("global error")
    plt.yscale('log')
    for i in range(n):
        plt.plot(glob_err2[0:iter2, i], label='agent ' + str(i + 1))
    plt.legend()
    plt.show()


# Exécution pour chaque système
results = []

for idx, system in enumerate(systems_data):
    A = np.array(system["A"])
    b = np.array(system["b"]).reshape(-1, 1)
    size = A.shape[0]
    condition_number = np.linalg.cond(A)

    if condition_number < 300:  # Filtrer les systèmes mal conditionnés
        iter_count, exec_time, solution = Inertial_Projected_Consensus_Algorithm(A, b)
        results.append({
            "system_index": idx + 1,
            "matrix_size": size,
            "condition_number": condition_number,
            "iterations": iter_count,
            "execution_time": exec_time,
            "solution": solution.flatten().tolist()
        })

# Affichage des résultats
for result in results:
    print(f"Système {result['system_index']}:")
    print(f"  Taille: {result['matrix_size']}x{result['matrix_size']}")
    print(f"  Nombre de conditionnement: {result['condition_number']:.2f}")
    print(f"  Nombre d'itérations: {result['iterations']}")
    print(f"  Temps d'exécution: {result['execution_time']:.4f}s")
    print(f"  Solution: {result['solution']}")
    print("true solution" ,np.linalg.solve(A,b))
