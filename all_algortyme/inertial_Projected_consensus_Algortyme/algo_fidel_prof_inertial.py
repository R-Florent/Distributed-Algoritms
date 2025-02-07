import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from Comunication_matrix import *


#il marche mais il y a un probléme, je sais plus ou je ais plus pourquoi j'ai oublier de notre le probléme
#DONC JUSTE LA POUR L'OBESERVER

# from numpy.linalg import solve

# Matrix A and vector b
n = 10
A = 2.0 * np.random.rand(n, n) - np.ones((n, n))
b = 2.0 * np.random.rand(n) - np.ones(n)

# normalization of each row of A
for i in range(n):
    norm = np.linalg.norm(A[i, :])
    A[i, :] = A[i, :] / norm
    b[i] = b[i] / norm

print("Condition Number of A:", LA.cond(A))

n = len(b)

# Weight matrix W (corrected)
# Matrice de comunnication
# W = (1/n) * np.ones((n,n))
# Symmetric weights for periodic neighbors-

W = complete_graph(n)  # Par exemple



def local_error(A, b, X, W):
    """Compute local error and disagreement for consensus."""
    err = np.zeros(n)
    for i in range(n):
        err[i] = abs(np.dot(A[i, :], X[:, i]) - b[i])  # Local error
    disagree = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if W[i, j] > 0.0:
                norm_i = LA.norm(X[:, i])
                if norm_i > 1e-9:  # Avoid division by zero
                    dis = LA.norm(X[:, i] - X[:, j]) / norm_i
                    if dis > disagree[i]:
                        disagree[i] = dis
    return err, disagree


def global_error(A, b, X):
    """Compute global error for consensus."""
    sol = LA.solve(A, b)  # Exact solution
    err = np.zeros(n)
    for i in range(n):
        norm_sol = LA.norm(sol)
        if norm_sol > 1e-9:  # Avoid division by zero
            err[i] = LA.norm(X[:, i] - sol) / norm_sol
    return err


def projection(a, b, x):
    """Project vector `x` onto the hyperplane defined by `a` and `b`."""
    norm_a = LA.norm(a)
    if norm_a > 1e-9:  # Avoid projection if `a` is zero
        return x - np.dot(np.outer(a, a), x) + b * a
    else:
        return x  # No projection needed if `a` is zero


# Algorithm parameters
max_iter = 15000
tol = 1e-5
eps = 1e-9

# Initialize error arrays
eqn_err2 = np.zeros((max_iter + 1, n))
cons_err2 = np.zeros((max_iter + 1, n))
glob_err2 = np.zeros((max_iter + 1, n))


def Inertial_Projected_Consensus_Algorithm(A, b):
    """Solve linear system iteratively using inertial projected consensus."""
    X_i = 2.0 * np.random.rand(n, n) - np.ones((n, n))  # Random initialization
    iter2 = 0
    Y = np.zeros((n, n))
    step = np.zeros(n)
    theta = np.zeros(n)

    # Initialize error metrics
    eqn_err2[0, :], cons_err2[0, :] = local_error(A, b, X_i, W)
    glob_err2[0, :] = global_error(A, b, X_i)

    # Main iterative loop
    while (max(eqn_err2[iter2, :]) > tol or max(cons_err2[iter2, :]) > tol) and (iter2 < max_iter):
        iter2 += 1

        # Step 1: Projection of X
        for i in range(n):
            X_i[:, i] = projection(A[i, :].T, b[i], X_i[:, i])
        X_i = np.dot(X_i, W.T)  # Consensus update

        # Step 2: Projection of Y
        for i in range(n):
            Y[:, i] = projection(A[i, :].T, b[i], X_i[:, i])
        Y = np.dot(Y, W.T)

        # Step 3: Update step sizes
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

        X_i = np.dot(X_i, W.T)  # Consensus update

        # Update errors
        eqn_err2[iter2, :], cons_err2[iter2, :] = local_error(A, b, X_i, W)
        glob_err2[iter2, :] = global_error(A, b, X_i)

    return iter2, X_i


# Run the algorithm
iterations, solution = Inertial_Projected_Consensus_Algorithm(A, b)

# Direct solution using LA.solve
direct_solution = LA.solve(A, b)

# Print results
print(f"Converged in {iterations} iterations")
print("Iterative Solution matrix X_i:")
print(solution)
print("Direct Solution:")
print(direct_solution)

# Plot error convergence
plt.figure(figsize=(12, 6))
plt.plot(np.max(eqn_err2[:iterations, :], axis=1), label='Equation Error')
plt.plot(np.max(cons_err2[:iterations, :], axis=1), label='Consensus Error')
plt.plot(np.max(glob_err2[:iterations, :], axis=1), label='Global Error')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Error (log scale)')
plt.title('Error Convergence in Inertial Projected Consensus Algorithm')
plt.legend()
plt.grid()
plt.show()
