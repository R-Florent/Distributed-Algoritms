import numpy as np

from all_algortyme.inertial_Projected_consensus_Algortyme.Inertial_Projected_Consensus import solution

A = np.array([[4, 1, 2],
              [3, 5, 1],
              [1, 1, 3]])

b = np.array([4, 5, 6])

x = np.random.random((len(b), 1))
true_solution = np.linalg.solve(A, b)

print("True solution is",true_solution)

def gauss_seidel(A, b, tol=1e-6, max_iter=10000, x_init=None):
    n = len(b)
    x = np.zeros(n) if x_init is None else x_init  # Initialisation
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sum_1 = np.dot(A[i, :i], x[:i])
            sum_2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - sum_1 - sum_2) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, k + 1
    raise ValueError("L'algorithme n'a pas convergé")