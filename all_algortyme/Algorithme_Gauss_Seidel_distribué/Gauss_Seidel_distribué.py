import numpy as np

tolerance = 1e-6
alpha = 0.9

def has_converged(x_new, x_old, tol):
    return np.linalg.norm(x_new - x_old) < tol


def distributed_gauss_seidel(A, b,max_iterations=10000):
    true_solution = np.linalg.solve(A,b)
    x = np.random.rand(len(b))
    n = len(b)
    x_old = np.zeros_like(x)
    iterations  = 0  # Compteur d'itérations

    for iteration in range(max_iterations):
        x_old[:] = x  # Sauvegarde de l'état précédent

        for i in range(n):  # Chaque agent résout son équation
            sum_except_i = np.dot(A[i, :], x) - A[i, i] * x[i]
            x[i] = (b[i] - sum_except_i) / A[i, i]

        # Appliquer le facteur de relaxation
        x = alpha * x + (1 - alpha) * x_old

        # Vérifier la convergence
        if has_converged(x, x_old, tolerance):
            iterations = iterations + 1
            print(f"Converged after {iteration + 1} iterations. Matrice size {len(A)} Solution:{x} | True solution:{true_solution} ")
            break

    return x,iterations