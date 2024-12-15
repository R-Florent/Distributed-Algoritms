import json
import numpy as np
import time
from numpy import linalg as LA
import matplotlib.pyplot as plt

from gradien_optimiser import eqn_err
tolerance = 1e-6

with open("../../ressource/System_of_linear_equations/systems_data_3x3_to_10x10.json", "r") as f:
    systems_data = json.load(f)

def matrice_W(n):
    W = np.zeros((n, n))
    for i in range(n):
        W[i, i] = 1 / 3
    for i in range(n - 1):
        W[i, i + 1] = 1 / 3
        W[i + 1, i] = 1 / 3
    if n == 10:  # Ajustement cyclique uniquement pour n = 10
        W[0, 9] = 1 / 3
        W[9, 0] = 1 / 3
    return W


def local_error(A,b,X,W,n):
    err = np.zeros(n)
    for i in range(n):
        err[i] = abs(np.dot(A[i, :], X[:, i]).item() - b[i])
    disagree = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if W[i,j] > 0.0:
                dis = LA.norm(X[:,i]-X[:,j])/LA.norm(X[:,i])
                if dis > disagree[i]:
                    disagree[i] = dis
    return err, disagree


def global_error(A,b,X,n):
    sol = LA.solve(A,b)
    err = np.zeros(n)
    for i in range(n):
        err[i] = LA.norm(X[:,i]-sol)/LA.norm(sol)
    return err


def projection(a,b,x):
    return x-np.dot(np.outer(a,a),x)+b*a


def Inertial_Projected_Consensus_Algorithm(A, b, max_iterations=10000):
    n = len(b)
    tol = 1e-5
    eps = 1e-9

    X_init = 2.0 * np.random.rand(n, n) - np.ones((n, n))
    X_i = X_init
    Y = np.zeros((n, n))
    step = np.full(n, float('inf'))
    W = matrice_W(n).T
    theta = np.zeros(n)

    iter = 0
    while iter < max_iterations:
        iter += 1

        # Mise à jour de X
        for i in range(n):
            X_i[:, i] = projection(A[i, :].T, b[i], X_i[:, i])
        X_i = np.dot(X_i, W)

        # Calcul de Y
        for i in range(n):
            Y[:, i] = projection(A[i, :].T, b[i], X_i[:, i])
        Y = np.dot(Y, W)

        # Calcul de theta et mise à jour de X
        for i in range(n):
            if abs(np.dot(A[i, :], Y[:, i] - X_i[:, i])) < eps:
                theta[i] = float('inf')
            else:
                theta[i] = (np.dot(A[i, :], X_i[:, i]) - b[i]).item() / np.dot(A[i, :], X_i[:, i] - Y[:, i]).item()

        for i in range(n):
            for j in range(n):
                if W[i, j] > 0.0 and eps < theta[j] < step[i]:
                    step[i] = theta[j]
            if step[i] < float('inf'):
                #improved the step
                step[i] = 1.0 / (1 + iter) * step[i]
                X_i[:, i] = X_i[:, i] + step[i] * (Y[:, i] - X_i[:, i])

        X_i = np.dot(X_i, W)

        # Vérifier la convergence
        def convergence(inconnue, x_old):
            return np.linalg.norm(inconnue - x_old) < tolerance


        eqn_err, cons_err = local_error(A, b, X_i, W, n)
        if max(eqn_err) < tol and max(cons_err) < tol:
            break

    return iter

#color for every systeme
colors = {
    3: "red",
    4: "blue",
    5: "green",
    6: "orange",
    7: "purple",
    8: "cyan",
    9: "brown",
    10: "pink"
}
# Stocker les résultats
condition_numbers = []
execution_times = []
matrix_sizes = []  # Taille de la matrice
point_colors = []  # Couleurs associées à chaque point

# Regrouper les données pour le calcul des moyennes
average_data = {size: {"condition_numbers": [], "execution_times": []} for size in colors.keys()}

    # Stocker les résultats si le nombre de condition est acceptable
for idx, system in enumerate(systems_data):
    A = np.array(system["A"])
    b = np.array(system["b"]).reshape(-1, 1)  # Assurer un vecteur colonne
    size = A.shape[0]
    condition_number = np.linalg.cond(A)

    # Mesurer le temps d'exécution
    start_time = time.time()
    iterations = Inertial_Projected_Consensus_Algorithm(A, b)
    end_time = time.time()
    execution_time = end_time - start_time

    if condition_number < 300:
        condition_numbers.append(condition_number)
        execution_times.append(execution_time)
        matrix_sizes.append(size)
        point_colors.append(colors[size])
        print(f"Système {idx + 1}: Taille={size}, Conditionnement={condition_number:.2f}, "
              f"Temps={execution_time:.4f}s, Iterations={iterations}")

# Tracer le graphique
plt.figure(figsize=(10, 6))

# Points individuels
for size in sorted(colors.keys()):
    indices = [i for i, s in enumerate(matrix_sizes) if s == size]
    plt.scatter(
        [condition_numbers[i] for i in indices],
        [execution_times[i] for i in indices],
        color=colors[size],
        label=f"{size}x{size}",
        alpha=0.7
    )

plt.xlabel("Condition number κ(A)")
plt.ylabel("Execution time (seconds)")
plt.title("Relationship between the condition number κ(A) and execution time Inertial")
plt.legend(title="Matrix size")
plt.grid()
plt.show()