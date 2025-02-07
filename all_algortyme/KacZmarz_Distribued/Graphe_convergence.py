import numpy as np
import matplotlib.pyplot as plt

# Définition du système Ax = b
A = np.array([[2, 3],
              [4, 1],
              [-1, 2]])

b = np.array([7, 5, 3])

# Paramètres
tolerance = 1e-6
alpha = 1  # Facteur de relaxation
max_iterations = 30  # Pour limiter les itérations
x_initial = np.array([5, 5])  # Point de départ
true_solution = np.linalg.lstsq(A, b, rcond=None)[0]  # Solution exacte


# 🔄 Fonction d'une itération du Kaczmarz
def Kaczmarz_one_iteration(A_row, b_value, x, norm, alpha):
    atixi = (b_value - np.dot(A_row, x)) / norm
    return x + atixi * A_row * alpha


# 📉 Fonction de convergence
def convergence(A, b, x):
    residu = np.linalg.norm(A @ x - b)
    return residu / np.linalg.norm(b) < tolerance


# 🔄 Algorithme de Kaczmarz avec stockage des étapes
def Kaczmarz(A, b, max_iterations, x_initial):
    x = x_initial.copy()
    norms = np.sum(A ** 2, axis=1)
    trajectory = [x.copy()]

    for _ in range(max_iterations):
        for i in range(len(A)):
            x = Kaczmarz_one_iteration(A[i], b[i], x, norms[i], alpha)
            trajectory.append(x.copy())

        if convergence(A, b, x):
            break

    return np.array(trajectory), x


# 🚀 Exécution de l’algorithme
trajectory, solution = Kaczmarz(A, b, max_iterations, x_initial)


# 🎨 Visualisation
fig, ax = plt.subplots(figsize=(6, 6))
x_vals = np.linspace(-10, 10, 100)

# Tracer les lignes des équations
colors = ['blue', 'orange', 'green']
for i in range(len(A)):
    if A[i, 1] != 0:
        y_vals = (b[i] - A[i, 0] * x_vals) / A[i, 1]
        ax.plot(x_vals, y_vals, color=colors[i], label='Line')

# Tracer le chemin de convergence
ax.plot(trajectory[:, 0], trajectory[:, 1], 'r--o', label="Kaczmarz Iterations", markersize=5)

# Tracer la solution
ax.scatter(*true_solution, color='green', marker='*', s=200, label="Solution")

# Configuration du graphique
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(r"Kaczmarz Algorithm in $\mathbb{R}^2$")
ax.legend()
ax.grid(True)

# Afficher le graphique
plt.show()