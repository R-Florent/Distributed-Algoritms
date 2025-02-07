import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 🔄 Projection sur un hyperplan défini par Ax = b
def projection(A_row, b_value, x):
    return x + (b_value - np.dot(A_row, x)) / np.dot(A_row, A_row) * A_row

# 🔍 Calcul de l'erreur locale
def local_error(A, b, X):
    return np.linalg.norm(A @ np.mean(X, axis=1) - b)

# 🔄 Algorithme du consensus projeté inertiel avec momentum
def inertial_projected_consensus(A, b, W, X_init, extension, max_iter, tol, momentum):
    n = A.shape[0]  # Nombre d'équations
    X = X_init.copy()
    Y = np.zeros_like(X)
    step = np.full(n, float('inf'))
    eps = 1e-9
    theta = np.zeros(n)
    err = local_error(A, b, X)
    err_history = [err]
    iter = 1
    trajectory = [X.copy()]  # Stocker les positions pour le traçage
    velocity = np.zeros_like(X)  # Stocker la "vitesse" (momentum)

    while err > tol and iter <= max_iter:
        # Projection sur chaque hyperplan
        for i in range(n):
            X[:, i] = projection(A[i, :], b[i], X[:, i])

        # Moyenne pondérée
        X = np.dot(X, W.T)

        # Nouvelle projection sur les hyperplans
        for i in range(n):
            Y[:, i] = projection(A[i, :], b[i], X[:, i])

        # Moyenne pondérée
        Y = np.dot(Y, W.T)

        # Calcul du pas de mise à jour
        for i in range(n):
            if abs(np.dot(A[i, :], Y[:, i] - X[:, i])) < eps:
                theta[i] = float('inf')
            else:
                theta[i] = (np.dot(A[i, :], X[:, i]) - b[i]) / np.dot(A[i, :], X[:, i] - Y[:, i])

        # Mise à jour avec le plus petit pas + momentum
        for i in range(n):
            for j in range(n):
                if W[i, j] > 0.0 and eps < theta[j] < step[i]:
                    step[i] = theta[j]
            if step[i] < float('inf'):
                step[i] *= extension
                velocity[:, i] = momentum * velocity[:, i] + step[i] * (Y[:, i] - X[:, i])
                X[:, i] += velocity[:, i]

        # Dernière projection
        for i in range(n):
            Y[:, i] = projection(A[i, :], b[i], X[:, i])

        # Moyenne pondérée finale
        X = np.dot(Y, W.T)

        # Évaluation de l'erreur
        err = local_error(A, b, X)
        err_history.append(err)
        iter += 1
        trajectory.append(X.copy())  # Sauvegarde de l'itération

    return X, iter, err_history, np.array(trajectory)

# 🔧 Définition du problème
A = np.array([[4, 1], [2, 3]])  # Coefficients des équations
b = np.array([5, 7])  # Second membre

# Matrice de consensus (identité pour deux agents)
W = np.array([[0.5, 0.5],
              [0.5, 0.5]])

# Initialisation aléatoire des agents
X_init = np.random.rand(2, 2) * 10 - 5  # Deux points aléatoires

# Paramètres
extension = 1.1
momentum = 0.8  # Facteur de relaxation
max_iter = 50
tol = 1e-6

# 🔄 Exécution de l'algorithme
X_final, iterations, errors, trajectory = inertial_projected_consensus(A, b, W, X_init, extension, max_iter, tol, momentum)

# 🌟 Animation avec Matplotlib
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_title("Évolution des agents avec Momentum")

# Tracé des hyperplans Ax = b
x_vals = np.linspace(-10, 10, 100)
for i in range(A.shape[0]):
    y_vals = (b[i] - A[i, 0] * x_vals) / A[i, 1]
    ax.plot(x_vals, y_vals, label=f"Hyperplan {i+1}")

# Tracé des trajectoires des agents
lines = [ax.plot([], [], 'o-', markersize=8)[0] for _ in range(2)]

def update(frame):
    for i in range(2):
        lines[i].set_data(trajectory[:frame+1, 0, i], trajectory[:frame+1, 1, i])
    return lines

ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=200, blit=True)

plt.legend()
plt.show()
