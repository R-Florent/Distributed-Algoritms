import numpy as np
import matplotlib.pyplot as plt


# =============================
# Fonction de projection
# =============================
def projection(b_i, c_i, x):
    """
    Projette x sur l'ensemble Xi = {x | b_i^T x = c_i, x >= 0}.
    Si l'ensemble est vide, retourne -1.
    """
    y = x.copy()
    if np.dot(b_i, b_i) == 0:  # Cas pathologique (b_i = 0)
        return -1 if c_i != 0 else y
    alpha = (c_i - np.dot(b_i, y)) / np.dot(b_i, b_i)  # Correction pour respecter b_i^T y = c_i
    y += alpha * b_i  # Ajuste y dans la direction de b_i
    y = np.maximum(y, 0)  # Applique la contrainte x >= 0
    return y


# =============================
# Matrice de communication
# =============================
def generate_weight_matrix(n):
    """
    Génère une matrice de communication symétrique (étoile, chaîne, ou autre).
    """
    W = np.ones((n, n)) / n  # Matrice uniformément pondérée
    return W


# =============================
# Algorithme de consensus projeté
# =============================
def simple_projected_consensus(A, b, X_init, W, max_iter=1000, tol=1e-6):
    """
    Algorithme simple de consensus projeté pour résoudre Ax = b.
    """
    n = len(b)  # Nombre d'agents
    iter1 = 0
    X_s = X_init.copy()
    true_solution = np.linalg.solve(A, b)
    # Erreurs pour l'analyse
    eqn_err = []
    cons_err = []
    x_erro_simple_projected_consensus = []

    while iter1 < max_iter:
        # Mettre à jour les erreurs pour les critères d'arrêt
        eqn_errors = [abs(np.dot(A[i], X_s[:, i]) - b[i]) for i in range(n)]
        consensus_errors = [np.linalg.norm(X_s[:, i] - np.mean(X_s, axis=1)) for i in range(n)]

        eqn_err.append(max(eqn_errors))
        cons_err.append(max(consensus_errors))

        if max(eqn_errors) < tol and max(consensus_errors) < tol:
            break

        # Algorithme : Projections et consensus
        for i in range(n):
            X_s[:, i] = projection(A[i], b[i], X_s[:, i])  # Projection locale
        X_s = np.dot(X_s, W.T)  # Mise à jour via la matrice de communication

        x_mean = np.mean(X_s, axis=1)  # Moyenne des estimations
        error = np.linalg.norm(true_solution - x_mean)  # Différence avec la solution exacte
        x_erro_simple_projected_consensus.append(error)

        iter1 += 1

    return X_s, eqn_err, cons_err, x_erro_simple_projected_consensus , iter1


A = np.array([[4, 1, 2],
              [3, 5, 1],
              [1, 1, 3]])

b = np.array([4, 7, 3])
n = len(b)
X_init = np.random.rand(len(A[0]), n)
W = np.ones((n, n)) / n

X_s, eqn_err, cons_err, tab_erro,iteration = simple_projected_consensus(A, b, X_init, W)

print(X_s, eqn_err, cons_err, tab_erro, iteration)