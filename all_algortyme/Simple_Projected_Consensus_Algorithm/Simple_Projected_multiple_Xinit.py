import numpy as np


def projection(A_row, b_val, x):
    """
    Projette x sur l'hyperplan défini par A_row @ x = b_val.
    """
    numerator = b_val - np.dot(A_row, x)
    denominator = np.dot(A_row, A_row)
    if denominator == 0:  # Pour éviter une division par zéro
        return x
    return x + (numerator / denominator) * A_row


def local_error(A, b, X_s, W):
    """
    Calcule les erreurs locales pour les contraintes et le consensus.
    """
    n = A.shape[0]
    eqn_err = np.zeros(n)
    cons_err = np.zeros(n)
    for i in range(n):
        eqn_err[i] = abs(np.dot(A[i, :], X_s[:, i]) - b[i])
        cons_err[i] = np.linalg.norm(X_s[:, i] - np.dot(X_s, W[:, i]))
    return eqn_err, cons_err


def global_error(A, b, X_s):
    """
    Calcule l'erreur globale pour l'ensemble des agents.
    """
    return np.linalg.norm(np.dot(A, X_s.mean(axis=1)) - b)


def simple_projected_consensus_algorithm(A, b, W, X_init, max_iter=1000, tol=1e-6):
    """
    Implémente l'algorithme de consensus projeté pour résoudre le système.

    Args:
        A (np.array): Matrice des coefficients (n x m).
        b (np.array): Vecteur des termes constants (n,).
        W (np.array): Matrice de poids (n x n).
        X_init (np.array): Matrice initiale des solutions (m x n).
        tol (float): Tolérance pour les critères d'arrêt.
        max_iter (int): Nombre maximum d'itérations.

    Returns:
        np.array: Solution finale X_s.
    """
    n = A.shape[0]
    iter1 = 0
    X_s = X_init.copy()  # Copie pour ne pas modifier l'original
    eqn_err1 = np.zeros((max_iter, n))
    cons_err1 = np.zeros((max_iter, n))
    glob_err1 = np.zeros(max_iter)


    # Boucle principale
    while iter1 < max_iter:
        # Vérifie les erreurs pour arrêter la boucle
        if iter1 > 0 and max(eqn_err1[iter1 - 1, :]) <= tol and max(cons_err1[iter1 - 1, :]) <= tol:
            break

        iter1 += 1

        # Mise à jour locale
        for i in range(n):
             X_s[:, i] = projection(A[i, :], b[i], X_s[:, i])

        # Consensus

        X_s = np.dot(X_s, W.T)

        # Calcul des erreurs
        eqn_err1[iter1 - 1, :], cons_err1[iter1 - 1, :] = local_error(A, b, X_s, W)
        glob_err1[iter1 - 1] = global_error(A, b, X_s)

    print(f"Algorithme convergé en {iter1} itérations avec une erreur globale de {glob_err1[iter1 - 1]:.6f}")
    return X_s.mean(axis=1),#glob_err1,iter1 # Retourne la moyenne des solutions comme consensus final


# Exemple d'utilisation
if __name__ == "__main__":
    A = np.array([[4, 1, 2],
                  [3, 5, 1],
                  [1, 1, 3]])

    b = np.array([4, 7, 3])

    W = np.array([[0.5, 0.25, 0.25],
                  [0.25, 0.5, 0.25],
                  [0.25, 0.25, 0.5]])  # Matrice de poids (symétrique et stochastique)

    X_init = np.random.rand(A.shape[1], A.shape[0])  # Initialisation aléatoire
    print(X_init)
    X_init = np.random.rand(len(b))
    print(X_init)
    solution = simple_projected_consensus_algorithm(A, b, W, X_init)
    print("Solution finale :", solution)
