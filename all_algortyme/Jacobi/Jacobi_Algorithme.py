import numpy as np

# Module pour générer des matrices de communication

# ============================
# Fonction de méthode de Jacobi distribuée
# ============================

def distributed_jacobi(A, b, W, tol=1e-6, max_iter=1000):
    """
    Résout un système d'équations linéaires Ax = b de manière distribuée avec la méthode de Jacobi.

    Arguments :
    - A : Matrice de coefficients (chaque agent ne possède qu'une ligne de A)
    - b : Vecteur des constantes (chaque agent ne possède qu'une valeur de b)
    - W : Matrice de communication (modélise les connexions entre les agents)
    - tol : Tolérance pour la convergence
    - max_iter : Nombre maximal d'itérations

    Retourne :
    - x : Solution approchée du système linéaire
    """
    n = len(b)  # Nombre d'agents
    x = np.zeros(n)  # Initialisation des estimations de la solution (vecteur partagé entre agents)
    x_prev = np.zeros(n)  # Stocke la solution de l'itération précédente
    x_erro_jacobi = []
    x_true = np.linalg.solve(A, b)
    iteration =0
    # Vérifie que la matrice de communication W est valide
    if not is_communication_graph_valid(W):
        raise ValueError("La matrice de communication W ne connecte pas tous les agents.")

    # Itérations de la méthode de Jacobi
    for iteration in range(max_iter):
        # Calcul de la nouvelle estimation locale de chaque agent
        for i in range(n):
            sum_neighbors = 0
            for j in range(n):
                if i != j:  # On ignore la diagonale d'A
                    sum_neighbors += A[i, j] * x_prev[j]
            x[i] = (b[i] - sum_neighbors) / A[i, i]  # Calcul de la solution locale

        # Communication entre agents (via W)
        x = np.dot(W, x)  # Partage des solutions intermédiaires via la matrice de communication
        x_erro_jacobi.append(np.linalg.norm(x_true - x))
        iteration +=1
        # Critère de convergence
        if np.linalg.norm(x - x_prev) < tol:
            print(f"Convergence atteinte en {iteration + 1} itérations.")
            return x,x_erro_jacobi,iteration

        # Mise à jour pour l'itération suivante
        x_prev = x.copy()

    print("La méthode de Jacobi n'a pas convergé après le nombre maximal d'itérations.")
    return x,x_erro_jacobi,iteration

# ============================
# Fonction pour valider la matrice de communication
# ============================

def is_communication_graph_valid(W):
    """
    Vérifie que tous les agents peuvent communiquer (le graphe de communication est connexe).

    Arguments :
    - W : Matrice de communication

    Retourne :
    - True si tous les agents sont connectés, False sinon.
    """
    n = W.shape[0]
    visited = np.zeros(n)

    # Parcours en profondeur pour vérifier la connexité
    def dfs(node):
        visited[node] = 1
        for neighbor in range(n):
            if W[node, neighbor] > 0 and visited[neighbor] == 0:
                dfs(neighbor)

    dfs(0)  # Commence avec le premier agent
    return np.all(visited)

# ============================
# Exemple d'utilisation
# ============================
# if __name__ == "__main__":
#     # Exemple : 3 agents (3 équations linéaires)
#     A = np.array([[4, -1, 0],
#                   [-1, 4, -1],
#                   [0, -1, 3]])  # Matrice locale de chaque agent
#     b = np.array([15, 10, 10])  # Les constantes connues par chaque agent
#
#     # Exemple de matrice de communication (graphe complet)
#     W = np.array([[1/3, 1/3, 1/3],
#                   [1/3, 1/3, 1/3],
#                   [1/3, 1/3, 1/3]])
#
#     # Résolution avec la méthode de Jacobi distribuée
#     x_approx,x_erro_jacobi =distributed_jacobi(A, b, W)
#
#     # Affichage de la solution approchée
#     print("Solution approchée :", x_approx)
#
#     print("Error : ", x_erro_jacobi)
#     # Vérification avec la solution exacte
#     x_true = np.linalg.solve(A, b)
#     print("Solution exacte :", x_true)