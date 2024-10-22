import numpy as np

# Matrix A and vector b
A = np.array([[4, 1, 2],
              [3, 5, 1],
              [1, 1, 3]])

b = np.array([4, 7, 3])

# Random initialization of vector x
x = np.random.rand(3)
print("Initial random x:", x)

# Parameters for the algorithm
tolerance = 1e-6
max_iterations = 10000

# Pré-calcul de la norme de chaque ligne de la matrice A (évitant de recalculer à chaque itération)
norms = np.sum(A ** 2, axis=1)


# Fonction Kaczmarz pour résoudre 1x
def Kaczmarz_one_iteration(A_row, b_value, inconnue, norm):
    # Produit scalaire entre A_row et x (A_row.dot(x))
    Transpose_a = np.dot(A_row, inconnue)

    # Mise à jour de x selon l'algorithme de Kaczmarz
    atixi = (b_value - Transpose_a) / norm
    inconnue += atixi * A_row  # Mise à jour de chaque inconnue

    return inconnue


# Fonction pour exécuter une itération de Kaczmarz sur toutes les équations
def Kaczmarz_one_loop(inconnue):
    all_inconnue = np.zeros_like(inconnue)  # Initialiser une nouvelle matrice pour stocker les résultats

    # Appliquer Kaczmarz à chaque ligne de A
    for i in range(len(A)):
        all_inconnue += Kaczmarz_one_iteration(A[i], b[i], inconnue, norms[i])

    # Retourner la moyenne des résultats avec numpy
    return all_inconnue / len(A)


# Fonction principale de l'algorithme avec contrôle de convergence
def Kaczmarz():
    x_copy = x.copy()  # Initialisation des inconnues
    x_old = np.zeros_like(x_copy)  # Conserver l'ancienne version de x pour la convergence

    for iteration in range(max_iterations):
        x_old[:] = x_copy  # Conserver l'ancienne itération pour vérification de la convergence

        # Effectuer une itération complète sur toutes les équations
        x_copy = Kaczmarz_one_loop(x_copy)

        # Vérification de la convergence
        if convergence(x_copy, x_old):
            print(f"Convergence atteinte après {iteration + 1} itérations.")
            break

    return x_copy


# Fonction de convergence
def convergence(inconnue, x_old):
    # Vérification de la différence entre l'ancienne et la nouvelle valeur de x
    return np.linalg.norm(inconnue - x_old) < tolerance


# Exécution de Kaczmarz et vérification de la convergence
final_inconnue = Kaczmarz()

print("La solution finale (x, y, z) est :", final_inconnue)