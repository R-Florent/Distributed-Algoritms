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

# Fonction Kaczmarz pour résoudre 1x
def KacZmarz_one_iteration(matrice_A, matrice_b, inconnue):
    Transpose_a = 0
    Norm = 0
    for i, A_elem in enumerate(matrice_A):
        Transpose_a += A_elem * inconnue[i]  # Produit scalaire entre a et x
        Norm += A_elem * A_elem  # Norme de a

    # Mise à jour de x selon l'algorithme de Kaczmarz
    atixi = (matrice_b - Transpose_a) / Norm
    for i, A_elem in enumerate(matrice_A):
        inconnue[i] = inconnue[i] + atixi * A_elem  # Mise à jour de chaque inconnue

    return inconnue

# Fonction pour exécuter une itération de Kaczmarz sur toutes les équations
def kaxZmarz_one_loop(inconnue):
    all_inconnue = []

    # Appliquer KacZmarz à chaque ligne de A pour obtenir les nouvelles inconnues
    for i in range(len(A)):
        new_inconnue = KacZmarz_one_iteration(A[i], b[i], inconnue.copy())
        all_inconnue.append(new_inconnue)

    # Calculer la moyenne des résultats
    return moyenne_inconnues(all_inconnue)

# Fonction pour calculer la moyenne des inconnues x, y, z
def moyenne_inconnues(inconnues):
    # Calculer la moyenne de chaque élément x, y, z
    moyenne_x = np.mean([inconnue[0] for inconnue in inconnues])
    moyenne_y = np.mean([inconnue[1] for inconnue in inconnues])
    moyenne_z = np.mean([inconnue[2] for inconnue in inconnues])

    # Retourner la moyenne sous forme de vecteur
    return np.array([moyenne_x, moyenne_y, moyenne_z])

# Fonction principale de l'algorithme avec contrôle de convergence
def kaxZmarz():
    x_copy = x.copy()  # Initialisation des inconnues
    x_old = x_copy.copy()  # Conserver l'ancienne version de x pour la convergence

    for iteration in range(max_iterations):
        x_old = x_copy.copy()  # Conserver l'ancienne itération pour vérification de la convergence

        # Effectuer une itération complète sur toutes les équations
        x_copy = kaxZmarz_one_loop(x_copy)

        # Vérification de la convergence
        if convergence(x_copy, x_old):
            print(f"Convergence atteinte après {iteration + 1} itérations.")
            break

    return x_copy

# Fonction de convergence
def convergence(inconnue, x_old):
    # Vérification de la différence entre l'ancienne et la nouvelle valeur de x
    return np.linalg.norm(np.array(inconnue) - np.array(x_old)) < tolerance

# Exécution de Kaczmarz et vérification de la convergence
final_inconnue = kaxZmarz()

print("La solution finale (x, y, z) est :", final_inconnue)