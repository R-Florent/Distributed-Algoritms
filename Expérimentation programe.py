import numpy as np

A = np.array([[1325, 5245, 4345345],
              [5, -44343, -40001],
              [1, 408, 4]])

b = np.array([10, -2, 30])

x = np.random.rand(3)  # x1, x2, x3 initialisés aléatoirement

# Paramètres pour l'algorithme itératif
tolerance = 1e-6
max_iterations = 1000
alpha = 0.9

print("Initial value of unknowns:", x)


def KacZmarg(matrice_A, matrice_b, inconnue):
    for iteration in range(max_iterations):
        x_old = inconnue.copy()  # Copie de l'ancienne solution
        for indise, equation in enumerate(matrice_A):
            transposer_a = 0
            norme = 0
            for j, element in enumerate(equation):
                transposer_a += element * inconnue[j]  # Produit scalaire de la ligne et de x
                norme += element * element  # Norme de la ligne

            # Mise à jour selon l'algorithme de Kaczmarz
            atixi = (matrice_b[indise] - transposer_a) / norme
            for j, element in enumerate(equation):
                inconnue[j] = inconnue[j] + atixi * element

        # Critère de convergence (tolérance)
        if np.linalg.norm(np.array(inconnue) - np.array(x_old)) < tolerance:
            print(f"Convergence atteinte après {iteration + 1} loop.")
            break

    return inconnue


# Exécution de l'algorithme
resulte = KacZmarg(A, b, x)

print("Solution approchée:", resulte)
