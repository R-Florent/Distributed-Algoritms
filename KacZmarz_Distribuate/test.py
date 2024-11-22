import numpy as np
import json
import os
import matplotlib.pyplot as plt
import time

# Paramètres pour le dossier de sortie
output_folder = "graphs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Paramètres de l'algorithme
tolerance = 1e-6
max_iterations = 10000

# Fonction Kaczmarz pour une itération unique
def Kaczmarz_one_iteration(A_row, b_value, inconnue, norm):
    atixi = (b_value - np.dot(A_row, inconnue)) / norm
    inconnue += atixi * A_row
    return inconnue

# Fonction pour exécuter une boucle d'itérations de Kaczmarz sur toutes les équations
def Kaczmarz_one_loop(A, b, inconnue, norms):
    for i in range(len(A)):
        inconnue = Kaczmarz_one_iteration(A[i], b[i], inconnue, norms[i])
    return inconnue

# Contrôle de convergence
def convergence(inconnue, x_old):
    return np.linalg.norm(inconnue - x_old) < tolerance

# Algorithme de Kaczmarz avec comptage des itérations
def Kaczmarz(A, b):
    norms = np.sum(A ** 2, axis=1)  # Pré-calcul des normes
    x = np.random.rand(len(b))  # Initialisation aléatoire
    x_old = np.zeros_like(x)

    for iteration in range(max_iterations):
        x_old[:] = x
        x = Kaczmarz_one_loop(A, b, x, norms)

        if convergence(x, x_old):
            return x, iteration + 1

    return x, max_iterations

# Lire les systèmes de `systems_data_3x3.json`
with open("Conditions_Number/systems_data_3x3.json", "r") as f:
    systems_data = json.load(f)

# Couleurs pour chaque taille de matrice
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

# Stocker les résultats pour le graphique
condition_numbers = []
execution_times = []
matrix_sizes = []  # Taille de la matrice pour colorer les points
point_colors = []  # Couleurs associées à chaque point

for idx, system in enumerate(systems_data):
    A = np.array(system["A"])
    b = np.array(system["b"])
    size = A.shape[0]

    # Calcul du nombre de condition de A
    eigenvalues = np.linalg.eigvals(A)
    lambda_max = np.max(np.abs(eigenvalues))
    lambda_min = np.min(np.abs(eigenvalues))
    condition_number = lambda_max / lambda_min

    # Mesurer le temps d'exécution de l'algorithme
    start_time = time.time()
    solution, iterations = Kaczmarz(A, b)
    end_time = time.time()

    execution_time = end_time - start_time

    # Stocker les résultats
    condition_numbers.append(condition_number)
    execution_times.append(execution_time)
    matrix_sizes.append(size)
    point_colors.append(colors[size])

    # Affichage des résultats
    print(f"Système {idx + 1}:")
    print(f"Taille de la matrice: {size}x{size}")
    print(f"Solution: {solution}")
    print(f"Nombre d'itérations: {iterations}")
    print(f"Nombre de condition κ(A): {condition_number:.2f}")
    print(f"Temps d'exécution: {execution_time:.4f} secondes\n")

# Tracer le graphique
plt.figure(figsize=(10, 6))

# Tracer chaque point avec la couleur correspondant à la taille de la matrice
for size in sorted(colors.keys()):
    indices = [i for i, s in enumerate(matrix_sizes) if s == size]
    plt.scatter(
        [condition_numbers[i] for i in indices],
        [execution_times[i] for i in indices],
        color=colors[size],
        label=f"{size}x{size}",
        alpha=0.7
    )

plt.xlabel("Nombre de condition κ(A)")
plt.ylabel("Temps d'exécution (secondes)")
plt.title("Relation entre le nombre de condition κ(A) et le temps d'exécution")
plt.legend(title="Taille de la matrice")
plt.grid()
plt.show()
