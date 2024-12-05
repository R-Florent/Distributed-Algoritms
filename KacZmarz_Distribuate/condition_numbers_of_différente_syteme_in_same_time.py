import numpy as np
import json
import os
import matplotlib.pyplot as plt
import time

# Création du dossier pour les graphiques
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

# Lire les systèmes depuis le fichier JSON
with open("../ressource/System_of_linear_equations/systems_data_3x3_to_10x10.json", "r") as f:
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

# Stocker les résultats
condition_numbers = []
execution_times = []
matrix_sizes = []  # Taille de la matrice
point_colors = []  # Couleurs associées à chaque point

# Regrouper les données pour le calcul des moyennes
average_data = {size: {"condition_numbers": [], "execution_times": []} for size in colors.keys()}

for idx, system in enumerate(systems_data):
    A = np.array(system["A"])
    b = np.array(system["b"])
    size = A.shape[0]
    condition_number = np.linalg.cond(A)

    # Mesurer le temps d'exécution de l'algorithme
    start_time = time.time()
    solution, iterations = Kaczmarz(A, b)
    end_time = time.time()

    execution_time = end_time - start_time

    # Stocker les résultats si le nombre de condition est acceptable
    if condition_number < 200:
        condition_numbers.append(condition_number)
        execution_times.append(execution_time)
        matrix_sizes.append(size)
        point_colors.append(colors[size])

        # Ajouter les données pour le calcul des moyennes
        average_data[size]["condition_numbers"].append(condition_number)
        average_data[size]["execution_times"].append(execution_time)

    # # Affichage des résultats
    # print(f"Système {idx + 1}:")
    # print(f"Taille de la matrice: {size}x{size}")
    # print(f"Solution: {solution}")
    # print(f"Nombre d'itérations: {iterations}")
    # print(f"Nombre de condition κ(A): {condition_number:.2f}")
    # print(f"Temps d'exécution: {execution_time:.4f} secondes\n")

# Calcul des moyennes
average_condition_numbers = []
average_execution_times = []
matrix_sizes_avg = []

for size in sorted(average_data.keys()):
    if average_data[size]["condition_numbers"]:
        avg_cond = np.mean(average_data[size]["condition_numbers"])
        avg_time = np.mean(average_data[size]["execution_times"])

        average_condition_numbers.append(avg_cond)
        average_execution_times.append(avg_time)
        matrix_sizes_avg.append(size)

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
plt.title("Relationship between the condition number κ(A) and execution time Kaczmarz")
plt.legend(title="Matrix size")
plt.grid()
plt.show()
