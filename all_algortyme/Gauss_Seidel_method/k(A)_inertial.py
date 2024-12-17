import numpy as np
import json
import os
import matplotlib.pyplot as plt
import time

from all_algortyme.G

# Création du dossier pour les graphiques
output_folder = "graphs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Lire les systèmes depuis le fichier JSON
with open("../../ressource/System_of_linear_equations/systems_data_3x3_to_10x10.json", "r") as f:
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
    solution, iterations = Inertial_Projected_Consensus_Algorithm(A, b)
    end_time = time.time()

    execution_time = end_time - start_time

    # Stocker les résultats si le nombre de condition est acceptable
    if condition_number < 500:
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
plt.title("Relationship between the condition number κ(A) and execution time Inertial")
plt.legend(title="Matrix size")
plt.grid()
plt.show()
