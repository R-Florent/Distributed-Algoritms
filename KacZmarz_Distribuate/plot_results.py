# plot_results.py

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from Kaczmarz_Distribuate_V1 import Kaczmarz  # Importation de l'algorithme de résolution

# Paramètres pour le dossier de sortie
output_folder = "graphs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Lire les systèmes de `systems_data_3x3.json`
with open("../ressource/System_of_linear_equations/systems_data_3x3.json", "r") as f:
    systems_data = json.load(f)

# Paramètres globaux
num_repetitions = 500  # Nombre de répétitions pour calculer la moyenne des itérations
results = []

# Calcul et génération des graphiques
for idx, system in enumerate(systems_data):
    A = np.array(system["A"])
    b = np.array(system["b"])
    taille_matrice = A.shape[0]

    # Exécuter plusieurs répétitions pour obtenir le nombre moyen d'itérations
    iterations_needed = [Kaczmarz(A, b)[1] for _ in range(num_repetitions)]
    average_iterations = np.mean(iterations_needed)
    std_dev_iterations = np.std(iterations_needed)

    # Stocker les résultats avec les informations pertinentes
    results.append({
        "index": idx,
        "matrix_size": taille_matrice,
        "average_iterations": average_iterations,
        "std_dev_iterations": std_dev_iterations
    })

    # Création du texte du système pour le graphique
    matrix_text = f"A = {A.tolist()}\n\nb = {b.tolist()}"

    # Génération du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_repetitions + 1), iterations_needed, label="Nombre d'itérations par essai", color='blue')
    plt.axhline(average_iterations, color='red', linestyle='--', label=f"Moyenne = {average_iterations:.2f}")
    plt.fill_between(
        range(1, num_repetitions + 1),
        average_iterations - std_dev_iterations,
        average_iterations + std_dev_iterations,
        color='gray', alpha=0.3,
        label=f"Écart type = {std_dev_iterations:.2f}"
    )

    # Affichage du système dans le graphique
    plt.text(1.02, 0.5, matrix_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    # Ajout de titres et légendes
    plt.xlabel("Essai")
    plt.ylabel("Nombre d'itérations pour convergence")
    plt.title(f"Système {idx + 1} : Analyse statistique des itérations")
    plt.legend()

    # Sauvegarde du graphique
    plt.savefig(f"{output_folder}/system_{idx + 1}_iterations.png")
    plt.close()

# Tri des résultats par taille de matrice et moyenne d'itérations
results_sorted = sorted(results, key=lambda x: (x["matrix_size"], x["average_iterations"]))

# Afficher les résultats triés et sauvegarder dans un nouveau fichier JSON
for result in results_sorted:
    print(f"Système {result['index'] + 1}: Taille de la matrice = {result['matrix_size']}, "
          f"Nombre moyen d'itérations = {result['average_iterations']:.2f}, "
          f"Écart type = {result['std_dev_iterations']:.2f}")

# Sauvegarder les résultats triés dans un fichier JSON
with open("sorted_systems_data.json", "w") as f:
    json.dump(results_sorted, f, indent=4)

print("Les systèmes ont été traités, les graphiques générés et les résultats stockés dans 'sorted_systems_data.json'.")
