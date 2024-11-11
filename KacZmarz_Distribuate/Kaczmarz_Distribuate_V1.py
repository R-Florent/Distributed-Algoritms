import numpy as np
import json
import os
import matplotlib.pyplot as plt
from conda.env.env import print_result

# Paramètres pour le dossier de sortie
output_folder = "graphs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Paramètres de l'algorithme
tolerance = 1e-6
max_iterations = 10000
num_repetitions = 500  # Nombre de répétitions pour calculer la moyenne des itérations


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


# Lire les systèmes de `systems_data.json`
with open("../systems_data.json", "r") as f:
    systems_data = json.load(f)

for systems in systems_data:
    results = Kaczmarz(systems_data[systems]["A"], systems_data[systems]["b"])
    print(results)

# Calcul et génération des graphiques
results = []

print(results)