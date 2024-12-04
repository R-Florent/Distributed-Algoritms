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
max_iterations = 1_000_000
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


#Convergence controle
#def convergence(inconnue, x_old):
#    return np.linalg.norm(inconnue - x_old) < tolerance

def convergence(A, b, inconnue):
    residu = np.linalg.norm(A @ inconnue - b)
    b_norm = np.linalg.norm(b)
    return residu / b_norm < tolerance



# Algorithme de Kaczmarz avec comptage des itérations
def Kaczmarz(A, b):
    norms = np.sum(A ** 2, axis=1)  # Pré-calcul des normes
    x = np.random.rand(len(b))  # Initialisation aléatoire
    x_old = np.zeros_like(x)

    for iteration in range(max_iterations):
        x_old[:] = x
        x = Kaczmarz_one_loop(A, b, x, norms)

        if convergence(A, b, x):
            return x, iteration + 1

    return x, max_iterations

condition_numbers = []
execution_times = []

# Lire les systèmes de `systems_data_3x3.json`
with open("../ressource/System_of_linear_equations/systems_data_30x30.json", "r") as f:
    systems_data = json.load(f)

for idx, system in enumerate(systems_data):
    A = np.array(system["A"])
    b = np.array(system["b"])

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

    # Affichage des résultats
    print(f"Système {idx + 1}:")
    print(f"Solution: {solution}")
    print(f"Nombre d'itérations: {iterations}")
    print(f"Nombre de condition κ(A): {condition_number:.2f}\n")

plt.figure(figsize=(10, 6))
plt.scatter(condition_numbers, execution_times, color="blue", alpha=0.7)
plt.xlabel("Number of conditions κ(A)")
plt.ylabel("Execution time (seconds)")
plt.title("Relationship between the number of conditions κ(A) and the execution time Matrix 30x30 (new condition)")
plt.grid()
plt.show()