import numpy as np
from all_algortyme.KacZmarz_Distribued.Kaczmar_matrice_comunication import true_solution

# Parameters for the algorithm
tolerance = 1e-6
max_iterations = 10000
alpha = 0.5  # Relaxation parameter (to be adjusted as needed)
beta = 0.1  # Momentum parameter (to be adjusted as needed)

# Pré-calcul de la norme de chaque ligne de la matrice A



# Fonction Kaczmarz pour résoudre 1x
def Kaczmarz_one_iteration(A_row, b_value, inconnue, norm):
    Transpose_a = np.dot(A_row, inconnue)
    atixi = (b_value - Transpose_a) / norm
    inconnue += atixi * A_row
    return inconnue


# Fonction pour exécuter une itération de Kaczmarz sur toutes les équations
def Kaczmarz_one_loop(inconnue,A,b,norms):
    all_inconnue = np.zeros_like(inconnue)
    for i in range(len(A)):
        all_inconnue += Kaczmarz_one_iteration(A[i], b[i], inconnue, norms[i])
    return all_inconnue / len(A)


# Fonction de convergence
def convergence(inconnue, x_old):
    return np.linalg.norm(inconnue - x_old) < tolerance


# Fonction principale de l'algorithme avec contrôle de convergence
def Kaczmarz_moyenne(A,b,max_iter):
    norms = np.sum(A ** 2, axis=1)
    true_solution = np.linalg.solve(A, b)
    tab_err= []
    x = np.random.rand(len(b))  # Initialisation aléatoire des inconnues
    x_old = np.zeros_like(x)  # Conserver l'ancienne version de x pour la convergence
    momentum = np.zeros_like(x)  # Pour ajouter le momentum
    iteration = 0

    while not convergence(x, x_old) and iteration < max_iterations:
        x_old[:] = x  # Conserver l'ancienne itération pour vérification de la convergence
        update = Kaczmarz_one_loop(x,A,b,norms)

        # Ajout de relaxation et de momentum dans la mise à jour de x
        new_x = (1 - alpha) * x + alpha * update
        x = new_x + beta * momentum  # Applique le momentum
        tab_err.append(np.linalg.norm(true_solution - x))
        # Met à jour le momentum
        momentum = x - x_old
        iteration +=1

    return x, iteration ,tab_err  # Si la convergence n'est pas atteinte


# Boucle pour calculer le nombre moyen d'itérations
#num_repetitions = 500  # Nombre de répétitions pour calculer la moyenne
#iterations_needed = []

#for i in range(num_repetitions):
#    i, iterations = Kaczmarz_moyenne(A,b)  # Exécute Kaczmarz et obtient le nombre d'itérations
#    iterations_needed.append(iterations)

# Calcul des indicateurs statistiques
###
# # average_iterations = np.mean(iterations_needed)
# std_dev_iterations = np.std(iterations_needed)
#
# # Tracé du graphique
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, num_repetitions + 1), iterations_needed, label="number of loop par try", color='blue')
# plt.axhline(average_iterations, color='red', linestyle='--', label=f"average = {average_iterations:.2f}")
# plt.fill_between(
#     range(1, num_repetitions + 1),
#     average_iterations - std_dev_iterations,
#     average_iterations + std_dev_iterations,
#     color='gray', alpha=0.3,
#     label=f"Standard deviation = {std_dev_iterations:.2f}"
# )
#
# # Ajout de titres et légendes
# plt.xlabel("Try")
# plt.ylabel("number of loop for convergence")
# plt.title(
#     "Statistical analysis of the number of iterations for the convergence of the Kaczmarz multe agent decentralized method")
# plt.legend()
# plt.show()

