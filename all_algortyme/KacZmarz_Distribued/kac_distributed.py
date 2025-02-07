import numpy as np
import matplotlib.pyplot as plt

# Matrix A and vector b
A = np.array([[4, 1, 2],
              [3, 5, 1],
              [1, 1, 3]])
b = np.array([4, 7, 3])

# Parameters for the algorithm
tolerance = 1e-6
max_iterations = 10000
alpha = 0.5  # Relaxation parameter

# Pre-calculate the norm of each row of matrix A
norms = np.sum(A ** 2, axis=1)

# Main Kaczmarz algorithm without momentum
def Kaczmarz_no_momentum():
    def Kaczmarz_one_iteration(A_row, b_value, inconnue, norm):
        Transpose_a = np.dot(A_row, inconnue)
        atixi = (b_value - Transpose_a) / norm
        inconnue += atixi * A_row
        return inconnue

    def Kaczmarz_one_loop(inconnue):
        all_inconnue = np.zeros_like(inconnue)
        for i in range(len(A)):
            all_inconnue += Kaczmarz_one_iteration(A[i], b[i], inconnue, norms[i])
        return all_inconnue / len(A)

    def convergence(inconnue, x_old):
        return np.linalg.norm(inconnue - x_old) < tolerance

    x = np.random.rand(3)  # Random initialization
    x_old = np.zeros_like(x)

    for iteration in range(max_iterations):
        x_old[:] = x
        update = Kaczmarz_one_loop(x)

        # Standard Kaczmarz update WITHOUT momentum
        x = (1 - alpha) * x + alpha * update

        if convergence(x, x_old):
            return x,iteration + 1  # Return the number of iterations

    return x,max_iterations  # If convergence is not reached

# Boucle pour calculer le nombre moyen d'itérations
num_repetitions = 100  # Nombre de répétitions pour calculer la moyenne
iterations_needed = []

for _ in range(num_repetitions):
    _, iterations = Kaczmarz_no_momentum()  # Exécute Kaczmarz et obtient le nombre d'itérations
    iterations_needed.append(iterations)

# Calcul des indicateurs statistiques
average_iterations = np.mean(iterations_needed)
std_dev_iterations = np.std(iterations_needed)

# Tracé du graphique
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_repetitions + 1), iterations_needed, label="number of loop par try", color='blue')
plt.axhline(average_iterations, color='red', linestyle='--', label=f"average = {average_iterations:.2f}")
plt.fill_between(
    range(1, num_repetitions + 1),
    average_iterations - std_dev_iterations,
    average_iterations + std_dev_iterations,
    color='gray', alpha=0.3,
    label=f"Standard deviation = {std_dev_iterations:.2f}"
)

# Ajout de titres et légendes
plt.xlabel("Try")
plt.ylabel("number of loop for convergence")
plt.title("Without Momentum (β)")
plt.legend()
plt.show()

