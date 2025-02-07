import numpy as np
import matplotlib.pyplot as plt

# Paramètres de l'algorithme
tolerance = 1e-6
max_iterations = 10000
alpha = 0.5  # Paramètre de relaxation
beta = 0.3   # Momentum (fixe pour tous les tests)
n = 3        # Taille des matrices

# Fonction principale de l'algorithme de Kaczmarz avec momentum
def Kaczmarz_momentum(A, b):
    norms = np.sum(A ** 2, axis=1)  # Pré-calcul des normes des lignes

    def Kaczmarz_one_iteration(A_row, b_value, inconnue, norm):
        atixi = (b_value - np.dot(A_row, inconnue)) / norm
        return inconnue + atixi * A_row

    def Kaczmarz_one_loop(inconnue):
        all_inconnue = np.zeros_like(inconnue)
        for i in range(len(A)):
            all_inconnue += Kaczmarz_one_iteration(A[i], b[i], inconnue, norms[i])
        return all_inconnue / len(A)

    def convergence(inconnue, x_old):
        return np.linalg.norm(inconnue - x_old) < tolerance

    x = np.random.rand(n)  # Initialisation aléatoire
    x_old = np.zeros_like(x)
    momentum = np.zeros_like(x)

    for iteration in range(max_iterations):
        x_old[:] = x
        update = Kaczmarz_one_loop(x)

        # Application du momentum
        new_x = (1 - alpha) * x + alpha * update
        x = new_x + beta * momentum
        momentum = x - x_old  # Mise à jour du momentum

        if convergence(x, x_old):
            return iteration + 1  # Nombre d'itérations pour converger

    return max_iterations  # Si pas de convergence


# 🔄 **Test sur 30 matrices différentes**
num_tests = 30
iteration_counts = []  # Stocke le nombre d'itérations pour chaque test

for _ in range(num_tests):
    # Génération d'une nouvelle matrice 3x3 normalisée et d'un nouveau vecteur b
    A = 2.0 * np.random.rand(n, n) - np.ones((n, n))
    for i in range(n):
        A[i, :] /= np.linalg.norm(A[i, :])
    b = 2.0 * np.random.rand(n) - np.ones(n)

    # Exécution de l'algorithme
    iterations = Kaczmarz_momentum(A, b)
    iteration_counts.append(iterations)

# 📊 **Affichage des résultats**
avg_iterations = np.mean(iteration_counts)  # Moyenne du nombre d'itérations

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_tests + 1), iteration_counts, marker='o', linestyle='-', color='blue', label="number of iterations")
plt.axhline(y=avg_iterations, color='red', linestyle='--', label=f"average: {avg_iterations:.2f} iterations")
plt.title(f'Convergence on 30 systemes 3x3 with β = {beta}')
plt.xlabel('Test n°')
plt.ylabel('Number of iterations')
plt.legend()
plt.grid(True)
plt.show()

# 🔥 **Affichage de la moyenne finale**
print(f"\n✅ average iteration {num_tests} tests: {avg_iterations:.2f}")
