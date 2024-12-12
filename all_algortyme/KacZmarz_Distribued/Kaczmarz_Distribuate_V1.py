import numpy as np

# Matrix A and vector b
A = np.array([[4, 1, 2],
              [3, 5, 1],
              [1, 1, 3]])

b = np.array([4, 7, 3])

# Parameters for the algorithm

tolerance = 1e-6
alpha = 1  # Relaxation parameter (to be adjusted as needed)
beta = 0.5  # Momentum parameter (to be adjusted as needed)


# Fonction Kaczmarz pour une itération unique
def Kaczmarz_one_iteration(A_row, b_value, unknown, norm,alpha):
    atixi = (b_value - np.dot(A_row, unknown)) / norm
    unknown += atixi * A_row*alpha
    return unknown


# Fonction pour exécuter une boucle d'itérations de Kaczmarz sur toutes les équations
def Kaczmarz_one_loop(A, b, unknown, norms):
    unknown_tab = []
    for i in range(len(A)):
        unknown = Kaczmarz_one_iteration(A[i], b[i], unknown, norms[i], alpha)
        unknown_tab.append(unknown)
    sums = [0] * len(b)
    for row in unknown_tab:
        for i, value in enumerate(row):
            sums[i] += value
    unknown = [s / len(b) for s in sums]
    return unknown


#Convergence controle
#def convergence(inconnue, x_old):
#    return np.linalg.norm(inconnue - x_old) < tolerance

def convergence(A, b, unknown):
    residu = np.linalg.norm(A @ unknown - b)
    b_norm = np.linalg.norm(b)
    return residu / b_norm < tolerance



# Algorithme de Kaczmarz avec comptage des itérations
def Kaczmarz(A, b, max_iterations=1000):
    norms = np.sum(A ** 2, axis=1)  # Pré-calcul des normes
    x = np.random.rand(len(b))  # Initialisation aléatoire
    x_old = np.zeros_like(x)
    iteration = 0

    for iteration in range(max_iterations):
        x_old[:] = x
        momentum =  beta * (x - x_old)
        x = Kaczmarz_one_loop(A, b, x, norms) #+ momentum
        iteration = iteration + 1
        if convergence(A, b, x):
            return x, iteration + 1

    return x, max_iterations,iteration

solution = Kaczmarz(A, b, 10000)

print("Kaczmar solution",solution,"true solution",np.linalg.solve(A,b))
