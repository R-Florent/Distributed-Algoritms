import numpy as np
import matplotlib.pyplot as plt

# Matrix A and vector b
A = np.array([[-15, 1, -10, -7, -12],
            [6, -6, 11, 7, 2],
            [-4, -9, 11, -15, 18],
            [17, 4, -17, -2, 6],
            [15, -8, -10, 16, 12]])

b = np.array([17, 2,-6 ,11 ,14 ])
true_solution = np.linalg.solve(A,b)
# Parameters for the algorithm
# Weight matrix W (corrected)

def Comunication_cycle_graphe (n):
    W = np.zeros((n, n))
    for i in range(n):
        W[i, i] = 1 / 3
    for i in range(n - 1):
        W[i, i + 1] = 1 / 3
        W[i + 1, i] = 1 / 3
    W[0, n - 1] = 1 / 3
    W[n - 1, 0] = 1 / 3
    # Symmetric weights for periodic neighbors-

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
def Kaczmarz(A, b, max_iterations=30000):

    tab_err= []
    norms = np.sum(A ** 2, axis=1)  # Pré-calcul des normes
    x = np.random.rand(len(b))  # Initialisation aléatoire
    x_old = np.zeros_like(x)
    iteration = 0

    for iteration in range(max_iterations):
        x_old[:] = x
        momentum =  beta * (x - x_old)
        x = Kaczmarz_one_loop(A, b, x, norms) #+ momentum
        iteration = iteration + 1
        tab_err.append(np.linalg.norm(true_solution - x))

        if convergence(A, b, x):
            plt.plot(range(len(tab_err)), tab_err)
            plt.show()
            return x, iteration + 1

    plt.plot(range(len(tab_err)),tab_err)
    plt.show()
    return x, max_iterations,#iteration


solution = Kaczmarz(A, b, 10000)
print("Kaczmar solution",solution,"true solution",np.linalg.solve(A,b))
