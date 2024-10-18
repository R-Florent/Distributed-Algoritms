import numpy as np

# Matrix A and vector b
A = np.array([[4, 1, 2],
              [3, 5, 1],
              [1, 1, 3]])

b = np.array([4, 7, 3])

# Random initialization of vector x
x = np.random.rand(3)

# Parameters for the algorithm
tolerance = 1e-6
max_iterations = 10000
alpha = 0.9

#fait la formule de kacmarz pour rouver 1x
def KacZmarz(matrice_A, matrice_b, inconnue):
    # use the formule of KacZmarz methode
    for equation in enumerate(matrice_A):
        Transpose_a = 0
        Norm = 0
        for i,element in enumerate(equation):
            Transpose_a += element * inconnue[i]  # Scalaire product of a and x ligne
            Norm += element * element  # Norm of vector

        # Mise à jour selon l'algorithme de Kaczmarz
        atixi = (matrice_b - Transpose_a) / Norm
        for i,element in enumerate(equation):
            inconnue[i] = inconnue[i] + atixi * element
        #return a vectore of x for 1 equation
        return inconnue

def KacZmarz_projection(matrice_A, matrice_b, inconnue):
    resultats = []
    for i in range(len(x)):
        resultats.append(KacZmarz(matrice_A[i], matrice_b[i],inconnue))

    resultats_array = np.array(resultats)
    average = np.mean(resultats_array, axis=0)
    return average

i = 0

while True:
    for i in range(len(x)):
        KacZmarz(A[i], b[i], x)

    print("pour valeur de x depart",x,"valer du vecteur ",A[0],"est valeur du resultat",b[0])
    print("la formule retourne comme valeur de x",KacZmarz(A[0], b, x))
    i += 1

    if i >= max_iterations:
        break

#%%
import numpy as np

# Matrix A and vector b
A = np.array([[4, 1, 2],
              [3, 5, 1],
              [1, 1, 3]])

b = np.array([4, 7, 3])

# Random initialization of vector x
x = np.random.rand(3)

# Parameters for the algorithm
tolerance = 1e-6
max_iterations = 10000
alpha = 0.9

#fait la formule de kacmarz pour rouver 1x
def KacZmarz(matrice_A, matrice_b, inconnue):
    Transpose_a = 0
    Norm = 0
    for i,A in enumerate(matrice_A):
        Transpose_a += A * inconnue[i]
        Norm += A * A

    atixi = (matrice_b - Transpose_a) / Norm
    for i,A in enumerate(matrice_A):
        inconnue[i] = inconnue[i] + atixi * A
    return inconnue

def kaxZmarz():
    all_inconnue = []
    for i in range(len(x)):
        all_inconnue.append(KacZmarz(A[i], b[i], x))
    for ligne in range(len(all_inconnue[0])):
        #for inco in range(len(inconnue_average[0])):
        print(all_inconnue[ligne])


kaxZmarz()