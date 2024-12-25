#ce fichier et pour appeller tout les algo qui sont dans leur fichier et plos une jolie courbe
from cProfile import label

import matplotlib.pyplot as plt
import numpy as np
from all_algortyme.KacZmarz_Distribued.KacZmarz_moyenne import Kaczmarz_moyenne
from all_algortyme.KacZmarz_Distribued.Kaczmarz_Distribuate_V1 import Kaczmarz

# Matrix A and vector b
A = np.array([[-15, 1, -10, -7, -12],
            [6, -6, 11, 7, 2],
            [-4, -9, 11, -15, 18],
            [17, 4, -17, -2, 6],
            [15, -8, -10, 16, 12]])

b = np.array([17, 2,-6 ,11 ,14 ])

x = np.random.rand(len(b))

Kaczmarz_v1_solution,Kaczmarz_v1_iterration_number,Kaczmarz_v1_erro = Kaczmarz(A, b, 10000,x)
Kaczmarz_moyenne_solution,Kaczmarz_moyenne_iterration_number,Kaczmarz_moyenne_erro = Kaczmarz_moyenne(A, b, 10000)

fig = plt.figure()
plt.xlabel("number of iterations")
plt.ylabel("global error")
plt.yscale('log')
plt.plot(range(Kaczmarz_v1_iterration_number), Kaczmarz_v1_erro,label="Kaczmarz_v1")
plt.plot(range(Kaczmarz_moyenne_iterration_number), Kaczmarz_moyenne_erro, label="Kaczmarz_moyenne")
plt.legend()
plt.show()