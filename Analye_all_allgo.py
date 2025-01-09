#ce fichier et pour appeller tout les algo qui sont dans leur fichier et plos une jolie courbe
from cProfile import label

import matplotlib.pyplot as plt
import numpy as np
from all_algortyme.KacZmarz_Distribued.KacZmarz_moyenne import Kaczmarz_moyenne
from all_algortyme.KacZmarz_Distribued.Kaczmarz_Distribuate_V1 import Kaczmarz
#from all_algortyme.inertial_Projected_consensus_Algortyme.JE_FAIT_NIMP import Inertial_Projected_Consensus_Algorithm
#from all_algortyme.Hirua_Algorithme.hirua_methode import hirua
from all_algortyme.Jacobi.Jacobi_Algorithme import distributed_jacobi
from Comunication_matrix import *
# Matrix A and vector b

A = np.array([[4, -1, 0],
              [-1, 4, -1],
              [0, -1, 3]])

b = np.array([15, 10, 10])

W = complete_graph(len(b))
x = np.random.rand(len(b))

Kaczmarz_v1_solution,Kaczmarz_v1_iterration_number,Kaczmarz_v1_erro = Kaczmarz(A, b, 10000,x)
Kaczmarz_moyenne_solution,Kaczmarz_moyenne_iterration_number,Kaczmarz_moyenne_erro = Kaczmarz_moyenne(A, b, 10000)
#Inertial_iterration,Inertial_solution = Inertial_Projected_Consensus_Algorithm(A, b, 10000,x)
solution_jacobi,x_erro_jacobi,iterration_jacobi =distributed_jacobi(A, b, W)


fig = plt.figure()
plt.xlabel("number of iterations")
plt.ylabel("global error")
plt.yscale('log')
plt.plot(range(Kaczmarz_v1_iterration_number), Kaczmarz_v1_erro,label="Kaczmarz_v1")
plt.plot(range(Kaczmarz_moyenne_iterration_number), Kaczmarz_moyenne_erro, label="Kaczmarz_moyenne")
#plt.plot(range(Kaczmarz_moyenne_iterration_number), Kaczmarz_moyenne_erro, label="Kaczmarz_moyenne")
plt.plot(range(iterration_jacobi), x_erro_jacobi, label="Jacobi")
plt.legend()
plt.show()