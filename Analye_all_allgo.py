#ce fichier et pour appeller tout les algo qui sont dans leur fichier et plos une jolie courbe
import matplotlib.pyplot as plt
import numpy as np
from all_algortyme.KacZmarz_Distribued.KacZmarz_moyenne import Kaczmarz_moyenne
from all_algortyme.KacZmarz_Distribued.Kaczmarz_Distribuate_V1 import Kaczmarz
from all_algortyme.inertial_Projected_consensus_Algortyme.JE_FAIT_NIMP import Inertial_Projected_Consensus_Algorithm
#from all_algortyme.Hirua_Algorithme.hirua_methode import hirua
from all_algortyme.Jacobi.Jacobi_Algorithme import distributed_jacobi
from all_algortyme.Simple_Projected_Consensus_Algorithm.Simple_Projected import simple_projected_consensus
from Comunication_matrix import *

# Matrix A and vector b

A = np.array([[4, 1, 2],
              [3, 5, 1],
              [1, 1, 3]])

b = np.array([4, 7, 3])

W = complete_graph(len(b))
x = np.random.rand(len(b))
#Sensei algortyme
X_s, eqn_err, cons_err, SPC_tab_erro,SPC_iteration = simple_projected_consensus(A, b, x, W ,10000)
SPC_X, IPC_moyenne_iterration_number, IPC_moyenne_erro, = Inertial_Projected_Consensus_Algorithm(A, b, x, W ,10000)

#Algtoyme allready exitse
solution_jacobi,x_erro_jacobi,iterration_jacobi =distributed_jacobi(A, b, W)

#My algortyme
Kaczmarz_v1_solution,Kaczmarz_v1_iterration_number,Kaczmarz_v1_erro = Kaczmarz(A, b, 10000,x)
Kaczmarz_moyenne_solution,Kaczmarz_moyenne_iterration_number,Kaczmarz_moyenne_erro = Kaczmarz_moyenne(A, b, 10000)



fig = plt.figure()
plt.xlabel("number of iterations")
plt.ylabel("global error")
plt.yscale('log')

#Sensei algortyme
plt.plot(range(SPC_iteration), SPC_tab_erro, label="simple_projected_consensus_algorithm")
#plt.plot(range(IPC_moyenne_iterration_number), IPC_moyenne_erro, label="SPCA")

#Algtoyme allready exitse
plt.plot(range(iterration_jacobi), x_erro_jacobi, label="Jacobi")

#My algortyme
plt.plot(range(Kaczmarz_v1_iterration_number), Kaczmarz_v1_erro,label="Kaczmarz_v1")
plt.plot(range(Kaczmarz_moyenne_iterration_number), Kaczmarz_moyenne_erro, label="Kaczmarz_moyenne")

plt.legend()
plt.show()