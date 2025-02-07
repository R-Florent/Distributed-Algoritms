import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def complete_graph(n):
    W = np.ones((n, n)) / n  # Tous les poids sont égaux
    return W

def star_graph(n):
    W = np.zeros((n, n))
    for i in range(1, n):
        W[0, i] = 1 / (n - 1)  # Le nœud central partage avec les autres
        W[i, 0] = 1            # Les autres partagent uniquement avec le centre
    return W


def path_graph(n):
    W = np.zeros((n, n))
    for i in range(n):
        if i > 0:  # Voisin gauche
            W[i, i-1] = 0.5
        if i < n-1:  # Voisin droit
            W[i, i+1] = 0.5
    return W


def cycle_graph(n):
    W = np.zeros((n, n))
    for i in range(n):
        W[i, i] = 1 / 3  # Conserve une partie pour lui-même
        W[i, (i-1) % n] = 1 / 3  # Voisin gauche (cyclique)
        W[i, (i+1) % n] = 1 / 3  # Voisin droit (cyclique)
    return W


def visualize_graph(W):
    G = nx.from_numpy_array(W, create_using=nx.DiGraph())
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500)
    plt.show()

visualize_graph(complete_graph(5))