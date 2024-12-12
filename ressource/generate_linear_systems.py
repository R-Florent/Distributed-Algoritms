import numpy as np
import json
import os

# Paramètres de génération
nombre_de_matrice = 100
taille_de_la_matrice = 7

# Création du dossier pour stocker les graphiques, si nécessaire
if not os.path.exists("graphs"):
    os.makedirs("graphs")

def generate_linear_systems(num_systems, matrix_size, min_val=-1, max_val=1):
    systems = []
    for _ in range(num_systems):
        # Génère une matrice A de taille `matrix_size` x `matrix_size` avec des valeurs aléatoires
        A = np.random.uniform(min_val, max_val, size=(matrix_size, matrix_size))
        # Génère un vecteur b de taille `matrix_size` avec des valeurs aléatoires
        b = np.random.uniform(min_val, max_val, size=(matrix_size,))

        # Ajoute le système seulement s'il est valide
        if verifier_validite_systeme(A, b):
            systems.append({
                "A": A.tolist(),
                "b": b.tolist()
            })

    return systems

def verifier_validite_systeme(A, b):
    # Calculer le rang de la matrice des coefficients et de la matrice augmentée
    rang_A = np.linalg.matrix_rank(A)
    rang_Aug = np.linalg.matrix_rank(np.hstack((A, b.reshape(-1, 1))))
    # Système valide si les rangs sont égaux
    return rang_A == rang_Aug

# Initialisation de la liste pour tous les systèmes
all_systems = []

# Génération des systèmes linéaires pour chaque taille de matrice
for size in range(3, 3 + taille_de_la_matrice):
    systems = generate_linear_systems(nombre_de_matrice, matrix_size=size)
    all_systems.extend(systems)  # Ajouter les systèmes validés à la liste totale

# Stockage de tous les systèmes valides dans un fichier JSON
with open("System_of_linear_equations/systems_data_3x3_to_10x10.json", "w") as f:
    formatted_systems = []
    for system in all_systems:
        formatted_systems.append({
            "A": system["A"],
            "b": system["b"]
        })
    json_str = json.dumps(formatted_systems, indent=4)
    json_str = json_str.replace("[\n                ", "[").replace("\n            ],", "],")
    json_str = json_str.replace(",\n                ", ", ")
    f.write(json_str)

print("Les systèmes ont été stockés avec succès dans le fichier 'systems_data_3x3.json'.")
