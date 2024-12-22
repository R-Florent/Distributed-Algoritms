import numpy as np
import json
import os
import matplotlib.pyplot as plt
import time

from all_algortyme.KacZmarz_Distribued.Kaczmarz_Distribuate_V1 import Kaczmarz

# Create a folder for storing the graphs
output_folder = "graphs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read the systems from the JSON file
with open("../../ressource/System_of_linear_equations/systems_data_3x3_to_10x10.json", "r") as f:
    systems_data = json.load(f)

# Colors for each matrix size
colors = {
    3: "red",
    4: "blue",
    5: "green",
    6: "orange",
    7: "purple",
    8: "cyan",
    9: "brown",
    10: "pink"
}

# Initialize results storage
condition_numbers = []
execution_times = []
matrix_sizes = []  # Matrix size for each system
point_colors = []  # Associated colors for each point

# Data for computing averages
average_data = {size: {"condition_numbers": [], "execution_times": []} for size in colors.keys()}

# Iterate through each linear system in the JSON file
for idx, system in enumerate(systems_data):
    A = np.array(system["A"])  # Extract the matrix A
    b = np.array(system["b"])  # Extract the vector b
    size = A.shape[0]          # Get the matrix size
    condition_number = np.linalg.cond(A)  # Compute the condition number of the matrix A

    # Measure the execution time of the Kaczmarz algorithm
    start_time = time.time()
    solution, iterations = Kaczmarz(A, b)  # Replace this with your own algorithm if desired
    end_time = time.time()

    execution_time = end_time - start_time

    # Store results only if the condition number is acceptable (below 500)
    if condition_number < 500:
        condition_numbers.append(condition_number)
        execution_times.append(execution_time)
        matrix_sizes.append(size)
        point_colors.append(colors[size])

        # Add data for computing averages
        average_data[size]["condition_numbers"].append(condition_number)
        average_data[size]["execution_times"].append(execution_time)

    # # Uncomment below to display results for each system
    # print(f"System {idx + 1}:")
    # print(f"Matrix size: {size}x{size}")
    # print(f"Solution: {solution}")
    # print(f"Iterations: {iterations}")
    # print(f"Condition number κ(A): {condition_number:.2f}")
    # print(f"Execution time: {execution_time:.4f} seconds\n")

# Compute the averages for each matrix size
average_condition_numbers = []
average_execution_times = []
matrix_sizes_avg = []

for size in sorted(average_data.keys()):
    if average_data[size]["condition_numbers"]:
        avg_cond = np.mean(average_data[size]["condition_numbers"])
        avg_time = np.mean(average_data[size]["execution_times"])

        average_condition_numbers.append(avg_cond)
        average_execution_times.append(avg_time)
        matrix_sizes_avg.append(size)

# Plot the graph
plt.figure(figsize=(10, 6))

# Individual points
for size in sorted(colors.keys()):
    indices = [i for i, s in enumerate(matrix_sizes) if s == size]
    plt.scatter(
        [condition_numbers[i] for i in indices],
        [execution_times[i] for i in indices],
        color=colors[size],
        label=f"{size}x{size}",
        alpha=0.7
    )

# Add labels and title to the graph
plt.xlabel("Condition number κ(A)")
plt.ylabel("Execution time (seconds)")
plt.title("Relationship between the condition number κ(A) and execution time (Kaczmarz Distributed)")
plt.legend(title="Matrix size")
plt.grid()
plt.show()