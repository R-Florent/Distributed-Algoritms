import json
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_system_analysis(systems_file, output_folder, num_repetitions, solver_algorithm):
    """
    Analyze and plot statistical results for solving systems of linear equations.

    Parameters:
        systems_file (str): Path to the JSON file containing linear systems.
        output_folder (str): Directory where plots will be saved.
        num_repetitions (int): Number of repetitions for each system.
        solver_algorithm (function): Function used to solve Ax = b, must return a tuple (solution, iterations).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load systems data
    with open(systems_file, "r") as f:
        systems_data = json.load(f)

    results = []

    # Process each system
    for idx, system in enumerate(systems_data):
        A = np.array(system["A"])
        b = np.array(system["b"])
        matrix_size = A.shape[0]
        condition_number = np.linalg.cond(A)  # Calculate the condition number

        # Perform multiple repetitions to compute iterations statistics
        iterations_needed = [solver_algorithm(A, b)[1] for _ in range(num_repetitions)]
        average_iterations = np.mean(iterations_needed)
        std_dev_iterations = np.std(iterations_needed)

        # Store results
        results.append({
            "index": idx,
            "matrix_size": matrix_size,
            "average_iterations": average_iterations,
            "std_dev_iterations": std_dev_iterations,
            "condition_number": condition_number
        })

        # Text of the matrix and b for the graph
        matrix_text = f"A = {A.tolist()}\n\nb = {b.tolist()}\n\nCondition Number = {condition_number:.2f}"

        # Generate the graph
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, num_repetitions + 1), iterations_needed, label="Iterations per trial", color='blue')
        plt.axhline(average_iterations, color='red', linestyle='--', label=f"Average = {average_iterations:.2f}")
        plt.fill_between(
            range(1, num_repetitions + 1),
            average_iterations - std_dev_iterations,
            average_iterations + std_dev_iterations,
            color='gray', alpha=0.3,
            label=f"Standard Deviation = {std_dev_iterations:.2f}"
        )

        # Display matrix A, b, and condition number on the side
        plt.text(1.02, 0.5, matrix_text, transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='center', bbox=dict(facecolor='white', alpha=0.6))

        # Titles and legend
        plt.xlabel("Trial")
        plt.ylabel("Iterations to Convergence")
        plt.title(f"System {idx + 1}: Statistical Analysis of Iterations")
        plt.legend()

        # Save the graph
        plt.savefig(f"{output_folder}/system_{idx + 1}_iterations.png")
        plt.close()

    # Sort results by matrix size and average iterations
    results_sorted = sorted(results, key=lambda x: (x["matrix_size"], x["average_iterations"]))

    # Display sorted results and save to a JSON file
    for result in results_sorted:
        print(f"System {result['index'] + 1}: Matrix size = {result['matrix_size']}, "
              f"Average iterations = {result['average_iterations']:.2f}, "
              f"Standard deviation = {result['std_dev_iterations']:.2f}, "
              f"Condition number = {result['condition_number']:.2f}")

    with open(f"{output_folder}/sorted_results.json", "w") as f:
        json.dump(results_sorted, f, indent=4)

    print(f"Results have been processed, graphs generated, and stored in '{output_folder}/sorted_results.json'.")

# Example usage
if __name__ == "__main__":
    from all_algortyme.KacZmarz_Distribued.Kaczmarz_Distribuate_V1 import Kaczmarz  # Replace with your solver

    def custom_solver(A, b):
        """
        Wrapper for the Kaczmarz solver to return iterations.
        """
        return Kaczmarz(A, b)

    # Parameters
    systems_file = "ressource/System_of_linear_equations/systems_data_3x3.json"
    output_folder = "graphs"
    num_repetitions = 500

    plot_system_analysis(systems_file, output_folder, num_repetitions, custom_solver)