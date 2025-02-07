import numpy as np
import matplotlib.pyplot as plt

# Parameters
TOLERANCE = 1e-6
MAX_ITERATIONS = 10000
ALPHA = 0.5  # Relaxation parameter
N_TRIALS = 10  # Number of repetitions per test
BETA_VALUES = np.linspace(0, 0.5, 10)  # Test beta values


# Function to generate a random well-conditioned system of size (n x n)
def generate_system(n):
    A = 2.0 * np.random.rand(n, n) - np.ones((n, n))
    for i in range(n):
        norm = np.linalg.norm(A[i, :])
        A[i, :] = A[i, :] / norm  # Normalize rows to avoid instability
    b = 2.0 * np.random.rand(n) - np.ones(n)
    return A, b


# Main Kaczmarz algorithm with momentum
def Kaczmarz_momentum(A, b, beta):
    norms = np.sum(A ** 2, axis=1)

    def Kaczmarz_one_iteration(A_row, b_value, inconnue, norm):
        atixi = (b_value - np.dot(A_row, inconnue)) / norm
        inconnue += atixi * A_row
        return inconnue

    def Kaczmarz_one_loop(inconnue):
        all_inconnue = np.zeros_like(inconnue)
        for i in range(len(A)):
            all_inconnue += Kaczmarz_one_iteration(A[i], b[i], inconnue, norms[i])
        return all_inconnue / len(A)

    def convergence(inconnue, x_old):
        return np.linalg.norm(inconnue - x_old) < TOLERANCE

    x = np.random.rand(len(b))  # Random initialization
    x_old = np.zeros_like(x)
    momentum = np.zeros_like(x)

    for iteration in range(MAX_ITERATIONS):
        x_old[:] = x
        update = Kaczmarz_one_loop(x)
        x = (1 - ALPHA) * x + ALPHA * update + beta * momentum
        momentum = x - x_old  # Update momentum

        if convergence(x, x_old):
            return iteration + 1  # Return the number of iterations

    return MAX_ITERATIONS  # If convergence is not reached


# Test momentum effect for different system sizes
def test_different_sizes(sizes, beta_values, n_trials):
    results = {size: {} for size in sizes}
    for size in sizes:
        print(f"Testing system size {size}x{size}...")
        A, b = generate_system(size)
        for beta in beta_values:
            iterations_list = [Kaczmarz_momentum(A, b, beta) for _ in range(n_trials)]
            avg_iterations = np.mean(iterations_list)
            results[size][beta] = avg_iterations
    return results


# Define different system sizes
SYSTEM_SIZES = [3, 5, 10, 20]  # Different complexities
results = test_different_sizes(SYSTEM_SIZES, BETA_VALUES, N_TRIALS)

# Visualization
plt.figure(figsize=(10, 6))
for size in SYSTEM_SIZES:
    plt.plot(BETA_VALUES, list(results[size].values()), marker='o', linestyle='-', label=f'{size}x{size}')
plt.title(f'Effect of Momentum (β) on Convergence for Different System Sizes')
plt.xlabel('Momentum (β)')
plt.ylabel('Average Number of Iterations to Converge')
plt.legend(title='System Size')
plt.grid(True)
plt.show()

# Display results
print("\nResults (average over multiple trials):")
for size, res in results.items():
    print(f"\nSystem Size: {size}x{size}")
    for beta, avg_iterations in res.items():
        print(f"β = {beta:.2f} → {avg_iterations:.2f} iterations (average)")
