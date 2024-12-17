from mpi4py import MPI
import numpy as np


def distributed_jacobi(A, b, max_iterations=1000, tolerance=1e-6):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n = len(b)
    x = np.zeros(n)  # Solution initiale
    local_rows = n // size  # Divise les lignes de A entre les agents
    local_A = A[rank * local_rows:(rank + 1) * local_rows]
    local_b = b[rank * local_rows:(rank + 1) * local_rows]

    for iteration in range(max_iterations):
        local_x = np.copy(x)
        for i, row in enumerate(local_A):
            sum_except_i = np.dot(row, x) - row[rank * local_rows + i] * x[rank * local_rows + i]
            local_x[rank * local_rows + i] = (local_b[i] - sum_except_i) / row[rank * local_rows + i]

        # Communication : chaque agent envoie sa partie de x aux autres
        comm.Allreduce(MPI.IN_PLACE, local_x, op=MPI.SUM)
        if np.linalg.norm(local_x - x) < tolerance:
            if rank == 0:
                print("Converged after", iteration + 1, "iterations.")
            break
        x[:] = local_x

    return x


if __name__ == "__main__":
    A = np.array([[4, 1, 2],
                  [3, 5, 1],
                  [1, 1, 3]])
    b = np.array([4, 7, 3])
    solution = distributed_jacobi(A, b)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Distributed Solution:", solution)
