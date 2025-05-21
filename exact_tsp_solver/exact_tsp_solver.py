import itertools
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# Precompute pairwise Euclidean distances between all city coordinates
# Returns an n x n distance matrix D where D[i, j] = distance between city i and city j
def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, np.newaxis] - coords[np.newaxis, :]  # shape: (n, n, 2)
    return np.linalg.norm(diff, axis=2)  # shape: (n, n)

# Compute the total length of a tour using a distance matrix.
# Early abandon: if the current total exceeds the best so far, return infinity.
def total_distance(tour: List[int], D: np.ndarray, best_so_far: float = float('inf')) -> float:
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += D[tour[i], tour[(i + 1) % n]]  # wrap around to form a cycle
        if total > best_so_far:  # prune search if worse than current best
            return float('inf')
    return total

# Brute-force solution to the TSP: tries all (n-1)! possible tours that start from city 0
# Returns the shortest tour found and its length
def tsp_brute_force(coords: np.ndarray) -> Tuple[List[int], float]:
    n = len(coords)
    D = compute_distance_matrix(coords)
    cities = list(range(n))
    best_tour: List[int] = []
    best_length = float('inf')

    # Fix city 0 as the starting point to avoid duplicate cycles with different rotations
    for perm in itertools.permutations(cities[1:]):
        tour = [0] + list(perm)
        length = total_distance(tour, D, best_length)
        if length < best_length:
            best_tour = tour
            best_length = length

    return best_tour, best_length

# Plot the TSP tour on a 2D plane
def plot_tour(coords: np.ndarray, tour: List[int], length: Optional[float] = None) -> None:
    coords = np.array(coords)
    tour = tour + [tour[0]]  # close the loop by returning to the starting city
    x = coords[tour, 0]
    y = coords[tour, 1]

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'o-', markersize=2)
    plt.title(f"TSP tour (length: {length:.2f})" if length else "TSP tour")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage for testing the brute-force TSP solver
if __name__ == "__main__":
    n = 11  # number of cities
    np.random.seed(42)
    coords = np.random.rand(n, 2) * 1000  # random coordinates in 1000x1000 space

    print(f"Running brute-force TSP for {n} cities...")
    start = time.time()
    tour, length = tsp_brute_force(coords)
    duration = time.time() - start

    print(f"Optimal tour: {tour}")
    print(f"Optimal length: {length:.2f}")
    print(f"Time: {duration:.2f} s")

    plot_tour(coords, tour, length)