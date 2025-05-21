import numpy as np
import itertools
import matplotlib.pyplot as plt
import time

# Precompute all distances for O(1) lookup
def compute_distance_matrix(coords):
    diff = coords[:, np.newaxis] - coords[np.newaxis, :]
    return np.linalg.norm(diff, axis=2)

def total_distance(tour, D, best_so_far=float('inf')):
    total = 0
    n = len(tour)
    for i in range(n):
        total += D[tour[i], tour[(i+1)%n]]
        if total > best_so_far:  # Early abandon
            return float('inf')
    return total

def tsp_brute_force(coords):
    n = len(coords)
    D = compute_distance_matrix(coords)
    cities = list(range(n))
    best_tour = None
    best_length = float('inf')

    for perm in itertools.permutations(cities[1:]):  # Fix city 0
        tour = [0] + list(perm)
        length = total_distance(tour, D, best_length)
        if length < best_length:
            best_tour = tour
            best_length = length

    return best_tour, best_length

# Plot the resulting TSP tour
def plot_tour(coords, tour, length=None):
    coords = np.array(coords)
    tour = tour + [tour[0]]  # Close the loop
    x = coords[tour, 0]
    y = coords[tour, 1]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'o-', markersize=2)
    plt.title(f"TSP-reitti (pituus: {length:.2f})" if length else "TSP-reitti")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show() 

# Example usage
if __name__ == "__main__":
    n = 12
    np.random.seed(42)
    coords = np.random.rand(n, 2) * 1000

    print(f"Running brute-force TSP for {n} cities...")
    start = time.time()
    tour, length = tsp_brute_force(coords)
    duration = time.time() - start

    print(f"Optimal tour: {tour}")
    print(f"Optimal length: {length:.2f}")
    print(f"Time: {duration:.2f} s")

    plot_tour(coords, tour, length)