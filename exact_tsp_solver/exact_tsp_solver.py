"""
Exact Brute-Force Traveling Salesperson Problem (TSP) Solver.

This script implements a brute-force algorithm to find the exact optimal solution
for the Traveling Salesperson Problem. It is suitable for small problem instances
due to its factorial time complexity.

The script includes functions to:
- Compute a pairwise Euclidean distance matrix from city coordinates.
- Calculate the total length of a given tour, with an option for early abandoning
  if the current partial tour length exceeds a known best.
- Solve the TSP using a brute-force approach by checking all permutations of
  cities (fixing the starting city to reduce redundant permutations).
- Plot the TSP tour using Matplotlib.
- Save the generated TSP problem instance in TSPLIB .tsp format.
- Save the found optimal tour in TSPLIB .opt.tour format.

When run as a script, it generates a random TSP instance with a specified number
of cities, solves it using the brute-force method, saves the problem and its
optimal tour to files, prints the optimal tour and its length, and displays a
plot of the tour. The output files are saved in a 'tsp' subdirectory relative
to this script's parent directory.
"""
import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np
# Instance saving without tsplib95 to ensure compatibility

# --- Constants for script execution ---
N_CITIES = 11  # Number of cities for the generated TSP instance
RANDOM_SEED = 42  # Seed for random number generation
OUTPUT_SUBDIRECTORY = "tsp"  # Subdirectory for saving TSP and tour files


# Precompute pairwise Euclidean distances between all city coordinates
def compute_distance_matrix(coords):
    diff = coords[:, np.newaxis] - coords[np.newaxis, :]
    return np.linalg.norm(diff, axis=2)


# Compute the total length of a tour using a distance matrix with early abandoning
def total_distance(tour, D, best_so_far=float('inf')):
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += D[tour[i], tour[(i + 1) % n]]
        if total > best_so_far:
            return float('inf')
    return total


# Brute-force TSP solver (fixing city 0)
def tsp_brute_force(coords):
    n = len(coords)
    D = compute_distance_matrix(coords)
    cities = list(range(n))
    best_tour = []
    best_length = float('inf')
    for perm in itertools.permutations(cities[1:]):
        tour = [0] + list(perm)
        length = total_distance(tour, D, best_length)
        if length < best_length:
            best_tour, best_length = tour, length
    return best_tour, best_length


# Plot the TSP tour
def plot_tour(coords, tour, length=None):
    path = tour + [tour[0]]
    coords = np.array(coords)
    x, y = coords[path, 0], coords[path, 1]
    plt.figure(figsize=(4, 4))
    plt.plot(x, y, 'o-')
    if length is not None:
        plt.title(f"TSP tour (length: {length:.2f})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Save optimal tour manually in TSPLIB .tour format
def save_tour(tour, filepath, name):
    with open(filepath, 'w') as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TOUR\n")
        f.write(f"DIMENSION: {len(tour)}\n")
        f.write("TOUR_SECTION\n")
        for node in tour:
            f.write(f"{node + 1}\n")
        f.write("-1\n")
        f.write("EOF\n")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    coords = np.random.randint(1, 1001, size=(N_CITIES, 2))  # integer coords in range [1,1000]

    # Create output directory
    # Construct the path relative to the script's parent directory
    script_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(script_parent_dir, OUTPUT_SUBDIRECTORY)
    os.makedirs(out_dir, exist_ok=True)

    # File names
    tsp_fname = f"rand{N_CITIES}.tsp"
    tour_fname = f"rand{N_CITIES}.opt.tour"
    tsp_path = os.path.join(out_dir, tsp_fname)
    tour_path = os.path.join(out_dir, tour_fname)

    # Save TSPLIB instance manually to ensure proper format
    with open(tsp_path, 'w') as f:
        f.write(f"NAME: {os.path.splitext(tsp_fname)[0]}\n")
        f.write("TYPE: TSP\n")
        f.write(f"COMMENT: Random instance with {N_CITIES} nodes\n")
        f.write(f"DIMENSION: {N_CITIES}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, start=1):
            f.write(f"{i} {x} {y}\n")
        f.write("EOF\n")
    print(f"Saved TSPLIB instance to {tsp_path}")

    # Solve brute-force
    print(f"Running brute-force TSP for {N_CITIES} cities...")
    start = time.time()
    tour, length = tsp_brute_force(coords)
    duration = time.time() - start
    print(f"Optimal tour: {tour}")
    print(f"Optimal length: {length:.2f}")
    print(f"Time: {duration:.2f} s")

    # Save optimal tour
    save_tour(tour, tour_path, os.path.splitext(tour_fname)[0])
    print(f"Saved optimal tour to {tour_path}")

    # Plot
    plot_tour(coords, tour, length)
