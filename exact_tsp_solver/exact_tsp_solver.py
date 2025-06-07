"""
exact_tsp_solver.py

Brute-force exact solver for the Traveling Salesperson Problem (TSP).

This script generates a random TSP instance, solves it exactly using a brute-force
algorithm (by checking all permutations of cities, fixing the starting city), and
saves the instance and optimal tour in TSPLIB format. The solution and its length
are printed and plotted. Intended for small instances due to factorial time complexity.

Output files are saved in a 'tsp' subdirectory relative to the script's parent directory.
"""

import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np

N_CITIES = 10
RANDOM_SEED = 42
OUTPUT_SUBDIRECTORY = "tsp"


def compute_distance_matrix(coords):
    """Compute the pairwise Euclidean distance matrix for city coordinates.

    Args:
        coords (np.ndarray): Array of shape (n, 2) with city coordinates.

    Returns:
        np.ndarray: Distance matrix of shape (n, n).
    """
    diff = coords[:, np.newaxis] - coords[np.newaxis, :]
    return np.linalg.norm(diff, axis=2)


def total_distance(tour, D, best_so_far=float('inf')):
    """Calculate the total length of a tour, with optional early abandoning.

    Args:
        tour (list[int]): Tour as a list of city indices.
        D (np.ndarray): Distance matrix.
        best_so_far (float, optional): If partial sum exceeds this, abandon. Defaults to inf.

    Returns:
        float: Total tour length, or inf if abandoned.
    """
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += D[tour[i], tour[(i + 1) % n]]
        if total > best_so_far:
            return float('inf')
    return total


def tsp_brute_force(coords):
    """Solve the TSP exactly by brute-force (fixing city 0).

    Args:
        coords (np.ndarray): Array of shape (n, 2) with city coordinates.

    Returns:
        tuple: (best_tour, best_length)
            best_tour (list[int]): Optimal tour as list of indices.
            best_length (float): Length of the optimal tour.
    """
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


def plot_tour(coords, tour, length=None):
    """Plot the TSP tour using Matplotlib.

    Args:
        coords (np.ndarray): Array of city coordinates.
        tour (list[int]): Tour as a list of city indices.
        length (float, optional): Tour length for title. Defaults to None.
    """
    path = tour + [tour[0]]
    coords = np.array(coords)
    x, y = coords[path, 0], coords[path, 1]
    plt.figure(figsize=(4, 4))
    plt.plot(x, y, '-')
    if length is not None:
        plt.title(f"TSP tour (length: {length:.2f})")
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def save_tour(tour, filepath, name):
    """Save a tour in TSPLIB .tour format.

    Args:
        tour (list[int]): Tour as a list of city indices.
        filepath (str): Output file path.
        name (str): Name for the tour.
    """
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
    coords = np.random.randint(1, 1001, size=(N_CITIES, 2))

    script_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(script_parent_dir, OUTPUT_SUBDIRECTORY)
    os.makedirs(out_dir, exist_ok=True)

    tsp_fname = f"rand{N_CITIES}.tsp"
    tour_fname = f"rand{N_CITIES}.opt.tour"
    tsp_path = os.path.join(out_dir, tsp_fname)
    tour_path = os.path.join(out_dir, tour_fname)

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

    print(f"Running brute-force TSP for {N_CITIES} cities...")
    start = time.time()
    tour, length = tsp_brute_force(coords)
    duration = time.time() - start
    print(f"Optimal tour: {tour}")
    print(f"Optimal length: {length:.2f}")
    print(f"Time: {duration:.2f} s")

    save_tour(tour, tour_path, os.path.splitext(tour_fname)[0])
    print(f"Saved optimal tour to {tour_path}")

    plot_tour(coords, tour, length)
