import os
import time
import math
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import tsplib95


"""
This is a simplified TSP solver that implements only the core features:
    - Basic k-opt moves
    - Recursive improvement (step)

Missing features of Lin-Kernighan heuristic algorithm:
    - Recursive improvement is only partially implemented
    - Flip tracking & rollback
    - Alternate first step (alternate_step)
    - Neighborhood ordering (lk-ordering)
    - Breadth & depth parameters
    - Kick / double-bridge restarts
    - Tour object abstraction (next, flip)
    - Delta-based tentative improvements
    - Chained Lin-Kernighan algorithm
"""


# Compute pairwise Euclidean distances for all city coordinates.
# Returns a distance matrix D where D[i, j] = distance between city i and j.
def compute_distance_matrix(coords):
    diff = coords[:, None] - coords[None, :]        # shape (n, n, 2)
    return np.linalg.norm(diff, axis=2)             # shape (n, n)


# Compute the total length of a tour given a distance matrix.
def total_distance(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[(i + 1) % len(tour)]]  # wrap around
               for i in range(len(tour)))


# Compute the gain from swapping two edges: (a-b) and (c-d) -> (a-c) and (b-d).
def compute_gain(dist_matrix, a, b, c, d):
    return dist_matrix[a][b] + dist_matrix[c][d] - dist_matrix[a][c] - dist_matrix[b][d]


# Reverse a segment between two indices [i+1, j] to simulate a 2-opt move.
def two_opt_swap(tour, i, j):
    return tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]


# Recursive search that attempts to improve the tour incrementally.
def recursive_k_opt(
    tour,
    dist_matrix,
    start_index,
    visited,
    gain_sum,
    max_k,
    deadline
):
    # Abort recursion if depth exceeded or time limit reached
    if max_k == 0 or time.time() >= deadline:
        return False

    n = len(tour)
    t1 = tour[start_index]
    t2 = tour[(start_index + 1) % n]

    for j in range(n):
        t3 = tour[j]
        t4 = tour[(j + 1) % n]

        # Skip invalid pairs (same node, adjacent edge, or already visited)
        if t3 == t1 or t4 == t2 or j == start_index or j in visited:
            continue

        gain = compute_gain(dist_matrix, t1, t2, t3, t4)
        if gain + gain_sum <= 0:
            continue  # Only consider moves that improve or might improve total cost

        new_tour = two_opt_swap(tour, start_index, j)
        new_length = total_distance(new_tour, dist_matrix)
        old_length = total_distance(tour, dist_matrix)

        if new_length < old_length:
            tour[:] = new_tour  # Commit the improved tour in-place
            return True

        # Try extending the move recursively to a deeper k-opt move
        if recursive_k_opt(new_tour, dist_matrix, start_index,
                           visited | {j}, gain + gain_sum, max_k - 1, deadline):
            tour[:] = new_tour
            return True

        if time.time() >= deadline:      # Time guard in deep recursion
            return False

    return False  # No improving move found at this level


# Simplified TSP solver with a wall-clock time limit.
def simple_tsp_solver(
    coords,
    max_k = 4,
    time_limit = 10.0          # Seconds allowed for search
):
    """
    Runs the simplified k-opt search until no improvement
    OR the wall-clock time limit is reached.
    Returns the best tour found so far.
    """
    deadline = time.time() + time_limit

    n            = len(coords)
    coords       = np.asarray(coords, float)
    dist_matrix  = compute_distance_matrix(coords)

    tour         = list(range(n))            # Initial tour: sequential node order
    best         = tour.copy()
    best_length  = total_distance(tour, dist_matrix)
    improved     = True

    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break                       # Stop outer loop if time is up
            visited = set()
            if recursive_k_opt(tour, dist_matrix, i, visited, 0.0, max_k, deadline):
                new_length = total_distance(tour, dist_matrix)
                if new_length < best_length:
                    best = tour.copy()
                    best_length = new_length
                    improved = True
                    break  # Restart outer loop to allow global improvement

    return best, best_length


# Visualize the tour in 2D using matplotlib.
def plot_tour(coords, tour, length = None):
    coords = np.asarray(coords)
    tour_cycle = tour + [tour[0]]             # Close the tour into a cycle
    x, y = coords[tour_cycle, 0], coords[tour_cycle, 1]

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'o-', markersize=2)
    plt.title(f"TSP Path (Length: {length:.2f})" if length is not None else "TSP Path")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# TSPLIB helpers
def read_tsp(path):
    prob = tsplib95.load(path)
    coords_map = dict(prob.node_coords)
    nodes = sorted(coords_map.keys())
    return np.array([coords_map[i] for i in nodes], float)


def read_opt_tour(path):
    tour, reading = [], False
    with open(path) as f:
        for line in f:
            tok = line.strip()
            if tok.upper().startswith('TOUR_SECTION'):
                reading = True
                continue
            if not reading:
                continue
            for p in tok.split():
                if p in ('-1', 'EOF'):
                    reading = False
                    break
                if p.isdigit() and int(p) > 0:
                    tour.append(int(p) - 1)
            if not reading:
                break
    return tour


# Batch run and plot
if __name__ == '__main__':
    folder = '../TSPLIB95/tsp'               # Change to your TSPLIB path
    results = []

    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith('.tsp'):
            continue
        base = fn[:-4]
        tsp_path = os.path.join(folder, fn)
        opt_path = os.path.join(folder, base + '.opt.tour')
        if not os.path.exists(opt_path):
            continue

        problem = tsplib95.load(tsp_path)
        if getattr(problem, 'edge_weight_type', '').upper() != 'EUC_2D':
            continue

        print(f"\nProcessing {base} (EUC_2D)...")
        coords   = read_tsp(tsp_path)
        D        = compute_distance_matrix(coords)
        opt_tour = read_opt_tour(opt_path)
        opt_len  = total_distance(opt_tour, D)
        print(f"  Optimal length: {opt_len:.2f}")

        start = time.time()
        heu_tour, heu_len = simple_tsp_solver(coords, max_k=4, time_limit=5.0)
        elapsed = time.time() - start
        gap = max(0.0, 100.0 * (heu_len - opt_len) / opt_len)
        print(f"  Heuristic length: {heu_len:.2f}  Gap: {gap:.2f}%  Time: {elapsed:.2f}s")

        results.append({'name': base, 'coords': coords,
                        'opt_tour': opt_tour, 'heu_tour': heu_tour,
                        'opt_len': opt_len, 'heu_len': heu_len,
                        'gap': gap, 'time': elapsed})

    print("\nInstance   OptLen   HeuLen   Gap(%)   Time(s)")
    for r in results:
        print(f"{r['name']:10s} {r['opt_len']:8.2f} {r['heu_len']:8.2f} {r['gap']:8.2f} {r['time']:8.2f}")
    if results:
        avg_opt = sum(r['opt_len'] for r in results) / len(results)
        avg_heu = sum(r['heu_len'] for r in results) / len(results)
        avg_gap = sum(r['gap'] for r in results) / len(results)
        avg_time = sum(r['time'] for r in results) / len(results)
        print(f"{'AVERAGE':10s} {avg_opt:8.2f} {avg_heu:8.2f} {avg_gap:8.2f} {avg_time:8.2f}")
    print("Done.")

    # Plot tours
    if results:
        cols = int(math.ceil(math.sqrt(len(results))))
        rows = int(math.ceil(len(results) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes_list = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for ax, r in zip(axes_list, results):
            coords = r['coords']
            opt = r['opt_tour'] + [r['opt_tour'][0]]
            heu = r['heu_tour'] + [r['heu_tour'][0]]
            ax.plot(coords[heu, 0], coords[heu, 1], '-', label='Heuristic')
            ax.plot(coords[opt, 0], coords[opt, 1], ':', label='Optimal')
            ax.set_title(f"{r['name']} gap={r['gap']:.2f}%")
            ax.axis('equal')
            ax.grid(True)
        for ax in axes_list[len(results):]:
            ax.axis('off')
        plt.tight_layout()
        plt.show()