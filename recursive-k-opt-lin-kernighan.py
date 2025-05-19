import numpy as np
import matplotlib.pyplot as plt
import time

# Calculate Euclidean distance between two points
def distance(a, b):
    return np.linalg.norm(a - b)

# Compute the total tour length (sum of distances between consecutive cities)
def total_distance(tour, coords):
    return sum(distance(coords[tour[i]], coords[tour[(i + 1) % len(tour)]])
               for i in range(len(tour)))

# Compute the gain from swapping edges (a-b) and (c-d) to (a-c) and (b-d)
def compute_gain(coords, a, b, c, d):
    return distance(coords[a], coords[b]) + distance(coords[c], coords[d]) - \
           distance(coords[a], coords[c]) - distance(coords[b], coords[d])

# Perform a 2-opt swap: reverse a section of the tour between indices i+1 and j
def two_opt_swap(tour, i, j):
    return tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]

# Recursive k-opt search starting from a given index
def recursive_k_opt(tour, coords, start_index, visited, gain_sum, max_k):
    n = len(tour)
    t1 = tour[start_index]
    t2 = tour[(start_index + 1) % n]

    if max_k == 0:
        return False  # Base case: stop recursion

    for j in range(1, n - 2):
        t3 = tour[j]
        t4 = tour[(j + 1) % n]

        # Avoid invalid or repeated swaps
        if t3 == t1 or t4 == t2 or j in visited:
            continue

        gain = compute_gain(coords, t1, t2, t3, t4)

        # Skip if gain is non-positive
        if gain + gain_sum <= 0:
            continue

        # Perform the 2-opt move
        new_tour = two_opt_swap(tour, start_index, j)
        new_length = total_distance(new_tour, coords)
        old_length = total_distance(tour, coords)

        # If the new tour is better, accept it
        if new_length < old_length:
            tour[:] = new_tour
            return True

        # Recursively continue improving the tour (with updated visited set and gain sum)
        if recursive_k_opt(new_tour, coords, start_index, visited | {j}, gain + gain_sum, max_k - 1):
            tour[:] = new_tour
            return True

    return False  # No improving move found

# Main Lin-Kernighan loop
def lin_kernighan(coords, max_k=5):
    tour = list(range(len(coords)))  # Start with a simple sequential tour
    best = tour.copy()
    best_length = total_distance(tour, coords)
    improved = True

    while improved:
        improved = False
        for i in range(len(tour)):
            visited = set()
            # Attempt recursive k-opt improvements from city i
            if recursive_k_opt(tour, coords, i, visited, 0, max_k):
                new_length = total_distance(tour, coords)
                if new_length < best_length:
                    best = tour.copy()
                    best_length = new_length
                    improved = True
                    break  # Restart search from scratch after improvement

    return best, best_length  # Return best tour and its length

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

# Test run for one example
if __name__ == "__main__":
    n = 20
    print(f"\nTestataan {n} solmulla")
    np.random.seed(40)
    coords = np.random.rand(n, 2) * 1000  # Generate n random points in 2D space

    start_time = time.time()
    tour, length = lin_kernighan(coords, max_k=4)  # Run LK with depth up to 4-opt
    duration = time.time() - start_time

    print(f"Pituus: {length:.2f}")
    print(f"Aika: {duration:.2f} s")

    if n <= 300:
        plot_tour(coords, tour, length)