"""
Performance profiling script for Lin-Kernighan algorithm.

This script profiles the LK algorithm to identify bottlenecks and opportunities
for Numba JIT optimization.
"""

import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

# Add the parent directory to the path so we can import the LK modules
sys.path.append(str(Path(__file__).parent.parent))

from lin_kernighan_tsp_solver.lk_algorithm import (
    chained_lin_kernighan, build_distance_matrix, delaunay_neighbors, Tour
)
from lin_kernighan_tsp_solver.starting_cycles import generate_starting_cycle


def create_test_problem(n_nodes: int, seed: int = 42) -> np.ndarray:
    """Create a test TSP problem with random coordinates."""
    np.random.seed(seed)
    return np.random.uniform(0, 1000, (n_nodes, 2))


def profile_lk_components():
    """Profile individual components of the LK algorithm."""
    print("=== Lin-Kernighan Performance Profiling ===\n")
    
    # Test different problem sizes
    problem_sizes = [20, 50, 100]
    
    for n in problem_sizes:
        print(f"--- Profiling problem size: {n} nodes ---")
        coords = create_test_problem(n)
        
        # Profile distance matrix computation
        start_time = time.time()
        D = build_distance_matrix(coords)
        distance_time = time.time() - start_time
        print(f"Distance matrix computation: {distance_time:.4f}s")
        
        # Profile neighbor list generation
        start_time = time.time()
        neigh = delaunay_neighbors(coords)
        neighbor_time = time.time() - start_time
        print(f"Delaunay neighbors computation: {neighbor_time:.4f}s")
        
        # Profile starting cycle generation
        start_time = time.time()
        initial_tour = generate_starting_cycle(coords, method="qboruvka")
        starting_cycle_time = time.time() - start_time
        print(f"Starting cycle generation: {starting_cycle_time:.4f}s")
        
        # Profile Tour operations
        tour = Tour(initial_tour, D)
        start_time = time.time()
        for _ in range(100):
            # Test flip operations
            if n > 4:
                tour.flip(0, 2)
                tour.flip(2, 0)  # Undo
        tour_ops_time = time.time() - start_time
        print(f"Tour operations (200 flips): {tour_ops_time:.4f}s")
        
        # Profile short LK run
        start_time = time.time()
        result_tour, result_cost = chained_lin_kernighan(
            coords, initial_tour, time_limit_seconds=1.0
        )
        lk_time = time.time() - start_time
        print(f"LK algorithm (1s limit): {lk_time:.4f}s, final cost: {result_cost:.2f}")
        print()


def detailed_profiling():
    """Run detailed cProfile analysis on LK algorithm."""
    print("=== Detailed cProfile Analysis ===\n")
    
    n = 50  # Medium-sized problem for detailed analysis
    coords = create_test_problem(n)
    initial_tour = generate_starting_cycle(coords, method="qboruvka")
    
    # Profile the chained LK algorithm
    profiler = cProfile.Profile()
    profiler.enable()
    
    result_tour, result_cost = chained_lin_kernighan(
        coords, initial_tour, time_limit_seconds=3.0
    )
    
    profiler.disable()
    
    # Save profiling results
    output_file = Path(__file__).parent / "lk_profile_results.prof"
    profiler.dump_stats(str(output_file))
    
    # Display top functions by cumulative time
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    print("Top 20 functions by cumulative time:")
    stats.print_stats(20)
    
    print("\nTop 20 functions by total time:")
    stats.sort_stats('tottime')
    stats.print_stats(20)
    
    return output_file


def analyze_hot_spots():
    """Analyze specific hot spots in the algorithm."""
    print("=== Hot Spot Analysis ===\n")
    
    n = 30
    coords = create_test_problem(n)
    D = build_distance_matrix(coords)
    neigh = delaunay_neighbors(coords)
    initial_tour = generate_starting_cycle(coords, method="qboruvka")
    tour = Tour(initial_tour, D)
    
    # Test Tour.next() and Tour.prev() performance
    start_time = time.time()
    for _ in range(10000):
        for i in range(min(n, 10)):
            next_node = tour.next(i)
            prev_node = tour.prev(i)
    next_prev_time = time.time() - start_time
    print(f"Tour.next()/prev() operations (100k calls): {next_prev_time:.4f}s")
    
    # Test Tour.sequence() performance
    start_time = time.time()
    for _ in range(1000):
        for i in range(min(n, 10)):
            for j in range(min(n, 10)):
                for k in range(min(n, 10)):
                    if i != j != k:
                        tour.sequence(i, j, k)
    sequence_time = time.time() - start_time
    print(f"Tour.sequence() operations (10k calls): {sequence_time:.4f}s")
    
    # Test distance matrix access patterns
    start_time = time.time()
    total_distance = 0.0
    for _ in range(10000):
        for i in range(min(n, 10)):
            for j in range(min(n, 10)):
                total_distance += D[i, j]
    distance_access_time = time.time() - start_time
    print(f"Distance matrix access (100k accesses): {distance_access_time:.4f}s")
    
    # Test neighbor list iteration
    start_time = time.time()
    neighbor_count = 0
    for _ in range(1000):
        for i in range(n):
            neighbor_count += len(neigh[i])
    neighbor_iter_time = time.time() - start_time
    print(f"Neighbor list iteration (1k full iterations): {neighbor_iter_time:.4f}s")


if __name__ == "__main__":
    profile_lk_components()
    analyze_hot_spots()
    profile_file = detailed_profiling()
    
    print(f"\nDetailed profiling results saved to: {profile_file}")
    print("Use 'python -m pstats {profile_file}' to analyze further.")
