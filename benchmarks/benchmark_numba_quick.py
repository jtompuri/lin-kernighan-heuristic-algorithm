#!/usr/bin/env python3
"""
Quick benchmark to test if Numba optimizations are actually working.
"""

import time
import numpy as np
from lin_kernighan_tsp_solver.lk_algorithm_integrated import (
    Tour, build_distance_matrix, 
    NUMBA_INTEGRATION_AVAILABLE, NUMBA_AVAILABLE
)

def benchmark_tour_operations():
    """Benchmark Tour operations with and without Numba."""
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print(f"Numba integration available: {NUMBA_INTEGRATION_AVAILABLE}")
    print()
    
    # Create test problem
    np.random.seed(42)
    n = 200  # Medium size problem
    coords = np.random.uniform(0, 1000, (n, 2))
    order = list(range(n))
    
    print(f"Testing with {n} nodes...")
    print()
    
    # Test distance matrix building
    print("=== Distance Matrix Building ===")
    
    start_time = time.time()
    D_original = build_distance_matrix(coords, use_numba=False)
    time_original = time.time() - start_time
    print(f"Original implementation: {time_original:.4f}s")
    
    if NUMBA_INTEGRATION_AVAILABLE:
        start_time = time.time()
        D_numba = build_distance_matrix(coords, use_numba=True)
        time_numba = time.time() - start_time
        print(f"Numba implementation: {time_numba:.4f}s")
        print(f"Speedup: {time_original/time_numba:.2f}x")
        print(f"Results match: {np.allclose(D_original, D_numba)}")
    else:
        print("Numba integration not available")
    
    print()
    
    # Test Tour operations
    print("=== Tour Operations ===")
    
    # Original Tour
    start_time = time.time()
    tour_original = Tour(order, D_original, use_numba=False)
    
    # Perform many operations
    for i in range(1000):
        next_node = tour_original.next(i % n)
        prev_node = tour_original.prev(i % n)
        seq_result = tour_original.sequence(i % n, (i + 1) % n, (i + 2) % n)
        
        # Occasional flip
        if i % 100 == 0:
            tour_original.flip(i % n, (i + 50) % n)
    
    time_original_ops = time.time() - start_time
    print(f"Original Tour ops: {time_original_ops:.4f}s")
    print(f"Original implementation: {tour_original.get_implementation_info()['implementation']}")
    
    # Numba Tour
    if NUMBA_INTEGRATION_AVAILABLE:
        start_time = time.time()
        tour_numba = Tour(order, D_original, use_numba=True)
        
        # Perform the same operations
        for i in range(1000):
            next_node = tour_numba.next(i % n)
            prev_node = tour_numba.prev(i % n)
            seq_result = tour_numba.sequence(i % n, (i + 1) % n, (i + 2) % n)
            
            # Occasional flip
            if i % 100 == 0:
                tour_numba.flip(i % n, (i + 50) % n)
        
        time_numba_ops = time.time() - start_time
        print(f"Numba Tour ops: {time_numba_ops:.4f}s")
        print(f"Numba implementation: {tour_numba.get_implementation_info()['implementation']}")
        print(f"Tour ops speedup: {time_original_ops/time_numba_ops:.2f}x")
    else:
        print("Numba Tour not available")
    
    print()

if __name__ == "__main__":
    benchmark_tour_operations()
