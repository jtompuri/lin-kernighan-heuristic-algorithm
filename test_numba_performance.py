#!/usr/bin/env python3
"""
Direct performance test of Numba optimizations.
"""

import time
import numpy as np
from lin_kernighan_tsp_solver.lk_algorithm import chained_lin_kernighan as original_chained_lin_kernighan
from lin_kernighan_tsp_solver.lk_algorithm_integrated import chained_lin_kernighan as integrated_chained_lin_kernighan

def run_performance_test():
    """Test the actual performance difference between original and Numba-optimized versions."""
    print("=== Numba Performance Test ===\n")
    
    # Test different problem sizes
    problem_sizes = [30, 50, 100, 200]
    
    for n in problem_sizes:
        print(f"--- Testing problem size: {n} nodes ---")
        
        # Create test problem
        np.random.seed(42)
        coords = np.random.rand(n, 2) * 100
        initial_tour = list(range(n))
        time_limit = 3.0  # 3 seconds per test
        
        # Test original implementation
        print("Running original implementation...")
        start_time = time.time()
        original_tour, original_cost = original_chained_lin_kernighan(
            coords, initial_tour, time_limit_seconds=time_limit
        )
        original_time = time.time() - start_time
        
        # Test integrated implementation with Numba
        print("Running Numba-optimized implementation...")
        start_time = time.time()
        integrated_tour, integrated_cost = integrated_chained_lin_kernighan(
            coords, initial_tour, time_limit_seconds=time_limit, 
            use_numba=True, verbose=True
        )
        integrated_time = time.time() - start_time
        
        # Calculate speedup
        speedup = original_time / integrated_time if integrated_time > 0 else float('inf')
        
        print(f"Results:")
        print(f"  Original:   time={original_time:.3f}s, cost={original_cost:.2f}")
        print(f"  Integrated: time={integrated_time:.3f}s, cost={integrated_cost:.2f}")
        print(f"  Speedup:    {speedup:.2f}x")
        print(f"  Cost diff:  {abs(original_cost - integrated_cost):.2f}")
        print()
        
        # Sanity check - costs should be similar
        if abs(original_cost - integrated_cost) > 1.0:
            print(f"WARNING: Large cost difference detected!")


if __name__ == "__main__":
    run_performance_test()
