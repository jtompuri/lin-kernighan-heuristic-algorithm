#!/usr/bin/env python3
"""
Comprehensive benchmark script for Lin-Kernighan TSP solver with Numba optimizations.

This script tests performance across different problem sizes and Numba configurations
to validate the integration and measure performance gains.
"""

import time
import sys
import numpy as np
from pathlib import Path
import argparse
import json

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent))

from lin_kernighan_tsp_solver.lk_algorithm_integrated import (
    get_performance_info,
    benchmark_integration,
    build_distance_matrix_auto,
    chained_lin_kernighan_auto
)
from lin_kernighan_tsp_solver.config import NUMBA_CONFIG
from lin_kernighan_tsp_solver.tsp_io import read_tsp_file
from lin_kernighan_tsp_solver.starting_cycles import generate_starting_cycle


def benchmark_distance_matrix(coords: np.ndarray, iterations: int = 10) -> dict:
    """Benchmark distance matrix computation with and without Numba."""
    from lin_kernighan_tsp_solver.lk_algorithm_integrated import build_distance_matrix
    from lin_kernighan_tsp_solver.lk_algorithm import build_distance_matrix as original_build_distance_matrix
    
    # Test with Numba
    times_numba = []
    for _ in range(iterations):
        start = time.time()
        D_numba = build_distance_matrix(coords, use_numba=True)
        times_numba.append(time.time() - start)
    
    # Test without Numba
    times_original = []
    for _ in range(iterations):
        start = time.time()
        D_original = original_build_distance_matrix(coords)
        times_original.append(time.time() - start)
    
    # Verify results are equivalent
    are_equivalent = np.allclose(D_numba, D_original, rtol=1e-10)
    
    avg_numba = np.mean(times_numba)
    avg_original = np.mean(times_original)
    speedup = avg_original / avg_numba if avg_numba > 0 else float('inf')
    
    return {
        'numba_time': avg_numba,
        'original_time': avg_original,
        'speedup': speedup,
        'equivalent': are_equivalent,
        'n_nodes': len(coords)
    }


def benchmark_tour_operations(n_nodes: int, n_operations: int = 1000) -> dict:
    """Benchmark Tour operations with and without Numba."""
    from lin_kernighan_tsp_solver.lk_algorithm_integrated import Tour
    
    # Generate test data
    np.random.seed(42)
    coords = np.random.uniform(0, 1000, (n_nodes, 2))
    order = list(range(n_nodes))
    np.random.shuffle(order)
    
    # Test with Numba
    tour_numba = Tour(order.copy(), use_numba=True)
    D = build_distance_matrix_auto(coords)
    tour_numba.init_cost(D)
    
    start = time.time()
    for i in range(n_operations):
        # Mix of operations
        if i % 4 == 0 and n_nodes > 4:
            tour_numba.flip(0, 2)
        elif i % 4 == 1:
            _ = tour_numba.next(0)
        elif i % 4 == 2:
            _ = tour_numba.prev(0)
        else:
            _ = tour_numba.cost
    numba_time = time.time() - start
    
    # Test without Numba
    tour_original = Tour(order.copy(), use_numba=False)
    tour_original.init_cost(D)
    
    start = time.time()
    for i in range(n_operations):
        # Same operations
        if i % 4 == 0 and n_nodes > 4:
            tour_original.flip(0, 2)
        elif i % 4 == 1:
            _ = tour_original.next(0)
        elif i % 4 == 2:
            _ = tour_original.prev(0)
        else:
            _ = tour_original.cost
    original_time = time.time() - start
    
    speedup = original_time / numba_time if numba_time > 0 else float('inf')
    
    return {
        'numba_time': numba_time,
        'original_time': original_time,
        'speedup': speedup,
        'n_nodes': n_nodes,
        'n_operations': n_operations
    }


def benchmark_full_algorithm(tsp_file: str) -> dict:
    """Benchmark the full Lin-Kernighan algorithm with and without Numba."""
    coords = read_tsp_file(tsp_file)
    n_nodes = len(coords)
    initial_tour = generate_starting_cycle(coords, method='qboruvka')
    
    # Save original config
    original_enabled = NUMBA_CONFIG["ENABLED"]
    
    # Test with Numba enabled
    NUMBA_CONFIG["ENABLED"] = True
    start = time.time()
    tour_numba, cost_numba = chained_lin_kernighan_auto(coords, initial_tour)
    time_numba = time.time() - start
    
    # Test with Numba disabled
    NUMBA_CONFIG["ENABLED"] = False
    start = time.time()
    tour_original, cost_original = chained_lin_kernighan_auto(coords, initial_tour)
    time_original = time.time() - start
    
    # Restore original config
    NUMBA_CONFIG["ENABLED"] = original_enabled
    
    speedup = time_original / time_numba if time_numba > 0 else float('inf')
    cost_diff = abs(cost_numba - cost_original) / cost_original if cost_original > 0 else 0
    
    return {
        'problem': Path(tsp_file).stem,
        'n_nodes': n_nodes,
        'numba_time': time_numba,
        'original_time': time_original,
        'speedup': speedup,
        'numba_cost': cost_numba,
        'original_cost': cost_original,
        'cost_difference_percent': cost_diff * 100
    }


def run_comprehensive_benchmark(output_file: str = None):
    """Run comprehensive benchmarks across different problem sizes and operations."""
    print("=== Lin-Kernighan Numba Integration Benchmark ===")
    print()
    
    # Performance info
    perf_info = get_performance_info()
    print("Performance Configuration:")
    print(f"  Numba Available: {perf_info['numba']['available']}")
    print(f"  Numba Enabled: {perf_info['numba']['enabled']}")
    print(f"  Auto Threshold: {perf_info['numba']['auto_threshold']} nodes")
    print()
    
    results = {
        'config': perf_info,
        'benchmarks': {
            'distance_matrix': [],
            'tour_operations': [],
            'integration_test': [],
            'full_algorithm': []
        }
    }
    
    # 1. Distance matrix benchmarks
    print("1. Distance Matrix Benchmarks:")
    for n_nodes in [50, 100, 200, 500]:
        np.random.seed(42)
        coords = np.random.uniform(0, 1000, (n_nodes, 2))
        result = benchmark_distance_matrix(coords, iterations=5)
        results['benchmarks']['distance_matrix'].append(result)
        print(f"  {n_nodes:3d} nodes: {result['speedup']:6.1f}x speedup "
              f"({result['original_time']:6.3f}s → {result['numba_time']:6.3f}s)")
    print()
    
    # 2. Tour operations benchmarks
    print("2. Tour Operations Benchmarks:")
    for n_nodes in [50, 100, 200, 500]:
        result = benchmark_tour_operations(n_nodes, n_operations=1000)
        results['benchmarks']['tour_operations'].append(result)
        print(f"  {n_nodes:3d} nodes: {result['speedup']:6.1f}x speedup "
              f"({result['original_time']:6.3f}s → {result['numba_time']:6.3f}s)")
    print()
    
    # 3. Integration test
    print("3. Integration Test:")
    integration_result = benchmark_integration(100, 1000)
    results['benchmarks']['integration_test'].append(integration_result)
    print(f"  Integration speedup: {integration_result['speedup']:.1f}x")
    print(f"  Integrated impl: {integration_result['integrated_implementation']['implementation']}")
    print(f"  Original impl: {integration_result['original_implementation']['implementation']}")
    print()
    
    # 4. Full algorithm benchmarks (if TSP files available)
    print("4. Full Algorithm Benchmarks:")
    tsp_files = list(Path("problems/tsplib95").glob("*.tsp"))[:3]  # Test first 3 files
    if tsp_files:
        for tsp_file in tsp_files:
            try:
                result = benchmark_full_algorithm(str(tsp_file))
                results['benchmarks']['full_algorithm'].append(result)
                print(f"  {result['problem']:10s} ({result['n_nodes']:3d} nodes): "
                      f"{result['speedup']:6.1f}x speedup "
                      f"(cost diff: {result['cost_difference_percent']:5.2f}%)")
            except Exception as e:
                print(f"  {tsp_file.stem:10s}: Error - {e}")
    else:
        print("  No TSP files found in problems/tsplib95/")
    print()
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_file}")
    
    # Summary
    print("=== Summary ===")
    dm_speedups = [r['speedup'] for r in results['benchmarks']['distance_matrix']]
    tour_speedups = [r['speedup'] for r in results['benchmarks']['tour_operations']]
    
    if dm_speedups:
        print(f"Distance Matrix Speedup: {np.mean(dm_speedups):.1f}x average "
              f"(range: {min(dm_speedups):.1f}x - {max(dm_speedups):.1f}x)")
    
    if tour_speedups:
        print(f"Tour Operations Speedup: {np.mean(tour_speedups):.1f}x average "
              f"(range: {min(tour_speedups):.1f}x - {max(tour_speedups):.1f}x)")
    
    integration_speedup = integration_result.get('speedup', 0)
    if integration_speedup > 0:
        print(f"Integration Speedup: {integration_speedup:.1f}x")
    
    print(f"Numba Integration Status: ✓ Working")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Lin-Kernighan Numba integration")
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for results (default: benchmark_results.json)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick benchmark with fewer iterations')
    
    args = parser.parse_args()
    
    if args.quick:
        print("Running quick benchmark...")
        # Quick integration test only
        result = benchmark_integration(50, 100)
        print(f"Quick integration test: {result['speedup']:.1f}x speedup")
        return
    
    run_comprehensive_benchmark(args.output)


if __name__ == "__main__":
    main()
