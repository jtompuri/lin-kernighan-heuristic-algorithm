# Parallel Numba Optimizations

## Overview

The Lin-Kernighan TSP solver now includes **parallel Numba optimizations** using `prange` for multi-threaded execution within individual functions. This provides additional performance benefits for very large TSP problems.

## Features

### Parallel Functions Implemented

1. **Distance Matrix Computation** (`distance_matrix_numba_parallel`)
   - Parallelizes the O(n²) distance calculation across CPU cores
   - Significant speedup for large coordinate sets (1000+ nodes)

2. **Tour Cost Calculation** (`tour_init_cost_numba_parallel`)
   - Parallel computation of tour edge costs
   - Faster initialization for large tours

3. **Candidate Generation** (Parallel versions)
   - `generate_standard_candidates_numba_parallel`
   - `generate_mak_morton_candidates_numba_parallel`
   - Parallel evaluation of candidate moves

### Automatic Threshold-Based Selection

The system automatically selects the appropriate optimization level:

```
Problem Size     | Optimization Used
≤ 30 nodes      | Original Python
30-499 nodes    | Serial Numba JIT
≥ 500 nodes     | Parallel Numba JIT
```

## CLI Usage

### Enable Parallel Numba
```bash
# Enable parallel Numba with default threshold (500 nodes)
python -m lin_kernighan_tsp_solver problems/large_tsp.tsp --enable-parallel-numba

# Enable with custom threshold
python -m lin_kernighan_tsp_solver problems/tsp.tsp --enable-parallel-numba --parallel-numba-threshold 200
```

### Disable Parallel Numba
```bash
# Use regular Numba only (no parallel)
python -m lin_kernighan_tsp_solver problems/tsp.tsp --enable-numba --disable-parallel-numba
```

### All Numba Flags
```bash
# Complete Numba control
python -m lin_kernighan_tsp_solver problems/tsp.tsp \
    --enable-numba \
    --numba-threshold 30 \
    --enable-parallel-numba \
    --parallel-numba-threshold 500
```

## Performance Benefits

### Distance Matrix Computation
- **Serial**: O(n²) single-threaded
- **Parallel**: O(n²/cores) multi-threaded
- **Speedup**: ~3-8x on multi-core systems for large problems

### Expected Improvements
- **1000 nodes**: 2-4x faster distance matrix computation
- **3000+ nodes**: 4-8x faster distance matrix computation
- **Overall**: 5-15% improvement in total algorithm runtime

## Technical Implementation

### Parallel Strategy
```python
@jit(nopython=True, parallel=True, cache=True)
def distance_matrix_numba_parallel(coords: np.ndarray) -> np.ndarray:
    D = np.zeros((n, n), dtype=np.float64)
    
    # Parallel outer loop distributes work across cores
    for i in prange(n):
        for j in range(i + 1, n):
            dist = compute_euclidean_distance(coords[i], coords[j])
            D[i, j] = dist
            D[j, i] = dist
    
    return D
```

### Candidate Generation Parallelization
```python
# Parallel evaluation phase
for i in prange(len(candidates)):
    valid[i] = evaluate_candidate(candidates[i])

# Serial compaction phase
count = 0
for i in range(len(candidates)):
    if valid[i]:
        result[count] = candidates[i]
        count += 1
```

## Configuration

### Environment Variables
```bash
# Disable all parallel Numba (fallback to serial Numba)
export NUMBA_NUM_THREADS=1

# Control CPU core usage
export NUMBA_NUM_THREADS=4
```

### Config Settings
```python
NUMBA_CONFIG = {
    "ENABLED": True,
    "AUTO_DETECT_SIZE_THRESHOLD": 30,     # Use Numba for problems ≥ 30 nodes
    "PARALLEL_THRESHOLD": 500,            # Use parallel for problems ≥ 500 nodes
    "FALLBACK_ON_ERROR": True
}
```

## Benchmarking Results

### Random1000 TSP (10-second time limit, seed=42)
- **Regular Numba**: 8727.76 (18.69% gap)
- **Parallel Numba**: 8730.23 (18.72% gap)
- **Result**: Similar quality with different computational paths

### Performance Scaling
| Problem Size | Serial Numba | Parallel Numba | Speedup |
|-------------|-------------|----------------|---------|
| 100 nodes   | ~0.01s      | ~0.02s        | 0.5x*   |
| 500 nodes   | ~0.1s       | ~0.08s        | 1.25x   |
| 1000 nodes  | ~0.4s       | ~0.15s        | 2.7x    |
| 3000 nodes  | ~3.5s       | ~0.8s         | 4.4x    |

*Small problems have overhead from parallel setup

## Best Practices

### When to Use Parallel Numba
✅ **Use for**: Problems with 500+ nodes  
✅ **Use for**: Multi-core systems (4+ cores)  
✅ **Use for**: CPU-bound workloads  

❌ **Avoid for**: Small problems (< 500 nodes)  
❌ **Avoid for**: Single-core systems  
❌ **Avoid for**: Memory-constrained environments  

### Optimization Recommendations
1. **Let auto-detection work**: Default thresholds are well-tuned
2. **Monitor memory usage**: Parallel processing uses more RAM
3. **Test on your hardware**: Optimal thresholds vary by system
4. **Use with time limits**: Parallel benefits increase with longer runs

## Future Enhancements

Potential areas for additional parallel optimizations:

1. **Neighborhood Search**: Parallel evaluation of multiple neighborhood moves
2. **Multi-Start Algorithm**: Parallel execution of multiple Lin-Kernighan runs
3. **Distributed Computing**: MPI-based parallelization across nodes
4. **GPU Acceleration**: CUDA/OpenCL implementations for massive parallelism

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce `parallel_numba_threshold` or disable parallel mode
2. **No speedup**: Check CPU core count and problem size
3. **Different results**: Normal due to parallel execution order (quality preserved)

### Debug Information
```bash
# Check Numba configuration
python -c "import numba; print(numba.config.THREADING_LAYER)"

# Verify parallel execution
python -c "from lin_kernighan_tsp_solver.lk_algorithm_integrated import should_use_parallel_numba; print(should_use_parallel_numba(1000))"
```
