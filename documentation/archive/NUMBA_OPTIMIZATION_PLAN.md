# Lin-Kernighan Algorithm Performance Optimization Plan with Numba JIT

## Executive Summary

Based on the performance profiling results, the Lin-Kernighan algorithm shows several optimization opportunities using Numba JIT compilation. The profiling reveals key bottlenecks that can achieve significant speedups with minimal code changes.

## Performance Profile Analysis

### Top Performance Bottlenecks (by total time):

1. **`_generate_mak_morton_flip_candidates`** - 19.5% of execution time (0.586s/3.000s)
2. **`flip` operations** - 17.5% of execution time (0.524s)
3. **`step` recursive function** - 13.3% of execution time (0.399s)
4. **`_generate_standard_flip_candidates`** - 13.3% of execution time (0.398s)
5. **`prev` method calls** - 11.3% of execution time (0.339s)
6. **`next` method calls** - 7.1% of execution time (0.214s)

### Hot Spot Analysis Results:

- **Tour.sequence()**: 175ms for 10K calls → ~17.5μs per call
- **Distance matrix access**: 100ms for 100K accesses → ~1μs per access
- **Tour.next()/prev()**: 42ms for 100K calls → ~0.42μs per call

### Potential Speedup Areas:

1. **High-frequency numeric operations** (flip, next, prev, sequence)
2. **Candidate generation loops** with heavy distance matrix access
3. **Tour state management** operations
4. **Nested loops in step() function**

## Numba JIT Optimization Strategy

### Phase 1: Core Data Structure Optimizations (Expected 3-5x speedup)

#### 1.1 Tour Class Method Optimizations
```python
# Target methods for @numba.jit(nopython=True):
- Tour.next()          # 769K calls
- Tour.prev()          # 1.29M calls  
- Tour.sequence()      # Heavy computational cost
- Tour.flip()          # 311K calls
```

**Implementation approach:**
- Convert Tour methods to standalone numba-compiled functions
- Pass tour data (order, pos arrays) as parameters
- Use numpy arrays for all data structures

#### 1.2 Distance Matrix Operations
```python
# Create numba-optimized distance functions:
- build_distance_matrix_numba()
- distance_lookup_numba()
```

### Phase 2: Algorithm Core Optimizations (Expected 2-3x additional speedup)

#### 2.1 Candidate Generation Functions
```python
# Target for optimization:
- _generate_standard_flip_candidates()    # 398ms total time
- _generate_mak_morton_flip_candidates()  # 586ms total time
```

**Key optimizations:**
- Replace list comprehensions with numba-compiled loops
- Optimize neighbor list iteration
- Pre-allocate result arrays

#### 2.2 Search Functions
```python
# Target functions:
- step() recursive function
- alternate_step()
- lk_search()
```

### Phase 3: Memory and Cache Optimizations (Expected 1.5-2x additional speedup)

#### 3.1 Data Layout Optimization
- Use contiguous numpy arrays for all tour data
- Optimize memory access patterns
- Reduce allocations in hot paths

#### 3.2 Algorithmic Improvements
- Cache frequently accessed values
- Reduce function call overhead
- Optimize loop structures

## Implementation Plan

### Step 1: Environment Setup
```bash
# Install Numba
pip install numba

# Add to requirements.txt
echo "numba>=0.58.0" >> requirements.txt
```

### Step 2: Create Numba-Optimized Module
Create `lin_kernighan_tsp_solver/lk_algorithm_numba.py`:

```python
import numba
import numpy as np
from numba import jit, types
from numba.typed import Dict, List

# Core tour operations
@jit(nopython=True)
def tour_next_numba(order, pos, v):
    """Numba-optimized tour.next() operation."""
    n = len(order)
    idx = pos[v] + 1
    return order[idx if idx < n else 0]

@jit(nopython=True) 
def tour_prev_numba(order, pos, v):
    """Numba-optimized tour.prev() operation."""
    n = len(order)
    idx = pos[v] - 1
    return order[idx if idx >= 0 else n - 1]

@jit(nopython=True)
def tour_sequence_numba(pos, node_a, node_b, node_c, n):
    """Numba-optimized tour.sequence() operation."""
    idx_a, idx_b, idx_c = pos[node_a], pos[node_b], pos[node_c]
    if idx_a <= idx_c:
        return idx_a <= idx_b <= idx_c
    return idx_a <= idx_b or idx_b <= idx_c

@jit(nopython=True)
def tour_flip_numba(order, pos, start_node, end_node):
    """Numba-optimized tour.flip() operation."""
    n = len(order)
    idx_a, idx_b = pos[start_node], pos[end_node]
    
    if idx_a == idx_b:
        return
        
    if idx_a <= idx_b:
        segment_len = idx_b - idx_a + 1
    else:
        segment_len = (n - idx_a) + (idx_b + 1)
    
    for i in range(segment_len // 2):
        left_idx = (idx_a + i) % n
        right_idx = (idx_b - i) % n
        
        # Swap nodes
        node_left = order[left_idx]
        node_right = order[right_idx]
        
        order[left_idx] = node_right
        order[right_idx] = node_left
        
        # Update positions
        pos[node_left] = right_idx
        pos[node_right] = left_idx
```

### Step 3: Progressive Integration

#### 3.1 Tour Class Hybrid Approach
```python
class TourNumba:
    """Tour class with Numba-optimized operations."""
    
    def __init__(self, order, D=None):
        self.order = np.array(order, dtype=np.int32)
        self.n = len(self.order)
        max_node = np.max(self.order) if self.n > 0 else 0
        self.pos = np.empty(max_node + 1, dtype=np.int32)
        
        # Initialize position mapping
        for i in range(self.n):
            self.pos[self.order[i]] = i
            
        self.cost = None
        if D is not None:
            self.init_cost_numba(D)
    
    def next(self, v):
        return tour_next_numba(self.order, self.pos, v)
    
    def prev(self, v):
        return tour_prev_numba(self.order, self.pos, v)
    
    def sequence(self, a, b, c):
        return tour_sequence_numba(self.pos, a, b, c, self.n)
    
    def flip(self, start, end):
        tour_flip_numba(self.order, self.pos, start, end)
```

#### 3.2 Candidate Generation Optimization
```python
@jit(nopython=True)
def generate_standard_candidates_numba(base, s1, order, pos, D, neigh, delta, tolerance):
    """Numba-optimized standard flip candidate generation."""
    candidates = []
    
    for y1_idx in range(len(neigh[s1])):
        y1_cand = neigh[s1][y1_idx]
        if y1_cand == base or y1_cand == s1:
            continue
            
        gain_G1 = D[base, s1] - D[s1, y1_cand]
        if gain_G1 <= tolerance:
            continue
            
        t3_node = tour_prev_numba(order, pos, y1_cand)
        gain_G2 = D[t3_node, y1_cand] - D[t3_node, base]
        total_gain = gain_G1 + gain_G2
        
        if delta + gain_G1 > tolerance:
            candidates.append((y1_cand, t3_node, total_gain))
    
    return candidates
```

### Step 4: Benchmarking and Validation

#### 4.1 Performance Benchmarks
```python
def benchmark_numba_vs_original():
    """Compare Numba-optimized vs original implementation."""
    sizes = [20, 50, 100, 200]
    
    for n in sizes:
        coords = create_test_problem(n)
        # Test both implementations
        # Measure speedup ratios
```

#### 4.2 Correctness Validation
```python
def validate_numba_correctness():
    """Ensure Numba optimizations produce identical results."""
    # Compare tour operations
    # Compare final tour costs
    # Validate against known optimal solutions
```

## Expected Performance Improvements

### Conservative Estimates:
- **Phase 1**: 3-5x speedup for core operations
- **Phase 2**: 2-3x additional speedup for algorithm functions  
- **Phase 3**: 1.5-2x additional speedup for memory optimizations

### Overall Expected Speedup: 5-15x

### Problem Size Scaling:
- **Small problems (n<50)**: 5-8x speedup
- **Medium problems (50<n<200)**: 8-12x speedup  
- **Large problems (n>200)**: 10-15x speedup

## Implementation Timeline

### Week 1: Setup and Core Optimizations
- Install Numba and dependencies
- Implement Tour class Numba optimizations
- Basic benchmarking

### Week 2: Algorithm Function Optimizations  
- Optimize candidate generation functions
- Optimize search functions
- Performance validation

### Week 3: Advanced Optimizations and Integration
- Memory layout optimizations
- Full algorithm integration
- Comprehensive testing

### Week 4: Refinement and Documentation
- Performance tuning
- Documentation updates
- Final benchmarking report

## Risk Mitigation

### Technical Risks:
1. **Numba compatibility issues** → Use feature detection and fallbacks
2. **Memory usage increase** → Monitor and optimize data structures
3. **Debugging complexity** → Maintain dual code paths during development

### Mitigation Strategies:
- Implement gradual rollout with feature flags
- Maintain original implementation as fallback
- Extensive unit testing for correctness
- Continuous benchmarking during development

## Success Metrics

### Primary Metrics:
- **5x minimum speedup** for 50-node problems
- **10x target speedup** for 100+ node problems
- **Zero regression** in solution quality

### Secondary Metrics:
- Reduced memory allocations
- Improved cache efficiency
- Better scaling characteristics

## Conclusion

The Lin-Kernighan algorithm shows excellent potential for Numba JIT optimization, with clear bottlenecks in numeric operations and tight loops. The proposed approach offers a systematic path to significant performance improvements while maintaining code correctness and maintainability.

The investment in Numba optimization will particularly benefit:
- Large problem instances (n>100)
- Longer optimization runs
- Research and benchmark scenarios
- Real-time applications requiring fast TSP solving
