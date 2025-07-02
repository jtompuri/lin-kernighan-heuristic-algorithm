# Starting Cycle Algorithms

The Lin-Kernighan TSP solver now supports multiple starting cycle algorithms ## Performance Considerations

The starting cycle algorithms are designed to be fast relative to the Lin-Kernighan optimization:

- **Natural**: O(n) - **Fastest possible**, original behavior
- **Random**: O(n) - Fast, ~5-15x slower than natural due to shuffling overhead
- **Nearest Neighbor**: O(n²) - Moderate for most instances
- **Greedy**: O(n² log n) - Moderate, falls back to nearest neighbor for n > 1000
- **Boruvka**: O(n² log n) - MST construction
- **QBoruvka**: O(n² log n) + 2-opt refinement with time limits

The QBoruvka method includes time-limited 2-opt improvement to prevent excessive startup time on large instances. The greedy method automatically falls back to nearest neighbor for very large instances (> 1000 nodes).

**Performance Tip**: Use `natural` for maximum startup speed when you need the fastest possible initialization, especially in benchmarking or performance-critical applications. initial tours. This can significantly impact the quality of the final solution.

## Available Methods

### 1. Natural (`natural`)
- Uses natural node order: [0, 1, 2, ..., n-1]
- **Fastest possible** - identical to original behavior before starting cycles were added
- Minimal overhead, best for performance-critical applications
- Good baseline for comparison

### 2. Random (`random`)
- Generates a random permutation of all nodes
- Fast but provides no structure
- Good for testing algorithm robustness and solution diversity

### 3. Nearest Neighbor (`nearest_neighbor`)
- Builds tour by always visiting the nearest unvisited node
- Classic greedy heuristic
- Usually produces reasonable initial solutions
- Configurable starting node
- Often produces good initial solutions
- Can limit number of edges considered for large instances

### 4. Borůvka (`boruvka`)
- Uses Borůvka's MST algorithm followed by DFS traversal
- Creates minimum spanning tree then converts to tour
- Good balance of quality and speed

### 5. QBorůvka (`qboruvka`) - **Default**
- Enhanced Borůvka with 2-opt refinement
- Default method used by Concorde TSP solver
- Applies multiple 2-opt improvement iterations
- Generally produces the best initial tours

## Usage

### Command Line
```bash
# Use nearest neighbor starting cycle
python -m lin_kernighan_tsp_solver --starting-cycle nearest_neighbor

# Use random starting cycle
python -m lin_kernighan_tsp_solver --starting-cycle random

# Use default (qboruvka)
python -m lin_kernighan_tsp_solver
```

### Programmatic
```python
from lin_kernighan_tsp_solver import generate_starting_cycle
import numpy as np

# Example coordinates
coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

# Generate tours with different methods
random_tour = generate_starting_cycle(coords, method="random")
nn_tour = generate_starting_cycle(coords, method="nearest_neighbor") 
greedy_tour = generate_starting_cycle(coords, method="greedy")
boruvka_tour = generate_starting_cycle(coords, method="boruvka")
qboruvka_tour = generate_starting_cycle(coords, method="qboruvka")

# Use default method from config
default_tour = generate_starting_cycle(coords)
```

### Configuration
```python
from lin_kernighan_tsp_solver.config import LK_CONFIG, STARTING_CYCLE_CONFIG

# Change default method
LK_CONFIG["STARTING_CYCLE"] = "nearest_neighbor"

# View available methods
print(STARTING_CYCLE_CONFIG["AVAILABLE_METHODS"])
# ['random', 'nearest_neighbor', 'greedy', 'boruvka', 'qboruvka']

# Configure method-specific parameters
STARTING_CYCLE_CONFIG["NEAREST_NEIGHBOR_START"] = 5  # Start from node 5
STARTING_CYCLE_CONFIG["QBORUVKA_ITERATIONS"] = 5     # More refinement iterations
```

## Method-Specific Parameters

### Nearest Neighbor
- `start_node`: Starting node index (default: 0)

### Greedy
- `max_edges`: Maximum edges to consider (default: None = all edges)

### QBorůvka
- `iterations`: Number of 2-opt refinement iterations (default: 3)

## Performance Characteristics

| Method | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| Random | Fastest | Poor | Testing, baseline |
| Nearest Neighbor | Fast | Good | Quick reasonable solutions |
| Greedy | Medium | Good | Balanced approach |
| Borůvka | Medium | Good | MST-based problems |
| QBorůvka | Slower | Best | Production use, best quality |

## Performance Considerations

The starting cycle algorithms are designed to be fast relative to the Lin-Kernighan optimization:

- **Random**: O(n) - Very fast
- **Nearest Neighbor**: O(n²) - Fast for most instances
- **Greedy**: O(n² log n) - Moderate, falls back to nearest neighbor for n > 1000
- **Borůvka**: O(n² log n) - MST construction
- **QBorůvka**: O(n² log n) + 2-opt refinement with time limits

The QBorůvka method includes time-limited 2-opt improvement to prevent excessive startup time on large instances. The greedy method automatically falls back to nearest neighbor for very large instances (> 1000 nodes).

## Integration with Lin-Kernighan

The starting cycle method is automatically used when processing TSP instances:

1. **Configuration**: Set method via CLI or config
2. **Generation**: Initial tour created using selected method
3. **Optimization**: Lin-Kernighan algorithm improves the initial tour
4. **Chaining**: Multiple LK runs with perturbations (double-bridge kicks)

Better starting tours often lead to:
- Faster convergence to good solutions
- Better final tour quality
- Reduced risk of getting stuck in poor local optima

## Examples

### Comparing Methods on a Small Instance
```python
import numpy as np
from lin_kernighan_tsp_solver import generate_starting_cycle, chained_lin_kernighan

# Create a small TSP instance
coords = np.random.rand(10, 2) * 100

methods = ["random", "nearest_neighbor", "greedy", "qboruvka"]
results = {}

for method in methods:
    initial_tour = generate_starting_cycle(coords, method=method)
    final_tour, final_cost = chained_lin_kernighan(
        coords, initial_tour, time_limit_seconds=5.0
    )
    results[method] = {
        'initial_tour': initial_tour,
        'final_cost': final_cost
    }
    print(f"{method:15}: Final cost = {final_cost:.2f}")
```

### Using Different Starting Nodes for Nearest Neighbor
```python
coords = np.array([[0, 0], [3, 0], [3, 4], [0, 4]])  # Rectangle

for start_node in range(4):
    tour = generate_starting_cycle(
        coords, 
        method="nearest_neighbor", 
        start_node=start_node
    )
    print(f"Start from node {start_node}: {tour}")
```

This flexibility allows users to experiment with different initialization strategies and find the best approach for their specific TSP instances.
