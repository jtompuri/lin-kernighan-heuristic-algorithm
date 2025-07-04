# Lin-Kernighan Heuristic for the Traveling Salesperson Problem (TSP)

A Python implementatio# Use different starting algorithm
python -m lin_kernighan_tsp_solver --starting-cycle greedy

# Set time limit and save tours
python -m lin_kernighan_tsp_solver --time-limit 10.0 --save-tours

# Enable performance optimizations (optional)
python -m lin_kernighan_tsp_solver --enable-numba Lin-Kernighan (LK) heuristic for solving the Traveling Salesperson Problem. This solver processes TSPLIB format instances with Euclidean 2D geometry and provides high-quality approximate solutions using a chained version of the LK algorithm.

**Key Features:**
- Complete implementation of the Lin-Kernighan local search heuristic
- Multiple starting cycle algorithms (natural, random, nearest neighbor, greedy, Borůvka, quick-Borůvka)
- Parallel and sequential processing modes for multiple TSP instances
- Automatic gap calculation against optimal tours (when available)
- Tour visualization and saving in TSPLIB format
- Comprehensive test coverage and helper tools
- Optional performance optimizations with automatic fallback

**Requirements:** Python 3.8+ (modern Python features with extensive type hinting)  
**Dependencies:** NumPy, SciPy, Matplotlib (see requirements.txt for details)

## Table of Contents
- [Quick Start](#quick-start)
- [Requirements & Compatibility](#requirements--compatibility)
- [Usage](#usage)
- [Example Output](#example-output)
- [Development](#development)
- [Configuration](#configuration)
- [References & Course Documentation](#references--course-documentation)

## Quick Start

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Verify installation (optional)
python test_requirements.py

# 2. Verify installation (optional)
python test_requirements.py

# 3. Run with default settings (processes all TSPLIB95 instances)
python -m lin_kernighan_tsp_solver

# 4. Process specific files
python -m lin_kernighan_tsp_solver problems/tsplib95/berlin52.tsp

# 5. Customize settings
python -m lin_kernighan_tsp_solver --time-limit 10.0 --starting-cycle greedy
```

## Requirements & Compatibility

### Core Dependencies
- **Python**: 3.8+ (tested with 3.12)
- **NumPy**: 2.2.x+ (stable and well-tested)
- **SciPy**: 1.16.0+
- **Matplotlib**: 3.10.3+

### Performance Optimization
- **Numba**: 0.58.0+ (optional automatic acceleration)

### Installation Notes
```bash
# Standard installation
pip install -r requirements.txt

# Development installation (includes testing/linting tools)
pip install -r requirements-dev.txt

# Verify installation
python test_requirements.py
```

### Compatibility Matrix
| Component | Status | Notes |
|-----------|--------|-------|
| Python 3.8-3.12 | ✅ Tested | Recommended: 3.10+ |
| NumPy 2.2.x+ | ✅ Supported | Stable and reliable |
| All platforms | ✅ Cross-platform | Windows, macOS, Linux |
```

## Usage

### Basic Commands

```bash
# Process all instances in problems/tsplib95/ (default)
python -m lin_kernighan_tsp_solver

# Process specific files
python -m lin_kernighan_tsp_solver file1.tsp file2.tsp

# Use different starting algorithm with Numba optimizations
python -m lin_kernighan_tsp_solver --starting-cycle greedy --enable-numba

# Set time limit and save tours with Numba auto-detection
python -m lin_kernighan_tsp_solver --time-limit 20.0 --save-tours

# Enable performance optimizations (optional)
python -m lin_kernighan_tsp_solver --enable-numba

# Use pure Python implementation
python -m lin_kernighan_tsp_solver --disable-numba
```

### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--starting-cycle` | Initial tour algorithm: `natural`, `random`, `nearest_neighbor`, `greedy`, `boruvka`, `qboruvka` | `qboruvka` |
| `--time-limit` | Time limit per instance (seconds) | `5.0` |
| `--workers` | Number of parallel workers | All CPU cores |
| `--sequential` | Force sequential processing | Parallel |
| `--save-tours` / `--no-save-tours` | Save/don't save heuristic tours | Save |
| `--seed` | Random seed for reproducible results | Random |

#### Performance Options (Optional)
| Option | Description | Default |
|--------|-------------|---------|
| `--enable-numba` / `--disable-numba` | Control performance optimizations | Auto-detect |
| `--numba-threshold` | Minimum nodes for auto-optimization | `30` |
| `--parallel-threshold` | Minimum nodes for parallel processing | `500` |

### Starting Cycle Algorithms

- **`qboruvka`** (default): Best balance of speed and quality
- **`greedy`**: Highest quality, slower for large instances  
- **`natural`**: Fastest, lowest quality
- **`nearest_neighbor`**: Good compromise
- **`boruvka`**, **`random`**: Alternative options

**Performance:** `natural` > `random` > `nearest_neighbor` > `qboruvka` > `boruvka` > `greedy`  
**Quality:** Often `greedy` ≥ `qboruvka` ≥ `boruvka` ≥ `nearest_neighbor` > `random` > `natural`

### Performance Notes

The solver includes automatic performance optimizations when beneficial:

- **Pure Python**: Reliable implementation works on all systems
- **Optional Acceleration**: Automatic performance enhancement for larger problems when available
- **Scalable Processing**: Efficient handling from small test cases to large instances

```bash
# Standard usage (automatic optimization)
python -m lin_kernighan_tsp_solver

# Force pure Python mode if needed
python -m lin_kernighan_tsp_solver --disable-numba
```

## Example Output

![Example output plots](/images/lin-kernighan-example-output-20s-parallel.png)

```
Found 18 TSP instances.
Processing using 12 parallel workers...

Configuration parameters:
  TIME_LIMIT  = 20.00
  STARTING_CYCLE = qboruvka

Instance     OptLen   HeuLen   Gap(%)  Time(s)
----------------------------------------------
berlin52    7544.37  7544.37     0.00     0.45
eil51        429.98   428.98     0.00    20.00
kroA100    21285.44 21285.44     0.00     4.07
...
----------------------------------------------
SUMMARY    910625.92 1104939.02     3.52    15.20
```

## Development

### Installation
```bash
pip install -r requirements.txt                # Includes Numba for JIT optimizations
pip install -r requirements-dev.txt          # Development: includes testing, linting tools
```

### Testing
```bash
python -m pytest                              # Run all tests
python -m pytest --cov=lin_kernighan_tsp_solver --cov-report=html  # With coverage
```

### Helper Tools
```bash
# Create random TSP instances with defaults
python helpers/create_tsp_problem.py 20                           # Creates rand20.tsp in problems/random/
python helpers/create_tsp_problem.py 15 --name custom --seed 123  # Custom name with seed

# Exact brute-force solver for small instances  
python helpers/exact_tsp_solver.py                                # Process problems/random/*.tsp
python helpers/exact_tsp_solver.py --input-dir problems/custom    # Custom input directory

# Simple k-opt TSP solver with time limits
python helpers/simple_tsp_solver.py                               # Process problems/tsplib95/*.tsp
python helpers/simple_tsp_solver.py --save-tours --plot           # Save tours and show plots
python helpers/simple_tsp_solver.py --input-dir problems/random --time-limit 10  # Custom config
```

## Configuration

- **TSP files**: Place `.tsp` files in `problems/tsplib95/` (default) or update `TSP_FOLDER_PATH` in `lin_kernighan_tsp_solver/config.py`
- **Optimal tours**: Place `.opt.tour` files alongside `.tsp` files for gap calculation
- **Output folders**: 
  - Lin-Kernighan heuristic tours: `solutions/lin_kernighan/` (when `--save-tours` is enabled)
  - Helper tools: `solutions/simple/` for simple TSP solver output
- **Input folders**: `problems/tsplib95/`, `problems/random/`, `problems/custom/` for organized problem storage
- **Algorithm parameters**: Modify `LK_CONFIG` and `STARTING_CYCLE_CONFIG` in `config.py`

## References & Course Documentation

**Algorithm based on:**
- Applegate, D.L., Bixby, R.E., Chvatál, V. & Cook, W.J. (2006): _The Traveling Salesman Problem: A Computational Study_, Princeton University Press
- Lin, S. & Kernighan, B.W. (1973): "An Effective Heuristic Algorithm for the Traveling-Salesman Problem", Operations Research, Vol. 21, No. 2, pp. 498–516

**Course documentation (Finnish):**
[Määrittelydokumentti](/documentation/requirements_specification.md) | [Toteutusdokumentti](/documentation/implementation_specification.md) | [Testausdokumentti](/documentation/testing_specification.md)

**Weekly reports:** [Week 1](/documentation/reports/weekly_report_1.md) | [Week 2](/documentation/reports/weekly_report_2.md) | [Week 3](/documentation/reports/weekly_report_3.md) | [Week 4](/documentation/reports/weekly_report_4.md) | [Week 5](/documentation/reports/weekly_report_5.md) | [Week 6](/documentation/reports/weekly_report_6.md) | [Week 7](/documentation/reports/weekly_report_7.md) | [Week 8](/documentation/reports/weekly_report_8.md)
