# Lin-Kernighan Heuristic for the Traveling Salesperson Problem (TSP)

The Python module `lin_kernighan_tsp_solver` implements the Lin-Kernighan (LK) heuristic, a
powerful local search algorithm for finding high-quality approximate solutions to the traveling 
salesperson problem. The implementation is based on the descriptions and algorithms presented in 
_The Traveling Salesman Problem: A Computational Study_ by Applegate, Bixby, Chvatál & Cook [^1] 
and "An Effective Heuristic Algorithm for the Traveling-Salesman Problem" by Lin & Kernighan [^2].

The module processes TSP instances from the TSPLIB format. The module supports only fully
connected networks with Euclidean 2D geometry. It computes heuristic solutions
using a chained version of the LK algorithm with configurable starting cycle algorithms.
If a corresponding optimal tour file is found, the module compares the heuristic solution
against the known optimal solution and calculates the percentage gap. If no optimal
tour file is available, the instance is still processed, but no gap calculation is
performed for it. The module displays a summary table and plots of the tours.

This project is expected to work with Python 3.12 or newer due to its use of modern Python features like f-strings, pathlib, and extensive type hinting.

## Installation

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    ```

2.  Activate virtual environment:

    Linux/Mac:
    ```bash
    source venv/bin/activate
    ```

    Windows:
    ```bash
    venv\Scripts\activate
    ```

3.  Ensure all dependencies are installed for production environment:
    ```bash
    pip install -r requirements.txt
    ```

4.  Optionally install dependencies for development environment:
    ```bash
    pip install -r requirements-dev.txt
    ```

5.  Linux may require installing `tkinter` for drawing `matplotlib` graphs in an OS window:
    Debian/Ubuntu:
    ```bash
    sudo apt install python3-tk
    ```

6.  Use default TSPLIB files or place your own TSPLIB `.tsp` files in a designated folder.
    Optionally, place corresponding `.opt.tour` files (if available) in the same
    folder. The default folder is `verifications/tsplib95/` relative to the project root.

7.  If you are using a different folder for TSP instances, update the `TSP_FOLDER_PATH`
    constant at the top of the `lin_kernighan_tsp_solver/config.py`.

8.  The solver will create a `solutions/` folder in the project root to save heuristic tours
    (when tour saving is enabled, which is the default).

## Usage

The solver can be used in several ways depending on your needs:

### Process All TSP Files (Default Behavior)

Process all TSP files in the configured folder with default settings:
```bash
python -m lin_kernighan_tsp_solver
```

### Process Specific Files

Process one or more specific TSP files:
```bash
# Single file (automatically uses sequential processing)
python -m lin_kernighan_tsp_solver path/to/file.tsp

# Multiple files (uses parallel processing by default)
python -m lin_kernighan_tsp_solver file1.tsp file2.tsp file3.tsp

# Force sequential processing for multiple files
python -m lin_kernighan_tsp_solver --sequential file1.tsp file2.tsp
```

### Command Line Options

```bash
# Sequential processing (one instance at a time)
python -m lin_kernighan_tsp_solver --sequential

# Parallel processing with specific number of workers
python -m lin_kernighan_tsp_solver --workers 4

# Set time limit per instance (in seconds)
python -m lin_kernighan_tsp_solver --time-limit 20.0

# Choose starting cycle algorithm
python -m lin_kernighan_tsp_solver --starting-cycle nearest_neighbor

# Save heuristic tours to solutions/ folder
python -m lin_kernighan_tsp_solver --save-tours

# Disable tour saving (overrides config default)
python -m lin_kernighan_tsp_solver --no-save-tours

# Combined options with specific files
python -m lin_kernighan_tsp_solver --starting-cycle greedy --time-limit 10.0 --save-tours file1.tsp file2.tsp

# Get help on available options
python -m lin_kernighan_tsp_solver --help
```

**Available command line options:**
- `files`: Positional arguments for specific TSP files to process (optional)
- `--sequential`: Use sequential processing instead of parallel
- `--workers N`: Number of parallel workers (default: all CPU cores)
- `--time-limit T`: Time limit per instance in seconds (overrides `config.py` setting)
- `--starting-cycle METHOD`: Starting cycle algorithm (see Starting Cycle Algorithms section)
- `--save-tours`: Save heuristic tours to `solutions/` folder in TSPLIB format
- `--no-save-tours`: Do not save heuristic tours (overrides config default)
- `--help`: Show help message with all available options

### Starting Cycle Algorithms

The solver supports multiple starting cycle algorithms that generate the initial tour for the Lin-Kernighan heuristic:

- **`natural`** (fastest): Uses the natural order [0, 1, 2, ..., n-1]. Fastest for large instances.
- **`random`**: Random permutation of cities. Good for diversity in multiple runs.
- **`nearest_neighbor`**: Greedy nearest neighbor heuristic. Often produces good initial tours.
- **`greedy`**: Greedy edge selection (shortest edges first). Can produce very good initial tours but is slower for large instances.
- **`boruvka`**: Borůvka's minimum spanning tree algorithm with 2-opt improvement. Balanced performance.
- **`qboruvka`** (default): Quick Borůvka - Concorde's default method. Good balance of quality and speed.

**Performance characteristics:**
- **Speed**: `natural` > `random` > `nearest_neighbor` > `qboruvka` > `boruvka` > `greedy`
- **Quality**: Often `greedy` ≥ `qboruvka` ≥ `boruvka` ≥ `nearest_neighbor` > `random` > `natural`

Choose based on your needs:
- For maximum speed: `--starting-cycle natural`
- For best quality initial tours: `--starting-cycle greedy`
- For balanced performance (default): `--starting-cycle qboruvka`

### File Discovery

When processing specific files, the solver automatically looks for optimal tour files in this order:
1. `filename.opt.tour` (TSPLIB standard)
2. `filename.opt` (alternative format)
3. `filename.tour` (simple format)

If no optimal tour file is found, the instance is still processed but no gap calculation is performed.

The module will process each EUC_2D TSP instance found in the specified folder or from the provided file list. By default, it uses parallel processing to handle multiple instances simultaneously for better performance. When processing a single file, sequential processing is automatically used for optimal performance.

Progress and results are printed to the console. The default time limit for each problem is set to 5 seconds in `config.py`, but can be overridden using the `--time-limit` option.

For instances with an optimal tour, the gap percentage is calculated and displayed. For instances without an optimal tour, the gap column shows "N/A". Finally, a plot of all processed tours is displayed (showing both optimal and heuristic tours if the optimal is available, otherwise just the heuristic tour).

Configuration parameters for the LK algorithm can be adjusted in the `LK_CONFIG` dictionary in `config.py`. Starting cycle algorithm preferences can be set in the `STARTING_CYCLE_CONFIG` dictionary.

### Saving Heuristic Tours

The solver can optionally save the computed heuristic tours to files in TSPLIB format:

- **Default behavior**: Tours are saved by default (configurable in `config.py` via `LK_CONFIG["SAVE_TOURS"]`)
- **Output location**: Tours are saved to the `solutions/` folder in the project root
- **File format**: Standard TSPLIB `.tour` format with `.heu.tour` extension
- **File naming**: `{problem_name}.heu.tour` (e.g., `berlin52.heu.tour`)
- **Content**: Includes tour metadata (name, length, dimension) and the complete tour sequence

**Control tour saving:**
```bash
# Enable tour saving (if disabled in config)
python -m lin_kernighan_tsp_solver --save-tours

# Disable tour saving (if enabled in config)
python -m lin_kernighan_tsp_solver --no-save-tours
```

**Tour file format example:**
```
NAME: berlin52.heu.tour
TYPE: TOUR
COMMENT: Heuristic tour (Lin-Kernighan), length 7544.37
DIMENSION: 52
TOUR_SECTION
1
22
...
-1
EOF
```

The saved tour files are compatible with TSPLIB tools and can be used for further analysis or comparison with other TSP solvers.

## Performance Notes

- **Parallel processing** (default): Processes multiple TSP instances simultaneously using all available CPU cores. Recommended for processing multiple instances.
- **Sequential processing** (`--sequential` or single file): Processes instances one at a time. Automatically used for single files and useful for debugging or when system resources are limited.
- **Custom worker count** (`--workers N`): Limits parallel processing to N workers. Useful for controlling resource usage.
- **Starting cycle performance**: Different starting cycle algorithms have varying speed/quality trade-offs (see Starting Cycle Algorithms section above).
- **Time limits**: For expensive starting cycle algorithms (greedy, qboruvka), time limits and fallbacks are implemented to prevent excessive computation times on large instances.

## Testing

The project includes comprehensive test coverage:

```bash
# Run all tests
python -m pytest

# Run tests with coverage report
python -m pytest --cov=lin_kernighan_tsp_solver --cov-report=html

# Run specific test modules
python -m pytest tests/test_starting_cycles.py -v
python -m pytest tests/test_workflow.py -v
python -m pytest tests/test_cli_files.py -v

# Run performance comparison tests
python -m pytest test_starting_performance.py -v
```

**Test modules:**
- `test_starting_cycles.py`: Tests for all starting cycle algorithms
- `test_workflow.py`: Integration and workflow tests
- `test_cli_files.py`: Tests for file-specific CLI functionality
- `test_lk_*.py`: Tests for Lin-Kernighan algorithm components
- `test_starting_performance.py`: Performance comparison tests

The test suite includes unit tests, integration tests, CLI tests, and performance benchmarks with nearly 100% code coverage.

## Helper Algorithms

Create and solve simple (4-12 nodes) TSP problems with Exact TSP Solver:
```bash
cd exact_tsp_solver
python exact_tsp_solver.py
```

Create random TSP problems:
```bash
cd problems
python create_tsp_problem.py 20 my_tsp20.tsp
python create_tsp_problem.py 50 random50.tsp --max_coord 500 --name Random50
```

Solve TSP problems with Simple TSP Solver:
```bash
cd simple_tsp_solver
python simple_tsp_solver.py
```

## Example output using TSPLIB95

Here is an example output using TSPLIB95 instances with optimal tours.
**Note:** The example output below was generated with `--time-limit 20.0` for illustrative purposes. The default time limit is 5.0 seconds, which may yield different heuristic lengths and runtimes.

![Example output plots](/images/lin-kernighan-example-output-20s-parallel.png)

```
Found 18 TSP instances.
Processing using 12 parallel workers...
[1/18] Completed: berlin52
[2/18] Completed: kroC100
[3/18] Completed: kroA100
...
[18/18] Completed: tsp225

Configuration parameters:
  MAX_LEVEL   = 12
  BREADTH     = [5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  BREADTH_A   = 5
  BREADTH_B   = 5
  BREADTH_D   = 1
  TIME_LIMIT  = 20.00
  STARTING_CYCLE = qboruvka

Instance     OptLen   HeuLen   Gap(%)  Time(s)
----------------------------------------------
a280        2586.77  2614.62     1.08    20.00
berlin52    7544.37  7544.37     0.00     0.45
ch130       6110.86  6134.56     0.39    20.00
ch150       6532.28  6552.30     0.31    20.00
eil101       642.31   640.21     0.00    20.00
eil51        429.98   428.98     0.00    20.00
eil76        545.39   544.37     0.00    20.00
kroA100    21285.44 21285.44     0.00     4.07
kroC100    20750.76 20750.76     0.00     0.62
kroD100    21294.29 21294.29     0.00    10.61
lin105     14383.00 14383.00     0.00    19.62
pcb442     50783.55 52109.68     2.61    20.00
pr1002     259066.66 323584.12    24.90    20.00
pr2392     378062.83 506459.61    33.96    20.00
pr76       108159.44 108159.44     0.00    16.81
rd100       7910.40  7910.40     0.00     1.38
st70         678.60   677.11     0.00    20.00
tsp225      3859.00  3865.77     0.18    20.00
----------------------------------------------
SUMMARY    910625.92 1104939.02     3.52    15.20
```

### Example: Processing Specific Files

```bash
# Process a single file with natural starting cycle
$ python -m lin_kernighan_tsp_solver --starting-cycle natural my_problem.tsp
Single file specified, using sequential processing.
Found 1 TSP instances.
Processing sequentially...
Processing my_problem (EUC_2D)...
  Optimal tour not available for my_problem.
  Heuristic length: 4162.62  Time: 5.00s

# Process multiple files with different starting cycle
$ python -m lin_kernighan_tsp_solver --starting-cycle greedy --time-limit 2.0 berlin52.tsp eil51.tsp
Found 2 TSP instances.
Processing using 2 parallel workers...
[1/2] Completed: berlin52
[2/2] Completed: eil51

Configuration parameters:
  STARTING_CYCLE = greedy
  TIME_LIMIT = 2.00

Instance     OptLen   HeuLen   Gap(%)  Time(s)
----------------------------------------------
berlin52    7544.37  7544.37     0.00     0.15
eil51        429.98   429.12     0.00     2.00
----------------------------------------------
SUMMARY     7974.35  7973.48     0.00     1.08
```

## Course documentation (in Finnish)

-   [Määrittelydokumentti](/documentation/requirements_specification.md)
-   [Toteutusdokumentti](/documentation/implementation_specification.md)
-   [Testausdokumentti](/documentation/testing_specification.md)
-   [Starting Cycle Algorithms Documentation](/documentation/starting_cycles.md)
-   [Viikkoraportti 1](/reports/weekly_report_1.md)
-   [Viikkoraportti 2](/reports/weekly_report_2.md)
-   [Viikkoraportti 3](/reports/weekly_report_3.md)
-   [Viikkoraportti 4](/reports/weekly_report_4.md)
-   [Viikkoraportti 5](/reports/weekly_report_5.md)
-   [Viikkoraportti 6](/reports/weekly_report_6.md)
-   [Viikkoraportti 7](/reports/weekly_report_7.md)

[^1]: Applegate, David L. & Bixby, Robert E. & Chvatál, Vašek & Cook, William J. (2006): _The Traveling Salesman Problem : A Computational Study_, Princeton University Press.

[^2]: Lin, Shen & Kernighan, Brian W. (1973): "An Effective Heuristic Algorithm for the Traveling-Salesman Problem", Operations Research, Vol. 21, No. 2, s. 498–516.
