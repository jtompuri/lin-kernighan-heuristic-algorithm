# Lin-Kernighan Heuristic for the Traveling Salesperson Problem (TSP)

The Python module `lin_kernighan_tsp_solver` implements the Lin-Kernighan (LK) heuristic, a
powerful local search algorithm for finding high-quality approximate solutions to the TSP.
The implementation is based on the descriptions and algorithms presented in _The Traveling
Salesman Problem: A Computational Study_ by Applegate, Bixby, Chvátal & Cook [^1] and
"An Effective Heuristic Algorithm for the Traveling-Salesman Problem" by Lin & Kernighan [^2].

The module processes TSP instances from the TSPLIB format. The module supports only fully
connected networks with Euclidian 2D geometry. It computes heuristic solutions
using a chained version of the LK algorithm. If a corresponding optimal tour file
`.opt.tour` is found, the module compares the heuristic solution
against the known optimal solution and calculates the percentage gap. If no optimal
tour file is available, the instance is still processed, but no gap calculation is
performed for it. The module displays a summary table and plots of the tours.

This project is expected to work with Python 3.7 or newer due to its use of modern Python features like f-strings, pathlib, and extensive type hinting.

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

5.  Use default TSPLIB files or place your own TSPLIB `.tsp` files in a designated folder.
    Optionally, place corresponding `.opt.tour` files (if available) in the same
    folder. The default folder is `verifications/tsplib95/` relative to the project root.

6.  If you are using a different folder for TSP instances, update the `TSP_FOLDER_PATH`
    constant at the top of the `lin_kernighan_tsp_solver/config.py`.

## Usage

**Basic usage (default parallel processing):**
```bash
python -m lin_kernighan_tsp_solver
```

**Command line options:**
```bash
# Sequential processing (one instance at a time)
python -m lin_kernighan_tsp_solver --sequential

# Parallel processing with specific number of workers
python -m lin_kernighan_tsp_solver --workers 4

# Set time limit per instance (in seconds)
python -m lin_kernighan_tsp_solver --time-limit 60.0

# Combined options
python -m lin_kernighan_tsp_solver --sequential --time-limit 30.0

# Get help on available options
python -m lin_kernighan_tsp_solver --help
```

**Available command line options:**
- `--sequential`: Use sequential processing instead of parallel (useful for debugging)
- `--workers N`: Number of parallel workers (default: all CPU cores)
- `--time-limit T`: Time limit per instance in seconds (overrides config.py setting)
- `--help`: Show help message with all available options

The module will process each EUC_2D TSP instance found in the specified folder. By default, 
it uses parallel processing to handle multiple instances simultaneously for better performance.
Progress and results are printed to the console. The default time limit for each problem is 
set to 5 seconds in `config.py`, but can be overridden using the `--time-limit` option.

For instances with an optimal tour, the gap percentage is calculated and displayed. For 
instances without an optimal tour, the gap column shows nothing. Finally, a plot of all 
processed tours is displayed (showing both optimal and heuristic tours if the optimal is 
available, otherwise just the heuristic tour).

Configuration parameters for the LK algorithm can be adjusted in the `LK_CONFIG` dictionary 
in `config.py`.

## Performance Notes

- **Parallel processing** (default): Processes multiple TSP instances simultaneously using all available CPU cores. Recommended for processing multiple instances.
- **Sequential processing** (`--sequential`): Processes instances one at a time. Useful for debugging or when system resources are limited.
- **Custom worker count** (`--workers N`): Limits parallel processing to N workers. Useful for controlling resource usage.

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

Here is an example output using TSPLIB95 instances.
**Note:** The example output below was generated with `--time-limit 20.0` for illustrative purposes. The default time limit is 5.0 seconds, which may yield different heuristic lengths and runtimes.

![Example output plots](/images/lk_verifications_tsplib95_20s_2025-06-30.png)

```
Found 18 TSP instances.
Processing using 8 parallel workers...
[1/18] Completed: a280
[2/18] Completed: berlin52
[3/18] Completed: ch130
...
[18/18] Completed: tsp225

Configuration parameters:
  MAX_LEVEL   = 12
  BREADTH     = [5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  BREADTH_A   = 5
  BREADTH_B   = 5
  BREADTH_D   = 1
  TIME_LIMIT  = 20.00

Instance     OptLen   HeuLen   Gap(%)  Time(s)
----------------------------------------------
a280        2586.77  2617.51     1.19    20.00
berlin52    7544.37  7544.37     0.00     0.13
ch130       6110.86  6152.99     0.69    20.00
ch150       6532.28  6566.55     0.52    20.00
eil101       642.31   640.21     0.00    20.00
eil51        429.98   428.87     0.00    20.00
eil76        545.39   544.37     0.00    20.00
kroA100    21285.44 21285.44     0.00     5.90
kroC100    20750.76 20750.76     0.00     4.57
kroD100    21294.29 21294.29     0.00     8.72
lin105     14383.00 14383.00     0.00     1.85
pcb442     50783.55 59765.25    17.69    20.00
pr1002     259066.66 336815.79  30.01    20.00
pr2392     378062.83 378062.83   0.00     8.97
pr76       108159.44 108159.44   0.00     3.68
rd100       7910.40  7910.40     0.00     1.66
st70         678.60   677.11     0.00    20.00
tsp225      3859.00  3910.96     1.35    20.00
----------------------------------------------
SUMMARY    910625.92 997510.13     2.86  13.08
```

## Course documentation (in Finnish)

-   [Määrittelydokumentti](/documentation/requirements_specification.md)
-   [Toteutusdokumentti](/documentation/implementation_specification.md)
-   [Testausdokumentti](/documentation/testing_specification.md)
-   [Viikkoraportti 1](/reports/weekly_report_1.md)
-   [Viikkoraportti 2](/reports/weekly_report_2.md)
-   [Viikkoraportti 3](/reports/weekly_report_3.md)
-   [Viikkoraportti 4](/reports/weekly_report_4.md)
-   [Viikkoraportti 5](/reports/weekly_report_5.md)
-   [Viikkoraportti 6](/reports/weekly_report_6.md)
-   [Viikkoraportti 7](/reports/weekly_report_7.md)

[^1]: Applegate, David L. & Bixby, Robert E. & Chvtal, Vaek & Cook, William J. (2006): _The Traveling Salesman Problem : A Computational Study_, Princeton University Press.

[^2]: Lin, Shen & Kernighan, Brian W. (1973): "An Effective Heuristic Algorithm for the Traveling-Salesman Problem", Operations Research, Vol. 21, No. 2, s. 498–516.
