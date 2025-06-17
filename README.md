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

## Usage 

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate virtual environment:

Linux/Mac:
```bash
source venv/bin/activate
```

Windows:
```bash
venv\Scripts\activate
```

3. Ensure all dependencies are installed for production environment:
   ```bash
   pip install -r requirements.txt
   ```

4. Optionally install dependencies for development environment:
   ```bash
   pip install -r requirements-dev.txt
   ```

5. Use default TSPLIB files or place your own TSPLIB `.tsp` files in a designated folder.
   Optionally, place corresponding `.opt.tour` files (if available) in the same
   folder. The default folder is `verifications/tsplib95/` relative to the project root.

6. If you are using a different folder for TSP instances, update the `TSP_FOLDER_PATH` 
   constant at the top of the `lin_kernighan_tsp_solver/config.py` module.

7. Run the Python module from the project's root directory:
   ```bash
   python -m lin_kernighan_tsp_solver
   ```

The module will then process each EUC_2D TSP instance found in the specified folder. It 
prints progress and results to the console. For instances with an optimal tour, the gap is 
shown. For instances without an optimal tour, nothing is displayed for optimal length and 
gap. Finally, a plot of all processed tours is displayed (showing both optimal and 
heuristic tours if the optimal is available, otherwise just the heuristic tour). Configuration 
parameters for the LK algorithm can be adjusted in the `LK_CONFIG` dictionary
in `config.py`.

## Helper Algorithms

Create and solve simple (4-12 nodes) TSP problems with Exact TSP Solver:
```bash
cd exact_tsp_solver
python exact_tsp_solver
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
python simple_tsp_solver
```

## Example output using TSPLIB95 

Here is the example output using defaults: 

![Example output plots](/images/lk_verifications_tsplib95_20s.png)

```
Configuration parameters:
  MAX_LEVEL   = 12
  BREADTH     = [5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  BREADTH_A   = 5
  BREADTH_B   = 5
  BREADTH_D   = 1
  TIME_LIMIT  = 20.00

Instance     OptLen   HeuLen   Gap(%)  Time(s)
----------------------------------------------
a280        2586.77  2617.49     1.19    20.00
berlin52    7544.37  7544.37     0.00     0.19
ch130       6110.86  6164.24     0.87    20.00
ch150       6532.28  6552.30     0.31    20.00
eil101       642.31   640.21     0.00    20.00
eil51        429.98   428.87     0.00    20.00
eil76        545.39   544.37     0.00    20.00
kroA100    21285.44 21285.44     0.00    11.69
kroC100    20750.76 20750.76     0.00     2.38
kroD100    21294.29 21294.29     0.00    11.22
lin105     14383.00 14383.00     0.00     2.70
pcb442     50783.55 74322.54    46.35    20.00
pr1002     259066.66 339278.81    30.96    20.00
pr2392     378062.83 378062.83     0.00    12.97
pr76       108159.44 108159.44     0.00     2.04
rd100       7910.40  7910.40     0.00     2.35
st70         678.60   677.11     0.00    20.00
tsp225      3859.00  3936.55     2.01    20.00
----------------------------------------------
SUMMARY    910625.92 1014553.02     4.54    13.64
```


## Course documentation (in Finnish)

- [Määrittelydokumentti](/documentation/requirements_specification.md)
- [Toteutusdokumentti](/documentation/implementation_specification.md)
- [Testausdokumentti](/documentation/testing_specification.md)
- [Viikkoraportti 1](/reports/weekly_report_1.md)
- [Viikkoraportti 2](/reports/weekly_report_2.md)
- [Viikkoraportti 3](/reports/weekly_report_3.md)
- [Viikkoraportti 4](/reports/weekly_report_4.md)
- [Viikkoraportti 5](/reports/weekly_report_5.md)

[^1]: Applegate, David L. & Bixby, Robert E. & Chvtal,  Vaek & Cook, William J. (2006): *The Traveling Salesman Problem : A Computational Study*, Princeton University Press.

[^2]: Lin, Shen & Kernighan, Brian W. (1973): ”An Effective Heuristic Algorithm for the Traveling-Salesman Problem”, Operations Research, Vol. 21, No. 2, s. 498–516.
