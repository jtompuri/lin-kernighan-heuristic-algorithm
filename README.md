# Lin-Kernighan Heuristic for the Traveling Salesperson Problem (TSP)

The script `lin_kernighan_tsp_solver.py`implements the Lin-Kernighan (LK) heuristic, a 
powerful local search algorithm for finding high-quality approximate solutions to the TSP. 
The implementation is based on the descriptions and algorithms presented in _The Traveling 
Salesman Problem: A Computational Study_"_ by Applegate, Bixby, Chvátal & Cook [^1] and 
"An Effective Heuristic Algorithm for the Traveling-Salesman Problem" by Lin & Kernighan [^2].

The script processes TSP instances from the TSPLIB format. It computes heuristic solutions
using a chained version of the LK algorithm. If a corresponding optimal tour file
(e.g., problem_name.opt.tour) is found, the script compares the heuristic solution
against the known optimal solution and calculates the percentage gap. If no optimal
tour file is available, the instance is still processed, but no gap calculation is
performed for it. The script displays a summary table and plots of the tours.

Usage:
  1. Ensure all dependencies are installed:
     pip install numpy matplotlib scipy

  2. Place your TSPLIB .tsp files in a designated folder.
     Optionally, place corresponding .opt.tour files (if available) in the same
       folder.

  3. Update the `TSP_FOLDER_PATH` constant at the top of this script
     (in the "--- Constants ---" section) to point to your TSPLIB folder.

  4. Run the script from the command line:
     python lin_kernighan_tsp_solver.py

The script will then process each EUC_2D TSP instance found. It prints progress
and results to the console. For instances with an optimal tour, the gap is shown.
For instances without an optimal tour, nothing is displayed for optimal length and gap.
Finally, a plot of all processed tours is displayed (showing both optimal and heuristic
tours if the optimal is available, otherwise just the heuristic tour). Configuration
parameters for the LK algorithm can be adjusted in the `LK_CONFIG` dictionary
within this script.

## Course documentation (in Finnish)

- [Määrittelydokumentti](/documentation/requirements_specification.md)
- [Viikkoraportti 1](/reports/weekly_report_1.md)
- [Viikkoraportti 2](/reports/weekly_report_2.md)

[^1]: Applegate, David L. & Bixby, Robert E. & Chvtal,  Vaek & Cook, William J. (2006): *The Traveling Salesman Problem : A Computational Study*, Princeton University Press.

[^2]: Lin, Shen & Kernighan, Brian W. (1973): ”An Effective Heuristic Algorithm for the Traveling-Salesman Problem”, Operations Research, Vol. 21, No. 2, s. 498–516.
