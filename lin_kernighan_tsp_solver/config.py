"""
Configuration constants for the Lin-Kernighan TSP solver.

This module centralizes all configuration parameters and constants used
throughout the project, such as algorithm settings, tolerances, and paths.
"""

from pathlib import Path

# Path to the folder containing TSPLIB .tsp files and optional .opt.tour files
TSP_FOLDER_PATH = (
    Path(__file__).resolve().parent.parent / "verifications" / "tsplib95"
)

# Tolerance for floating point comparisons
FLOAT_COMPARISON_TOLERANCE = 1e-12

# Maximum number of subplots in the tour visualization
MAX_SUBPLOTS_IN_PLOT = 25

# Lin-Kernighan algorithm configuration parameters
LK_CONFIG = {
    "MAX_LEVEL": 12,  # Max recursion depth for k-opt moves in step()
    "BREADTH": [5, 5] + [1] * 20,  # Search breadth at each level in step()
    "BREADTH_A": 5,  # Search breadth for y1 in alternate_step()
    "BREADTH_B": 5,  # Search breadth for y2 in alternate_step()
    "BREADTH_D": 1,  # Search breadth for y4 in alternate_step()
    "TIME_LIMIT": 5.0,  # Time limit for LK search (seconds)
}
