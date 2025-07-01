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

# Tour class configuration constants
TOUR_CONFIG = {
    "MAX_NODE_INDEX": 1_000_000,  # Maximum allowed node index
    "EMPTY_TOUR_COST": 0.0,      # Cost of an empty tour
    "SINGLE_NODE_SEGMENT": 1,     # Length of single-node segment
}

# Resource limits for problem sizes
RESOURCE_LIMITS = {
    "MAX_NODES": 50000,           # Maximum number of nodes
    "MAX_TIME_LIMIT": 7200,       # Maximum time limit (2 hours)
    "MAX_MEMORY_ESTIMATE_MB": 2000  # Maximum estimated memory usage
}

# Delaunay triangulation constants
DELAUNAY_CONFIG = {
    "MIN_POINTS_FOR_TRIANGULATION": 3,  # Minimum points needed for Delaunay
}

# Double bridge perturbation constants
DOUBLE_BRIDGE_CONFIG = {
    "MIN_TOUR_SIZE": 4,           # Minimum tour size for meaningful perturbation
    "NUM_CUT_POINTS": 4,          # Number of cut points for double bridge
    "CUT_RANGE_START": 1,         # Start of range for cut point selection
}

# Error handling and validation constants
VALIDATION_CONFIG = {
    "DEADLINE_SAFETY_MARGIN": 0.1,              # Safety margin for deadline checks (seconds)
    "RELATIVE_TOLERANCE": 1e-7,          # Relative tolerance for optimality check
    "ABSOLUTE_TOLERANCE_MULTIPLIER": 10,  # Multiplier for absolute tolerance
    "MAX_STAGNATION_ITERATIONS": 1000,          # Max iterations without improvement
    "MAX_TOTAL_ITERATIONS": 10000,              # Max total iterations regardless of improvement
    "MAX_STAGNATION_TIME": 30.0,                # Max time without improvement (seconds)
    "LOOP_BREAK_CHECK_INTERVAL": 100,           # Check for breaks every N iterations
    "RECURSIVE_DEPTH_LIMIT": 50,                # Max recursive depth before forced return
}

# Memory calculation constants
MEMORY_CONFIG = {
    "BYTES_PER_FLOAT64": 8,
    "BYTES_TO_MB_DIVISOR": 1024 * 1024,
}

# Coordinate system constants
COORDINATE_CONFIG = {
    "EXPECTED_DIMENSIONS": 2,     # Expected number of dimensions (x, y)
    "COORDINATE_ARRAY_NDIM": 2,   # Expected ndim for coordinate arrays
    "MAX_COORDINATE_VALUE": 1e6,  # Maximum allowed coordinate magnitude
    "LARGE_COORDINATE_THRESHOLD": 1e4,  # Threshold for using stable computation
    "MAX_SQUARED_DISTANCE": 1e12,  # Maximum allowed squared distance
    "MAX_REASONABLE_DISTANCE": 1e8,  # Maximum reasonable distance value
}

# Segment flip constants
FLIP_CONFIG = {
    "SEGMENT_HALF_DIVISOR": 2,    # Divisor for calculating half segment length
    "MAX_SEGMENT_SIZE": 1000,     # Maximum segment size for flip operation
}
