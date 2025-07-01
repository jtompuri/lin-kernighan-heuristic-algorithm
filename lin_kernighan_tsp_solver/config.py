"""
Configuration constants for the Lin-Kernighan TSP solver.

This module centralizes all configuration parameters and constants used
throughout the project, such as algorithm settings, tolerances, and paths.
"""

from pathlib import Path
from dataclasses import dataclass


# Path to the folder containing TSPLIB .tsp files and optional .opt.tour files
TSP_FOLDER_PATH = (
    Path(__file__).resolve().parent.parent / "verifications" / "tsplib95"
)

# Tolerance for floating point comparisons
FLOAT_COMPARISON_TOLERANCE = 1e-12

# Maximum number of subplots in the tour visualization
MAX_SUBPLOTS_IN_PLOT = 25

# Double-bridge perturbation constants
DOUBLE_BRIDGE_MIN_SIZE = 5  # Minimum tour size for meaningful double-bridge
DOUBLE_BRIDGE_CUT_POINTS = 4  # Number of cut points for double-bridge

# Optimality checking tolerances
OPTIMALITY_REL_TOLERANCE = 1e-7  # Relative tolerance for optimality check
OPTIMALITY_ABS_TOLERANCE = FLOAT_COMPARISON_TOLERANCE * 10  # Absolute tolerance

# Delaunay triangulation constants
MIN_DELAUNAY_POINTS = 3  # Minimum points required for Delaunay triangulation

# Tour and algorithm constants
EMPTY_TOUR_SIZE = 0  # Size of an empty tour
SINGLE_NODE_TOUR_SIZE = 1  # Size of a single-node tour
MIN_TOUR_SIZE_FOR_FLIP = 2  # Minimum tour size where flipping makes sense
NODE_POSITION_BUFFER = 1  # Buffer for node position array sizing
RANDOM_CHOICE_START_INDEX = 1  # Start index for random choice in double_bridge
AXIS_2D = 2  # Axis for 2D distance calculation
PAIR_COMBINATION_SIZE = 2  # Size for pairwise combinations in Delaunay

# Distance matrix constants
EMPTY_MATRIX_SIZE = (0, 0)  # Size for empty distance matrix
SINGLE_POINT_DISTANCE = 0.0  # Distance for single point to itself

# Very close points warning threshold
DUPLICATE_POINTS_THRESHOLD = 1e-10  # Threshold for detecting very close/duplicate points


@dataclass(frozen=True)
class LinKernighanConfig:
    """Configuration parameters for the Lin-Kernighan algorithm."""

    # Recursion and search limits
    max_level: int = 12  # Max recursion depth for k-opt moves in step()
    time_limit: float = 5.0  # Time limit for LK search (seconds)

    # Search breadth parameters
    breadth: list[int] = [5, 5] + [1] * 20  # Search breadth at each level in step()
    breadth_a: int = 5  # Search breadth for y1 in alternate_step()
    breadth_b: int = 5  # Search breadth for y2 in alternate_step()
    breadth_d: int = 1  # Search breadth for y4 in alternate_step()


# Global configuration instance
LK_CONFIG = LinKernighanConfig()
