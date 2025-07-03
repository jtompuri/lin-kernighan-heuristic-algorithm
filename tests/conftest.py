import pytest
import numpy as np
from pathlib import Path
from lin_kernighan_tsp_solver.lk_algorithm import (
    Tour,
    build_distance_matrix,
    delaunay_neighbors
)

# Path constants for test files
VERIFICATION_RANDOM_PATH = Path(__file__).resolve().parent.parent / "problems" / "random"
VERIFICATION_SOLUTIONS_PATH = Path(__file__).resolve().parent.parent / "solutions" / "exact"


@pytest.fixture
def simple_tsp_setup():
    """
    Provides a simple 5-node TSP instance for testing Lin-Kernighan components.

    This fixture sets up:
    - Coordinates for 5 nodes, slightly perturbed to avoid co-linearity issues
      for Delaunay triangulation.
    - The corresponding distance matrix.
    - An initial tour object ([0, 1, 2, 3, 4]) and its cost.
    - Delaunay neighbors for the given coordinates.
    - A known optimal tour ([0, 2, 4, 3, 1]) for this specific problem setup
      and its pre-calculated cost, which serves as a benchmark.

    Returns:
        tuple: A tuple containing:
            - coords (np.ndarray): Node coordinates.
            - dist_matrix (np.ndarray): Pairwise distance matrix.
            - initial_tour_obj (Tour): Tour object for the initial tour.
            - neighbors (List[List[int]]): Delaunay neighbors for each node.
            - initial_cost (float): Cost of the initial tour.
            - lk_optimal_cost (float): Known optimal cost for this instance.
            - lk_optimal_order (List[int]): Node order of the known optimal tour.
    """
    coords = np.array([
        [0.0, 0.01],  # Node 0
        [1.0, -0.01],  # Node 1
        [3.0, 0.02],  # Node 2
        [2.0, -0.02],  # Node 3
        [4.0, 0.0]    # Node 4
    ])
    dist_matrix = build_distance_matrix(coords)

    initial_tour_order_nodes = [0, 1, 2, 3, 4]
    initial_tour_obj = Tour(initial_tour_order_nodes, dist_matrix)
    initial_cost = initial_tour_obj.cost

    # Known optimal tour and cost for this specific 5-node problem,
    # consistently found by the Lin-Kernighan implementation.
    # This serves as a benchmark for tests.
    lk_optimal_order_nodes = [0, 2, 4, 3, 1]
    # Pre-calculated optimal cost for high precision in assertions.
    # This value was obtained by running the LK algorithm on this instance.
    lk_optimal_cost = 8.000566622878555
    # As a verification, one could also calculate it:
    # optimal_cost_calculated = Tour(lk_optimal_order_nodes, dist_matrix).cost
    # assert np.isclose(lk_optimal_cost, optimal_cost_calculated)

    neighbors = delaunay_neighbors(coords)

    return (
        coords,
        dist_matrix,
        initial_tour_obj,
        neighbors,
        initial_cost,
        lk_optimal_cost,
        lk_optimal_order_nodes
    )
