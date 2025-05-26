"""
Unit tests for the Lin-Kernighan and Chained Lin-Kernighan algorithms.

This module tests the convergence, deadline respect, and other specific
behaviors of the LK and Chained LK TSP solving algorithms, as well as
auxiliary functions like the tour perturbation (kick) function.
It utilizes a common TSP setup provided by the `simple_tsp_setup` fixture.
"""
import pytest
import time
import numpy as np
from lin_kernighan_tsp_solver.lin_kernighan_tsp_solver import (
    Tour,
    lin_kernighan,
    chained_lin_kernighan,
    LK_CONFIG,
    double_bridge,
    build_distance_matrix,
    delaunay_neighbors
)


def test_lin_kernighan_converges_on_simple_tsp(simple_tsp_setup):
    """
    Tests if lin_kernighan converges to the known optimal solution
    for a simple TSP instance.
    """
    coords, dist_matrix, initial_tour_obj, neighbors, \
        _, optimal_cost_fixture, optimal_order_fixture = simple_tsp_setup
    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 5
    LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]
    LK_CONFIG["BREADTH_A"], LK_CONFIG["BREADTH_B"], LK_CONFIG["BREADTH_D"] = 5, 5, 1
    deadline = time.time() + 10  # Generous deadline for a small problem

    best_tour_obj, best_cost = lin_kernighan(
        coords, initial_tour_obj.get_tour(), dist_matrix, neighbors, deadline
    )

    # Sanity check: returned cost should match the cost of the returned tour object
    assert best_tour_obj.cost == pytest.approx(best_cost)
    assert best_cost == pytest.approx(optimal_cost_fixture)
    assert best_tour_obj.get_tour() == optimal_order_fixture
    assert best_tour_obj.cost == pytest.approx(optimal_cost_fixture)

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_lin_kernighan_no_change_on_optimal_tour(simple_tsp_setup):
    """
    Tests if lin_kernighan makes no changes when starting with an
    already optimal tour.
    """
    coords, dist_matrix, _, neighbors, \
        _, optimal_cost_fixture, optimal_order_fixture = simple_tsp_setup
    # Ensure fixture's optimal tour has the correct cost before testing
    assert Tour(optimal_order_fixture, dist_matrix).cost == \
        pytest.approx(optimal_cost_fixture)

    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 5
    LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]
    LK_CONFIG["BREADTH_A"], LK_CONFIG["BREADTH_B"], LK_CONFIG["BREADTH_D"] = 5, 5, 1
    deadline = time.time() + 10

    best_tour_obj, best_cost = lin_kernighan(
        coords, optimal_order_fixture, dist_matrix, neighbors, deadline
    )

    assert best_cost == pytest.approx(optimal_cost_fixture)
    assert best_tour_obj.get_tour() == optimal_order_fixture

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_lin_kernighan_respects_deadline(simple_tsp_setup):
    """
    Tests if lin_kernighan respects the deadline and returns the initial
    tour if the time is up before any improvements can be made.
    """
    coords, dist_matrix, initial_tour_obj, neighbors, \
        initial_cost_fixture, _, _ = simple_tsp_setup
    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 5
    LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]
    immediate_deadline = time.time() - 1  # Deadline in the past

    best_tour_obj, best_cost = lin_kernighan(
        coords, initial_tour_obj.get_tour(), dist_matrix, neighbors, immediate_deadline
    )

    assert best_cost == pytest.approx(initial_cost_fixture)
    assert best_tour_obj.get_tour() == initial_tour_obj.get_tour()

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_chained_lin_kernighan_converges_on_simple_tsp(simple_tsp_setup):
    """
    Tests if chained_lin_kernighan converges to the known optimal solution
    for a simple TSP instance.
    """
    coords, dist_matrix_fixture, initial_tour_obj, _, \
        _, optimal_cost_fixture, optimal_order_fixture = simple_tsp_setup
    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 5
    LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]
    LK_CONFIG["BREADTH_A"], LK_CONFIG["BREADTH_B"], LK_CONFIG["BREADTH_D"] = 5, 5, 1
    LK_CONFIG["KICK_STRENGTH"] = 4
    LK_CONFIG["MAX_NO_IMPROVEMENT_ITERATIONS"] = coords.shape[0]
    time_limit_for_test = 10.0

    best_tour_order_list, best_cost = chained_lin_kernighan(
        coords, initial_tour_obj.get_tour(),
        known_optimal_length=optimal_cost_fixture,
        time_limit_seconds=time_limit_for_test
    )

    best_tour_obj = Tour(best_tour_order_list, dist_matrix_fixture)

    assert best_cost == pytest.approx(optimal_cost_fixture)
    assert best_tour_obj.get_tour() == optimal_order_fixture
    assert best_tour_obj.cost == pytest.approx(optimal_cost_fixture)

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_chained_lin_kernighan_respects_overall_deadline(simple_tsp_setup):
    """
    Tests if chained_lin_kernighan respects the overall time limit,
    returning the initial tour if the deadline is in the past.
    """
    coords, dist_matrix_fixture, initial_tour_obj, _, \
        initial_cost_fixture, _, _ = simple_tsp_setup
    initial_order = initial_tour_obj.get_tour()
    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 5
    LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]

    time_limit_in_past = -1.0  # Deadline effectively in the past

    best_tour_order_list, best_cost = chained_lin_kernighan(
        coords, initial_order, time_limit_seconds=time_limit_in_past
    )

    best_tour_obj = Tour(best_tour_order_list, dist_matrix_fixture)

    assert best_cost == pytest.approx(initial_cost_fixture)
    assert best_tour_obj.get_tour() == initial_order

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_chained_lin_kernighan_with_optimal_start(simple_tsp_setup):
    """
    Tests chained_lin_kernighan's behavior when starting with an
    already optimal tour.
    """
    coords, dist_matrix_fixture, _, _, \
        _, optimal_cost_fixture, optimal_order_fixture = simple_tsp_setup
    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 5
    LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]
    LK_CONFIG["BREADTH_A"], LK_CONFIG["BREADTH_B"], LK_CONFIG["BREADTH_D"] = 5, 5, 1
    LK_CONFIG["KICK_STRENGTH"] = 4
    LK_CONFIG["MAX_NO_IMPROVEMENT_ITERATIONS"] = coords.shape[0]
    time_limit_for_test = 10.0

    best_tour_order_list, best_cost = chained_lin_kernighan(
        coords, optimal_order_fixture,
        known_optimal_length=optimal_cost_fixture,
        time_limit_seconds=time_limit_for_test
    )

    best_tour_obj = Tour(best_tour_order_list, dist_matrix_fixture)

    assert best_cost == pytest.approx(optimal_cost_fixture)
    assert best_tour_obj.get_tour() == optimal_order_fixture

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_chained_lin_kernighan_stops_on_max_no_improvement(simple_tsp_setup):
    """
    Tests if chained_lin_kernighan stops after MAX_NO_IMPROVEMENT_ITERATIONS
    when starting with an optimal tour and no further improvement is possible.
    """
    coords, dist_matrix_fixture, _, _, \
        _, optimal_cost_fixture, optimal_order_fixture = simple_tsp_setup

    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 5
    LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]
    LK_CONFIG["BREADTH_A"], LK_CONFIG["BREADTH_B"], LK_CONFIG["BREADTH_D"] = 5, 5, 1
    LK_CONFIG["KICK_STRENGTH"] = 4
    LK_CONFIG["MAX_NO_IMPROVEMENT_ITERATIONS"] = 1  # Stop quickly if no improvement

    time_limit_for_test = 30.0  # Generous time limit
    # Known optimal is set slightly better, so it's not the stopping reason
    slightly_better_than_optimal = optimal_cost_fixture - 0.0001

    best_tour_order_list, best_cost = chained_lin_kernighan(
        coords,
        optimal_order_fixture,
        known_optimal_length=slightly_better_than_optimal,
        time_limit_seconds=time_limit_for_test
    )

    best_tour_obj = Tour(best_tour_order_list, dist_matrix_fixture)

    assert best_cost == pytest.approx(optimal_cost_fixture), \
        "Cost should be optimal if starting optimal and no improvement found."
    assert best_tour_obj.get_tour() == optimal_order_fixture, \
        "Tour should be optimal if starting optimal and no improvement found."

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_kick_function_perturbs_tour(simple_tsp_setup):
    """
    Tests the double_bridge kick function to ensure it perturbs the tour
    correctly, producing a valid permutation different from the original
    for tours of more than 4 nodes.
    """
    _, dist_matrix_fixture, _, _, \
        _, _, optimal_order_fixture = simple_tsp_setup

    original_tour_nodes = list(optimal_order_fixture)

    kicked_tour_nodes = double_bridge(original_tour_nodes)

    assert kicked_tour_nodes is not None, "Kick function should return a tour."
    assert len(kicked_tour_nodes) == len(original_tour_nodes), \
        "Kicked tour should have the same number of nodes."
    assert sorted(kicked_tour_nodes) == sorted(original_tour_nodes), \
        "Kicked tour should be a permutation of the original nodes."

    # For simple_tsp_setup (5 nodes), double_bridge should change the tour.
    if len(original_tour_nodes) > 4:
        assert kicked_tour_nodes != original_tour_nodes, \
            "Kicked tour should be different from original for n > 4."
    else:
        assert kicked_tour_nodes == original_tour_nodes, \
            "Kicked tour should be the same as original for n <= 4."

    kicked_tour_obj = Tour(kicked_tour_nodes, dist_matrix_fixture)
    assert (kicked_tour_obj.cost is not None and kicked_tour_obj.cost > 0) \
        or len(kicked_tour_nodes) == 0, \
        "Kicked tour should have a valid positive cost (or zero if empty)."


def test_lin_kernighan_on_2_node_tsp():
    """
    Tests lin_kernighan on a 2-node TSP instance.
    The tour and cost are trivial and should be found correctly.
    """
    coords = np.array([[0, 0], [1, 1]])
    dist_matrix = build_distance_matrix(coords)
    initial_tour_nodes = [0, 1]

    # For 2 nodes, neighbors are fixed.
    # delaunay_neighbors might return [[1],[0]] or similar.
    # If it fails or returns empty lists, provide a fallback.
    try:
        neighbors = delaunay_neighbors(coords)
        if not (len(neighbors) == 2 and neighbors[0] and neighbors[1]):
            neighbors = [[1], [0]]  # Ensure correct list-of-lists format
    except Exception:
        neighbors = [[1], [0]]


    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 2
    LK_CONFIG["BREADTH"] = [1] * LK_CONFIG["MAX_LEVEL"]
    deadline = time.time() + 5

    best_tour_obj, best_cost = lin_kernighan(
        coords, initial_tour_nodes, dist_matrix, neighbors, deadline
    )

    expected_tour_nodes = [0, 1]
    expected_cost = dist_matrix[0, 1] + dist_matrix[1, 0]

    assert best_tour_obj.get_tour() == expected_tour_nodes or \
        best_tour_obj.get_tour() == expected_tour_nodes[::-1]
    assert best_cost == pytest.approx(expected_cost)
    assert best_tour_obj.cost == pytest.approx(expected_cost)

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_lin_kernighan_on_3_node_tsp():
    """
    Tests lin_kernighan on a 3-node TSP instance (a simple triangle).
    The optimal tour should be found.
    """
    coords = np.array([[0, 0], [1, 0], [0, 1]])
    dist_matrix = build_distance_matrix(coords)
    initial_tour_nodes = [0, 1, 2]

    neighbors = delaunay_neighbors(coords)  # For 3 nodes, this is [[1,2],[0,2],[0,1]]

    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 2
    LK_CONFIG["BREADTH"] = [1] * LK_CONFIG["MAX_LEVEL"]
    deadline = time.time() + 5

    best_tour_obj, best_cost = lin_kernighan(
        coords, initial_tour_nodes, dist_matrix, neighbors, deadline
    )

    # Expected cost: d(0,1)+d(1,2)+d(2,0) = 1 + sqrt(2) + 1 = 2 + sqrt(2)
    expected_cost = 1.0 + np.sqrt(2) + 1.0

    assert best_cost == pytest.approx(expected_cost)
    assert best_tour_obj.cost == pytest.approx(expected_cost)
    assert len(best_tour_obj.get_tour()) == 3
    assert sorted(best_tour_obj.get_tour()) == [0, 1, 2]

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)
