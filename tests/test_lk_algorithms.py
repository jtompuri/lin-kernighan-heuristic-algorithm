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

# simple_tsp_setup fixture is automatically available from conftest.py

# --- Tests for lin_kernighan function ---


def test_lin_kernighan_converges_on_simple_tsp(simple_tsp_setup):
    coords, dist_matrix, initial_tour_obj, neighbors, \
        initial_cost_fixture, optimal_cost_fixture, optimal_order_fixture = simple_tsp_setup
    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 5
    LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]
    LK_CONFIG["BREADTH_A"], LK_CONFIG["BREADTH_B"], LK_CONFIG["BREADTH_D"] = 5, 5, 1
    deadline = time.time() + 10
    best_tour_obj, best_cost = lin_kernighan(
        coords, initial_tour_obj.get_tour(), dist_matrix, neighbors, deadline
    )

    assert best_tour_obj.cost == pytest.approx(best_cost)  # This can stay, it's a good sanity check

    assert best_cost == pytest.approx(optimal_cost_fixture)
    assert best_tour_obj.get_tour() == optimal_order_fixture
    assert best_tour_obj.cost == pytest.approx(optimal_cost_fixture)
    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_lin_kernighan_no_change_on_optimal_tour(simple_tsp_setup):
    coords, dist_matrix, _initial_tour_obj, neighbors, \
        _initial_cost, optimal_cost_fixture, optimal_order_fixture = simple_tsp_setup
    # Ensure fixture's optimal tour has the correct cost before testing
    assert Tour(optimal_order_fixture, dist_matrix).cost == pytest.approx(optimal_cost_fixture)

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
    coords, dist_matrix, initial_tour_obj, neighbors, \
        initial_cost_fixture, _oc, _oo = simple_tsp_setup  # _oc, _oo are optimal_cost, optimal_order
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


# --- Tests for chained_lin_kernighan function ---


def test_chained_lin_kernighan_converges_on_simple_tsp(simple_tsp_setup):
    coords, dist_matrix_fixture, initial_tour_obj, _neighbors_fixture, \
        initial_cost_fixture, optimal_cost_fixture, optimal_order_fixture = simple_tsp_setup
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
    coords, dist_matrix_fixture, initial_tour_obj, _neighbors_fixture, \
        initial_cost_fixture, _oc, _oo = simple_tsp_setup
    initial_order = initial_tour_obj.get_tour()
    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 5
    LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]

    # Use a deadline that is definitively in the past
    # This ensures that the very first check of the deadline should cause an early exit.
    time_limit_in_past = -1.0  # Or time.time() - 1.0, but -1.0 is simpler if using total_runtime_start

    best_tour_order_list, best_cost = chained_lin_kernighan(
        coords, initial_order, time_limit_seconds=time_limit_in_past
    )

    best_tour_obj = Tour(best_tour_order_list, dist_matrix_fixture)
    # The print can be helpful for debugging if it still fails
    print(f"ChainedLK deadline test (past): Found tour {best_tour_obj.get_tour()} with cost {best_cost}. Initial cost was {initial_cost_fixture}")

    assert best_cost == pytest.approx(initial_cost_fixture)
    assert best_tour_obj.get_tour() == initial_order  # Expecting the exact initial tour
    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_chained_lin_kernighan_with_optimal_start(simple_tsp_setup):
    coords, dist_matrix_fixture, _initial_tour_obj, _neighbors_fixture, \
        _initial_cost, optimal_cost_fixture, optimal_order_fixture = simple_tsp_setup
    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 5
    LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]
    LK_CONFIG["BREADTH_A"], LK_CONFIG["BREADTH_B"], LK_CONFIG["BREADTH_D"] = 5, 5, 1
    LK_CONFIG["KICK_STRENGTH"] = 4
    LK_CONFIG["MAX_NO_IMPROVEMENT_ITERATIONS"] = coords.shape[0]
    time_limit_for_test = 10.0

    best_tour_order_list, best_cost = chained_lin_kernighan(
        coords, optimal_order_fixture,
        known_optimal_length=optimal_cost_fixture, time_limit_seconds=time_limit_for_test
    )

    best_tour_obj = Tour(best_tour_order_list, dist_matrix_fixture)

    assert best_cost == pytest.approx(optimal_cost_fixture)
    assert best_tour_obj.get_tour() == optimal_order_fixture
    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_chained_lin_kernighan_stops_on_max_no_improvement(simple_tsp_setup):
    coords, dist_matrix_fixture, _initial_tour_obj, _neighbors_fixture, \
        _initial_cost, optimal_cost_fixture, optimal_order_fixture = simple_tsp_setup

    original_lk_config = LK_CONFIG.copy()
    # Configure for a quick stop due to no improvement
    LK_CONFIG["MAX_LEVEL"] = 5
    LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]
    LK_CONFIG["BREADTH_A"], LK_CONFIG["BREADTH_B"], LK_CONFIG["BREADTH_D"] = 5, 5, 1
    LK_CONFIG["KICK_STRENGTH"] = 4  # Ensure kicks happen
    LK_CONFIG["MAX_NO_IMPROVEMENT_ITERATIONS"] = 1  # Stop after 1 iteration without improvement

    # Set a generous time limit that won't be hit
    time_limit_for_test = 30.0
    # Set known_optimal_length to something slightly better than actual, so it's not the stop reason
    # Or set to None if your implementation handles that to mean "run until other conditions"
    slightly_better_than_optimal = optimal_cost_fixture - 0.0001

    # Start with the optimal tour
    best_tour_order_list, best_cost = chained_lin_kernighan(
        coords,
        optimal_order_fixture,  # Start with the optimal tour
        known_optimal_length=slightly_better_than_optimal,
        time_limit_seconds=time_limit_for_test
    )

    best_tour_obj = Tour(best_tour_order_list, dist_matrix_fixture)

    # Expect to get the optimal tour back, as no improvement was possible
    assert best_cost == pytest.approx(optimal_cost_fixture), \
        "Cost should be optimal if starting with optimal and no improvement found."
    assert best_tour_obj.get_tour() == optimal_order_fixture, \
        "Tour should be optimal if starting with optimal and no improvement found."

    # We can't directly check the number of iterations without instrumenting the code,
    # but if the test passes quickly and returns the optimal, it implies
    # MAX_NO_IMPROVEMENT_ITERATIONS was likely the stopping condition,
    # as the time limit was high and known_optimal_length was unachievable.

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_kick_function_perturbs_tour(simple_tsp_setup):
    _coords, dist_matrix_fixture, _initial_tour_obj, _neighbors_fixture, \
        _initial_cost, _optimal_cost_fixture, optimal_order_fixture = simple_tsp_setup

    original_tour_nodes = list(optimal_order_fixture)  # Make a mutable copy

    # Call the kick function with the correct signature
    kicked_tour_nodes = double_bridge(original_tour_nodes)

    assert kicked_tour_nodes is not None, "Kick function should return a tour."
    assert len(kicked_tour_nodes) == len(original_tour_nodes), \
        "Kicked tour should have the same number of nodes."
    assert sorted(kicked_tour_nodes) == sorted(original_tour_nodes), \
        "Kicked tour should be a permutation of the original nodes."

    # For a 5-node problem, a double bridge kick should change the tour
    # if n > 4 as per the double_bridge function's condition.
    # The simple_tsp_setup has 5 nodes.
    if len(original_tour_nodes) > 4:
        assert kicked_tour_nodes != original_tour_nodes, \
            "Kicked tour should be different from the original tour for n > 4."
    else:  # If n <= 4, double_bridge returns the original tour
        assert kicked_tour_nodes == original_tour_nodes, \
            "Kicked tour should be the same as original for n <= 4."

    # Optionally, check if the new tour is valid and has a cost
    kicked_tour_obj = Tour(kicked_tour_nodes, dist_matrix_fixture)
    assert (kicked_tour_obj.cost is not None and kicked_tour_obj.cost > 0) or len(kicked_tour_nodes) == 0, \
        "Kicked tour should have a valid positive cost (or zero if empty)."


def test_lin_kernighan_on_2_node_tsp():
    coords = np.array([[0, 0], [1, 1]])
    dist_matrix = build_distance_matrix(coords)
    initial_tour_nodes = [0, 1]
    # For a 2-node TSP, neighbors are fixed, or Delaunay might be trivial
    # Let's assume a fully connected graph for neighbors for simplicity here,
    # or rely on delaunay_neighbors to handle it.
    try:
        neighbors = delaunay_neighbors(coords)
        if not neighbors[0] or not neighbors[1]:  # Ensure delaunay gives meaningful result
            neighbors = {0: [1], 1: [0]}
    except Exception:  # Fallback if Delaunay fails for 2 nodes (e.g. QhullError)
        neighbors = {0: [1], 1: [0]}
    if isinstance(neighbors, dict):
        neighbors = [neighbors[i] for i in range(len(neighbors))]

    original_lk_config = LK_CONFIG.copy()
    # Use default or simple LK_CONFIG for this small case
    LK_CONFIG["MAX_LEVEL"] = 2
    LK_CONFIG["BREADTH"] = [1] * LK_CONFIG["MAX_LEVEL"]
    deadline = time.time() + 5  # Short deadline

    best_tour_obj, best_cost = lin_kernighan(
        coords, initial_tour_nodes, dist_matrix, neighbors, deadline
    )

    expected_tour_nodes = [0, 1]  # Or [1, 0] normalized
    expected_cost = dist_matrix[0, 1] + dist_matrix[1, 0]

    assert best_tour_obj.get_tour() == expected_tour_nodes or best_tour_obj.get_tour() == expected_tour_nodes[::-1]
    assert best_cost == pytest.approx(expected_cost)
    assert best_tour_obj.cost == pytest.approx(expected_cost)
    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_lin_kernighan_on_3_node_tsp():
    coords = np.array([[0, 0], [1, 0], [0, 1]])  # Simple triangle
    dist_matrix = build_distance_matrix(coords)
    initial_tour_nodes = [0, 1, 2]

    # For 3 nodes, Delaunay should work fine and give full connectivity
    neighbors = delaunay_neighbors(coords)

    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 2
    LK_CONFIG["BREADTH"] = [1] * LK_CONFIG["MAX_LEVEL"]
    deadline = time.time() + 5

    best_tour_obj, best_cost = lin_kernighan(
        coords, initial_tour_nodes, dist_matrix, neighbors, deadline
    )

    # Expected tour is 0-1-2-0 (or its reverse/rotation)
    # Cost: d(0, 1) + d(1, 2) + d(2, 0) = 1 + sqrt(1^2+1^2) + 1 = 2 + sqrt(2)
    expected_cost = 1.0 + np.sqrt(2) + 1.0

    # Check cost first, as tour order can vary by rotation/reversal
    assert best_cost == pytest.approx(expected_cost)
    assert best_tour_obj.cost == pytest.approx(expected_cost)
    assert len(best_tour_obj.get_tour()) == 3
    assert sorted(best_tour_obj.get_tour()) == [0, 1, 2]  # Ensure all nodes are present

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)
