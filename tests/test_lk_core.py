"""
Unit tests for the core components of the Lin-Kernighan algorithm.

This module tests the `step` function (for finding k-opt moves),
the `lk_search` function (a single pass of LK from a base node),
and various utility functions like `build_distance_matrix` and
`delaunay_neighbors`. It also includes tests for the main
`lin_kernighan` and `chained_lin_kernighan` algorithms, focusing on
their behavior with different configurations and scenarios, including
deadline respect and convergence.
"""
from pathlib import Path
import numpy as np
import time
import pytest

from lin_kernighan_tsp_solver.lin_kernighan_tsp_solver import (
    Tour,
    step,
    alternate_step,
    lk_search,
    lin_kernighan,
    chained_lin_kernighan,
    double_bridge,
    LK_CONFIG,
    FLOAT_COMPARISON_TOLERANCE,
    read_tsp_file,
    read_opt_tour,
    build_distance_matrix,
    delaunay_neighbors,
)

# simple_tsp_setup fixture is automatically available from conftest.py

VERIFICATION_RANDOM_PATH = Path(__file__).resolve().parent.parent / "verifications" / "random"


def test_step_finds_simple_2_opt(simple_tsp_setup):
    """
    Tests if the `step` function (configured for 2-opt) can find a specific
    2-opt improvement on the simple_tsp_setup instance.
    """
    _coords, dist_matrix, initial_tour_obj, neighbors, initial_cost, \
        _lk_optimal_cost, _lk_optimal_order = simple_tsp_setup

    # This test focuses on a 2-opt step, not necessarily reaching the full LK optimum.
    # Expected outcome of flipping segment (2,3) in tour [0,1,2,3,4] (nodes 2 and 3)
    # results in tour [0,1,3,2,4]
    expected_tour_after_2_opt_nodes = [0, 1, 3, 2, 4]
    expected_cost_after_2_opt = Tour(expected_tour_after_2_opt_nodes, dist_matrix).cost

    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 1  # Corresponds to 2-opt
    LK_CONFIG["BREADTH"] = [1] * LK_CONFIG["MAX_LEVEL"]  # Minimal breadth

    improved_overall = False
    best_flip_sequence_found = None
    found_the_specific_2_opt_move = False

    for base_node_val in initial_tour_obj.order:
        current_tour_for_step = Tour(initial_tour_obj.get_tour(), dist_matrix)
        improved, flip_sequence = step(
            level=1, delta=0.0, base=base_node_val, tour=current_tour_for_step,
            D=dist_matrix, neigh=neighbors, flip_seq=[],
            start_cost=initial_cost, best_cost=initial_cost, deadline=time.time() + 10
        )
        if improved and flip_sequence:
            improved_overall = True
            if best_flip_sequence_found is None:
                best_flip_sequence_found = flip_sequence

            temp_tour_check = Tour(initial_tour_obj.get_tour(), dist_matrix)
            for fs, fe in flip_sequence:
                temp_tour_check.flip_and_update_cost(fs, fe, dist_matrix)

            if temp_tour_check.get_tour() == expected_tour_after_2_opt_nodes and \
               abs(temp_tour_check.cost - expected_cost_after_2_opt) < FLOAT_COMPARISON_TOLERANCE * 100:
                best_flip_sequence_found = flip_sequence
                found_the_specific_2_opt_move = True
                break

    assert improved_overall, "Step function should have found an improvement."
    assert found_the_specific_2_opt_move, \
        (f"Step did not find the expected 2-opt move leading to "
         f"{expected_tour_after_2_opt_nodes}. Best sequence found: {best_flip_sequence_found}")

    if best_flip_sequence_found:
        assert len(best_flip_sequence_found) == 1, "Expected a single flip for this 2-opt."
        # The flip in Tour.flip(a,b) reverses segment from node a to node b.
        # For [0,1,2,3,4] to become [0,1,3,2,4], the segment [2,3] was flipped.
        # The `step` function returns the pair of nodes defining the segment to flip.
        expected_flip_nodes = tuple(sorted((2, 3)))
        actual_flip_nodes = tuple(sorted(best_flip_sequence_found[0]))
        assert actual_flip_nodes == expected_flip_nodes, \
            f"Expected flip involving nodes {expected_flip_nodes}, but got {actual_flip_nodes}"

        final_tour_check = Tour(initial_tour_obj.get_tour(), dist_matrix)
        for fs, fe in best_flip_sequence_found:
            final_tour_check.flip_and_update_cost(fs, fe, dist_matrix)
        assert final_tour_check.cost == pytest.approx(expected_cost_after_2_opt)
        assert final_tour_check.get_tour() == expected_tour_after_2_opt_nodes

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_step_no_improvement_on_optimal_tour(simple_tsp_setup):
    """
    Tests that the `step` function (configured for 2-opt) makes no
    improvements on a tour that is already 2-optimal.
    """
    _coords, dist_matrix, initial_tour_obj_from_fixture, neighbors, \
        _initial_cost_fixture, _optimal_cost_fixture_setup, \
        _optimal_order_fixture_setup = simple_tsp_setup

    current_processing_tour = Tour(initial_tour_obj_from_fixture.get_tour(), dist_matrix)
    original_test_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 1  # 2-opt
    LK_CONFIG["BREADTH"] = [1] * LK_CONFIG["MAX_LEVEL"]
    max_2opt_passes = current_processing_tour.n * 5  # Heuristic limit for stabilization

    # First, stabilize the tour to a 2-optimal state
    for _pass_num in range(max_2opt_passes):
        made_improvement_in_this_pass = False
        nodes_to_iterate_for_base = list(current_processing_tour.order)
        for base_node_val in nodes_to_iterate_for_base:
            cost_before_this_step_attempt = current_processing_tour.cost
            tour_for_step_to_modify_order = Tour(current_processing_tour.get_tour(), dist_matrix)
            improved_by_step, flip_sequence_from_step = step(
                level=1, delta=0.0, base=base_node_val, tour=tour_for_step_to_modify_order,
                D=dist_matrix, neigh=neighbors, flip_seq=[],
                start_cost=cost_before_this_step_attempt, best_cost=cost_before_this_step_attempt,
                deadline=time.time() + 10
            )
            if improved_by_step and flip_sequence_from_step:
                candidate_tour_after_flips = Tour(current_processing_tour.get_tour(), dist_matrix)
                for f_start, f_end in flip_sequence_from_step:
                    candidate_tour_after_flips.flip_and_update_cost(f_start, f_end, dist_matrix)
                actual_new_cost = candidate_tour_after_flips.cost
                if actual_new_cost < cost_before_this_step_attempt - (FLOAT_COMPARISON_TOLERANCE / 10.0):
                    current_processing_tour = candidate_tour_after_flips
                    made_improvement_in_this_pass = True
                    break
        if not made_improvement_in_this_pass:
            break
    else:
        # This warning indicates the stabilization loop might not have converged.
        print(f"Warning: 2-opt stabilization loop reached max_passes in {test_step_no_improvement_on_optimal_tour.__name__}.")

    truly_2_optimal_tour = current_processing_tour
    cost_of_2_optimal_tour = truly_2_optimal_tour.cost
    order_of_2_optimal_tour = truly_2_optimal_tour.get_tour()

    # Now, test that `step` finds no improvement on this 2-optimal tour
    LK_CONFIG.clear()
    LK_CONFIG.update(original_test_lk_config)  # Restore config before this test's modifications
    LK_CONFIG["MAX_LEVEL"] = 1
    LK_CONFIG["BREADTH"] = [1] * LK_CONFIG["MAX_LEVEL"]
    deadline_for_final_check = time.time() + 10

    for base_node_val_check in order_of_2_optimal_tour:
        tour_arg_for_final_step = Tour(order_of_2_optimal_tour, dist_matrix)
        improved_final, flip_sequence_final = step(
            level=1, delta=0.0, base=base_node_val_check, tour=tour_arg_for_final_step,
            D=dist_matrix, neigh=neighbors, flip_seq=[],
            start_cost=cost_of_2_optimal_tour, best_cost=cost_of_2_optimal_tour,
            deadline=deadline_for_final_check
        )
        final_tour_cost_after_step_if_improved = cost_of_2_optimal_tour
        if improved_final and flip_sequence_final:
            temp_tour_for_cost_check = Tour(order_of_2_optimal_tour, dist_matrix)
            for f_s, f_e in flip_sequence_final:
                temp_tour_for_cost_check.flip_and_update_cost(f_s, f_e, dist_matrix)
            final_tour_cost_after_step_if_improved = temp_tour_for_cost_check.cost

        assert not improved_final, \
            (f"Step found improvement from base {base_node_val_check} on 2-optimal tour "
             f"(cost {cost_of_2_optimal_tour:.10f}). Seq: {flip_sequence_final}. "
             f"New cost: {final_tour_cost_after_step_if_improved:.10f}")
        if not improved_final:  # Should always be true if the above assert passes
            assert tour_arg_for_final_step.get_tour() == order_of_2_optimal_tour
            assert tour_arg_for_final_step.cost == pytest.approx(cost_of_2_optimal_tour)

    LK_CONFIG.clear()
    LK_CONFIG.update(original_test_lk_config)


def test_lk_search_finds_optimum_for_simple_tsp(simple_tsp_setup):
    """
    Tests if `lk_search` (a single LK pass from multiple base nodes)
    finds the known optimum for the simple_tsp_setup instance.
    """
    _coords, dist_matrix, initial_tour_obj, neighbors, \
        initial_cost_fixture, optimal_cost_fixture, \
        optimal_order_fixture = simple_tsp_setup

    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 5
    LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]
    LK_CONFIG["BREADTH_A"], LK_CONFIG["BREADTH_B"], LK_CONFIG["BREADTH_D"] = 5, 5, 1

    current_tour = Tour(initial_tour_obj.get_tour(), dist_matrix)
    deadline = time.time() + 20
    overall_improved_flag, made_change_in_iteration = False, True
    max_iterations, iter_count = current_tour.n * 2, 0  # Heuristic iteration limit

    while made_change_in_iteration and iter_count < max_iterations:
        if abs(current_tour.cost - optimal_cost_fixture) < FLOAT_COMPARISON_TOLERANCE:
            overall_improved_flag = True
            break
        made_change_in_iteration = False
        iter_count += 1
        nodes_to_try_as_base = list(current_tour.order)  # Iterate from current tour's nodes
        for start_node_for_search in nodes_to_try_as_base:
            tour_for_lk_search_call = Tour(current_tour.get_tour(), dist_matrix)
            improving_sequence = lk_search(
                start_node_for_search, tour_for_lk_search_call, dist_matrix, neighbors, deadline
            )
            if improving_sequence:
                cost_before_flips = current_tour.cost
                for fs, fe in improving_sequence:
                    current_tour.flip_and_update_cost(fs, fe, dist_matrix)

                if current_tour.cost < cost_before_flips - FLOAT_COMPARISON_TOLERANCE:
                    overall_improved_flag = True
                    made_change_in_iteration = True
                    break  # Improvement found, restart iteration with new tour
                else:
                    # Sequence found but no strict improvement, revert and try next base
                    current_tour = Tour(nodes_to_try_as_base, dist_matrix)
                    # No break here, continue with other base nodes in this iteration
            if time.time() > deadline:  # Check deadline within the loop
                break
        if time.time() > deadline:
            break


    assert overall_improved_flag or abs(initial_cost_fixture - optimal_cost_fixture) < FLOAT_COMPARISON_TOLERANCE, \
        (f"lk_search failed to improve. Initial: {initial_cost_fixture}, "
         f"Optimal: {optimal_cost_fixture}, Final: {current_tour.cost}")
    assert current_tour.cost == pytest.approx(optimal_cost_fixture)
    assert current_tour.get_tour() == optimal_order_fixture

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_lk_search_no_improvement_on_optimal_tour(simple_tsp_setup):
    """
    Tests that `lk_search` makes no improvements when starting with an
    already optimal tour.
    """
    coords, dist_matrix, initial_tour_obj_from_fixture, neighbors, \
        _initial_cost, _optimal_cost_fixture, \
        _optimal_order_fixture = simple_tsp_setup

    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 5
    LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]
    LK_CONFIG["BREADTH_A"], LK_CONFIG["BREADTH_B"], LK_CONFIG["BREADTH_D"] = 5, 5, 1
    deadline_for_lk = time.time() + 15

    # First, obtain a converged (optimal for this setup) tour using lin_kernighan
    converged_tour_obj, converged_cost = lin_kernighan(
        coords, initial_tour_obj_from_fixture.get_tour(), dist_matrix, neighbors, deadline_for_lk
    )

    deadline_for_test = time.time() + 20
    for i in range(converged_tour_obj.n):
        start_node_for_search = converged_tour_obj.order[i]
        tour_copy_for_lk_search = Tour(converged_tour_obj.get_tour(), dist_matrix)
        assert tour_copy_for_lk_search.cost == pytest.approx(converged_cost)

        improving_sequence = lk_search(
            start_node_for_search, tour_copy_for_lk_search, dist_matrix, neighbors, deadline_for_test
        )
        assert improving_sequence is None, \
            (f"lk_search found improvement from node {start_node_for_search} "
             f"on converged tour. Sequence: {improving_sequence}")
        # Ensure the tour passed to lk_search was not modified if no improvement
        assert tour_copy_for_lk_search.get_tour() == converged_tour_obj.get_tour()
        assert tour_copy_for_lk_search.cost == pytest.approx(converged_cost)

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


# --- Utility Function Tests ---


def test_build_distance_matrix_simple_cases():
    """Tests `build_distance_matrix` with simple coordinate sets."""
    # Case 1: Two points
    coords1 = np.array([[0, 0], [3, 4]])  # Distance should be 5
    dist_matrix1 = build_distance_matrix(coords1)
    assert dist_matrix1.shape == (2, 2)
    assert dist_matrix1[0, 0] == pytest.approx(0.0)
    assert dist_matrix1[1, 1] == pytest.approx(0.0)
    assert dist_matrix1[0, 1] == pytest.approx(5.0)
    assert dist_matrix1[1, 0] == pytest.approx(5.0)

    # Case 2: Three points (3-4-5 triangle and one other)
    coords2 = np.array([[0, 0], [3, 0], [3, 4]])
    dist_matrix2 = build_distance_matrix(coords2)
    assert dist_matrix2.shape == (3, 3)
    assert dist_matrix2[0, 1] == pytest.approx(3.0)  # 0 to 1
    assert dist_matrix2[0, 2] == pytest.approx(5.0)  # 0 to 2
    assert dist_matrix2[1, 2] == pytest.approx(4.0)  # 1 to 2
    for i in range(3):  # Check symmetry and diagonal
        assert dist_matrix2[i, i] == pytest.approx(0.0)
        for j in range(i + 1, 3):
            assert dist_matrix2[i, j] == pytest.approx(dist_matrix2[j, i])

    # Case 3: Points forming a square
    coords3 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    dist_matrix3 = build_distance_matrix(coords3)
    assert dist_matrix3.shape == (4, 4)
    assert dist_matrix3[0, 1] == pytest.approx(1.0)
    assert dist_matrix3[0, 2] == pytest.approx(1.0)
    assert dist_matrix3[0, 3] == pytest.approx(np.sqrt(2))
    assert dist_matrix3[1, 2] == pytest.approx(np.sqrt(2))
    assert dist_matrix3[1, 3] == pytest.approx(1.0)
    assert dist_matrix3[2, 3] == pytest.approx(1.0)
    for i in range(4):
        assert dist_matrix3[i, i] == pytest.approx(0.0)
        for j in range(i + 1, 4):
            assert dist_matrix3[i, j] == pytest.approx(dist_matrix3[j, i])


def test_build_distance_matrix_edge_cases():
    """Tests `build_distance_matrix` with edge case inputs."""
    # Case 1: Empty coordinates
    coords_empty_1d = np.array([])
    dist_matrix_empty_1d = build_distance_matrix(coords_empty_1d)
    assert dist_matrix_empty_1d.shape == (0, 0)

    coords_empty_2d = np.empty((0, 2))
    dist_matrix_empty_2d = build_distance_matrix(coords_empty_2d)
    assert dist_matrix_empty_2d.shape == (0, 0)

    # Case 2: Single point
    coords_single = np.array([[10, 20]])
    dist_matrix_single = build_distance_matrix(coords_single)
    assert dist_matrix_single.shape == (1, 1)
    assert dist_matrix_single[0, 0] == pytest.approx(0.0)


def test_delaunay_neighbors_few_points():
    """Tests `delaunay_neighbors` with fewer than 3 points."""
    # Case 1: 0 points
    coords0 = np.empty((0, 2))
    neighbors0 = delaunay_neighbors(coords0)
    assert neighbors0 == []

    # Case 2: 1 point
    coords1 = np.array([[0, 0]])
    neighbors1 = delaunay_neighbors(coords1)
    assert len(neighbors1) == 1
    assert neighbors1[0] == []

    # Case 3: 2 points
    coords2 = np.array([[0, 0], [1, 1]])
    neighbors2 = delaunay_neighbors(coords2)
    assert len(neighbors2) == 2
    assert neighbors2[0] == [1]
    assert neighbors2[1] == [0]


def test_delaunay_neighbors_triangle():
    """Tests `delaunay_neighbors` with 3 points forming a triangle."""
    coords = np.array([[0, 0], [1, 0], [0, 1]])
    neighbors = delaunay_neighbors(coords)
    assert len(neighbors) == 3
    expected_neighbors = [[1, 2], [0, 2], [0, 1]]
    for i in range(3):
        assert sorted(neighbors[i]) == sorted(expected_neighbors[i]), \
            f"Neighbors for node {i} incorrect"


def test_delaunay_neighbors_square():
    """
    Tests `delaunay_neighbors` with 4 points forming a square.
    Checks for correct structure and properties of Delaunay neighbors.
    """
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    neighbors = delaunay_neighbors(coords)
    assert len(neighbors) == 4

    for i in range(len(coords)):  # General properties
        assert neighbors[i] == sorted(neighbors[i]), f"Neighbors for node {i} not sorted"
        assert i not in neighbors[i], f"Node {i} is its own neighbor"
        for neighbor_node in neighbors[i]:
            assert i in neighbors[neighbor_node], \
                f"Asymmetric neighborhood: {i} -> {neighbor_node} but not vice-versa"

    # Perimeter edges must exist
    assert 1 in neighbors[0] and 0 in neighbors[1]  # Edge 0-1
    assert 2 in neighbors[1] and 1 in neighbors[2]  # Edge 1-2
    assert 3 in neighbors[2] and 2 in neighbors[3]  # Edge 2-3
    assert 0 in neighbors[3] and 3 in neighbors[0]  # Edge 3-0

    # For a convex quadrilateral, Delaunay triangulation has 5 edges.
    # Sum of degrees = 2 * num_edges.
    total_degree = sum(len(n_list) for n_list in neighbors)
    assert total_degree == 2 * 5


def test_delaunay_neighbors_from_fixture(simple_tsp_setup):
    """
    Tests `delaunay_neighbors` using coordinates from `simple_tsp_setup`
    and compares against the fixture's expected neighbors.
    """
    coords, _dist_matrix, _initial_tour_obj, \
        expected_neighbors_from_fixture, _initial_cost, \
        _lk_optimal_cost, _lk_optimal_order = simple_tsp_setup

    calculated_neighbors = delaunay_neighbors(coords)

    assert len(calculated_neighbors) == len(coords)
    for i in range(len(coords)):
        assert calculated_neighbors[i] == sorted(calculated_neighbors[i]), \
            f"Calculated neighbors for node {i} are not sorted."
        assert i not in calculated_neighbors[i], \
            f"Node {i} is its own neighbor in calculated set."
        assert calculated_neighbors[i] == expected_neighbors_from_fixture[i], \
            f"Calculated neighbors for node {i} do not match fixture's expected neighbors."


def test_lin_kernighan_respects_overall_deadline(simple_tsp_setup):
    """
    Tests that `lin_kernighan` respects the `deadline` parameter
    and terminates in a timely manner.
    """
    coords, dist_matrix, initial_tour_obj, neighbors, \
        _initial_cost, _optimal_cost, _optimal_order = simple_tsp_setup

    original_lk_config = LK_CONFIG.copy()
    short_deadline = time.time() + 0.001  # Very short deadline

    start_time = time.time()
    final_tour_obj, final_cost = lin_kernighan(
        coords, initial_tour_obj.get_tour(), dist_matrix,
        neighbors, deadline=short_deadline
    )
    end_time = time.time()
    execution_time = end_time - start_time

    # Execution time should be very short, indicating deadline was likely hit.
    # Allow for some overhead (e.g., up to 0.1 seconds).
    assert execution_time < 0.1, \
        f"lin_kernighan took too long ({execution_time:.4f}s), deadline likely not respected."

    assert final_tour_obj is not None, "lin_kernighan should return a Tour object."
    assert isinstance(final_tour_obj, Tour), "Returned object is not a Tour."
    assert final_tour_obj.n == initial_tour_obj.n, "Returned tour has incorrect number of nodes."
    assert final_cost is not None, "lin_kernighan should return a cost."
    assert final_cost == pytest.approx(final_tour_obj.cost), "Returned cost does not match tour's cost."

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_lin_kernighan_improves_to_optimum_simple_case(simple_tsp_setup):
    """
    Tests that `lin_kernighan` improves a non-optimal initial tour to the
    known optimal solution for the `simple_tsp_setup` instance.
    """
    coords, dist_matrix, initial_tour_obj, neighbors, \
        _initial_cost, lk_optimal_cost, lk_optimal_order = simple_tsp_setup

    original_lk_config = LK_CONFIG.copy()
    deadline = time.time() + 10  # Sufficient time

    final_tour_obj, final_cost = lin_kernighan(
        coords, initial_tour_obj.get_tour(), dist_matrix, neighbors, deadline=deadline
    )

    assert final_tour_obj is not None
    assert final_cost is not None
    assert final_cost == pytest.approx(lk_optimal_cost), \
        f"Expected optimal cost {lk_optimal_cost}, but got {final_cost}"
    returned_tour_normalized = final_tour_obj.get_tour()
    assert returned_tour_normalized == lk_optimal_order, \
        f"Expected optimal tour {lk_optimal_order}, but got {returned_tour_normalized}"
    assert final_tour_obj.cost == pytest.approx(lk_optimal_cost)

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_lin_kernighan_on_optimal_tour_simple_case(simple_tsp_setup):
    """
    Tests that `lin_kernighan`, given an already optimal tour for
    `simple_tsp_setup`, returns it without finding further 'improvements'.
    """
    coords, dist_matrix, _initial_tour_obj, neighbors, \
        _initial_cost, lk_optimal_cost, lk_optimal_order = simple_tsp_setup

    original_lk_config = LK_CONFIG.copy()
    optimal_initial_tour_obj = Tour(lk_optimal_order, dist_matrix)
    assert optimal_initial_tour_obj.cost == pytest.approx(lk_optimal_cost)

    deadline = time.time() + 10

    final_tour_obj, final_cost = lin_kernighan(
        coords, optimal_initial_tour_obj.get_tour(), dist_matrix, neighbors, deadline=deadline
    )

    assert final_tour_obj is not None
    assert final_cost is not None
    assert final_cost == pytest.approx(lk_optimal_cost)
    returned_tour_normalized = final_tour_obj.get_tour()
    assert returned_tour_normalized == lk_optimal_order
    assert final_tour_obj.cost == pytest.approx(lk_optimal_cost)

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_lin_kernighan_config_sensitivity(simple_tsp_setup):
    """
    Tests that `LK_CONFIG` settings affect `lin_kernighan`'s ability
    to find the optimum. A restrictive config should fail, while a
    default/good config should succeed on `simple_tsp_setup`.
    """
    coords, dist_matrix, initial_tour_obj, neighbors, \
        initial_cost, lk_optimal_cost, _lk_optimal_order = simple_tsp_setup

    assert initial_cost != pytest.approx(lk_optimal_cost), \
        "Initial tour cost is already optimal, test may not be indicative."

    original_lk_config_snapshot = LK_CONFIG.copy()
    deadline = time.time() + 10

    try:
        # Scenario 1: Highly restrictive LK_CONFIG
        restrictive_settings = {
            "MAX_LEVEL": 0, "BREADTH": [], "MAX_CANDIDATES": 1,
            "BREADTH_A": 0, "BREADTH_B": 0, "BREADTH_D": 0
        }
        current_config_for_test = original_lk_config_snapshot.copy()
        current_config_for_test.update(restrictive_settings)
        LK_CONFIG.clear()
        LK_CONFIG.update(current_config_for_test)

        suboptimal_tour_obj, suboptimal_cost = lin_kernighan(
            coords, initial_tour_obj.get_tour(), dist_matrix, neighbors, deadline
        )
        assert suboptimal_cost == pytest.approx(initial_cost), \
            "Restrictive LK_CONFIG (MAX_LEVEL=0) should not have improved the tour."
        assert suboptimal_cost != pytest.approx(lk_optimal_cost), \
            "Restrictive LK_CONFIG (MAX_LEVEL=0) unexpectedly resulted in optimal cost."

        # Scenario 2: Restore to original (presumably good) LK_CONFIG
        LK_CONFIG.clear()
        LK_CONFIG.update(original_lk_config_snapshot)
        optimal_tour_obj_again, optimal_cost_again = lin_kernighan(
            coords, initial_tour_obj.get_tour(), dist_matrix, neighbors, deadline
        )
        assert optimal_cost_again == pytest.approx(lk_optimal_cost), \
            f"Default/Original LK_CONFIG should find the optimum ({lk_optimal_cost}). Found: {optimal_cost_again}"
    finally:
        LK_CONFIG.clear()
        LK_CONFIG.update(original_lk_config_snapshot)


def test_chained_lk_max_iterations_one(simple_tsp_setup):
    """
    Tests that `chained_lin_kernighan` (when effectively performing one LK run
    due to problem simplicity or short time) behaves comparably to a single
    `lin_kernighan` run on `simple_tsp_setup`.
    """
    coords, dist_matrix, initial_tour_obj, neighbors, \
        _initial_cost, lk_optimal_cost, lk_optimal_order = simple_tsp_setup

    original_lk_config = LK_CONFIG.copy()
    deadline_val = time.time() + 10
    time_limit_sec = 10.0  # Sufficient for simple case

    lk_tour_obj, lk_cost = lin_kernighan(
        coords, initial_tour_obj.get_tour(), dist_matrix, neighbors, deadline_val
    )
    chained_lk_tour_order, chained_lk_cost = chained_lin_kernighan(
        coords, initial_tour_obj.get_tour(), time_limit_seconds=time_limit_sec
    )

    assert chained_lk_cost == pytest.approx(lk_cost)
    assert chained_lk_tour_order == lk_tour_obj.get_tour()
    assert chained_lk_cost == pytest.approx(lk_optimal_cost)
    assert chained_lk_tour_order == lk_optimal_order

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_chained_lk_multiple_iterations(simple_tsp_setup):
    """
    Tests that `chained_lin_kernighan` with a reasonable time limit
    runs and finds the optimum for `simple_tsp_setup`.
    """
    coords, dist_matrix, initial_tour_obj, _neighbors, \
        _initial_cost, lk_optimal_cost, lk_optimal_order = simple_tsp_setup

    original_lk_config = LK_CONFIG.copy()
    time_limit_sec = 15.0

    chained_lk_tour_order, chained_lk_cost = chained_lin_kernighan(
        coords, initial_tour_obj.get_tour(), time_limit_seconds=time_limit_sec
    )

    assert chained_lk_tour_order is not None
    assert isinstance(chained_lk_tour_order, list)
    assert chained_lk_cost is not None
    assert chained_lk_cost == pytest.approx(lk_optimal_cost), \
        f"Chained LK failed to find optimal cost. Got {chained_lk_cost}, expected {lk_optimal_cost}."
    assert chained_lk_tour_order == lk_optimal_order, \
        f"Chained LK failed to find optimal tour. Got {chained_lk_tour_order}, expected {lk_optimal_order}."

    reconstructed_tour_from_chained = Tour(chained_lk_tour_order, dist_matrix)
    assert reconstructed_tour_from_chained.cost == pytest.approx(lk_optimal_cost)
    assert reconstructed_tour_from_chained.cost == pytest.approx(chained_lk_cost)

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_double_bridge_small_tours():
    """
    Tests `double_bridge` with small tours (n <= 4), where it should
    return the original tour.
    """
    for n in range(5):  # Test n = 0, 1, 2, 3, 4
        original_tour = list(range(n))
        perturbed_tour = double_bridge(original_tour.copy())
        assert perturbed_tour == original_tour, \
            f"Double bridge should not change tour with {n} nodes. Got {perturbed_tour}"


def test_double_bridge_larger_tour():
    """
    Tests `double_bridge` with tours large enough (n > 4) to be perturbed,
    ensuring the perturbed tour is a valid, different permutation.
    """
    for n in [5, 8, 10]:
        original_tour = list(range(n))
        different_tour_found = False
        attempts = 5  # Try a few times to mitigate unlucky random choices
        for _ in range(attempts):
            perturbed_tour = double_bridge(original_tour.copy())
            assert len(perturbed_tour) == n
            assert sorted(perturbed_tour) == sorted(original_tour)
            assert len(set(perturbed_tour)) == n
            if perturbed_tour != original_tour:
                different_tour_found = True
                break
        assert different_tour_found, \
            f"Double bridge did not change tour with {n} nodes after {attempts} attempts."


def test_double_bridge_very_large_tour_is_different():
    """
    Tests that `double_bridge` on a sufficiently large tour (n=20)
    produces a different tour.
    """
    n = 20
    original_tour = list(range(n))
    perturbed_tour = double_bridge(original_tour.copy())

    assert len(perturbed_tour) == n
    assert sorted(perturbed_tour) == sorted(original_tour)
    assert len(set(perturbed_tour)) == n
    assert perturbed_tour != original_tour, \
        f"Double bridge perturbation resulted in the same tour for n={n}."


def test_alternate_step_finds_improvement(simple_tsp_setup):
    """
    Tests if `alternate_step` can find an improvement on a non-optimal tour
    from the `simple_tsp_setup`.
    """
    _coords, dist_matrix, initial_tour_obj, neighbors, \
        _initial_cost, _lk_optimal_cost, _lk_optimal_order = simple_tsp_setup

    tour_to_test_order = initial_tour_obj.get_tour()
    tour_to_test = Tour(tour_to_test_order, dist_matrix)
    initial_test_cost = tour_to_test.cost

    original_lk_config = LK_CONFIG.copy()
    deadline = time.time() + 5
    improvement_found = False
    best_improving_sequence = None

    for base_node_idx in range(tour_to_test.n):
        current_call_tour = Tour(tour_to_test_order, dist_matrix)
        found, sequence = alternate_step(
            base_node=current_call_tour.order[base_node_idx],
            tour=current_call_tour, D=dist_matrix, neigh=neighbors, deadline=deadline
        )
        if found and sequence:
            temp_tour = Tour(tour_to_test_order, dist_matrix)
            cost_before_apply = temp_tour.cost
            for f_start, f_end in sequence:
                temp_tour.flip_and_update_cost(f_start, f_end, dist_matrix)
            if temp_tour.cost < cost_before_apply - 1e-9:  # Strict improvement
                improvement_found = True
                best_improving_sequence = sequence
                break

    assert improvement_found, "alternate_step failed to find an improvement."
    if best_improving_sequence:  # Should be true if improvement_found
        final_tour = Tour(tour_to_test_order, dist_matrix)
        for f_start, f_end in best_improving_sequence:
            final_tour.flip_and_update_cost(f_start, f_end, dist_matrix)
        assert final_tour.cost < initial_test_cost - 1e-9, \
            "Applying sequence from alternate_step did not reduce cost."

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_alternate_step_no_improvement_on_optimal(simple_tsp_setup):
    """
    Tests that `alternate_step` does not find an improvement on an
    already optimal tour from `simple_tsp_setup`.
    """
    _coords, dist_matrix, _initial_tour_obj, neighbors, \
        _initial_cost, _lk_optimal_cost, lk_optimal_order = simple_tsp_setup

    optimal_tour = Tour(lk_optimal_order, dist_matrix)
    deadline = time.time() + 5
    original_lk_config = LK_CONFIG.copy()

    for base_node_idx in range(optimal_tour.n):
        current_call_tour = Tour(lk_optimal_order, dist_matrix)
        cost_before_call = current_call_tour.cost
        found, sequence = alternate_step(
            base_node=current_call_tour.order[base_node_idx],
            tour=current_call_tour, D=dist_matrix, neigh=neighbors, deadline=deadline
        )
        if found and sequence:
            temp_tour = Tour(lk_optimal_order, dist_matrix)
            for f_start, f_end in sequence:
                temp_tour.flip_and_update_cost(f_start, f_end, dist_matrix)
            assert temp_tour.cost >= cost_before_call - 1e-9, \
                (f"alternate_step found sequence {sequence} from base "
                 f"{current_call_tour.order[base_node_idx]} that 'improved' an optimal tour. "
                 f"New cost: {temp_tour.cost}, Optimal: {cost_before_call}")

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_alternate_step_restrictive_breadth(simple_tsp_setup):
    """
    Tests that `alternate_step` with zero breadth settings (BREADTH_A/B/D = 0)
    does not find an improvement and does not crash.
    """
    _coords, dist_matrix, initial_tour_obj, neighbors, \
        _initial_cost, _lk_optimal_cost, _lk_optimal_order = simple_tsp_setup

    test_tour = Tour(initial_tour_obj.get_tour(), dist_matrix)
    deadline = time.time() + 5
    original_lk_config = LK_CONFIG.copy()
    restrictive_config = original_lk_config.copy()
    restrictive_config.update({"BREADTH_A": 0, "BREADTH_B": 0, "BREADTH_D": 0})
    LK_CONFIG.clear()
    LK_CONFIG.update(restrictive_config)

    for base_node_idx in range(test_tour.n):
        current_call_tour = Tour(test_tour.get_tour(), dist_matrix)
        found, sequence = alternate_step(
            base_node=current_call_tour.order[base_node_idx],
            tour=current_call_tour, D=dist_matrix, neigh=neighbors, deadline=deadline
        )
        assert not (found and sequence), \
            (f"alternate_step found improvement {sequence} from base "
             f"{current_call_tour.order[base_node_idx]} with zero breadth settings.")

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


@pytest.mark.parametrize("instance_base_name", ["rand4"])
def test_chained_lk_terminates_at_known_optimum(instance_base_name):
    """
    Tests if `chained_lin_kernighan` terminates early when the
    `known_optimal_length` is reached for a given TSP instance.
    """
    tsp_file = VERIFICATION_RANDOM_PATH / f"{instance_base_name}.tsp"
    opt_tour_file = VERIFICATION_RANDOM_PATH / f"{instance_base_name}.opt.tour"

    assert tsp_file.exists(), f"TSP file not found: {tsp_file}"
    assert opt_tour_file.exists(), f"Optimal tour file not found: {opt_tour_file}"

    coords = read_tsp_file(str(tsp_file))
    assert coords is not None and coords.size > 0, f"Could not load coordinates from {tsp_file}"

    optimal_tour_nodes = read_opt_tour(str(opt_tour_file))
    assert optimal_tour_nodes is not None, f"Could not load optimal tour from {opt_tour_file}"

    if coords.shape[0] < 4:
        pytest.skip(f"Skipping {instance_base_name} as it has fewer than 4 nodes ({coords.shape[0]}).")
        return
    if len(optimal_tour_nodes) != coords.shape[0]:
        pytest.fail(
            f"Optimal tour node count ({len(optimal_tour_nodes)}) does not match "
            f"coordinate count ({coords.shape[0]}) for {instance_base_name}."
        )

    dist_matrix = build_distance_matrix(coords)
    calculated_known_opt_len = 0.0
    for i in range(len(optimal_tour_nodes)):
        u_node = optimal_tour_nodes[i]
        v_node = optimal_tour_nodes[(i + 1) % len(optimal_tour_nodes)]
        calculated_known_opt_len += dist_matrix[u_node, v_node]

    initial_tour_order = list(range(coords.shape[0]))

    original_lk_config = LK_CONFIG.copy()
    # Use a long time limit to ensure termination is due to finding optimum, not timeout
    long_time_limit_seconds = 15.0  # Increased for potentially harder small instances
    temp_config = original_lk_config.copy()
    # LK_CONFIG["TIME_LIMIT"] is not a standard LK_CONFIG key used by chained_lin_kernighan directly
    # The time_limit_seconds parameter to chained_lin_kernighan is what matters.
    LK_CONFIG.clear()
    LK_CONFIG.update(temp_config)  # Apply any other desired default LK settings

    start_time = time.time()
    final_tour_order, final_cost = chained_lin_kernighan(
        coords, initial_tour_order,
        known_optimal_length=calculated_known_opt_len,
        time_limit_seconds=long_time_limit_seconds
    )
    end_time = time.time()
    execution_time = end_time - start_time

    assert final_cost == pytest.approx(calculated_known_opt_len), \
        (f"Chained LK for {instance_base_name} did not reach known optimal length "
         f"{calculated_known_opt_len:.4f}. Got {final_cost:.4f}.")

    # Heuristic check for early termination: execution time should be noticeably
    # less than the long_time_limit if early exit due to known_optimal_length occurred.
    # This is a soft check as very easy instances might solve quickly anyway.
    # For rand4 (4 nodes), it will be extremely fast.
    if coords.shape[0] > 4:  # Only apply this heuristic for non-trivial cases
        assert execution_time < (long_time_limit_seconds * 0.75), \
            (f"Chained LK for {instance_base_name} took {execution_time:.3f}s. "
             f"This might be too long if early exit due to known_optimal_length "
             f"(at {calculated_known_opt_len:.4f}) was expected relative to "
             f"time_limit_seconds={long_time_limit_seconds}s.")

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)
