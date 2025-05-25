import pytest
import time
import numpy as np
from lin_kernighan_tsp_solver.lin_kernighan_tsp_solver import (
    Tour,
    step,
    lk_search,
    lin_kernighan,
    LK_CONFIG,
    FLOAT_COMPARISON_TOLERANCE,
    build_distance_matrix,
    delaunay_neighbors
)

# simple_tsp_setup fixture is automatically available from conftest.py


def test_step_finds_simple_2_opt(simple_tsp_setup):
    _coords, dist_matrix, initial_tour_obj, neighbors, initial_cost, _lk_optimal_cost, _lk_optimal_order = simple_tsp_setup
    # This test focuses on a 2-opt step, not necessarily reaching the full LK optimum in one go.
    # Expected outcome of flipping (2, 3) on initial tour [0, 1, 2, 3, 4] is [0, 1, 3, 2, 4]
    expected_tour_after_2_opt_nodes = [0, 1, 3, 2, 4]
    expected_cost_after_2_opt = Tour(expected_tour_after_2_opt_nodes, dist_matrix).cost

    original_lk_config = LK_CONFIG.copy()
    # Ensure FLOAT_COMPARISON_TOLERANCE is available if not already imported in this file
    # from lin_kernighan_tsp_solver.lin_kernighan_tsp_solver import FLOAT_COMPARISON_TOLERANCE
    # (It seems to be imported at the top level of test_lk_core.py already)

    LK_CONFIG["MAX_LEVEL"] = 1
    LK_CONFIG["BREADTH"] = [1] * LK_CONFIG["MAX_LEVEL"]

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

            if best_flip_sequence_found is None:  # Store the first improvement sequence
                best_flip_sequence_found = flip_sequence

            temp_tour_check = Tour(initial_tour_obj.get_tour(), dist_matrix)
            for fs, fe in flip_sequence: temp_tour_check.flip_and_update_cost(fs, fe, dist_matrix)

            # Check if this particular improvement matches our expected 2-opt
            if temp_tour_check.get_tour() == expected_tour_after_2_opt_nodes and \
               abs(temp_tour_check.cost - expected_cost_after_2_opt) < FLOAT_COMPARISON_TOLERANCE * 100:  # Using existing tolerance logic
                best_flip_sequence_found = flip_sequence  # Ensure this is the one we validate against
                found_the_specific_2_opt_move = True
                break

    assert improved_overall, "Step function should have found an improvement."
    assert found_the_specific_2_opt_move, f"Step did not find the expected 2-opt move leading to {expected_tour_after_2_opt_nodes}. Best sequence found: {best_flip_sequence_found}"

    if best_flip_sequence_found:  # This check is now somewhat redundant due to the assert above, but fine
        assert len(best_flip_sequence_found) == 1, "Expected a single flip for this 2-opt."

        expected_flip_nodes = tuple(sorted((2, 3)))
        actual_flip_nodes = tuple(sorted(best_flip_sequence_found[0]))
        assert actual_flip_nodes == expected_flip_nodes, \
            f"Expected flip involving nodes {expected_flip_nodes}, but got {actual_flip_nodes}"

        final_tour_check = Tour(initial_tour_obj.get_tour(), dist_matrix)
        for fs, fe in best_flip_sequence_found: final_tour_check.flip_and_update_cost(fs, fe, dist_matrix)
        assert final_tour_check.cost == pytest.approx(expected_cost_after_2_opt)
        assert final_tour_check.get_tour() == expected_tour_after_2_opt_nodes

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_step_no_improvement_on_optimal_tour(simple_tsp_setup):
    _coords, dist_matrix, initial_tour_obj_from_fixture, neighbors, \
        _initial_cost_fixture, _optimal_cost_fixture_setup, _optimal_order_fixture_setup = simple_tsp_setup

    current_processing_tour = Tour(initial_tour_obj_from_fixture.get_tour(), dist_matrix)
    original_test_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 1
    LK_CONFIG["BREADTH"] = [1] * LK_CONFIG["MAX_LEVEL"]
    max_2opt_passes = current_processing_tour.n * 5

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
        if not made_improvement_in_this_pass: break
    else:
        print(f"Warning: 2-opt stabilization loop reached max_passes in {test_step_no_improvement_on_optimal_tour.__name__}.")

    truly_2_optimal_tour = current_processing_tour
    cost_of_2_optimal_tour = truly_2_optimal_tour.cost
    order_of_2_optimal_tour = truly_2_optimal_tour.get_tour()

    LK_CONFIG.clear()
    LK_CONFIG.update(original_test_lk_config)
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
            for f_s, f_e in flip_sequence_final: temp_tour_for_cost_check.flip_and_update_cost(f_s, f_e, dist_matrix)
            final_tour_cost_after_step_if_improved = temp_tour_for_cost_check.cost
        assert not improved_final, \
            (f"Step found improvement from base {base_node_val_check} on 2-optimal tour (cost {cost_of_2_optimal_tour:.10f}). "
             f"Seq: {flip_sequence_final}. New cost: {final_tour_cost_after_step_if_improved:.10f}")
        if not improved_final:
            assert tour_arg_for_final_step.get_tour() == order_of_2_optimal_tour
            assert tour_arg_for_final_step.cost == pytest.approx(cost_of_2_optimal_tour)
    LK_CONFIG.clear()
    LK_CONFIG.update(original_test_lk_config)


def test_lk_search_finds_optimum_for_simple_tsp(simple_tsp_setup):
    _coords, dist_matrix, initial_tour_obj, neighbors, \
        initial_cost_fixture, optimal_cost_fixture, optimal_order_fixture = simple_tsp_setup
    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 5
    LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]  # Corrected: LK_CONFIG
    LK_CONFIG["BREADTH_A"], LK_CONFIG["BREADTH_B"], LK_CONFIG["BREADTH_D"] = 5, 5, 1

    current_tour = Tour(initial_tour_obj.get_tour(), dist_matrix)
    deadline = time.time() + 20
    overall_improved_flag, made_change_in_iteration = False, True
    max_iterations, iter_count = current_tour.n * 2, 0

    while made_change_in_iteration and iter_count < max_iterations:
        if abs(current_tour.cost - optimal_cost_fixture) < FLOAT_COMPARISON_TOLERANCE:
            overall_improved_flag = True; break
        made_change_in_iteration = False; iter_count += 1
        nodes_to_try_as_base = list(current_tour.order)
        for start_node_for_search in nodes_to_try_as_base:
            tour_for_lk_search_call = Tour(current_tour.get_tour(), dist_matrix)
            improving_sequence = lk_search(start_node_for_search, tour_for_lk_search_call, dist_matrix, neighbors, deadline)
            if improving_sequence:
                cost_before_flips = current_tour.cost
                for fs, fe in improving_sequence: current_tour.flip_and_update_cost(fs, fe, dist_matrix)
                if current_tour.cost < cost_before_flips - FLOAT_COMPARISON_TOLERANCE:
                    overall_improved_flag, made_change_in_iteration = True, True; break
                else:
                    current_tour = Tour(nodes_to_try_as_base, dist_matrix)  # Revert
                    made_change_in_iteration = False; break
    assert overall_improved_flag or abs(initial_cost_fixture - optimal_cost_fixture) < FLOAT_COMPARISON_TOLERANCE, \
        f"lk_search failed to improve. Initial: {initial_cost_fixture}, Optimal: {optimal_cost_fixture}, Final: {current_tour.cost}"
    assert current_tour.cost == pytest.approx(optimal_cost_fixture)
    assert current_tour.get_tour() == optimal_order_fixture
    LK_CONFIG.clear(); LK_CONFIG.update(original_lk_config)


def test_lk_search_no_improvement_on_optimal_tour(simple_tsp_setup):
    coords, dist_matrix, initial_tour_obj_from_fixture, neighbors, \
        _initial_cost, _optimal_cost_fixture, _optimal_order_fixture = simple_tsp_setup
    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 5
    LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]
    LK_CONFIG["BREADTH_A"], LK_CONFIG["BREADTH_B"], LK_CONFIG["BREADTH_D"] = 5, 5, 1
    deadline_for_lk = time.time() + 15

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
            f"lk_search found improvement from node {start_node_for_search} on converged tour. Seq: {flip_sequence_final}"
        assert tour_copy_for_lk_search.get_tour() == converged_tour_obj.get_tour()
        assert tour_copy_for_lk_search.cost == pytest.approx(converged_cost)
    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


# --- Utility Function Tests ---


def test_build_distance_matrix_simple_cases():
    """Tests build_distance_matrix with simple coordinate sets."""
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
    # Check symmetry and diagonal
    for i in range(3):
        assert dist_matrix2[i, i] == pytest.approx(0.0)
        for j in range(i + 1, 3):
            assert dist_matrix2[i, j] == pytest.approx(dist_matrix2[j, i])

    # Case 3: Points forming a square
    coords3 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    dist_matrix3 = build_distance_matrix(coords3)
    assert dist_matrix3.shape == (4, 4)
    assert dist_matrix3[0, 1] == pytest.approx(1.0)  # (0, 0) to (1, 0)
    assert dist_matrix3[0, 2] == pytest.approx(1.0)  # (0, 0) to (0, 1)
    assert dist_matrix3[0, 3] == pytest.approx(np.sqrt(2))  # (0, 0) to (1, 1)
    assert dist_matrix3[1, 2] == pytest.approx(np.sqrt(2))  # (1, 0) to (0, 1)
    assert dist_matrix3[1, 3] == pytest.approx(1.0)  # (1, 0) to (1, 1)
    assert dist_matrix3[2, 3] == pytest.approx(1.0)  # (0, 1) to (1, 1)
    for i in range(4):
        assert dist_matrix3[i, i] == pytest.approx(0.0)
        for j in range(i + 1, 4):
            assert dist_matrix3[i, j] == pytest.approx(dist_matrix3[j, i])


def test_build_distance_matrix_edge_cases():
    """Tests build_distance_matrix with edge case inputs."""
    # Case 1: Empty coordinates (input shape (0,) or (0,d))
    coords_empty_1d = np.array([])  # shape (0,)
    dist_matrix_empty_1d = build_distance_matrix(coords_empty_1d)
    assert dist_matrix_empty_1d.shape == (0, 0)

    coords_empty_2d = np.empty((0, 2))  # shape (0, 2)
    dist_matrix_empty_2d = build_distance_matrix(coords_empty_2d)
    assert dist_matrix_empty_2d.shape == (0, 0)

    # Case 2: Single point
    coords_single = np.array([[10, 20]])
    dist_matrix_single = build_distance_matrix(coords_single)
    assert dist_matrix_single.shape == (1, 1)
    assert dist_matrix_single[0, 0] == pytest.approx(0.0)


def test_delaunay_neighbors_few_points():
    """Tests delaunay_neighbors with fewer than 3 points."""
    # Case 1: 0 points
    coords0 = np.empty((0, 2))
    neighbors0 = delaunay_neighbors(coords0)
    assert neighbors0 == []

    # Case 2: 1 point
    coords1 = np.array([[0, 0]])
    neighbors1 = delaunay_neighbors(coords1)
    assert len(neighbors1) == 1
    assert neighbors1[0] == []  # Node 0 has no other neighbors

    # Case 3: 2 points
    coords2 = np.array([[0, 0], [1, 1]])
    neighbors2 = delaunay_neighbors(coords2)
    assert len(neighbors2) == 2
    assert neighbors2[0] == [1]  # Node 0 is neighbor with Node 1
    assert neighbors2[1] == [0]  # Node 1 is neighbor with Node 0


def test_delaunay_neighbors_triangle():
    """Tests delaunay_neighbors with 3 points (a single triangle)."""
    coords = np.array([[0, 0], [1, 0], [0, 1]])  # Forms a triangle
    neighbors = delaunay_neighbors(coords)
    assert len(neighbors) == 3
    # Each node should be connected to the other two
    expected_neighbors = [
        [1, 2],  # Neighbors of node 0
        [0, 2],  # Neighbors of node 1
        [0, 1]  # Neighbors of node 2
    ]
    for i in range(3):
        assert sorted(neighbors[i]) == sorted(expected_neighbors[i]), f"Neighbors for node {i} incorrect"


def test_delaunay_neighbors_square():
    """Tests delaunay_neighbors with 4 points forming a square."""
    # For a square, Delaunay typically gives the 4 outer edges and one diagonal.
    # The choice of diagonal can depend on slight perturbations or library implementation.
    # We'll test for a common outcome.
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    neighbors = delaunay_neighbors(coords)
    assert len(neighbors) == 4

    # Check general properties: sorted, no self-loops
    for i in range(len(coords)):
        assert neighbors[i] == sorted(neighbors[i]), f"Neighbors for node {i} not sorted"
        assert i not in neighbors[i], f"Node {i} is its own neighbor"
        for neighbor_node in neighbors[i]:
            assert i in neighbors[neighbor_node], f"Asymmetric neighborhood: {i} -> {neighbor_node} but not vice-versa"

    # Expected connections for a square (0-1-2-3-0)
    # Node 0: connected to 1, 3. One diagonal (e.g., to 2)
    # Node 1: connected to 0, 2. One diagonal (e.g., to 3)
    # Node 2: connected to 1, 3. One diagonal (e.g., to 0)
    # Node 3: connected to 0, 2. One diagonal (e.g., to 1)
    # The exact diagonal depends on the triangulation.
    # Let's check that each node has degree 3 (2 square edges + 1 diagonal)
    # or degree 2 if it's a very flat quadrilateral that only forms 2 triangles along one diagonal.
    # For a unit square, it's typically 2 triangles, so most nodes have degree 3.
    # Example: diagonal (0, 2) -> neighbors[0]=[1, 2, 3], neighbors[1]=[0, 2], neighbors[2]=[0, 1, 3], neighbors[3]=[0, 2]
    # Example: diagonal (1, 3) -> neighbors[0]=[1, 3], neighbors[1]=[0, 2, 3], neighbors[2]=[1, 3], neighbors[3]=[0, 1, 2]

    # A robust check: ensure all perimeter edges exist
    assert 1 in neighbors[0] and 0 in neighbors[1]  # Edge 0-1
    assert 2 in neighbors[1] and 1 in neighbors[2]  # Edge 1-2
    assert 3 in neighbors[2] and 2 in neighbors[3]  # Edge 2-3
    assert 0 in neighbors[3] and 3 in neighbors[0]  # Edge 3-0

    # Check that the sum of degrees is 2 * num_edges.
    # For 4 points, Delaunay gives 2*N-2-k triangles (k=num hull edges), so 2*4-2-4 = 2 triangles.
    # Number of edges in 2 triangles = 3 edges/triangle * 2 triangles - shared_edges.
    # If 2 triangles share 1 edge (the diagonal), total edges = 3+3-1 = 5.
    # Sum of degrees = 2 * 5 = 10.
    # If 2 triangles share 0 edges (not possible for convex hull), total edges = 6. Sum of degrees = 12.
    # For a convex quad, it's 5 edges.
    total_degree = sum(len(n_list) for n_list in neighbors)
    assert total_degree == 2 * 5  # 5 edges in a typical Delaunay of a convex quadrilateral


def test_delaunay_neighbors_from_fixture(simple_tsp_setup):
    """Tests delaunay_neighbors using the coordinates from simple_tsp_setup."""
    coords, _, _, expected_neighbors_from_fixture, _, _, _ = simple_tsp_setup

    # Re-calculate neighbors to test the function directly
    calculated_neighbors = delaunay_neighbors(coords)

    assert len(calculated_neighbors) == len(coords)
    for i in range(len(coords)):
        assert calculated_neighbors[i] == sorted(calculated_neighbors[i]), f"Neighbors for node {i} do not match fixture"
        assert i not in calculated_neighbors[i], f"Node {i} is its own neighbor"
        # Compare with the neighbors provided by the fixture (which should be correct)
        assert calculated_neighbors[i] == expected_neighbors_from_fixture[i], \
            f"Calculated neighbors for node {i} do not match fixture"


def test_lin_kernighan_respects_overall_deadline(simple_tsp_setup):
    """
    Tests that lin_kernighan respects the overall_deadline and terminates.
    """
    coords, dist_matrix, initial_tour_obj, neighbors, \
        _initial_cost, _optimal_cost, _optimal_order = simple_tsp_setup

    original_lk_config = LK_CONFIG.copy()
    # Use default LK_CONFIG settings for this test, focusing on deadline

    # Set a very short deadline (e.g., 1 millisecond)
    # This should be short enough to likely interrupt the search.
    short_deadline = time.time() + 0.001 

    start_time = time.time()
    # Run lin_kernighan with the short deadline
    final_tour_obj, final_cost = lin_kernighan(
        coords,
        initial_tour_obj.get_tour(),
        dist_matrix,
        neighbors,
        deadline=short_deadline  # Correct: uses 'deadline'
    )
    end_time = time.time()

    execution_time = end_time - start_time

    # Assert that the execution time was reasonably short,
    # indicating the deadline was likely hit.
    # Allow for some overhead, e.g., up to 0.1 seconds.
    # This threshold might need adjustment based on system speed.
    assert execution_time < 0.1, \
        f"lin_kernighan took too long ({execution_time:.4f}s), deadline likely not respected."

    # Assert that some valid tour is returned
    assert final_tour_obj is not None, "lin_kernighan should return a Tour object."
    assert isinstance(final_tour_obj, Tour), "Returned object is not a Tour."
    assert final_tour_obj.n == initial_tour_obj.n, "Returned tour has incorrect number of nodes."
    assert final_cost is not None, "lin_kernighan should return a cost."
    assert final_cost == pytest.approx(final_tour_obj.cost), "Returned cost does not match tour's cost."
    
    # It's hard to assert non-optimality deterministically here,
    # as for a tiny problem and a lucky very short deadline, it might find the optimum.
    # The main check is timely termination.

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_lin_kernighan_improves_to_optimum_simple_case(simple_tsp_setup):
    """
    Tests that lin_kernighan can improve a non-optimal initial tour to the
    known optimal solution for the simple_tsp_setup instance.
    """
    coords, dist_matrix, initial_tour_obj, neighbors, \
        initial_cost, lk_optimal_cost, lk_optimal_order = simple_tsp_setup

    original_lk_config = LK_CONFIG.copy()
    # Use default LK_CONFIG or specific settings if needed for convergence on this simple case
    # For simple_tsp_setup, defaults should be fine.
    # LK_CONFIG["MAX_LEVEL"] = 5  # Example if defaults weren't enough
    # LK_CONFIG["BREADTH"] = [2] * LK_CONFIG["MAX_LEVEL"]
    # LK_CONFIG["BREADTH_A"], LK_CONFIG["BREADTH_B"], LK_CONFIG["BREADTH_D"] = 5,5,1


    # Set a reasonable deadline, long enough for this small problem to solve
    deadline = time.time() + 10  # 10 seconds, more than enough

    final_tour_obj, final_cost = lin_kernighan(
        coords,
        initial_tour_obj.get_tour(),  # Start with the non-optimal initial tour
        dist_matrix,
        neighbors,
        deadline=deadline
    )

    assert final_tour_obj is not None, "lin_kernighan should return a Tour object."
    assert final_cost is not None, "lin_kernighan should return a cost."

    # Check if the cost matches the known optimal cost
    assert final_cost == pytest.approx(lk_optimal_cost), \
        f"Expected optimal cost {lk_optimal_cost}, but got {final_cost}"

    # Check if the tour order matches the known optimal order
    # Normalizing the returned tour to start with node 0 for consistent comparison
    returned_tour_normalized = final_tour_obj.get_tour()  # get_tour() normalizes
    assert returned_tour_normalized == lk_optimal_order, \
        f"Expected optimal tour {lk_optimal_order}, but got {returned_tour_normalized}"
    
    assert final_tour_obj.cost == pytest.approx(lk_optimal_cost), \
        "Tour object's internal cost does not match the returned optimal cost."

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_lin_kernighan_on_optimal_tour_simple_case(simple_tsp_setup):
    """
    Tests that lin_kernighan, when given an already optimal tour for the
    simple_tsp_setup instance, returns it without finding further 'improvements'.
    """
    coords, dist_matrix, _initial_tour_obj, neighbors, \
        _initial_cost, lk_optimal_cost, lk_optimal_order = simple_tsp_setup

    original_lk_config = LK_CONFIG.copy()
    # Use default LK_CONFIG settings.

    # Create a Tour object with the known optimal order
    optimal_initial_tour_obj = Tour(lk_optimal_order, dist_matrix)
    assert optimal_initial_tour_obj.cost == pytest.approx(lk_optimal_cost), \
        "Optimal tour object for setup has unexpected cost."

    # Set a reasonable deadline
    deadline = time.time() + 10  # 10 seconds

    final_tour_obj, final_cost = lin_kernighan(
        coords,
        optimal_initial_tour_obj.get_tour(),  # Start with the known optimal tour
        dist_matrix,
        neighbors,
        deadline=deadline
    )

    assert final_tour_obj is not None, "lin_kernighan should return a Tour object."
    assert final_cost is not None, "lin_kernighan should return a cost."

    # Check if the cost matches the known optimal cost
    assert final_cost == pytest.approx(lk_optimal_cost), \
        f"Expected optimal cost {lk_optimal_cost}, but got {final_cost} when starting optimal."

    # Check if the tour order matches the known optimal order
    returned_tour_normalized = final_tour_obj.get_tour()  # get_tour() normalizes
    assert returned_tour_normalized == lk_optimal_order, \
        f"Expected optimal tour {lk_optimal_order}, but got {returned_tour_normalized} when starting optimal."

    assert final_tour_obj.cost == pytest.approx(lk_optimal_cost), \
        "Tour object's internal cost does not match the returned optimal cost."

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)

# ... (rest of test_lk_core.py)
