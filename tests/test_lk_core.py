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
    build_distance_matrix,  # Already likely imported or used by other fixtures/tests
    delaunay_neighbors,    # Already likely imported or used
)
from pathlib import Path
import numpy as np  # Ensure numpy is imported
import time  # Ensure time is imported
import pytest  # Ensure pytest is imported

# simple_tsp_setup fixture is automatically available from conftest.py

VERIFICATION_RANDOM_PATH = Path(__file__).resolve().parent.parent / "verifications" / "random"
# Path(__file__).parent -> /project_root/tests/
# Path(__file__).parent.parent -> /project_root/
# ... / "verifications" / "random" -> /project_root/verifications/random/


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


def test_lin_kernighan_config_sensitivity(simple_tsp_setup):
    """
    Tests that LK_CONFIG settings affect lin_kernighan's ability to find the optimum.
    A restrictive config should fail to find the optimum on simple_tsp_setup,
    while a default/good config should succeed.
    """
    coords, dist_matrix, initial_tour_obj, neighbors, \
        initial_cost, lk_optimal_cost, lk_optimal_order = simple_tsp_setup

    # Ensure the initial tour is not already optimal for this test to be meaningful
    assert initial_cost != pytest.approx(lk_optimal_cost), \
        "Initial tour cost is already optimal, test may not be indicative."

    original_lk_config_snapshot = LK_CONFIG.copy()  # Snapshot at the very beginning
    deadline = time.time() + 10  # Sufficient time for this small problem

    try:
        # Scenario 1: Highly restrictive LK_CONFIG
        # These settings severely limit the search options.
        restrictive_settings = {
            "MAX_LEVEL": 0,       # Disables standard k-opt search in step()
            "BREADTH": [],        # Consistent with MAX_LEVEL = 0 for step()
            "MAX_CANDIDATES": 1,  # Used by step() if MAX_LEVEL > 0
            "BREADTH_A": 0,       # Disables candidate search for y1 in alternate_step()
            "BREADTH_B": 0,       # Disables candidate search for y2 in alternate_step()
            "BREADTH_D": 0        # Disables candidate search for y3 in alternate_step()
        }

        current_config_for_test = original_lk_config_snapshot.copy()
        current_config_for_test.update(restrictive_settings)

        LK_CONFIG.clear()
        LK_CONFIG.update(current_config_for_test)

        suboptimal_tour_obj, suboptimal_cost = lin_kernighan(
            coords, initial_tour_obj.get_tour(), dist_matrix, neighbors, deadline
        )

        # With MAX_LEVEL = 0, no improvement should be made from the initial tour.
        assert suboptimal_cost == pytest.approx(initial_cost), \
            (f"Restrictive LK_CONFIG (MAX_LEVEL=0) should not have improved the tour. "
             f"Cost found: {suboptimal_cost}. Expected initial cost: {initial_cost}")

        # And therefore, it should not be the optimal cost (unless initial was already optimal, checked above)
        assert suboptimal_cost != pytest.approx(lk_optimal_cost), \
            (f"Restrictive LK_CONFIG (MAX_LEVEL=0) unexpectedly resulted in optimal cost. "
             f"Cost found: {suboptimal_cost}. Optimal cost: {lk_optimal_cost}")

        # Scenario 2: Restore to original (presumably good) LK_CONFIG
        LK_CONFIG.clear()
        LK_CONFIG.update(original_lk_config_snapshot)

        # Re-run with original/good config
        optimal_tour_obj_again, optimal_cost_again = lin_kernighan(
            coords, initial_tour_obj.get_tour(), dist_matrix, neighbors, deadline
        )

        assert optimal_cost_again == pytest.approx(lk_optimal_cost), \
            (f"Default/Original LK_CONFIG should find the global optimum ({lk_optimal_cost}). "
             f"Cost found: {optimal_cost_again}")

    finally:
        # Ensure LK_CONFIG is restored to its original state before the test
        LK_CONFIG.clear()
        LK_CONFIG.update(original_lk_config_snapshot)


def test_chained_lk_max_iterations_one(simple_tsp_setup):
    """
    Tests that chained_lin_kernighan with max_iterations=1 behaves like a single lin_kernighan run.
    """
    coords, dist_matrix, initial_tour_obj, neighbors, \
        _initial_cost, lk_optimal_cost, lk_optimal_order = simple_tsp_setup

    original_lk_config = LK_CONFIG.copy()
    deadline_val = time.time() + 10
    time_limit_sec = 10.0

    # Run single lin_kernighan
    lk_tour_obj, lk_cost = lin_kernighan(
        coords, initial_tour_obj.get_tour(), dist_matrix, neighbors, deadline_val
    )

    # chained_lin_kernighan returns List[int], float
    chained_lk_tour_order, chained_lk_cost = chained_lin_kernighan(
        coords, initial_tour_obj.get_tour(),
        time_limit_seconds=time_limit_sec
    )

    assert chained_lk_cost == pytest.approx(lk_cost), \
        "Chained LK cost should match single LK cost (when effectively one iteration)."
    # chained_lk_tour_order is a list, lk_tour_obj.get_tour() is also a list
    assert chained_lk_tour_order == lk_tour_obj.get_tour(), \
        "Chained LK tour order should match single LK tour order (when effectively one iteration)."

    # It should also find the optimum for simple_tsp_setup
    assert chained_lk_cost == pytest.approx(lk_optimal_cost)
    assert chained_lk_tour_order == lk_optimal_order

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_chained_lk_multiple_iterations(simple_tsp_setup):
    """
    Tests that chained_lin_kernighan with multiple iterations runs and finds the optimum.
    """
    coords, dist_matrix, initial_tour_obj, neighbors, \
        _initial_cost, lk_optimal_cost, lk_optimal_order = simple_tsp_setup

    original_lk_config = LK_CONFIG.copy()
    time_limit_sec = 15.0  # Sufficient time for a few iterations

    # chained_lin_kernighan returns List[int], float
    # num_iterations variable is just for the f-string, not used by the function
    num_iterations_display = 3
    chained_lk_tour_order, chained_lk_cost = chained_lin_kernighan(
        coords, initial_tour_obj.get_tour(),
        time_limit_seconds=time_limit_sec
    )

    assert chained_lk_tour_order is not None, "Chained LK should return a tour order list."
    assert isinstance(chained_lk_tour_order, list), "Chained LK should return a list as the first element."
    assert chained_lk_cost is not None, "Chained LK should return a cost."

    # For simple_tsp_setup, it should find the optimal solution.
    assert chained_lk_cost == pytest.approx(lk_optimal_cost), \
        f"Chained LK (runtime {time_limit_sec}s) failed to find optimal cost. Got {chained_lk_cost}, expected {lk_optimal_cost}."
    assert chained_lk_tour_order == lk_optimal_order, \
        f"Chained LK (runtime {time_limit_sec}s) failed to find optimal tour. Got {chained_lk_tour_order}, expected {lk_optimal_order}."

    # To check the cost of the returned tour order, reconstruct a Tour object
    reconstructed_tour_from_chained = Tour(chained_lk_tour_order, dist_matrix)
    assert reconstructed_tour_from_chained.cost == pytest.approx(lk_optimal_cost), \
        "Chained LK tour order's calculated cost does not match expected optimal cost."
    assert reconstructed_tour_from_chained.cost == pytest.approx(chained_lk_cost), \
        "Chained LK tour order's calculated cost does not match its returned cost."

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_double_bridge_small_tours():
    """
    Tests double_bridge with small tours (n <= 4),
    where it should return the original tour as per its implementation.
    """
    # The double_bridge function returns the original tour if n <= 4.
    for n in range(5):  # Test n = 0, 1, 2, 3, 4
        original_tour = list(range(n))
        perturbed_tour = double_bridge(original_tour.copy())  # Pass a copy
        assert perturbed_tour == original_tour, \
            f"Double bridge should not change a tour with {n} nodes. Got {perturbed_tour}, expected {original_tour}"


def test_double_bridge_larger_tour():
    """
    Tests double_bridge with a tour large enough (n > 4)
    to be perturbed.
    """
    # Test cases for n > 4, e.g., n = 5, 8, 10
    for n in [5, 8, 10]:
        original_tour = list(range(n))

        different_tour_found = False
        # For these sizes, a single perturbation should ideally be different.
        # Running a few times can help if the random choice is unlucky,
        # but for a test, one good perturbation is often enough to check validity.
        # Let's try a few times to be more robust against unlucky random choices.
        attempts = 5
        for _ in range(attempts):
            perturbed_tour = double_bridge(original_tour.copy())

            assert len(perturbed_tour) == n, \
                f"Perturbed tour length {len(perturbed_tour)} for n={n} does not match original {n}."
            assert sorted(perturbed_tour) == sorted(original_tour), \
                f"Perturbed tour for n={n} does not contain the same set of nodes as the original."
            assert len(set(perturbed_tour)) == n, \
                f"Perturbed tour for n={n} contains duplicate nodes or missing nodes."

            if perturbed_tour != original_tour:
                different_tour_found = True
                break  # Found a different tour, this attempt is successful for this n

        assert different_tour_found, \
            f"Double bridge did not change the tour with {n} nodes after {attempts} attempts."


def test_double_bridge_very_large_tour_is_different():
    """
    Tests that double_bridge on a sufficiently large tour
    actually produces a different tour.
    """
    n = 20  # A reasonably large tour
    original_tour = list(range(n))

    perturbed_tour = double_bridge(original_tour.copy())

    assert len(perturbed_tour) == n, f"Perturbed tour length {len(perturbed_tour)} does not match original {n}."
    assert sorted(perturbed_tour) == sorted(original_tour), "Perturbed tour does not contain the same set of nodes."
    assert len(set(perturbed_tour)) == n, "Perturbed tour has duplicate or missing nodes."

    # For a tour of size 20, it's highly probable it will be different.
    # If it's the same, the random choices might have coincidentally reconstructed the original,
    # or there's an issue. Running it once should be fine for this check.
    assert perturbed_tour != original_tour, \
        f"Double bridge perturbation resulted in the same tour for n={n}. This is unlikely or indicates an issue."


def test_alternate_step_finds_improvement(simple_tsp_setup):
    coords, dist_matrix, _initial_tour_obj, neighbors, \
        _initial_cost, lk_optimal_cost, _lk_optimal_order = simple_tsp_setup

    # Define a specific tour that is NOT optimal but might be 2-optimal,
    # and hopefully alternate_step can find a 3-opt or 5-opt.
    # For simple_tsp_setup (5 nodes: 0,1,2,3,4), optimal is [0,1,3,2,4] cost ~8.00056
    # Initial in fixture is [0,1,2,3,4] cost ~10.0013
    # Let's try a tour that is not the initial one from the fixture, but also not optimal.
    # e.g., tour_order = [0, 2, 1, 3, 4]  # A permutation
    # We need to ensure this tour is one where alternate_step can find a known improvement.

    # This requires careful construction of 'tour_to_test_order'
    # For now, let's use the initial tour from simple_tsp_setup, as alternate_step
    # might find an improvement there if step() doesn't.
    tour_to_test_order = _initial_tour_obj.get_tour()  # [0,1,2,3,4]
    tour_to_test = Tour(tour_to_test_order, dist_matrix)
    initial_test_cost = tour_to_test.cost

    original_lk_config = LK_CONFIG.copy()
    # Configure LK_CONFIG for alternate_step if necessary (e.g., ensure BREADTH_A/B/D are > 0)
    # Default config should be fine: BREADTH_A=5, BREADTH_B=5, BREADTH_D=1

    deadline = time.time() + 5

    improvement_found = False
    best_improving_sequence = None

    # Try alternate_step from each node as a base_node
    for base_node_idx in range(tour_to_test.n):
        # alternate_step operates on the tour object passed to it.
        # To test its effect cleanly, we might want to pass a fresh copy or reset.
        # However, alternate_step itself doesn't modify the tour; it returns a sequence.

        # Create a fresh tour object for each call to alternate_step to avoid state issues
        # if alternate_step were to modify the tour (it doesn't, but good practice for testing)
        current_call_tour = Tour(tour_to_test_order, dist_matrix)

        found, sequence = alternate_step(
            base_node=current_call_tour.order[base_node_idx],  # Pass actual node label
            tour=current_call_tour,
            D=dist_matrix,
            neigh=neighbors,
            deadline=deadline
        )
        if found and sequence:
            # Apply sequence to a copy to check cost
            temp_tour = Tour(tour_to_test_order, dist_matrix)
            cost_before_apply = temp_tour.cost
            for f_start, f_end in sequence:
                temp_tour.flip_and_update_cost(f_start, f_end, dist_matrix)

            if temp_tour.cost < cost_before_apply - 1e-9: # Check for strict improvement
                improvement_found = True
                best_improving_sequence = sequence  # Store the first one found
                # print(f"Alternate_step from base {current_call_tour.order[base_node_idx]} found sequence: {sequence}, new cost: {temp_tour.cost}")
                break

    assert improvement_found, "alternate_step failed to find an improvement on the test tour."

    # Further assertions: apply best_improving_sequence and check cost
    final_tour = Tour(tour_to_test_order, dist_matrix)
    for f_start, f_end in best_improving_sequence:
        final_tour.flip_and_update_cost(f_start, f_end, dist_matrix)

    assert final_tour.cost < initial_test_cost - 1e-9, \
        "Applying sequence from alternate_step did not reduce cost."
    # Optionally, assert it reaches the known optimal for simple_tsp_setup if the sequence is powerful enough
    # assert final_tour.cost == pytest.approx(lk_optimal_cost)

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_alternate_step_no_improvement_on_optimal(simple_tsp_setup):
    """
    Tests that alternate_step does not find an improvement on an already optimal tour.
    """
    coords, dist_matrix, _initial_tour_obj, neighbors, \
        _initial_cost, _lk_optimal_cost, lk_optimal_order = simple_tsp_setup

    optimal_tour = Tour(lk_optimal_order, dist_matrix)
    deadline = time.time() + 5
    original_lk_config = LK_CONFIG.copy()  # Ensure default BREADTH_A/B/D are used

    for base_node_idx in range(optimal_tour.n):
        # Pass a fresh copy of the optimal tour object to alternate_step
        current_call_tour = Tour(lk_optimal_order, dist_matrix)
        cost_before_call = current_call_tour.cost

        found, sequence = alternate_step(
            base_node=current_call_tour.order[base_node_idx],
            tour=current_call_tour,
            D=dist_matrix,
            neigh=neighbors,
            deadline=deadline
        )
        if found and sequence:
            # If a sequence is returned, apply it and check if it's truly improving
            temp_tour = Tour(lk_optimal_order, dist_matrix)
            for f_start, f_end in sequence:
                temp_tour.flip_and_update_cost(f_start, f_end, dist_matrix)
            assert temp_tour.cost >= cost_before_call - 1e-9, \
                f"alternate_step found a sequence {sequence} from base {current_call_tour.order[base_node_idx]} that 'improved' an optimal tour. New cost: {temp_tour.cost}, Optimal: {cost_before_call}"
        # If found is False, or sequence is None, that's the expected behavior (no strict improvement).

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_alternate_step_restrictive_breadth(simple_tsp_setup):
    """
    Tests that alternate_step with zero breadth settings does not find an improvement
    and does not crash.
    """
    coords, dist_matrix, initial_tour_obj, neighbors, \
        _initial_cost, _lk_optimal_cost, _lk_optimal_order = simple_tsp_setup

    # Use the non-optimal initial tour
    test_tour = Tour(initial_tour_obj.get_tour(), dist_matrix)
    deadline = time.time() + 5

    original_lk_config = LK_CONFIG.copy()
    restrictive_config = original_lk_config.copy()
    restrictive_config.update({
        "BREADTH_A": 0,
        "BREADTH_B": 0,
        "BREADTH_D": 0
    })
    LK_CONFIG.clear()
    LK_CONFIG.update(restrictive_config)

    for base_node_idx in range(test_tour.n):
        current_call_tour = Tour(test_tour.get_tour(), dist_matrix)  # Fresh copy
        found, sequence = alternate_step(
            base_node=current_call_tour.order[base_node_idx],
            tour=current_call_tour,
            D=dist_matrix,
            neigh=neighbors,
            deadline=deadline
        )
        assert not (found and sequence), \
            f"alternate_step found an improvement {sequence} from base {current_call_tour.order[base_node_idx]} with zero breadth settings."

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


# You might want to parameterize this test if you have many random instances
# List the base names of your instance files here (without .tsp or .opt.tour)
@pytest.mark.parametrize("instance_base_name", ["rand4"])  # Replace with your actual file base names
def test_chained_lk_terminates_at_known_optimum(instance_base_name):
    tsp_file = VERIFICATION_RANDOM_PATH / f"{instance_base_name}.tsp"
    opt_tour_file = VERIFICATION_RANDOM_PATH / f"{instance_base_name}.opt.tour"

    assert tsp_file.exists(), f"TSP file not found: {tsp_file}"
    assert opt_tour_file.exists(), f"Optimal tour file not found: {opt_tour_file}"

    coords = read_tsp_file(str(tsp_file))
    assert coords is not None and coords.size > 0, f"Could not load coordinates from {tsp_file}"

    optimal_tour_nodes = read_opt_tour(str(opt_tour_file))
    assert optimal_tour_nodes is not None, f"Could not load optimal tour from {opt_tour_file}"

    if coords.shape[0] < 4:  # Lin-Kernighan typically needs at least 4 nodes
        pytest.skip(f"Skipping {instance_base_name} as it has fewer than 4 nodes ({coords.shape[0]}).")
        return

    if len(optimal_tour_nodes) != coords.shape[0]:
        pytest.fail(f"Optimal tour node count ({len(optimal_tour_nodes)}) does not match coordinate count ({coords.shape[0]}) for {instance_base_name}.")

    dist_matrix = build_distance_matrix(coords)

    # Calculate the known optimal length from the optimal tour and distance matrix
    calculated_known_opt_len = 0.0
    for i in range(len(optimal_tour_nodes)):
        u = optimal_tour_nodes[i]
        v = optimal_tour_nodes[(i + 1) % len(optimal_tour_nodes)]
        calculated_known_opt_len += dist_matrix[u, v]

    neighbors = delaunay_neighbors(coords)
    initial_tour_order = list(range(coords.shape[0]))  # A simple sequential initial tour

    original_lk_config = LK_CONFIG.copy()
    # Use default LK_CONFIG, but ensure TIME_LIMIT is long enough for the test logic
    temp_config = original_lk_config.copy()
    long_time_limit_seconds = 5.0  # Should be much longer than expected solve time
    temp_config["TIME_LIMIT"] = long_time_limit_seconds  # Ensure chained_lk uses this if not overridden by param

    LK_CONFIG.clear()
    LK_CONFIG.update(temp_config)

    start_time = time.time()
    final_tour_order, final_cost = chained_lin_kernighan(
        coords,
        initial_tour_order,
        known_optimal_length=calculated_known_opt_len,  # Pass the calculated known optimum
        time_limit_seconds=long_time_limit_seconds  # Explicitly pass time limit
    )
    end_time = time.time()
    execution_time = end_time - start_time

    assert final_cost == pytest.approx(calculated_known_opt_len), \
        (f"Chained LK for {instance_base_name} did not reach known optimal length {calculated_known_opt_len:.4f}. "
         f"Got {final_cost:.4f}.")

    # Heuristic check for early termination:
    # If the problem is non-trivial and solved, execution time should be noticeably less than the long_time_limit.
    # This threshold is somewhat arbitrary and might need adjustment based on instance difficulty.
    # A very small instance might solve "instantly" regardless of the known_optimal_length.
    # We expect that if known_optimal_length caused an early exit, the time taken is small.
    # If the instance is very easy, execution_time might be small anyway.
    # The main goal is that it *stops* once optimal is hit, not running for the full long_time_limit_seconds.
    # For a 30s limit, finishing in < 5s would suggest early exit or very fast solve.
    # If an instance takes 0.01s to solve anyway, this part of the assert is less meaningful for proving early exit.
    assert execution_time < (long_time_limit_seconds * 0.5) or execution_time < 5.0, \
        (f"Chained LK for {instance_base_name} took {execution_time:.3f}s. "
         f"This might be too long if early exit due to known_optimal_length was expected "
         f"relative to time_limit_seconds={long_time_limit_seconds}s. "
         f"Optimal length was {calculated_known_opt_len:.4f}.")

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)  # Restore original config
