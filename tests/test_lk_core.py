import pytest
import time
import numpy as np
from unittest.mock import patch, MagicMock

from lin_kernighan_tsp_solver.lk_algorithm import (
    Tour,
    step,
    alternate_step,
    lk_search,
    lin_kernighan,
)
from lin_kernighan_tsp_solver.lk_algorithm import (
    build_distance_matrix,
    delaunay_neighbors,
)
from lin_kernighan_tsp_solver.config import (
    LK_CONFIG,
    FLOAT_COMPARISON_TOLERANCE,
)


def test_alternate_step_deadline_exceeded(simple_tsp_setup):
    """
    Tests that alternate_step returns (False, None) if the deadline is exceeded.
    Covers lines like 448, 471, 500-503.
    """
    # Unpack all 7 items from the fixture
    coords, D_matrix, initial_tour_obj, neighbors, _initial_cost, _optimal_cost, _optimal_order = simple_tsp_setup
    tour_obj = Tour(initial_tour_obj.get_tour(), D_matrix)

    # Mock time.time() to simulate an immediate deadline
    # Use a fixed past time for the deadline to avoid issues with time.time() in the patch value itself
    fixed_current_time = time.time()
    with patch('time.time', return_value=fixed_current_time + 1000):  # Mock time to be far in the future
        found, seq = alternate_step(
            base_node=0,
            tour=tour_obj,
            D=D_matrix,
            neigh=neighbors,
            deadline=fixed_current_time - 1  # Deadline is in the past relative to fixed_current_time
        )
    assert not found
    assert seq is None


def test_lin_kernighan_deadline_exceeded(simple_tsp_setup):
    """
    Tests that lin_kernighan exits early if the deadline is hit.
    Covers lines like 560-570.
    """
    # Unpack all 7 items from the fixture
    coords, D_matrix, initial_tour_obj, neighbors, _initial_cost, _optimal_cost, _optimal_order = simple_tsp_setup
    initial_tour_nodes = initial_tour_obj.get_tour()

    side_effect_time = [time.time(), time.time() + 0.001, time.time() + 1000]

    with patch('time.time', side_effect=side_effect_time):
        deadline_for_lk = side_effect_time[0] + 0.0001

        result_tour_obj, result_cost = lin_kernighan(
            coords=coords,
            init=initial_tour_nodes,
            D=D_matrix,
            neigh=neighbors,
            deadline=deadline_for_lk
        )

    expected_initial_tour_obj = Tour(initial_tour_nodes, D_matrix)
    assert result_tour_obj.get_tour() == expected_initial_tour_obj.get_tour()
    assert result_cost is not None and expected_initial_tour_obj.cost is not None and np.isclose(result_cost, expected_initial_tour_obj.cost)


def test_step_hits_break_due_to_breadth_limit_zero(simple_tsp_setup):
    """
    Tests that the 'break' in the step function's candidate loop is hit
    when breadth_limit is 0.
    lin_kernighan should return the initial tour if step (due to this break)
    and alternate_step (mocked) find no improvements.
    This covers line approx. 351 (was 448 in report).
    """
    coords, dist_matrix, initial_tour_obj, initial_neighbors, \
        initial_cost, _, _ = simple_tsp_setup

    initial_tour_nodes = initial_tour_obj.get_tour()

    original_lk_config = LK_CONFIG.copy()
    # Set breadth to 0 for all levels, forcing the break if candidates exist
    LK_CONFIG["BREADTH"] = [0] * (LK_CONFIG.get("MAX_LEVEL", 12) + 5)  # Ensure enough zeros
    LK_CONFIG["MAX_LEVEL"] = 5  # Keep MAX_LEVEL reasonable for test speed

    deadline = time.time() + 5  # Generous deadline

    # Mock alternate_step to ensure it doesn't find an improvement,
    # isolating the behavior of 'step' hitting the break.
    with patch('lin_kernighan_tsp_solver.lk_algorithm.alternate_step', return_value=(False, None)) as mock_alt_step:
        final_tour_obj, final_cost = lin_kernighan(
            coords,
            initial_tour_nodes,
            dist_matrix,
            initial_neighbors,
            deadline
        )

    # Assertions:
    # 1. lin_kernighan should return the initial tour and cost because
    #    'step' will break early and 'alternate_step' is mocked to find nothing.
    assert final_cost == pytest.approx(initial_cost)
    assert final_tour_obj.get_tour() == initial_tour_nodes

    # 2. (Optional but good) Check if alternate_step was called,
    #    implying lk_search proceeded past the 'step' call.
    #    The number of calls to alternate_step depends on how many start_nodes lk_search tries.
    #    If simple_tsp_setup has 5 nodes, it could be called up to 5 times if step always returns (False, None).
    assert mock_alt_step.call_count > 0, "alternate_step should have been called."

    # Restore original LK_CONFIG
    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_alternate_step_y2_candidate_continues(simple_tsp_setup):
    """
    Tests that alternate_step correctly handles the 'continue' for y2_candidate
    if y2_candidate is t1, t2, or chosen_y1.
    This should lead to alternate_step returning (False, None) if no other
    paths yield a sequence.
    This covers line 438 and potentially the final return (line 473).
    """
    # Setup a 5-node scenario
    # Nodes: 0, 1, 2, 3, 4
    # Tour: [0, 1, 2, 3, 4]
    # t1 = 0, t2 = 1
    # We want chosen_y1 = 3.
    # Then t4 = tour.next(chosen_y1) = tour.next(3) = 4.
    # We want neigh[4] (neighbors of t4) to only contain t1, t2, or chosen_y1.
    # So, neigh[4] = [0, 1, 3] or subsets.

    coords = np.array([
        [0, 0],  # Node 0 (t1)
        [10, 0],  # Node 1 (t2)
        [20, 0],  # Node 2
        [10, 1],  # Node 3 (chosen_y1) - D[1, 3] should be small for positive gain_G1
        [0, 10]  # Node 4 (t4)
    ])
    dist_matrix = build_distance_matrix(coords)
    tour_nodes = [0, 1, 2, 3, 4]
    tour_obj = Tour(tour_nodes, dist_matrix)

    # Ensure gain_G1 = D[t1,t2] - D[t2,y1_cand] is positive for y1_cand=3
    # D[0, 1] = 10
    # D[1, 3] = D( (10, 0), (10, 1) ) = 1.
    # gain_G1 = 10 - 1 = 9. This is > 0. So y1=3 can be chosen.

    # Define neighbor lists
    # neigh[t2=1] should make y1_candidate=3 the primary choice.
    # neigh[t4=4] should only contain {t1=0, t2=1, chosen_y1=3}
    neighbors = [
        [1, 4],             # Neighbors of 0 (t1)
        [0, 2, 3],          # Neighbors of 1 (t2) -> make 3 a candidate for y1
        [1, 3],             # Neighbors of 2
        [1, 2, 4],          # Neighbors of 3 (chosen_y1)
        [0, 1, 3]           # Neighbors of 4 (t4) -> these are t1, t2, chosen_y1
    ]

    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["BREADTH_A"] = 1  # Consider only the best y1 candidate
    LK_CONFIG["BREADTH_B"] = 3  # Consider all candidates for y2 from neigh[t4]

    deadline = time.time() + 5  # Generous deadline

    found, seq = alternate_step(
        base_node=0,  # t1
        tour=tour_obj,
        D=dist_matrix,
        neigh=neighbors,
        deadline=deadline
    )

    assert not found, "alternate_step should not find a sequence."
    assert seq is None, "Sequence should be None."

    # Restore original LK_CONFIG
    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_step_no_candidates_for_t1_t2_pair():
    """
    Tests that the 'if not candidates: continue' line (approx. 341) in 'step'
    is hit when no valid y1_candidates are found for a (t1, t2) pair,
    leading to an empty 'candidates' list for that pair.
    """
    # Setup a 3-node scenario: 0-1-2
    coords = np.array([
        [0, 0],  # Node 0
        [10, 0],  # Node 1
        [10, 10]  # Node 2
    ])
    dist_matrix = build_distance_matrix(coords)
    tour_nodes = [0, 1, 2]
    tour_obj = Tour(tour_nodes, dist_matrix)
    initial_cost = tour_obj.cost
    assert initial_cost is not None, "Tour cost should be initialized."

    neighbors = [
        [1, 2],  # Neighbors of 0
        [0, 2],  # Neighbors of 1
        [0, 1]   # Neighbors of 2
    ]

    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 1
    LK_CONFIG["BREADTH"] = [1] * (LK_CONFIG["MAX_LEVEL"] + 5)  # Ensure enough breadth entries

    deadline = time.time() + 5

    from lin_kernighan_tsp_solver.lk_algorithm import step

    # Call step.
    # We are interested in the behavior when t1=0.
    # The loop in step iterates through t1 (base_node_outer_loop)
    # and then t2 (s1 = tour.next(base_node_outer_loop)).
    # To directly test the inner logic for a specific t1, we'd typically
    # rely on lk_search or lin_kernighan to set up the call to step.
    # However, for this specific coverage, we call step directly.
    # We expect it to not find an improvement.

    # For the first iteration of the outer loop in step (which is not explicit in its signature)
    # let's assume base_node_outer_loop would be 0.
    # The inner loop iterates over t2 (s1).
    # The 'if not candidates: continue' is inside the loop over s1 (t2).
    # To hit this, we need to ensure that for a specific base (t1) and s1 (t2),
    # no y1_candidates are found.

    # The test is designed such that for base=0, s1=1, no candidates are generated.
    # step itself iterates over base nodes. We can't directly force a single (t1,t2) pair
    # into step's parameters easily without it running its own loops.
    # The most straightforward way to test this line is to ensure that for *some* (t1,t2)
    # pair encountered by step's natural iteration, 'candidates' becomes empty.
    # The current setup for (t1=0, t2=1) should achieve this.

    improved, best_sequence = step(
        level=1,  # Start at level 1 as per lk_search
        delta=0.0,
        base=0,   # Start with base node 0
        tour=tour_obj,  # Corrected argument name
        D=dist_matrix,
        neigh=neighbors,
        flip_seq=[],
        start_cost=initial_cost,  # Cost of the tour when step is called
        best_cost=initial_cost,  # Global best cost known so far
        deadline=deadline
    )

    assert not improved, "Step should not find an improvement in this specific scenario."
    assert best_sequence is None, "Best sequence should be None if no improvement found."

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_lk_search_deadline_after_step_before_alternate(simple_tsp_setup):
    """
    Tests that lk_search returns None if the deadline is met after 'step'
    returns no improvement, but before 'alternate_step' is called.
    This covers lines approx. 507-508 in lk_search.
    """
    coords, dist_matrix, initial_tour_obj, initial_neighbors, \
        _, _, _ = simple_tsp_setup

    # Define a sequence of time values for time.time() mock
    start_time = time.time()
    time_before_deadline_for_first_check = start_time + 0.01
    time_after_deadline_for_second_check = start_time + 0.2

    # Deadline for lk_search argument
    lk_search_deadline = start_time + 0.1  # Deadline is between the two mocked times

    # Mock time.time() to control its return values sequentially
    mock_time_func = MagicMock(side_effect=[
        time_before_deadline_for_first_check,  # For initial check in lk_search
        time_after_deadline_for_second_check   # For check after step, before alternate_step
    ])

    # Mock 'step' to return no improvement
    mock_step_func = MagicMock(return_value=(False, None))

    # Mock 'alternate_step' to ensure it's not called
    mock_alternate_step_func = MagicMock()

    with patch('time.time', mock_time_func), \
            patch('lin_kernighan_tsp_solver.lk_algorithm.step', mock_step_func), \
            patch('lin_kernighan_tsp_solver.lk_algorithm.alternate_step', mock_alternate_step_func):

        result_sequence = lk_search(
            start_node_for_search=0,
            current_tour_obj=initial_tour_obj,
            D=dist_matrix,
            neigh=initial_neighbors,
            deadline=lk_search_deadline
        )

    assert result_sequence is None, "lk_search should return None due to deadline after step."

    # Assertions about mock calls:
    # time.time() should be called twice: once at the start, once after step.
    assert mock_time_func.call_count == 2, "time.time() should have been called twice."

    # step should be called once.
    mock_step_func.assert_called_once()

    # alternate_step should NOT be called because of the deadline.
    mock_alternate_step_func.assert_not_called()


def test_lk_search_returns_none_if_deadline_at_start(simple_tsp_setup):
    """
    Tests that lk_search returns None immediately if the deadline has passed
    at the very beginning of the function.
    This covers lines 490-491 in lk_search.
    """
    coords, dist_matrix, initial_tour_obj, initial_neighbors, \
        _, _, _ = simple_tsp_setup

    # Set a deadline that has effectively already passed or will be met immediately.
    # We want time.time() called inside lk_search to be >= deadline.
    current_simulated_time = time.time()
    expired_deadline = current_simulated_time - 1  # A deadline in the past

    # Mock time.time() to return a value that ensures the deadline is met.
    with patch('time.time', return_value=current_simulated_time) as mock_time_func:
        result_sequence = lk_search(
            start_node_for_search=0,
            current_tour_obj=initial_tour_obj,
            D=dist_matrix,
            neigh=initial_neighbors,
            deadline=expired_deadline  # Pass the already expired deadline
        )

    assert result_sequence is None, "lk_search should return None if deadline met at start."
    mock_time_func.assert_called()  # Ensure time.time() was checked by lk_search


def test_lk_search_handles_non_improving_alternate_step(simple_tsp_setup):
    """
    Tests that lk_search (via lin_kernighan) correctly handles a scenario
    where alternate_step is called and returns a valid but non-improving sequence.
    lk_search should not adopt this non-improving sequence.
    """
    coords, dist_matrix, _, neigh, \
        _, optimal_cost_fixture, optimal_order_fixture = simple_tsp_setup

    # Start with the optimal tour, so 'step' is unlikely to find improvements,
    # making it more likely for lk_search to try 'alternate_step'.
    initial_tour_nodes = list(optimal_order_fixture)
    initial_tour_cost = optimal_cost_fixture

    mock_alternate_step = MagicMock()

    def alternate_step_side_effect(base_node, tour, D, neigh, deadline):  # Match keyword args
        # This side effect simulates 'alternate_step'.
        # It's called by 'lk_search' to test non-improving behavior.

        # Assert that the tour cost passed to alternate_step matches the initial (optimal) cost.
        assert tour.cost == pytest.approx(initial_tour_cost)

        # Return a "successful" indication (True) but with an empty list of flips ([]).
        # This simulates alternate_step finding a sequence of operations (flips)
        # that results in no change to the tour and thus no cost improvement.
        # lk_search should then correctly handle this by not adopting the non-improving result.
        return True, []  # Simulate finding a "valid" sequence of (no) flips that doesn't improve cost

    mock_alternate_step.side_effect = alternate_step_side_effect

    deadline_val = time.time() + 5  # Generous deadline for lin_kernighan call

    # Patch 'alternate_step' in the module where it's defined and used by lk_search
    with patch('lin_kernighan_tsp_solver.lk_algorithm.alternate_step', mock_alternate_step):
        # lin_kernighan calls lk_search, which calls step and potentially alternate_step
        final_tour_obj, final_cost = lin_kernighan(
            coords,
            initial_tour_nodes,
            dist_matrix,
            neigh,
            deadline_val
        )

    # Assertions:
    # 1. alternate_step should have been called by lk_search.
    assert mock_alternate_step.call_count > 0, "Mocked alternate_step was not called."

    # 2. Since alternate_step returned a non-improving sequence,
    #    and we started with an optimal tour (so 'step' likely found no improvements either),
    #    the final tour should be the same as the initial optimal tour.
    assert final_cost == pytest.approx(initial_tour_cost)
    assert final_tour_obj.get_tour() == initial_tour_nodes
    assert final_tour_obj.cost == pytest.approx(initial_tour_cost)


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

    neighbors = delaunay_neighbors(coords)  # For 3 nodes, this is [[1, 2],[0, 2],[0, 1]]

    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 2
    LK_CONFIG["BREADTH"] = [1] * LK_CONFIG["MAX_LEVEL"]
    deadline = time.time() + 5

    best_tour_obj, best_cost = lin_kernighan(
        coords, initial_tour_nodes, dist_matrix, neighbors, deadline
    )

    # Expected cost: d(0, 1)+d(1, 2)+d(2, 0) = 1 + sqrt(2) + 1 = 2 + sqrt(2)
    expected_cost = 1.0 + np.sqrt(2) + 1.0

    assert best_cost == pytest.approx(expected_cost)
    assert best_tour_obj.cost == pytest.approx(expected_cost)
    assert len(best_tour_obj.get_tour()) == 3
    assert sorted(best_tour_obj.get_tour()) == [0, 1, 2]

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


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
    Tests that lin_kernighan stops execution if the deadline is reached.
    """
    coords, dist_matrix, initial_tour_obj, neigh, _, _, _ = simple_tsp_setup  # Get dist_matrix

    # Mock time.time() to control its return value
    mock_time = MagicMock()
    start_time = time.time()  # Get a real start time for reference

    # The deadline for lin_kernighan function argument
    test_deadline = start_time + 0.1  # Deadline is relatively soon

    # Use a callable for side_effect to handle an arbitrary number of calls
    # before returning the timeout value.
    # Let's allow, for example, up to 200 "normal" time checks before timeout.
    MAX_NORMAL_CALLS = 200
    call_count = 0

    def time_side_effect():
        nonlocal call_count
        call_count += 1
        if call_count <= MAX_NORMAL_CALLS:
            return start_time + (call_count * 0.0001)  # Small, increasing time steps
        else:
            return start_time + 1000  # Timeout value

    mock_time.side_effect = time_side_effect

    with patch('time.time', mock_time):
        result_tour, result_cost = lin_kernighan(
            coords,
            initial_tour_obj.get_tour(),  # Pass the tour node list
            dist_matrix,                 # Pass the distance matrix
            neigh,                       # Pass the neighbors
            test_deadline                # Pass the calculated deadline
        )

    # Check that time.time() was called.
    # The number of calls depends on how quickly the mocked deadline is hit.
    assert mock_time.call_count >= 1
    assert result_tour is not None
    assert result_cost is not None


def test_step_finds_simple_2_opt(simple_tsp_setup):
    _coords, dist_matrix, initial_tour_obj, neighbors, initial_cost, _lk_optimal_cost, _lk_optimal_order = simple_tsp_setup
    # This test focuses on a 2-opt step, not necessarily reaching the full LK optimum in one go.
    # Expected outcome of flipping (2, 3) on initial tour [0, 1, 2, 3, 4] is [0, 1, 3, 2, 4]
    expected_tour_after_2_opt_nodes = [0, 1, 3, 2, 4]
    expected_cost_after_2_opt = Tour(expected_tour_after_2_opt_nodes, dist_matrix).cost

    original_lk_config = LK_CONFIG.copy()
    # Ensure FLOAT_COMPARISON_TOLERANCE is available if not already imported in this file
    # from lin_kernighan_tsp_solver.lk_algorithm import FLOAT_COMPARISON_TOLERANCE
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
            for fs, fe in flip_sequence:
                temp_tour_check.flip_and_update_cost(fs, fe, dist_matrix)

            # Check if this particular improvement matches our expected 2-opt
            if temp_tour_check.get_tour() == expected_tour_after_2_opt_nodes and \
               temp_tour_check.cost is not None and expected_cost_after_2_opt is not None and \
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
        for fs, fe in best_flip_sequence_found:
            final_tour_check.flip_and_update_cost(fs, fe, dist_matrix)
        assert final_tour_check.cost == pytest.approx(expected_cost_after_2_opt)
        assert final_tour_check.get_tour() == expected_tour_after_2_opt_nodes

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_step_no_improvement_on_optimal_tour(simple_tsp_setup):
    _coords, dist_matrix, initial_tour_obj_from_fixture, neighbors, \
        _initial_cost_fixture, _optimal_cost_fixture_setup, _optimal_order_fixture = simple_tsp_setup

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
            assert cost_before_this_step_attempt is not None, "Tour cost should be a float, got None"
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
                if actual_new_cost is not None and cost_before_this_step_attempt is not None and actual_new_cost < cost_before_this_step_attempt - (FLOAT_COMPARISON_TOLERANCE / 10.0):
                    current_processing_tour = candidate_tour_after_flips
                    made_improvement_in_this_pass = True
                    break
        if not made_improvement_in_this_pass:
            break
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
        assert cost_of_2_optimal_tour is not None, "Tour cost should not be None"
        improved_final, flip_sequence_final = step(
            level=1, delta=0.0, base=base_node_val_check, tour=tour_arg_for_final_step,
            D=dist_matrix, neigh=neighbors, flip_seq=[],
            start_cost=float(cost_of_2_optimal_tour), best_cost=float(cost_of_2_optimal_tour),
            deadline=deadline_for_final_check
        )
        final_tour_cost_after_step_if_improved = cost_of_2_optimal_tour
        if improved_final and flip_sequence_final:
            temp_tour_for_cost_check = Tour(order_of_2_optimal_tour, dist_matrix)
            for f_s, f_e in flip_sequence_final:
                temp_tour_for_cost_check.flip_and_update_cost(f_s, f_e, dist_matrix)
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
            overall_improved_flag = True
            break
        made_change_in_iteration = False
        iter_count += 1
        nodes_to_try_as_base = list(current_tour.order)
        for start_node_for_search in nodes_to_try_as_base:
            tour_for_lk_search_call = Tour(current_tour.get_tour(), dist_matrix)
            improving_sequence = lk_search(start_node_for_search, tour_for_lk_search_call, dist_matrix, neighbors, deadline)
            if improving_sequence:
                cost_before_flips = current_tour.cost
                for fs, fe in improving_sequence:
                    current_tour.flip_and_update_cost(fs, fe, dist_matrix)
                if current_tour.cost is not None and cost_before_flips is not None and current_tour.cost < cost_before_flips - FLOAT_COMPARISON_TOLERANCE:
                    overall_improved_flag, made_change_in_iteration = True, True
                    break
                else:
                    current_tour = Tour(nodes_to_try_as_base, dist_matrix)  # Revert
                    made_change_in_iteration = False
                    break
    assert overall_improved_flag or abs(initial_cost_fixture - optimal_cost_fixture) < FLOAT_COMPARISON_TOLERANCE, \
        f"lk_search failed to improve. Initial: {initial_cost_fixture}, Optimal: {optimal_cost_fixture}, Final: {current_tour.cost}"
    assert current_tour.cost == pytest.approx(optimal_cost_fixture)
    assert current_tour.get_tour() == optimal_order_fixture
    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


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
            f"lk_search found improvement from node {start_node_for_search} on converged tour. Seq: {improving_sequence}"
        assert tour_copy_for_lk_search.get_tour() == converged_tour_obj.get_tour()
        assert tour_copy_for_lk_search.cost == pytest.approx(converged_cost)
    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


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

            if temp_tour.cost is not None and cost_before_apply is not None and temp_tour.cost < cost_before_apply - 1e-9:  # Check for strict improvement
                improvement_found = True
                best_improving_sequence = sequence  # Store the first one found
                # print(f"Alternate_step from base {current_call_tour.order[base_node_idx]} found sequence: {sequence}, new cost: {temp_tour.cost}")
                break

    assert improvement_found, "alternate_step failed to find an improvement on the test tour."

    # Further assertions: apply best_improving_sequence and check cost
    final_tour = Tour(tour_to_test_order, dist_matrix)
    if best_improving_sequence:
        for f_start, f_end in best_improving_sequence:
            final_tour.flip_and_update_cost(f_start, f_end, dist_matrix)

    assert final_tour.cost is not None and initial_test_cost is not None and final_tour.cost < initial_test_cost - 1e-9, \
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
            assert temp_tour.cost is not None and cost_before_call is not None and (temp_tour.cost >= cost_before_call - 1e-9), \
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


def test_alternate_step_deadline_passed():
    coords = np.array([[0, 0], [1, 1]])
    D = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    tour = Tour([0, 1], D)
    neigh = [[1], [0]]
    deadline = time.time() - 10  # Already passed
    found, seq = alternate_step(0, tour, D, neigh, deadline)
    assert not found and seq is None


def test_alternate_step_no_candidates():
    coords = np.array([[0, 0], [1, 1], [2, 2]])
    D = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    tour = Tour([0, 1, 2], D)
    # Neighbors such that no candidate is valid
    neigh = [[1], [0], [1]]
    deadline = time.time() + 10
    found, seq = alternate_step(0, tour, D, neigh, deadline)
    assert not found and seq is None


def test_lk_search_deadline_passed():
    coords = np.array([[0, 0], [1, 1]])
    D = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    tour = Tour([0, 1], D)
    neigh = [[1], [0]]
    deadline = time.time() - 10
    assert lk_search(0, tour, D, neigh, deadline) is None


def test_lk_search_no_improvement():
    coords = np.array([[0, 0], [1, 1]])
    D = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    tour = Tour([0, 1], D)
    neigh = [[1], [0]]
    deadline = time.time() + 10
    assert lk_search(0, tour, D, neigh, deadline) is None


def test_flip_and_update_cost_recomputes_cost_if_none():
    from lin_kernighan_tsp_solver.lk_algorithm import Tour
    import numpy as np
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    # Create Tour without initializing cost
    tour = Tour([0, 1], None)
    # At this point, tour.cost is None
    delta = tour.flip_and_update_cost(0, 1, D)
    # After the call, cost should be recomputed and not None
    assert tour.cost == 2.0
    assert isinstance(delta, float)


def test_step_returns_false_none_when_no_candidates():
    from lin_kernighan_tsp_solver.lk_algorithm import Tour, step
    import numpy as np
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    tour = Tour([0, 1], D)
    neigh = [[1], [0]]
    start_cost = tour.cost if tour.cost is not None else 0.0
    result = step(1, 0.0, 0, tour, D, neigh, [], start_cost, start_cost, float('inf'))
    assert result == (False, None)


def test_alternate_step_deadline_and_no_candidates():
    from lin_kernighan_tsp_solver.lk_algorithm import Tour, alternate_step
    import numpy as np
    import time
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    tour = Tour([0, 1], D)
    neigh = [[1], [0]]
    # Deadline already passed
    found, seq = alternate_step(0, tour, D, neigh, time.time() - 1)
    assert not found and seq is None

    # No candidates (neighbors are self)
    neigh = [[0], [1]]
    found, seq = alternate_step(0, tour, D, neigh, time.time() + 10)
    assert not found and seq is None


def test_lk_search_deadline_and_no_improvement():
    from lin_kernighan_tsp_solver.lk_algorithm import Tour, lk_search
    import numpy as np
    import time
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    tour = Tour([0, 1], D)
    neigh = [[1], [0]]
    # Deadline already passed
    assert lk_search(0, tour, D, neigh, time.time() - 1) is None
    # No improvement possible
    assert lk_search(0, tour, D, neigh, time.time() + 10) is None


def test_alternate_step_deadline_returns(monkeypatch):
    import numpy as np
    import time
    from lin_kernighan_tsp_solver.lk_algorithm import Tour, alternate_step

    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    tour = Tour([0, 1], D)
    neigh = [[1], [0]]

    # Line 449: Deadline already passed at function entry
    found, seq = alternate_step(0, tour, D, neigh, time.time() - 10)
    assert found is False and seq is None

    # Line 465: Deadline passed inside first for-loop
    # Patch time.time to simulate deadline passing after function entry
    call_count = {'count': 0}
    real_time = time.time

    def fake_time():
        call_count['count'] += 1
        # First call: not passed, second call: passed
        return real_time() if call_count['count'] < 2 else real_time() + 1000
    monkeypatch.setattr('time.time', fake_time)
    found, seq = alternate_step(0, tour, D, neigh, real_time() + 500)
    assert found is False and seq is None

    # Line 484: Deadline passed inside second for-loop
    # To reach this, need at least one candidate for y1 and y2
    # Use a 3-node tour to ensure candidates exist
    D3 = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    tour3 = Tour([0, 1, 2], D3)
    neigh3 = [[1, 2], [0, 2], [0, 1]]
    call_count['count'] = 0

    def fake_time2():
        call_count['count'] += 1
        # Let the first few calls pass, then simulate deadline passed
        return real_time() if call_count['count'] < 6 else real_time() + 1000
    monkeypatch.setattr('time.time', fake_time2)
    found, seq = alternate_step(0, tour3, D3, neigh3, real_time() + 500)
    assert found is False and seq is None


def test_delaunay_neighbors_small_cases():
    assert delaunay_neighbors(np.empty((0, 2))) == []
    assert delaunay_neighbors(np.array([[0, 0]])) == [[]]
    assert delaunay_neighbors(np.array([[0, 0], [1, 1]])) == [[1], [0]]


def test_tour_init_cost_empty():
    from lin_kernighan_tsp_solver.lk_algorithm import Tour
    import numpy as np
    tour = Tour([], None)
    tour.init_cost(np.empty((0, 0)))
    assert tour.cost == 0.0


def test_alternate_step_deadline_in_nested_loops(monkeypatch):
    import numpy as np
    from lin_kernighan_tsp_solver.lk_algorithm import Tour, alternate_step

    # Setup a 3-node tour to ensure all loops are entered
    coords = np.array([[0, 0], [1, 0], [0, 1]])
    D = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    tour = Tour([0, 1, 2], D)
    neigh = [[1, 2], [0, 2], [0, 1]]
    deadline = 1000.0  # Arbitrary future time

    # Patch time.time() to simulate deadline being passed at different loop depths
    call_counter = {'count': 0}
    real_time = time.time()

    def fake_time():
        call_counter['count'] += 1
        # 1st call: before first for-loop (ok)
        # 2nd call: inside first for-loop (simulate deadline passed)
        # 3rd call: inside second for-loop (simulate deadline passed)
        # 4th call: inside third for-loop (simulate deadline passed)
        if call_counter['count'] == 2:
            return deadline + 1  # Triggers line 449
        elif call_counter['count'] == 4:
            return deadline + 1  # Triggers line 465
        elif call_counter['count'] == 6:
            return deadline + 1  # Triggers line 484
        return real_time

    monkeypatch.setattr('time.time', fake_time)

    # Line 449: Deadline passed in first for-loop
    found, seq = alternate_step(0, tour, D, neigh, deadline)
    assert found is False and seq is None

    # Reset counter for next test
    call_counter['count'] = 0

    # Line 465: Deadline passed in second for-loop
    # To reach the second for-loop, we need at least one candidate for y1
    # So, ensure gain_G1 > 0 for at least one neighbor
    D2 = np.array([[0, 1, 10], [1, 0, 1], [10, 1, 0]], dtype=float)
    tour2 = Tour([0, 1, 2], D2)
    neigh2 = [[1, 2], [0, 2], [0, 1]]
    monkeypatch.setattr('time.time', lambda: real_time if call_counter['count'] < 4 else deadline + 1)
    found, seq = alternate_step(0, tour2, D2, neigh2, deadline)
    assert found is False and seq is None

    # Reset counter for next test
    call_counter['count'] = 0

    # Line 484: Deadline passed in third for-loop
    # To reach the third for-loop, we need at least one candidate for y1 and y2
    D3 = np.array([[0, 1, 10], [1, 0, 0.5], [10, 0.5, 0]], dtype=float)
    tour3 = Tour([0, 1, 2], D3)
    neigh3 = [[1, 2], [0, 2], [0, 1]]
    monkeypatch.setattr('time.time', lambda: real_time if call_counter['count'] < 6 else deadline + 1)
    found, seq = alternate_step(0, tour3, D3, neigh3, deadline)
    assert found is False and seq is None


def test_lk_search_returns_seq_alt_on_improvement(monkeypatch):
    import numpy as np
    from lin_kernighan_tsp_solver.lk_algorithm import Tour, lk_search

    coords = np.array([[0, 0], [1, 0], [0, 1]])
    D = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    tour = Tour([0, 1, 2], D)
    neigh = [[1, 2], [0, 2], [0, 1]]
    deadline = time.time() + 10

    # Patch alternate_step to return a sequence
    def fake_alternate_step(base_node, tour, D, neigh, deadline):
        return True, [(0, 1)]

    monkeypatch.setattr('lin_kernighan_tsp_solver.lk_algorithm.alternate_step', fake_alternate_step)

    # Patch Tour.flip_and_update_cost to always reduce cost
    original_flip_and_update_cost = Tour.flip_and_update_cost

    def fake_flip_and_update_cost(self, node_a, node_b, D):
        self.cost = 0.0  # Simulate strict improvement
        return -1.0
    Tour.flip_and_update_cost = fake_flip_and_update_cost

    try:
        result = lk_search(0, tour, D, neigh, deadline)
        assert result == [(0, 1)]
    finally:
        Tour.flip_and_update_cost = original_flip_and_update_cost


def test_alternate_step_deadline_first_for(monkeypatch):  # Covers line 449
    import numpy as np
    from lin_kernighan_tsp_solver.lk_algorithm import Tour, alternate_step, LK_CONFIG

    original_breadth_a = LK_CONFIG["BREADTH_A"]
    LK_CONFIG["BREADTH_A"] = 1  # Consider only one y1 candidate

    # 3-node tour: [0, 1, 2]. t1=0, t2=1.
    # To make gain_G1 > 0 for y1_candidate=2: D[0,1] - D[1,2] > TOLERANCE
    D = np.array([
        [0, 1.0, 0.1],
        [1.0, 0, 0.5],  # D[t2=1, y1_cand=2] = 0.5
        [0.1, 0.5, 0]
    ], dtype=float)  # gain_G1 = D[0,1] - D[1,2] = 1.0 - 0.5 = 0.5 > TOLERANCE
    tour = Tour([0, 1, 2], D)
    neigh = [[1, 2], [0, 2], [0, 1]]  # Ensure y1_candidate=2 is in neigh[t2=1]

    deadline_val = 1000.0
    call_count = {'count': 0}

    def fake_time():
        call_count['count'] += 1
        # Call 1: initial check in alternate_step (line 431) - pass
        # Call 2: check in first for-loop (line 449) - fail (hit deadline)
        if call_count['count'] == 1:
            return deadline_val - 1.0
        else:
            return deadline_val + 1.0

    monkeypatch.setattr('time.time', fake_time)

    found, seq = alternate_step(0, tour, D, neigh, deadline_val)
    assert found is False and seq is None
    assert call_count['count'] == 2  # Ensure fake_time was called as expected

    LK_CONFIG["BREADTH_A"] = original_breadth_a  # Restore config


def test_alternate_step_deadline_second_for(monkeypatch):  # Targets line 465
    import numpy as np
    from lin_kernighan_tsp_solver.lk_algorithm import Tour, alternate_step, LK_CONFIG

    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["BREADTH_A"] = 1
    LK_CONFIG["BREADTH_B"] = 1

    # 4-node tour: [0,1,2,3]. t1=0, t2=1.
    # D matrix from your failing test, which populated candidates but found a 3-opt.
    # This D ensures candidates_for_y1 and candidates_for_y2 are non-empty.
    D = np.array([
        [0, 1.0, 0.1, 1.0],  # Node 0 (t1, also t4)
        [1.0, 0, 1.0, 0.1],  # Node 1 (t2)
        [0.1, 1.0, 0, 1.0],  # Node 2 (chosen_y2 if t4=0)
        [1.0, 0.1, 1.0, 0]  # Node 3 (chosen_y1)
    ], dtype=float)
    # With this D and tour [0,1,2,3]:
    # t1=0, t2=1.
    # For y1_cand=3 (from neigh[1]): gain_G1 = D[0,1]-D[1,3] = 1.0-0.1 = 0.9. Good.
    # chosen_y1=3, chosen_t3=tour.prev(3)=2.
    # t4=tour.next(chosen_y1=3)=0.
    # For y2_cand from neigh[t4=0]. Let neigh[0]=[1,2,3].
    # y2_cand=2 (not t1,t2,chosen_y1). chosen_y2=2.
    # The 3-opt check tour.sequence(t2=1, chosen_y2=2, chosen_y1=3) on tour [0,1,2,3] is True.
    # The deadline check is BEFORE this 3-opt check.
    tour = Tour([0, 1, 2, 3], D)
    neigh = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]

    deadline_val = 100.0
    call_count = {'value': 0}

    def fake_time():
        call_count['value'] += 1
        # Call 1: Initial check (line ~431) -> pass
        # Call 2: Check in first for-loop (line 449) -> pass (BREADTH_A=1)
        # Call 3: Check at start of second for-loop (line 465) -> hit deadline
        if call_count['value'] <= 2:  # Calls 1 and 2
            return deadline_val - 10.0  # Before deadline
        else:  # call_count['value'] == 3
            return deadline_val + 10.0  # After deadline

    monkeypatch.setattr('time.time', fake_time)

    found, seq = alternate_step(0, tour, D, neigh, deadline_val)

    assert found is False and seq is None
    assert call_count['value'] == 3, "time.time() was not called the expected number of times"

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_alternate_step_deadline_third_for(monkeypatch):  # Targets line 484
    import numpy as np
    from lin_kernighan_tsp_solver.lk_algorithm import Tour, alternate_step, LK_CONFIG

    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["BREADTH_A"] = 1
    LK_CONFIG["BREADTH_B"] = 1
    LK_CONFIG["BREADTH_D"] = 1

    # 6-node setup. Tour: [0,1,2,3,4,5]. t1=0, t2=1.
    # Goal: chosen_y1=2, chosen_y2=4. This makes 3-opt tour.sequence(1,4,2) False.

    D = np.full((6, 6), 100.0)
    np.fill_diagonal(D, 0)

    def set_dist(i, j, val):
        D[i, j] = val
        D[j, i] = val

    # Step 1: Choose chosen_y1 = 2 (from t2=1)
    # gain_G1 = D[t1=0,t2=1] - D[t2=1,y1_cand=2]
    set_dist(0, 1, 20)  # D[0,1]
    set_dist(1, 2, 1)   # D[1,2] -> gain_G1 = 19. chosen_y1=2
    set_dist(1, 3, 50)
    set_dist(1, 4, 50)
    set_dist(1, 5, 50)
    # chosen_y1=2. chosen_t3=tour.prev(2)=1. t4=tour.next(2)=3.

    # Step 2: Choose chosen_y2 = 4 (from t4=3)
    # gain_G2 = D[chosen_t3=1,t4=3] - D[t4=3,y2_cand=4]
    set_dist(1, 3, 20)  # D[chosen_t3=1, t4=3]
    set_dist(3, 4, 1)   # D[t4=3, y2_cand=4] -> gain_G2 = 19. chosen_y2=4
    # Excluded for y2: t1=0, t2=1, chosen_y1=2.
    # Valid y2_cand from t4=3 could be 0,1,4,5. We want 4.
    set_dist(3, 0, 50)
    set_dist(3, 1, 50)  # Skipped
    set_dist(3, 5, 50)
    # chosen_y2=4.
    # tour.sequence(t2=1, chosen_y2=4, chosen_y1=2) is False.

    # Step 3: Populate y3_candidates
    # chosen_t5=tour.prev(chosen_y2=4)=3. chosen_t6=tour.next(chosen_y2=4)=5.
    # y3_cand from neigh[chosen_t6=5].
    # Excluded for y3: t1=0, t2=1, chosen_y1=2, chosen_y2=4, chosen_t6=5. (Nodes: 0,1,2,4,5)
    # Node 3 is available. Let chosen_y3 = 3.
    # gain_G3 = D[chosen_t5=3,chosen_t6=5] - D[chosen_t6=5,y3_cand=3]
    # D[3,5] was 50. D[5,3] is 50.
    # To make gain_G3 positive for y3_cand=3 (from t6=5):
    set_dist(3, 5, 20)  # D[chosen_t5=3, chosen_t6=5]
    set_dist(5, 3, 1)  # D[chosen_t6=5, y3_cand=3] -> gain_G3 = 19
    # Ensure other y3_cand from t6=5 are less attractive or invalid
    set_dist(5, 0, 50)  # Skipped
    set_dist(5, 1, 50)  # Skipped
    set_dist(5, 2, 50)  # Skipped
    set_dist(5, 4, 50)  # Skipped

    tour_obj = Tour(list(range(6)), D)
    neigh = [  # Ensure choices are picked by being first in neigh list with BREADTH=1
        [1, 2, 3, 4, 5],  # 0
        [2, 0, 3, 4, 5],  # 1 (t2) -> pick 2 for y1
        [0, 1, 3, 4, 5],  # 2
        [4, 0, 1, 2, 5],  # 3 (t4) -> pick 4 for y2
        [0, 1, 2, 3, 5],  # 4
        [3, 0, 1, 2, 4]  # 5 (chosen_t6) -> pick 3 for y3
    ]

    deadline_val = 100.0
    call_count = {'value': 0}

    def fake_time():
        call_count['value'] += 1
        if call_count['value'] <= 3:
            return deadline_val - 10.0
        else:
            return deadline_val + 10.0

    monkeypatch.setattr('time.time', fake_time)

    found, seq = alternate_step(0, tour_obj, D, neigh, deadline_val)

    assert found is False and seq is None, f"Expected (False, None), got ({found}, {seq})"
    assert call_count['value'] == 4, f"time.time() called {call_count['value']} times, expected 4"

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)
