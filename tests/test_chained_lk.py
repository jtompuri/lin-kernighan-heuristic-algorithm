import pytest
import time
import numpy as np
from pathlib import Path
from lin_kernighan_tsp_solver.lk_algorithm import (
    Tour,
    lin_kernighan,
    chained_lin_kernighan,
    build_distance_matrix,
)
from lin_kernighan_tsp_solver.tsp_io import (
    read_opt_tour,
    read_tsp_file,
)
from lin_kernighan_tsp_solver.config import LK_CONFIG

# Definition of paths for test files
VERIFICATION_RANDOM_PATH = Path(__file__).resolve().parent.parent / "problems" / "random"
VERIFICATION_SOLUTIONS_PATH = Path(__file__).resolve().parent.parent / "problems" / "random"


@pytest.mark.parametrize("instance_base_name", ["rand4", "rand5", "rand6", "rand7", "rand8", "rand9", "rand10", "rand11", "rand12"])  # Replace with your actual file base names
def test_chained_lk_terminates_at_known_optimum(instance_base_name):
    tsp_file = VERIFICATION_RANDOM_PATH / f"{instance_base_name}.tsp"
    opt_tour_file = VERIFICATION_SOLUTIONS_PATH / f"{instance_base_name}.opt.tour"

    assert tsp_file.exists(), f"TSP file not found: {tsp_file}"
    if not opt_tour_file.exists():
        pytest.skip(f"Optimal tour file not found: {opt_tour_file} (expected for random instances)")

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
    # num_iterations_display = 3 # Unused variable
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


def test_chained_lin_kernighan_empty_tour(simple_tsp_setup):
    """
    Tests chained_lin_kernighan with an empty initial tour.
    It should correctly handle this and return an empty tour with zero cost.
    This also helps cover the final cost re-computation for n=0.
    """
    # Coords for an empty problem
    empty_coords = np.array([])
    empty_initial_tour = []

    # Use a short time limit as it should return quickly.
    time_limit_for_test = 0.1

    # known_optimal_length for an empty tour is 0.
    known_opt = 0.0

    best_tour_order_list, best_cost = chained_lin_kernighan(
        empty_coords,
        empty_initial_tour,
        known_optimal_length=known_opt,
        time_limit_seconds=time_limit_for_test
    )

    assert best_tour_order_list == [], "Expected an empty tour list."
    assert best_cost == pytest.approx(0.0), "Expected zero cost for an empty tour."

    # Test without known_optimal_length
    best_tour_order_list_no_opt, best_cost_no_opt = chained_lin_kernighan(
        empty_coords,
        empty_initial_tour,
        time_limit_seconds=time_limit_for_test
    )
    assert best_tour_order_list_no_opt == [], "Expected an empty tour list (no opt)."
    assert best_cost_no_opt == pytest.approx(0.0), "Expected zero cost (no opt)."


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


def test_chained_lin_kernighan_breaks_on_optimum_found_single_call():
    import numpy as np
    from lin_kernighan_tsp_solver.lk_algorithm import chained_lin_kernighan, FLOAT_COMPARISON_TOLERANCE, Tour

    coords = np.array([[0, 0], [1, 0], [0, 1]])
    initial_tour = [0, 1, 2]
    known_optimal_length = 3.0

    # Patch lin_kernighan to simulate improvement to optimal
    from lin_kernighan_tsp_solver import lk_algorithm

    class DummyTour(Tour):
        def __init__(self, order, cost):
            self._order = order
            self.cost = cost
            self.n = len(order)

        def get_tour(self):
            return self._order

    def fake_lin_kernighan(*args, **kwargs):
        return (DummyTour([0, 2, 1], known_optimal_length), known_optimal_length)

    original_lk = lk_algorithm.lin_kernighan
    lk_algorithm.lin_kernighan = fake_lin_kernighan

    try:
        best_tour, best_cost = chained_lin_kernighan(
            coords,
            initial_tour,
            known_optimal_length=known_optimal_length
        )
        assert abs(best_cost - known_optimal_length) < FLOAT_COMPARISON_TOLERANCE * 10
        assert best_tour == [0, 2, 1]
    finally:
        lk_algorithm.lin_kernighan = original_lk


def test_chained_lin_kernighan_breaks_on_optimum_found_after_improvement(monkeypatch):
    import numpy as np
    from lin_kernighan_tsp_solver.lk_algorithm import chained_lin_kernighan, Tour

    coords = np.array([[0, 0], [1, 0], [0, 1]])
    initial_tour = [0, 1, 2]
    known_optimal_length = 3.0

    from lin_kernighan_tsp_solver import lk_algorithm

    class DummyTour(Tour):
        def __init__(self, order, cost):
            self._order = order
            self.cost = cost
            self.n = len(order)

        def get_tour(self):
            return self._order

    call_count = {'count': 0}

    def fake_lin_kernighan(*args, **kwargs):
        call_count['count'] += 1
        if call_count['count'] == 1:
            return DummyTour([0, 1, 2], 4.0), 4.0
        else:
            return DummyTour([0, 2, 1], known_optimal_length), known_optimal_length

    original_lk = lk_algorithm.lin_kernighan
    lk_algorithm.lin_kernighan = fake_lin_kernighan

    try:
        best_tour, best_cost = chained_lin_kernighan(
            coords,
            initial_tour,
            known_optimal_length=known_optimal_length,
            time_limit_seconds=10
        )
        # The best_tour should be [0, 2, 1]
        assert best_tour == [0, 2, 1]
        # The best_cost should be the actual cost for this tour
        expected_cost = (
            np.linalg.norm(coords[0] - coords[2]) + np.linalg.norm(coords[2] - coords[1]) + np.linalg.norm(coords[1] - coords[0])
        )
        assert abs(best_cost - expected_cost) < 1e-9
    finally:
        lk_algorithm.lin_kernighan = original_lk
