"""
Unit tests for the Tour class in the Lin-Kernighan TSP solver.

This module contains tests for various functionalities of the Tour class,
including initialization, node traversal (next, prev), segment checking
(sequence), tour modification (flip, flip_and_update_cost), and
cost calculation.
"""
import pytest
import numpy as np

from lin_kernighan_tsp_solver.lin_kernighan_tsp_solver import (
    Tour,
    build_distance_matrix
)


def test_init_and_get_tour():
    """Tests basic Tour initialization and the get_tour() method."""
    order = [0, 1, 2, 3, 4]
    tour = Tour(order)
    assert tour.get_tour() == order, "get_tour() should return the initial order."


def test_next_prev():
    """Tests the next() and prev() methods of the Tour class."""
    order = [2, 0, 3, 1]  # Cycle: 2->0->3->1->2
    tour = Tour(order)
    assert tour.next(2) == 0, "Next node after 2 should be 0."
    assert tour.next(0) == 3, "Next node after 0 should be 3."
    assert tour.prev(2) == 1, "Previous node before 2 should be 1."
    assert tour.prev(0) == 2, "Previous node before 0 should be 2."


def test_sequence_wrap_and_nonwrap():
    """Tests the sequence() method for segments with and without wrapping."""
    order = [0, 1, 2, 3, 4]
    tour = Tour(order)
    # Non-wrap segment: from 1 to 3 includes 2
    assert tour.sequence(1, 2, 3), "Node 2 should be in sequence (1, 2, 3)."
    assert not tour.sequence(1, 4, 3), \
        "Node 4 should not be in sequence (1, _, 3) non-wrapping."
    # Wrap segment: from 3 to 1 wraps through 4->0->1
    assert tour.sequence(3, 4, 1), \
        "Node 4 should be in sequence (3, 4, 1) with wrap."
    assert tour.sequence(3, 0, 1), \
        "Node 0 should be in sequence (3, 0, 1) with wrap."
    assert not tour.sequence(3, 2, 1), \
        "Node 2 should not be in sequence (3, _, 1) with wrap."


def test_flip_no_wrap():
    """
    Tests the flip() method for a non-wrapping segment.
    The flip(a,b) method reverses the segment of nodes from a to b inclusive.
    """
    order = [0, 1, 2, 3, 4]
    tour = Tour(order)
    # Flip segment [1,2,3] (nodes 1,2,3) to [3,2,1]
    tour.flip(1, 3)
    # Expected internal order: [0,3,2,1,4]
    # get_tour() normalizes this to [0,3,2,1,4]
    expected_get_tour_output = [0, 3, 2, 1, 4]
    assert tour.get_tour() == expected_get_tour_output, \
        (f"Flipping non-wrapping segment (1,3) "
         f"resulted in {tour.get_tour()}, expected {expected_get_tour_output}.")
    assert list(tour.order) == [0, 3, 2, 1, 4], "Internal order mismatch."


def test_flip_wrap():
    """
    Tests the flip() method for a wrapping segment.
    The flip(a,b) method reverses the segment of nodes from a to b inclusive.
    """
    order = [0, 1, 2, 3, 4]
    tour = Tour(order)
    # Flip segment [3,4,0,1] (nodes 3,4,0,1) to [1,0,4,3]
    tour.flip(3, 1)
    # Expected internal order: [4,3,2,1,0]
    # get_tour() normalizes this to [0,4,3,2,1]
    expected_get_tour_output = [0, 4, 3, 2, 1]
    assert tour.get_tour() == expected_get_tour_output, \
        (f"Flipping wrapping segment (3,1) "
         f"resulted in {tour.get_tour()}, expected {expected_get_tour_output}.")
    assert list(tour.order) == [4, 3, 2, 1, 0], "Internal order mismatch."


def test_tour_init_with_cost_specific_sequence():
    """Tests Tour initialization with a distance matrix and cost calculation."""
    order = [0, 1, 2]
    coords = np.array([[0, 0], [3, 0], [0, 4]])  # Cost = 3+5+4 = 12
    dist_matrix = build_distance_matrix(coords)

    tour = Tour(order, dist_matrix)
    assert tour.get_tour() == order, "Tour order mismatch after init with cost."
    assert tour.n == 3, "Tour size mismatch."
    assert tour.cost == pytest.approx(12.0), "Tour cost calculation incorrect."

    tour_empty_with_d = Tour([], dist_matrix)
    assert tour_empty_with_d.cost == 0.0, \
        "Empty tour with dist_matrix should have cost 0."
    tour_empty_no_d = Tour([])
    assert tour_empty_no_d.cost == 0.0, \
        "Empty tour without dist_matrix should have cost 0."

    tour_one_node = Tour([0], dist_matrix)
    assert tour_one_node.cost == 0.0, \
        "Single-node tour with dist_matrix should have cost 0."
    tour_one_node_no_d = Tour([0])
    assert tour_one_node_no_d.cost is None, \
        "Single-node tour without dist_matrix should have None cost."


def test_tour_flip_and_update_cost_basic():
    """
    Tests flip_and_update_cost for a basic scenario,
    checking delta, final cost, and order.
    """
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    dist_matrix = build_distance_matrix(coords)
    initial_order_list = [0, 1, 2, 3]  # Cost = 4.0
    tour = Tour(initial_order_list, dist_matrix)
    initial_cost = tour.cost
    assert initial_cost == pytest.approx(4.0), "Initial tour cost incorrect."

    # Flip segment from node 0 to node 2 (nodes 0,1,2).
    # Original tour.order: [0,1,2,3]. Segment [0,1,2] -> reversed [2,1,0].
    # New internal order: [2,1,0,3].
    segment_start_node, segment_end_node = 0, 2
    delta_cost1 = tour.flip_and_update_cost(
        segment_start_node, segment_end_node, dist_matrix
    )

    expected_internal_order_after_flip1 = [2, 1, 0, 3]
    expected_get_tour_after_flip1 = [0, 3, 2, 1]  # Normalized from [2,1,0,3]
    # Cost of [2,1,0,3]: d(2,1)+d(1,0)+d(0,3)+d(3,2) = 1+1+1+sqrt(2) = 3+sqrt(2) ~ 4.414
    expected_cost_after_flip1 = (
        dist_matrix[2, 1] + dist_matrix[1, 0] +
        dist_matrix[0, 3] + dist_matrix[3, 2]
    )

    assert delta_cost1 == pytest.approx(expected_cost_after_flip1 - initial_cost), \
        "Delta cost calculation incorrect."
    assert list(tour.order) == expected_internal_order_after_flip1, \
        "Tour internal order after flip is incorrect."
    assert tour.cost == pytest.approx(expected_cost_after_flip1), \
        "Tour cost after flip is incorrect."
    assert tour.get_tour() == expected_get_tour_after_flip1, \
        "get_tour() output after flip is incorrect."

    # Flip back: current order [2,1,0,3]. Flip segment from node 2 to node 0.
    # Segment [2,1,0] -> reversed [0,1,2].
    # New internal order: [0,1,2,3].
    delta_cost2 = tour.flip_and_update_cost(2, 0, dist_matrix)
    assert delta_cost2 == pytest.approx(initial_cost - expected_cost_after_flip1), \
        "Delta cost for flip back incorrect."
    assert list(tour.order) == initial_order_list, \
        "Tour internal order after flipping back is incorrect."
    assert tour.cost == pytest.approx(initial_cost), \
        "Tour cost after flipping back is incorrect."
    assert tour.get_tour() == initial_order_list, \
        "get_tour() output after flipping back is incorrect."


@pytest.mark.parametrize(
    "initial_order, flip_start_node, flip_end_node, "
    "expected_internal_order_after_flip",
    [
        # Case 1: Flip segment [1,2,3] in [0,1,2,3,4] -> [0,3,2,1,4]
        ([0, 1, 2, 3, 4], 1, 3, [0, 3, 2, 1, 4]),
        # Case 2: Flip segment [0,1,2] in [0,1,2,3,4] -> [2,1,0,3,4]
        ([0, 1, 2, 3, 4], 0, 2, [2, 1, 0, 3, 4]),
        # Case 3: Flip segment [2,3,4] in [0,1,2,3,4] -> [0,1,4,3,2]
        ([0, 1, 2, 3, 4], 2, 4, [0, 1, 4, 3, 2]),
        # Case 4: Flip segment [3,4,0,1] in [0,1,2,3,4] -> [4,3,2,1,0]
        ([0, 1, 2, 3, 4], 3, 1, [4, 3, 2, 1, 0]),
        # Case 5: Flip segment [1,2] in [0,1,2,3,4] -> [0,2,1,3,4]
        ([0, 1, 2, 3, 4], 1, 2, [0, 2, 1, 3, 4]),
        # Case 6: Flip segment [0,1,2,3,4] in [0,1,2,3,4] -> [4,3,2,1,0]
        ([0, 1, 2, 3, 4], 0, 4, [4, 3, 2, 1, 0]),
        # Case 7: Flip segment [2] in [0,1,2,3,4] -> [0,1,2,3,4] (no change)
        ([0, 1, 2, 3, 4], 2, 2, [0, 1, 2, 3, 4]),
        # Case 8: Flip segment [4,5,0,1] in [0,1,2,3,4,5] -> [5,4,2,3,1,0]
        ([0, 1, 2, 3, 4, 5], 4, 1, [5, 4, 2, 3, 1, 0]),
        # Case 9: Flip segment [1,4,0] in [3,1,4,0,2] -> [3,0,4,1,2]
        ([3, 1, 4, 0, 2], 1, 0, [3, 0, 4, 1, 2]),
    ]
)
def test_tour_flip_scenarios(
    initial_order, flip_start_node, flip_end_node,
    expected_internal_order_after_flip
):
    """
    Tests Tour.flip() for various scenarios, checking internal order and pos.
    """
    tour = Tour(initial_order)
    tour.flip(flip_start_node, flip_end_node)

    assert list(tour.order) == expected_internal_order_after_flip, \
        (f"Tour order incorrect after flip({flip_start_node}, "
         f"{flip_end_node}) on {initial_order}. Expected "
         f"{expected_internal_order_after_flip}, got {list(tour.order)}")

    expected_pos = np.empty_like(tour.pos)
    for i, node_val in enumerate(expected_internal_order_after_flip):
        expected_pos[node_val] = i

    for node_in_initial_tour in initial_order:
        assert tour.pos[node_in_initial_tour] == \
            expected_pos[node_in_initial_tour], \
            (f"Position of node {node_in_initial_tour} is incorrect. "
             f"Expected {expected_pos[node_in_initial_tour]}, "
             f"got {tour.pos[node_in_initial_tour]}")

    assert sorted(list(tour.order)) == sorted(initial_order), \
        "Nodes in tour changed after flip."
    assert tour.n == len(initial_order), \
        "Number of nodes in tour changed."


@pytest.mark.parametrize(
    "tour_order, node_a, node_b, node_c, expected_result",
    [
        # Simple cases (no wrap-around)
        ([0, 1, 2, 3, 4], 0, 1, 2, True),
        ([0, 1, 2, 3, 4], 0, 2, 1, False),
        ([0, 1, 2, 3, 4], 0, 3, 4, True),
        ([0, 1, 2, 3, 4], 0, 0, 2, True),   # b is a
        ([0, 1, 2, 3, 4], 0, 2, 2, True),   # b is c
        ([0, 1, 2, 3, 4], 0, 4, 1, False),
        # Wrap-around cases
        ([0, 1, 2, 3, 4], 3, 4, 0, True),
        ([0, 1, 2, 3, 4], 3, 0, 1, True),
        ([0, 1, 2, 3, 4], 4, 0, 2, True),
        ([0, 1, 2, 3, 4], 3, 2, 0, False),
        ([0, 1, 2, 3, 4], 1, 0, 4, False),
        # Edge cases
        ([0, 1, 2], 0, 1, 2, True),
        ([0, 1, 2], 1, 2, 0, True),
        ([0, 1, 2], 2, 0, 1, True),
        ([0, 1, 2], 0, 2, 1, False),
        ([0, 1, 2], 1, 0, 2, False),
        ([0, 1, 2, 3, 4], 1, 1, 1, True),   # a, b, c are the same
        # Cases where a == c: sequence(a,b,a) is True iff b == a.
        ([0, 1, 2, 3, 4], 0, 1, 0, False),
        ([0, 1, 2, 3, 4], 0, 0, 0, True),
        ([0, 1, 2], 0, 2, 0, False),
        # Longer sequence
        ([0, 1, 2, 3, 4, 5, 6], 1, 3, 5, True),
        ([0, 1, 2, 3, 4, 5, 6], 1, 5, 3, False),
        ([0, 1, 2, 3, 4, 5, 6], 5, 0, 2, True),
        ([0, 1, 2, 3, 4, 5, 6], 5, 3, 0, False),
    ]
)
def test_tour_sequence(
    tour_order, node_a, node_b, node_c, expected_result
):
    """
    Tests Tour.sequence(a,b,c), checking if b is on path a to c.
    """
    tour = Tour(tour_order)
    result = tour.sequence(node_a, node_b, node_c)
    assert result == expected_result, \
        (f"Tour({tour_order}).sequence({node_a}, {node_b}, {node_c}) "
         f"expected {expected_result}, got {result}")


@pytest.mark.parametrize(
    "initial_order_nodes, flip_start_node, flip_end_node, "
    "expected_internal_order_after_flip_nodes",
    [
        # Test cases match those in test_tour_flip_scenarios
        ([0, 1, 2, 3, 4], 1, 3, [0, 3, 2, 1, 4]),
        ([0, 1, 2, 3, 4], 0, 2, [2, 1, 0, 3, 4]),
        ([0, 1, 2, 3, 4], 2, 4, [0, 1, 4, 3, 2]),
        ([0, 1, 2, 3, 4], 3, 1, [4, 3, 2, 1, 0]),
        ([0, 1, 2, 3, 4], 1, 2, [0, 2, 1, 3, 4]),
        ([0, 1, 2, 3, 4], 0, 4, [4, 3, 2, 1, 0]),
        ([0, 1, 2, 3, 4], 2, 2, [0, 1, 2, 3, 4]),
    ]
)
def test_tour_flip_and_update_cost_parametrized(
    simple_tsp_setup, initial_order_nodes, flip_start_node, flip_end_node,
    expected_internal_order_after_flip_nodes
):
    """
    Tests Tour.flip_and_update_cost() ensuring order, delta, and final cost.
    """
    _coords, dist_matrix, _initial_tour_obj_fixture, _neighbors, \
        _initial_cost_fixture, _lk_optimal_cost, _lk_optimal_order = \
        simple_tsp_setup

    tour = Tour(initial_order_nodes, dist_matrix)
    initial_cost_for_case = tour.cost

    # Calculate expected cost based on the expected *internal* order after flip
    expected_cost_after_flip = 0.0
    if expected_internal_order_after_flip_nodes:
        n_exp = len(expected_internal_order_after_flip_nodes)
        for i in range(n_exp):
            u_node = expected_internal_order_after_flip_nodes[i]
            v_node = expected_internal_order_after_flip_nodes[(i + 1) % n_exp]
            expected_cost_after_flip += dist_matrix[u_node, v_node]

    expected_delta_cost = expected_cost_after_flip - initial_cost_for_case

    actual_delta_cost = tour.flip_and_update_cost(
        flip_start_node, flip_end_node, dist_matrix
    )

    assert actual_delta_cost == pytest.approx(expected_delta_cost), \
        (f"Incorrect delta_cost for flip({flip_start_node}, "
         f"{flip_end_node}). Expected {expected_delta_cost}, "
         f"got {actual_delta_cost}")
    assert tour.cost == pytest.approx(expected_cost_after_flip), \
        (f"Incorrect tour.cost after flip. Expected "
         f"{expected_cost_after_flip}, got {tour.cost}")
    assert tour.cost == pytest.approx(initial_cost_for_case + actual_delta_cost), \
        "tour.cost is not consistent with initial_cost + actual_delta_cost"
    assert list(tour.order) == expected_internal_order_after_flip_nodes, \
        (f"Tour internal order incorrect after flip. "
         f"Expected {expected_internal_order_after_flip_nodes}, "
         f"got {list(tour.order)}")

    for i, node_val in enumerate(tour.order):
        assert tour.pos[node_val] == i, \
            f"Position of node {node_val} is incorrect in tour.pos"


@pytest.mark.parametrize(
    "tour_order, node_v, expected_next, expected_prev",
    [
        ([0, 1, 2, 3, 4], 1, 2, 0),
        ([0, 1, 2, 3, 4], 0, 1, 4),
        ([0, 1, 2, 3, 4], 4, 0, 3),
        ([7, 5, 9], 7, 5, 9),
        ([7, 5, 9], 5, 9, 7),
        ([7, 5, 9], 9, 7, 5),
        ([10, 20], 10, 20, 20),
        ([10, 20], 20, 10, 10),
        ([5], 5, 5, 5),
    ]
)
def test_tour_next_prev_parametrized(
    tour_order, node_v, expected_next, expected_prev
):
    """
    Tests Tour.next() and Tour.prev() methods for various tour configurations.
    """
    tour = Tour(tour_order)
    assert tour.next(node_v) == expected_next, \
        (f"For tour {tour_order}, next({node_v}) expected {expected_next}, "
         f"got {tour.next(node_v)}")
    assert tour.prev(node_v) == expected_prev, \
        (f"For tour {tour_order}, prev({node_v}) expected {expected_prev}, "
         f"got {tour.prev(node_v)}")


def test_tour_next_prev_empty_tour():
    """
    Tests that Tour.next() and Tour.prev() raise IndexError for an empty tour.
    """
    empty_tour = Tour([])
    with pytest.raises(IndexError, match="Cannot get next node from an empty tour."):
        empty_tour.next(0)
    with pytest.raises(IndexError, match="Cannot get previous node from an empty tour."):
        empty_tour.prev(0)


def test_tour_next_prev_node_not_in_tour_if_pos_small():
    """
    Tests behavior of next()/prev() if pos array is too small for queried node.
    """
    tour = Tour([0, 1, 2])  # Max node label is 2, so pos array is size 3.
    with pytest.raises(IndexError):
        tour.next(5)  # Node 5 is out of bounds for self.pos
    with pytest.raises(IndexError):
        tour.prev(5)  # Node 5 is out of bounds for self.pos


@pytest.mark.parametrize(
    "initial_order_nodes",
    [
        ([0, 1, 2, 3, 4]),
        ([0, 2, 1, 4, 3]),
        ([4, 3, 2, 1, 0]),
        ([0, 1, 2]),
        ([3, 0, 2, 4, 1]),
    ]
)
def test_tour_init_cost_parametrized(simple_tsp_setup, initial_order_nodes):
    """
    Tests Tour.init_cost() by comparing calculated cost against manual sum.
    """
    _coords, dist_matrix, _initial_tour_obj_fixture, _neighbors, \
        _initial_cost_fixture, _lk_optimal_cost, _lk_optimal_order = \
        simple_tsp_setup

    current_n = len(initial_order_nodes)
    expected_cost = 0.0
    if current_n > 0:
        for i in range(current_n):
            node1 = initial_order_nodes[i]
            node2 = initial_order_nodes[(i + 1) % current_n]
            # Ensure nodes are within bounds of the dist_matrix
            if not (0 <= node1 < dist_matrix.shape[0] and
                    0 <= node2 < dist_matrix.shape[0]):
                pytest.fail(
                    f"Node {node1} or {node2} out of bounds for "
                    f"dist_matrix of shape {dist_matrix.shape}"
                )
            expected_cost += dist_matrix[node1, node2]

    tour = Tour(initial_order_nodes, dist_matrix)

    assert tour.n == current_n, \
        f"Tour.n incorrect. Expected {current_n}, got {tour.n}"
    if current_n > 0:
        assert tour.cost == pytest.approx(expected_cost), \
            (f"Tour.cost incorrect for order {initial_order_nodes}. "
             f"Expected {expected_cost}, got {tour.cost}")
    else:
        assert tour.cost == 0.0, "Cost of an empty tour should be 0.0"


def test_tour_init_cost_empty():
    """Tests Tour.init_cost() with an empty tour, ensuring cost is 0."""
    dummy_dist_matrix = np.array([[]])
    tour = Tour([], dummy_dist_matrix)
    assert tour.n == 0, "Empty tour should have n=0."
    assert tour.cost == 0.0, "Empty tour should have cost 0."


@pytest.mark.parametrize(
    "initial_order, expected_get_tour_output",
    [
        # Case 1: Already starts with 0
        ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
        # Case 2: Contains 0, but not at the start
        ([1, 2, 0, 3, 4], [0, 3, 4, 1, 2]),
        ([4, 3, 2, 1, 0], [0, 4, 3, 2, 1]),
        # Case 3: Does not contain 0
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([5, 4, 3, 2, 1], [5, 4, 3, 2, 1]),
        # Case 4: Empty tour
        ([], []),
        # Case 5: Single node tour (with 0)
        ([0], [0]),
        # Case 6: Single node tour (without 0)
        ([7], [7]),
        # Case 7: Tour with 0 at the end
        ([1, 2, 3, 0], [0, 1, 2, 3]),
    ]
)
def test_tour_get_tour_normalization(
    initial_order, expected_get_tour_output
):
    """
    Tests Tour.get_tour() normalization (starts with 0 if present).
    """
    tour = Tour(initial_order)
    retrieved_tour = tour.get_tour()
    assert retrieved_tour == expected_get_tour_output, \
        (f"get_tour() for initial order {initial_order} "
         f"expected {expected_get_tour_output}, got {retrieved_tour}")

    # Check that the internal tour.order remains unchanged by get_tour()
    if initial_order:
        assert list(tour.order) == initial_order, \
            "get_tour() should not modify internal tour.order"
    else:
        assert len(tour.order) == 0, \
            "Internal order should be empty for an empty initial tour"
