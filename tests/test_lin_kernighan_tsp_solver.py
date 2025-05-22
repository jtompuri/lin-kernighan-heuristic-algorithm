"""
Unit tests for the Tour class in the Lin-Kernighan TSP solver.

This module contains a suite of pytest tests designed to verify the correctness
of the Tour data structure, which is a core component of the Lin-Kernighan
heuristic implementation. Tests cover initialization, neighbor finding (next/prev),
sequence checking (both wrapping and non-wrapping segments), and the critical
flip operation for various scenarios.
"""
from lin_kernighan_tsp_solver import Tour


def test_init_and_get_tour():
    # Basic initialization and round-trip
    order = [0, 1, 2, 3, 4]
    t = Tour(order)
    assert t.get_tour() == order


def test_next_prev():
    # Check next() and prev() in a simple 4-node cycle
    order = [2, 0, 3, 1]
    # Cycle: 2->0->3->1->2
    t = Tour(order)
    assert t.next(2) == 0
    assert t.next(0) == 3
    assert t.prev(2) == 1
    assert t.prev(0) == 2


def test_sequence_wrap_and_nonwrap():
    # Using order [0,1,2,3,4]
    order = [0, 1, 2, 3, 4]
    t = Tour(order)
    # Non-wrap segment: from 1 to 3 includes 2
    assert t.sequence(1, 2, 3)
    assert not t.sequence(1, 4, 3)
    # Wrap segment: from 3 to 1 wraps through 4->0->1
    assert t.sequence(3, 4, 1)
    assert t.sequence(3, 0, 1)
    assert not t.sequence(3, 2, 1)


def test_flip_no_wrap():
    # Flip the segment [1,2,3] in 0-4 cycle
    order = [0, 1, 2, 3, 4]
    t = Tour(order)
    t.flip(1, 3)
    # Expect: [0,3,2,1,4]
    assert t.get_tour() == [0, 3, 2, 1, 4]


def test_flip_wrap():
    # Flip the segment 3->4->0->1
    order = [0, 1, 2, 3, 4]
    t = Tour(order)
    t.flip(3, 1)
    # Expect: [0,4,3,2,1]
    assert t.get_tour() == [0, 4, 3, 2, 1]


def test_flip_idempotence():
    # Flipping the same segment twice returns to original
    order = [0, 1, 2, 3, 4, 5]
    t = Tour(order)
    original = t.get_tour()
    # choose a and b
    a, b = 2, 5
    t.flip(a, b)
    t.flip(a, b)
    assert t.get_tour() == original
