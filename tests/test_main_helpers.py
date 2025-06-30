import numpy as np
import pytest
from lin_kernighan_tsp_solver.main import (
    _calculate_tour_length,
    _calculate_gap
)

# --- Tests for _calculate_tour_length ---


def test_calculate_tour_length_empty_tour():
    """Test that an empty tour has a length of 0."""
    assert _calculate_tour_length([], np.array([])) == 0.0


def test_calculate_tour_length_single_node():
    """Test that a tour with a single node has a length of 0."""
    D = np.array([[0]])
    assert _calculate_tour_length([0], D) == 0.0


def test_calculate_tour_length_simple_tour():
    """Test a simple valid tour calculation."""
    # A square with side length 1. Tour 0-1-2-3-0. Length = 4.
    D = np.array([
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0]
    ])
    tour = [0, 1, 2, 3]
    assert _calculate_tour_length(tour, D) == pytest.approx(4.0)

# --- Tests for _calculate_gap ---


def test_calculate_gap_no_opt_len():
    """Test that gap is None if optimal length is not provided."""
    assert _calculate_gap(100.0, None) is None


def test_calculate_gap_normal_case():
    """Test a standard gap calculation."""
    assert _calculate_gap(110.0, 100.0) == pytest.approx(10.0)


def test_calculate_gap_no_improvement():
    """Test that gap is 0 if heuristic length equals optimal length."""
    assert _calculate_gap(100.0, 100.0) == pytest.approx(0.0)


def test_calculate_gap_better_than_optimal():
    """Test that gap is 0 if heuristic finds a (theoretically impossible) better tour."""
    assert _calculate_gap(90.0, 100.0) == pytest.approx(0.0)


def test_calculate_gap_zero_optimal_length():
    """Test handling of division by zero when optimal length is 0."""
    # If both are 0, gap is 0.
    assert _calculate_gap(0.0, 0.0) == pytest.approx(0.0)
    # If heuristic is > 0 but optimal is 0, gap is infinite.
    assert _calculate_gap(10.0, 0.0) == float('inf')


def test_calculate_gap_very_small_opt_len():
    """Test that gap is None for very small, non-zero optimal lengths."""
    from lin_kernighan_tsp_solver.config import FLOAT_COMPARISON_TOLERANCE
    small_val = FLOAT_COMPARISON_TOLERANCE
    assert _calculate_gap(10.0, small_val) is None
