"""
Unit tests for the starting cycles module.

This module tests all the different starting cycle algorithms including:
- Random permutation
- Nearest neighbor heuristic
- Greedy edge selection
- Borůvka's MST-based construction
- Quick Borůvka (QBorůvka)
"""

import pytest
import numpy as np
from lin_kernighan_tsp_solver.starting_cycles import (
    generate_starting_cycle,
    _natural_tour,
    _random_tour,
    _nearest_neighbor_tour,
    _greedy_tour,
    _boruvka_tour,
    _qboruvka_tour,
    _edges_to_tour,
    _boruvka_mst,
    _mst_to_tour,
    _improve_tour_2opt
)
from lin_kernighan_tsp_solver.config import STARTING_CYCLE_CONFIG


@pytest.fixture
def simple_coords():
    """Simple 4-node coordinates for testing."""
    return np.array([
        [0.0, 0.0],  # Node 0
        [1.0, 0.0],  # Node 1
        [0.0, 1.0],  # Node 2
        [1.0, 1.0]   # Node 3
    ])


@pytest.fixture
def triangle_coords():
    """Three-node triangle coordinates."""
    return np.array([
        [0.0, 0.0],  # Node 0
        [1.0, 0.0],  # Node 1
        [0.5, 0.866]  # Node 2 (equilateral triangle)
    ])


class TestGenerateStartingCycle:
    """Tests for the main generate_starting_cycle function."""

    def test_default_method(self, simple_coords):
        """Test that default method works."""
        tour = generate_starting_cycle(simple_coords)
        assert len(tour) == 4
        assert set(tour) == {0, 1, 2, 3}

    def test_explicit_methods(self, simple_coords):
        """Test all methods explicitly."""
        methods = STARTING_CYCLE_CONFIG["AVAILABLE_METHODS"]
        for method in methods:
            tour = generate_starting_cycle(simple_coords, method=method)
            assert len(tour) == 4
            assert set(tour) == {0, 1, 2, 3}

    def test_invalid_method(self, simple_coords):
        """Test error handling for invalid method."""
        with pytest.raises(ValueError, match="Unknown starting cycle method"):
            generate_starting_cycle(simple_coords, method="invalid_method")

    def test_empty_coords(self):
        """Test handling of empty coordinates."""
        empty_coords = np.array([]).reshape(0, 2)
        tour = generate_starting_cycle(empty_coords)
        assert tour == []

    def test_single_node(self):
        """Test handling of single node."""
        single_node = np.array([[0.0, 0.0]])
        tour = generate_starting_cycle(single_node)
        assert tour == [0]

    def test_invalid_coords_shape(self):
        """Test error handling for invalid coordinate shape."""
        invalid_coords = np.array([1, 2, 3])  # 1D array
        with pytest.raises(ValueError, match="Coordinates must be a 2D array"):
            generate_starting_cycle(invalid_coords)

        invalid_coords_3d = np.array([[[1, 2], [3, 4]]])  # 3D array
        with pytest.raises(ValueError, match="Coordinates must be a 2D array"):
            generate_starting_cycle(invalid_coords_3d)

        invalid_dims = np.array([[1, 2, 3], [4, 5, 6]])  # 3 dimensions per point
        with pytest.raises(ValueError, match="Coordinates must be a 2D array"):
            generate_starting_cycle(invalid_dims)


class TestRandomTour:
    """Tests for random tour generation."""

    def test_random_tour_length(self):
        """Test that random tour has correct length."""
        for n in [3, 5, 10, 20]:
            tour = _random_tour(n)
            assert len(tour) == n
            assert set(tour) == set(range(n))

    def test_random_tour_different(self):
        """Test that random tours are different (probabilistically)."""
        tours = [_random_tour(10) for _ in range(10)]
        # Very unlikely all tours are identical
        assert len(set(tuple(tour) for tour in tours)) > 1


class TestNearestNeighborTour:
    """Tests for nearest neighbor tour generation."""

    def test_nearest_neighbor_basic(self, simple_coords):
        """Test basic nearest neighbor functionality."""
        tour = _nearest_neighbor_tour(simple_coords)
        assert len(tour) == 4
        assert set(tour) == {0, 1, 2, 3}
        assert tour[0] == 0  # Should start from node 0 by default

    def test_nearest_neighbor_custom_start(self, simple_coords):
        """Test nearest neighbor with custom start node."""
        tour = _nearest_neighbor_tour(simple_coords, start_node=2)
        assert len(tour) == 4
        assert set(tour) == {0, 1, 2, 3}
        assert tour[0] == 2

    def test_nearest_neighbor_invalid_start(self, simple_coords):
        """Test nearest neighbor with invalid start node."""
        tour = _nearest_neighbor_tour(simple_coords, start_node=10)
        assert len(tour) == 4
        assert set(tour) == {0, 1, 2, 3}
        assert tour[0] == 0  # Should fallback to 0

    def test_nearest_neighbor_triangle(self, triangle_coords):
        """Test nearest neighbor on triangle."""
        tour = _nearest_neighbor_tour(triangle_coords)
        assert len(tour) == 3
        assert set(tour) == {0, 1, 2}


class TestGreedyTour:
    """Tests for greedy tour generation."""

    def test_greedy_basic(self, simple_coords):
        """Test basic greedy functionality."""
        tour = _greedy_tour(simple_coords)
        assert len(tour) == 4
        assert set(tour) == {0, 1, 2, 3}

    def test_greedy_max_edges(self, simple_coords):
        """Test greedy with edge limit."""
        tour = _greedy_tour(simple_coords, max_edges=3)
        assert len(tour) == 4
        assert set(tour) == {0, 1, 2, 3}

    def test_greedy_triangle(self, triangle_coords):
        """Test greedy on triangle."""
        tour = _greedy_tour(triangle_coords)
        assert len(tour) == 3
        assert set(tour) == {0, 1, 2}


class TestBoruvkaTour:
    """Tests for Borůvka tour generation."""

    def test_boruvka_basic(self, simple_coords):
        """Test basic Borůvka functionality."""
        tour = _boruvka_tour(simple_coords)
        assert len(tour) == 4
        assert set(tour) == {0, 1, 2, 3}

    def test_boruvka_triangle(self, triangle_coords):
        """Test Borůvka on triangle."""
        tour = _boruvka_tour(triangle_coords)
        assert len(tour) == 3
        assert set(tour) == {0, 1, 2}

    def test_boruvka_two_nodes(self):
        """Test Borůvka on two nodes."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        tour = _boruvka_tour(coords)
        assert len(tour) == 2
        assert set(tour) == {0, 1}


class TestQBoruvkaTour:
    """Tests for Quick Borůvka tour generation."""

    def test_qboruvka_basic(self, simple_coords):
        """Test basic QBorůvka functionality."""
        tour = _qboruvka_tour(simple_coords)
        assert len(tour) == 4
        assert set(tour) == {0, 1, 2, 3}

    def test_qboruvka_iterations(self, simple_coords):
        """Test QBorůvka with different iteration counts."""
        tour1 = _qboruvka_tour(simple_coords, iterations=1)
        tour2 = _qboruvka_tour(simple_coords, iterations=5)
        assert len(tour1) == 4
        assert len(tour2) == 4
        assert set(tour1) == {0, 1, 2, 3}
        assert set(tour2) == {0, 1, 2, 3}

    def test_qboruvka_time_limit(self):
        """Test that QBoruvka respects time limits during 2-opt improvement."""
        import time
        from lin_kernighan_tsp_solver.starting_cycles import _improve_tour_2opt

        # Create a larger tour that would take time to optimize
        n = 100
        coords = np.random.rand(n, 2) * 100
        distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
        tour = list(range(n))

        # Test with very short time limit
        start_time = time.time()
        _improve_tour_2opt(tour, distances, max_time=0.01)
        elapsed = time.time() - start_time

        # Should respect the time limit (allowing some overhead)
        assert elapsed < 0.1, f"2-opt took {elapsed:.3f}s, should be < 0.1s"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_edges_to_tour(self):
        """Test edge list to tour conversion."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        tour = _edges_to_tour(edges, 4)
        assert len(tour) == 4
        assert set(tour) == {0, 1, 2, 3}

    def test_edges_to_tour_empty(self):
        """Test edge list to tour with empty edges."""
        tour = _edges_to_tour([], 3)
        assert len(tour) == 3
        assert set(tour) == {0, 1, 2}

    def test_boruvka_mst(self, simple_coords):
        """Test Borůvka MST algorithm."""
        distances = np.linalg.norm(simple_coords[:, None] - simple_coords[None, :], axis=2)
        mst_edges = _boruvka_mst(distances, 4)
        assert len(mst_edges) == 3  # MST has n-1 edges

    def test_mst_to_tour(self):
        """Test MST to tour conversion."""
        mst_edges = [(0, 1), (1, 2), (2, 3)]
        tour = _mst_to_tour(mst_edges, 4)
        assert len(tour) == 4
        assert set(tour) == {0, 1, 2, 3}

    def test_improve_tour_2opt(self, simple_coords):
        """Test 2-opt improvement."""
        distances = np.linalg.norm(simple_coords[:, None] - simple_coords[None, :], axis=2)
        initial_tour = [0, 2, 1, 3]
        improved_tour = _improve_tour_2opt(initial_tour, distances)
        assert len(improved_tour) == 4
        assert set(improved_tour) == {0, 1, 2, 3}

    def test_improve_tour_2opt_small(self):
        """Test 2-opt on small tours."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
        tour = [0, 1]
        improved_tour = _improve_tour_2opt(tour, distances)
        assert improved_tour == [0, 1]

        # Test single node
        improved_single = _improve_tour_2opt([0], distances[:1, :1])
        assert improved_single == [0]


class TestIntegration:
    """Integration tests for starting cycles."""

    def test_all_methods_consistent_output(self, simple_coords):
        """Test that all methods produce valid tours."""
        methods = STARTING_CYCLE_CONFIG["AVAILABLE_METHODS"]
        for method in methods:
            tour = generate_starting_cycle(simple_coords, method=method)
            assert len(tour) == 4
            assert set(tour) == {0, 1, 2, 3}
            # Each node appears exactly once
            assert len(set(tour)) == len(tour)

    def test_performance_comparison(self, simple_coords):
        """Test that different methods can produce different quality tours."""
        distances = np.linalg.norm(simple_coords[:, None] - simple_coords[None, :], axis=2)

        def tour_length(tour):
            length = 0.0
            for i in range(len(tour)):
                length += distances[tour[i], tour[(i + 1) % len(tour)]]
            return length

        methods = ["nearest_neighbor", "greedy", "qboruvka"]
        lengths = {}

        for method in methods:
            tour = generate_starting_cycle(simple_coords, method=method)
            lengths[method] = tour_length(tour)

        # All should produce finite, positive lengths
        for method, length in lengths.items():
            assert length > 0
            assert np.isfinite(length)

    def test_larger_instance(self):
        """Test on a larger instance."""
        np.random.seed(42)  # For reproducibility
        n = 10
        coords = np.random.rand(n, 2) * 100

        methods = STARTING_CYCLE_CONFIG["AVAILABLE_METHODS"]
        for method in methods:
            tour = generate_starting_cycle(coords, method=method)
            assert len(tour) == n
            assert set(tour) == set(range(n))

    def test_config_integration(self):
        """Test integration with configuration."""
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

        # Test default method from config
        default_method = STARTING_CYCLE_CONFIG["DEFAULT_METHOD"]
        tour1 = generate_starting_cycle(coords)
        tour2 = generate_starting_cycle(coords, method=default_method)

        # Both should use the same method
        assert len(tour1) == len(tour2) == 4
        assert set(tour1) == set(tour2) == {0, 1, 2, 3}


class TestNaturalTour:
    """Tests for natural order tour generation."""

    def test_natural_tour_length(self):
        """Test that natural tour has correct length."""
        tour = _natural_tour(5)
        assert len(tour) == 5

        tour = _natural_tour(10)
        assert len(tour) == 10

    def test_natural_tour_order(self):
        """Test that natural tour is in natural order."""
        tour = _natural_tour(5)
        assert tour == [0, 1, 2, 3, 4]

        tour = _natural_tour(10)
        assert tour == list(range(10))

    def test_natural_tour_empty(self):
        """Test natural tour with zero nodes."""
        tour = _natural_tour(0)
        assert tour == []

    def test_natural_tour_single(self):
        """Test natural tour with single node."""
        tour = _natural_tour(1)
        assert tour == [0]
