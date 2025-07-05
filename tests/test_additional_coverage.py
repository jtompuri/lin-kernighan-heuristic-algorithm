"""Additional tests to improve coverage of edge cases and error handling."""

import pytest
from unittest.mock import patch

from lin_kernighan_tsp_solver.main import process_single_instance
from lin_kernighan_tsp_solver.starting_cycles import generate_starting_cycle
from lin_kernighan_tsp_solver.utils import save_heuristic_tour
import numpy as np


class TestMainIOErrorHandling:
    """Test error handling in main.py process_single_instance function."""

    def test_process_single_instance_save_tour_failure(self, capsys, tmp_path):
        """Test handling of IOError when saving tours fails."""
        # Create a simple test TSP file
        tsp_content = """NAME: test_save_error
TYPE: TSP
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 1.0 0.0
3 0.5 1.0
EOF
"""
        tsp_file = tmp_path / "test_save_error.tsp"
        tsp_file.write_text(tsp_content)

        # Mock save_heuristic_tour to raise IOError
        with patch('lin_kernighan_tsp_solver.main.save_heuristic_tour',
                   side_effect=IOError("Permission denied")):

            result = process_single_instance(
                str(tsp_file),
                "",  # No opt tour file
                time_limit=1.0,
                verbose=True
            )

        # Should complete successfully despite save error
        assert result is not None
        assert 'error' not in result or not result['error']

        # Check that warning was printed
        captured = capsys.readouterr()
        assert "Warning: Failed to save tour: Permission denied" in captured.out


class TestStartingCyclesLargeInstances:
    """Test fallback behavior for large instances in starting cycles."""

    def test_greedy_tour_large_instance_fallback(self):
        """Test that greedy tour falls back to nearest neighbor for large instances."""
        # Create a large instance (>1000 nodes)
        n_nodes = 1001
        coords = np.random.RandomState(42).randint(0, 100, size=(n_nodes, 2))

        with patch('lin_kernighan_tsp_solver.starting_cycles._nearest_neighbor_tour') as mock_nn:
            mock_nn.return_value = list(range(n_nodes))

            result = generate_starting_cycle(coords, method="greedy")

            # Should have called nearest neighbor fallback
            mock_nn.assert_called_once_with(coords)
            assert result == list(range(n_nodes))

    def test_mst_tour_empty_edges(self):
        """Test MST tour building with empty edge list."""
        from lin_kernighan_tsp_solver.starting_cycles import _mst_to_tour

        n_nodes = 5
        mst_edges = []  # Empty edge list

        result = _mst_to_tour(mst_edges, n_nodes)

        # Should return natural order when no edges
        assert result == list(range(n_nodes))

    def test_mst_tour_unvisited_nodes(self):
        """Test MST tour building handles unvisited nodes."""
        from lin_kernighan_tsp_solver.starting_cycles import _mst_to_tour

        n_nodes = 5
        # Create MST that doesn't connect all nodes (edge case)
        mst_edges = [(0, 1), (1, 2)]  # Nodes 3, 4 not connected

        result = _mst_to_tour(mst_edges, n_nodes)

        # Should include all nodes
        assert len(result) == n_nodes
        assert set(result) == set(range(n_nodes))

    def test_2opt_improvement_time_limit(self):
        """Test 2-opt improvement respects time limit."""
        from lin_kernighan_tsp_solver.starting_cycles import _improve_tour_2opt

        tour = [0, 1, 2, 3, 4]
        n = len(tour)
        distances = np.random.RandomState(42).rand(n, n)
        # Make symmetric
        distances = (distances + distances.T) / 2
        np.fill_diagonal(distances, 0)

        # Very short time limit should cause early exit
        result = _improve_tour_2opt(tour.copy(), distances, max_time=0.0001)

        # Should return some result (may or may not be improved)
        assert len(result) == len(tour)
        assert set(result) == set(tour)


class TestUtilsImportErrorHandling:
    """Test import error handling in utils.py."""

    def test_matplotlib_import_with_tkinter_error(self):
        """Test matplotlib import error handling when tkinter is missing."""
        # This is tricky to test because imports happen at module level
        # We can test the _check_tkinter_available function behavior
        from lin_kernighan_tsp_solver.utils import _check_tkinter_available

        # Mock import to raise ImportError for tkinter
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'tkinter':
                raise ImportError("No module named 'tkinter'")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            result = _check_tkinter_available()
            assert result is False


class TestEdgeCaseScenarios:
    """Test additional edge cases for better coverage."""

    def test_save_heuristic_tour_io_error_with_permissions(self, tmp_path):
        """Test save_heuristic_tour with permission errors."""
        tour = [0, 1, 2]
        problem_name = "test_permission"
        tour_length = 123.45

        # Create a directory with restrictive permissions
        restricted_dir = tmp_path / "restricted"
        restricted_dir.mkdir()
        restricted_dir.chmod(0o444)  # Read-only

        try:
            with pytest.raises(IOError, match="Failed to save tour"):
                save_heuristic_tour(tour, problem_name, tour_length, str(restricted_dir))
        finally:
            # Restore permissions for cleanup
            restricted_dir.chmod(0o755)

    def test_natural_tour_edge_cases(self):
        """Test natural tour generation edge cases."""
        from lin_kernighan_tsp_solver.starting_cycles import _natural_tour

        # Test with 0 nodes
        result = _natural_tour(0)
        assert result == []

        # Test with 1 node
        result = _natural_tour(1)
        assert result == [0]

        # Test with large number
        result = _natural_tour(1000)
        assert result == list(range(1000))


if __name__ == "__main__":
    pytest.main([__file__])
