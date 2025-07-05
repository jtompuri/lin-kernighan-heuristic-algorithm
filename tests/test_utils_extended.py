"""
Extended unit tests for utility functions in utils.py to improve test coverage.

This module tests functions that were previously not comprehensively covered:
- save_heuristic_tour()
- _configure_matplotlib_backend()
- _check_tkinter_available()
- Additional edge cases for plot_all_tours() and display_summary_table()
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import builtins

from lin_kernighan_tsp_solver.utils import (
    save_heuristic_tour,
    _configure_matplotlib_backend,
    _check_tkinter_available,
    plot_all_tours,
    display_summary_table
)


class TestSaveHeuristicTour:
    """Test cases for save_heuristic_tour function."""

    def test_save_heuristic_tour_success(self, tmp_path):
        """Test successful tour saving with default folder."""
        tour = [0, 1, 2, 3]
        problem_name = "test_problem"
        tour_length = 123.45
        
        with patch('lin_kernighan_tsp_solver.config.SOLUTIONS_FOLDER_PATH', str(tmp_path)):
            result_path = save_heuristic_tour(tour, problem_name, tour_length)
            
        expected_path = tmp_path / "test_problem.heu.tour"
        assert result_path == str(expected_path)
        assert expected_path.exists()
        
        # Verify file content
        content = expected_path.read_text()
        assert "NAME: test_problem.heu.tour" in content
        assert "TYPE: TOUR" in content
        assert "COMMENT: Heuristic tour (Lin-Kernighan), length 123.45" in content
        assert "DIMENSION: 4" in content
        assert "TOUR_SECTION" in content
        assert "1\n2\n3\n4\n" in content  # 1-indexed nodes
        assert "-1" in content
        assert "EOF" in content

    def test_save_heuristic_tour_custom_folder(self, tmp_path):
        """Test tour saving with custom solutions folder."""
        tour = [0, 1, 2]
        problem_name = "custom_test"
        tour_length = 456.78
        custom_folder = tmp_path / "custom_solutions"
        
        result_path = save_heuristic_tour(tour, problem_name, tour_length, str(custom_folder))
        
        expected_path = custom_folder / "custom_test.heu.tour"
        assert result_path == str(expected_path)
        assert expected_path.exists()
        assert custom_folder.exists()  # Folder should be created

    def test_save_heuristic_tour_empty_tour(self, tmp_path):
        """Test saving empty tour."""
        tour = []
        problem_name = "empty_tour"
        tour_length = 0.0
        
        with patch('lin_kernighan_tsp_solver.config.SOLUTIONS_FOLDER_PATH', str(tmp_path)):
            save_heuristic_tour(tour, problem_name, tour_length)
            
        expected_path = tmp_path / "empty_tour.heu.tour"
        content = expected_path.read_text()
        assert "DIMENSION: 0" in content
        assert "TOUR_SECTION" in content
        assert "-1" in content

    def test_save_heuristic_tour_io_error(self, tmp_path):
        """Test IOError handling when file cannot be written."""
        tour = [0, 1, 2]
        problem_name = "test_problem"
        tour_length = 123.45
        
        # Create a file that will cause permission error
        test_file = tmp_path / "test_problem.heu.tour"
        test_file.write_text("existing")
        test_file.chmod(0o444)  # Read-only
        
        with patch('lin_kernighan_tsp_solver.config.SOLUTIONS_FOLDER_PATH', str(tmp_path)):
            with pytest.raises(IOError, match="Failed to save tour"):
                save_heuristic_tour(tour, problem_name, tour_length)

    def test_save_heuristic_tour_large_tour(self, tmp_path):
        """Test saving large tour with many nodes."""
        tour = list(range(100))  # 100 nodes
        problem_name = "large_tour"
        tour_length = 9999.99
        
        with patch('lin_kernighan_tsp_solver.config.SOLUTIONS_FOLDER_PATH', str(tmp_path)):
            save_heuristic_tour(tour, problem_name, tour_length)
            
        expected_path = tmp_path / "large_tour.heu.tour"
        content = expected_path.read_text()
        assert "DIMENSION: 100" in content
        
        # Check that all nodes are present (1-indexed)
        for i in range(1, 101):
            assert f"{i}\n" in content


class TestMatplotlibBackendConfiguration:
    """Test cases for matplotlib backend configuration functions."""

    def test_configure_matplotlib_backend_non_interactive(self):
        """Test non-interactive backend configuration."""
        with patch('matplotlib.use') as mock_use:
            result = _configure_matplotlib_backend(interactive=False)
            
        mock_use.assert_called_once_with('Agg')
        assert result is False

    def test_configure_matplotlib_backend_interactive_success(self):
        """Test interactive backend configuration when tkinter is available."""
        with patch('matplotlib.use') as mock_use, \
             patch('importlib.import_module') as mock_import:
            # Mock successful tkinter import
            mock_import.return_value = MagicMock()
            
            result = _configure_matplotlib_backend(interactive=True)
            
        mock_use.assert_called_once_with('TkAgg')
        assert result is True

    def test_configure_matplotlib_backend_interactive_fallback(self, capsys):
        """Test interactive backend fallback when tkinter is not available."""
        # Mock the import to raise ImportError when tkinter is imported
        original_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'tkinter':
                raise ImportError("No module named 'tkinter'")
            return original_import(name, *args, **kwargs)
        
        with patch('matplotlib.use') as mock_use, \
             patch('builtins.__import__', side_effect=mock_import):
            
            result = _configure_matplotlib_backend(interactive=True)
            
        # Should call matplotlib.use once for Agg after tkinter import fails
        mock_use.assert_called_once_with('Agg')
        assert result is False
        
        captured = capsys.readouterr()
        assert "Warning: Interactive plotting requested but tkinter not available" in captured.out
        assert "Falling back to non-interactive plotting" in captured.out

    def test_check_tkinter_available_success(self):
        """Test _check_tkinter_available when tkinter is available."""
        # Mock the import to succeed when tkinter is imported
        original_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'tkinter':
                return MagicMock()  # Return a mock tkinter module
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = _check_tkinter_available()
            
        assert result is True

    def test_check_tkinter_available_failure(self):
        """Test _check_tkinter_available when tkinter is not available."""
        # Mock the import to raise ImportError when tkinter is imported
        original_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'tkinter':
                raise ImportError("No module named 'tkinter'")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = _check_tkinter_available()
            
        assert result is False


class TestPlotAllToursExtended:
    """Extended test cases for plot_all_tours function."""

    def test_plot_all_tours_force_save_plot(self, tmp_path):
        """Test plot_all_tours with force_save_plot=True."""
        results_data = [{
            'name': 'test_instance',
            'error': False,
            'coords': np.array([[0, 0], [1, 1], [2, 0]]),
            'heu_tour': [0, 1, 2],
            'opt_tour': [0, 2, 1],
            'gap': 5.0
        }]
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close, \
             patch('pathlib.Path.mkdir'), \
             patch('lin_kernighan_tsp_solver.utils._tkinter_available', True):
            
            # Mock figure and axes - use a simple object that behaves like ndarray
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            
            # Create a mock axes array that supports flatten()
            mock_axes = MagicMock()
            mock_axes.flatten.return_value = [mock_ax]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            plot_all_tours(results_data, force_save_plot=True)
            
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        # Should not call plt.show() when force_save_plot=True

    def test_plot_all_tours_interactive_mode(self):
        """Test plot_all_tours in interactive mode."""
        results_data = [{
            'name': 'test_instance',
            'error': False,
            'coords': np.array([[0, 0], [1, 1]]),
            'heu_tour': [0, 1],
            'opt_tour': None,
            'gap': None
        }]
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.show') as mock_show, \
             patch('matplotlib.pyplot.close') as mock_close, \
             patch('lin_kernighan_tsp_solver.utils._tkinter_available', True), \
             patch('lin_kernighan_tsp_solver.utils._configure_matplotlib_backend', return_value=True):
            
            # Mock figure and axes - use a simple object that behaves like ndarray
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            
            # Create a mock axes array that supports flatten()
            mock_axes = MagicMock()
            mock_axes.flatten.return_value = [mock_ax]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            plot_all_tours(results_data, force_save_plot=False)
            
        mock_show.assert_called_once()
        mock_close.assert_called_once()

    def test_plot_all_tours_filename_generation_single(self, tmp_path):
        """Test filename generation for single problem."""
        results_data = [{
            'name': 'single_problem',
            'error': False,
            'coords': np.array([[0, 0], [1, 1]]),
            'heu_tour': [0, 1],
            'opt_tour': None
        }]
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close'), \
             patch('pathlib.Path.mkdir'):
            
            # Mock figure and axes - use a simple object that behaves like ndarray
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            
            # Create a mock axes array that supports flatten()
            mock_axes = MagicMock()
            mock_axes.flatten.return_value = [mock_ax]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            plot_all_tours(results_data, force_save_plot=True)
            
        # Check that savefig was called with correct filename
        args, kwargs = mock_savefig.call_args
        assert 'single_problem_tour.png' in str(args[0])

    def test_plot_all_tours_filename_generation_multiple_small(self):
        """Test filename generation for multiple problems (<=3)."""
        results_data = [
            {
                'name': 'prob1',
                'error': False,
                'coords': np.array([[0, 0], [1, 1]]),
                'heu_tour': [0, 1]
            },
            {
                'name': 'prob2',
                'error': False,
                'coords': np.array([[0, 0], [1, 1]]),
                'heu_tour': [0, 1]
            }
        ]
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close'), \
             patch('pathlib.Path.mkdir'):
            
            # Mock figure and axes - use a simple object that behaves like ndarray
            mock_fig = MagicMock()
            mock_ax1, mock_ax2 = MagicMock(), MagicMock()
            
            # Create a mock axes array that supports flatten()
            mock_axes = MagicMock()
            mock_axes.flatten.return_value = [mock_ax1, mock_ax2]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            plot_all_tours(results_data, force_save_plot=True)
            
        # Check that savefig was called with correct filename
        args, kwargs = mock_savefig.call_args
        assert 'prob1_prob2_tours.png' in str(args[0])

    def test_plot_all_tours_filename_generation_multiple_large(self):
        """Test filename generation for many problems (>3)."""
        results_data = [
            {
                'name': f'prob{i}',
                'error': False,
                'coords': np.array([[0, 0], [1, 1]]),
                'heu_tour': [0, 1]
            }
            for i in range(5)
        ]
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close'), \
             patch('pathlib.Path.mkdir'):
            
            # Mock figure and axes - use a simple object that behaves like ndarray
            mock_fig = MagicMock()
            mock_axes_list = [MagicMock() for _ in range(6)]  # 3x2 grid has 6 slots
            
            # Create a mock axes array that supports flatten()
            mock_axes = MagicMock()
            mock_axes.flatten.return_value = mock_axes_list
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            plot_all_tours(results_data, force_save_plot=True)
            
        # Check that savefig was called with correct filename format
        args, kwargs = mock_savefig.call_args
        filename = str(args[0])
        assert 'prob0_to_prob4_5_tours.png' in filename

    def test_plot_all_tours_backend_fallback(self, capsys):
        """Test backend fallback when interactive mode fails."""
        results_data = [{
            'name': 'test',
            'error': False,
            'coords': np.array([[0, 0], [1, 1]]),
            'heu_tour': [0, 1]
        }]
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('pathlib.Path.mkdir'), \
             patch('lin_kernighan_tsp_solver.utils._tkinter_available', True), \
             patch('lin_kernighan_tsp_solver.utils._configure_matplotlib_backend', return_value=False):
            
            # Mock figure and axes - use a simple object that behaves like ndarray
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            
            # Create a mock axes array that supports flatten()
            mock_axes = MagicMock()
            mock_axes.flatten.return_value = [mock_ax]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            plot_all_tours(results_data, force_save_plot=False)
            
        captured = capsys.readouterr()
        assert "Falling back to saving plot to file" in captured.out

    def test_plot_all_tours_empty_coordinates(self, capsys):
        """Test plot_all_tours with empty coordinates."""
        results_data = [{
            'name': 'empty_coords',
            'error': False,
            'coords': np.array([]),
            'heu_tour': []
        }]
        
        plot_all_tours(results_data)
        
        captured = capsys.readouterr()
        assert "No valid results with coordinates to plot" in captured.out

    def test_plot_all_tours_no_tkinter_message(self, capsys):
        """Test that no tkinter message is displayed when saving plots."""
        results_data = [{
            'name': 'test',
            'error': False,
            'coords': np.array([[0, 0], [1, 1]]),
            'heu_tour': [0, 1]
        }]
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('pathlib.Path.mkdir'), \
             patch('lin_kernighan_tsp_solver.utils._tkinter_available', False):
            
            # Mock figure and axes - use a simple object that behaves like ndarray
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            
            # Create a mock axes array that supports flatten()
            mock_axes = MagicMock()
            mock_axes.flatten.return_value = [mock_ax]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            plot_all_tours(results_data, force_save_plot=True)
            
        captured = capsys.readouterr()
        assert "Note: Install tkinter for interactive plots" in captured.out


class TestDisplaySummaryTableExtended:
    """Extended test cases for display_summary_table function."""

    def test_display_summary_table_with_override_config(self, capsys):
        """Test display_summary_table with override_config parameter."""
        results_data = [{
            'name': 'test1',
            'error': False,
            'opt_len': 100.0,
            'heu_len': 105.0,
            'gap': 5.0,
            'time': 1.0
        }]
        
        override_config = {
            'TIME_LIMIT': 30.0,
            'CUSTOM_PARAM': 'test_value'
        }
        
        with patch('lin_kernighan_tsp_solver.utils.config.LK_CONFIG', {'MAX_LEVEL': 10}):
            display_summary_table(results_data, override_config)
            
        captured = capsys.readouterr()
        assert "TIME_LIMIT  = 30.00" in captured.out
        assert "CUSTOM_PARAM = test_value" in captured.out
        assert "MAX_LEVEL   = 10" in captured.out

    def test_display_summary_table_mixed_data_types_in_config(self, capsys):
        """Test display_summary_table with mixed data types in config."""
        results_data = [{
            'name': 'test1',
            'error': False,
            'opt_len': 100.0,
            'heu_len': 105.0,
            'gap': 5.0,
            'time': 1.0
        }]
        
        config_data = {
            'FLOAT_PARAM': 3.14159,
            'INT_PARAM': 42,
            'STRING_PARAM': 'hello',
            'LIST_PARAM': [1, 2, 3],
            'BOOL_PARAM': True
        }
        
        with patch('lin_kernighan_tsp_solver.utils.config.LK_CONFIG', config_data):
            display_summary_table(results_data)
            
        captured = capsys.readouterr()
        assert "FLOAT_PARAM = 3.14" in captured.out  # Should be formatted to 2 decimal places
        assert "INT_PARAM   = 42" in captured.out
        assert "STRING_PARAM = hello" in captured.out
        assert "LIST_PARAM  = [1, 2, 3]" in captured.out
        assert "BOOL_PARAM  = True" in captured.out

    def test_display_summary_table_no_valid_results(self, capsys):
        """Test display_summary_table when all results have errors."""
        results_data = [
            {'name': 'err1', 'error': True},
            {'name': 'err2', 'error': True}
        ]
        
        with patch('lin_kernighan_tsp_solver.utils.config.LK_CONFIG', {}):
            display_summary_table(results_data)
            
        captured = capsys.readouterr()
        # Should not print any summary statistics
        assert "SUMMARY" not in captured.out
        assert "Done." in captured.out

    def test_display_summary_table_infinite_heuristic_lengths(self, capsys):
        """Test display_summary_table with infinite heuristic lengths."""
        results_data = [
            {
                'name': 'test1',
                'error': False,
                'opt_len': 100.0,
                'heu_len': float('inf'),
                'gap': float('inf'),
                'time': 1.0
            },
            {
                'name': 'test2',
                'error': False,
                'opt_len': 200.0,
                'heu_len': 205.0,
                'gap': 2.5,
                'time': 2.0
            }
        ]
        
        with patch('lin_kernighan_tsp_solver.utils.config.LK_CONFIG', {}):
            display_summary_table(results_data)
            
        captured = capsys.readouterr()
        # Should handle infinite values properly in summary
        assert "SUMMARY" in captured.out
        # Should sum only finite heuristic lengths (205.0)
        lines = captured.out.split('\n')
        summary_line = [line for line in lines if 'SUMMARY' in line][0]
        assert "205.00" in summary_line  # Only finite heu_len should be summed


if __name__ == "__main__":
    pytest.main([__file__])
