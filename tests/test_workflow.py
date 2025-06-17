import pytest
import numpy as np
from unittest.mock import patch
from lin_kernighan_tsp_solver.main import (
    process_single_instance
)


def test_process_single_instance_no_coords_loaded(tmp_path):
    """
    Tests process_single_instance when coordinate loading fails.

    It should return a result dictionary indicating an error.
    """
    tsp_file = tmp_path / "empty.tsp"
    # Provide a minimal TSP file structure for the mock to be called.
    tsp_file.write_text("EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\nEOF")
    opt_tour_file = tmp_path / "empty.opt.tour"
    opt_tour_file.write_text("TOUR_SECTION\n-1\nEOF")  # Dummy opt tour file.

    # Mock read_tsp_file to simulate failed coordinate loading.
    with patch('lin_kernighan_tsp_solver.main.read_tsp_file', return_value=np.array([])) as mock_read_tsp:
        result = process_single_instance(str(tsp_file), str(opt_tour_file))
        mock_read_tsp.assert_called_once_with(str(tsp_file))

        assert result is not None, "Result dictionary should not be None."
        assert result.get('error') is True, "Result should indicate an error."
        assert result.get('name') == "empty", "Instance name should be in the result."
        # Additional checks for 'heu_len' or 'gap' could be added if specific
        # error values are expected (e.g., float('inf') or None).


def test_process_single_instance_no_opt_tour_loaded(tmp_path, simple_tsp_setup):
    """
    Tests process_single_instance when coordinates load but no optimal tour is found.

    Chained LK should still run, and the result should reflect the missing
    optimal tour information (e.g., opt_len and gap should be None).
    """
    coords, _, initial_tour_obj, _, \
        initial_cost_fixture, _, _ = simple_tsp_setup

    instance_name = "test_instance_no_opt"
    tsp_file = tmp_path / f"{instance_name}.tsp"
    opt_tour_file = tmp_path / f"{instance_name}.opt.tour"

    # Create dummy TSP and opt_tour files; their content is not critical
    # as the relevant file reading functions will be mocked.
    tsp_file.write_text(f"NAME: {instance_name}\nTYPE: TSP\nDIMENSION: 5\nNODE_COORD_SECTION\n1 0 0\nEOF")
    opt_tour_file.write_text("TOUR_SECTION\n-1\nEOF")

    mock_coords = coords
    mock_clk_result_tour = initial_tour_obj.get_tour()
    mock_clk_result_cost = initial_cost_fixture

    with patch('lin_kernighan_tsp_solver.main.read_tsp_file', return_value=mock_coords) as mock_read_tsp, \
         patch('lin_kernighan_tsp_solver.main.read_opt_tour', return_value=None) as mock_read_opt, \
         patch('lin_kernighan_tsp_solver.main.chained_lin_kernighan',
               return_value=(mock_clk_result_tour, mock_clk_result_cost)) as mock_clk:

        result = process_single_instance(str(tsp_file), str(opt_tour_file))

        mock_read_tsp.assert_called_once_with(str(tsp_file))
        mock_read_opt.assert_called_once_with(str(opt_tour_file))
        mock_clk.assert_called_once()

        # Check that known_optimal_length was None in the call to chained_lin_kernighan.
        args, kwargs = mock_clk.call_args
        assert kwargs.get('known_optimal_length') is None

        assert result is not None
        assert result.get('name') == instance_name
        assert result.get('error') is False  # No error in processing itself.
        assert result.get('opt_len') is None  # Optimal length is unknown.
        assert result.get('heu_len') == pytest.approx(mock_clk_result_cost)
        assert result.get('gap') is None  # Gap cannot be calculated.
        assert result.get('time') is not None
        assert result.get('nodes') == len(mock_coords)


def test_process_single_instance_handles_chained_lk_exception(tmp_path, simple_tsp_setup):
    """
    Tests that process_single_instance correctly handles a generic Exception
    raised by chained_lin_kernighan and sets error flags in the results.
    This covers the generic 'except Exception' block.
    """
    coords, _, _, _, _, _, _ = simple_tsp_setup
    instance_name = "test_instance_clk_fail"
    tsp_file = tmp_path / f"{instance_name}.tsp"
    opt_tour_file = tmp_path / f"{instance_name}.opt.tour"

    # Create dummy TSP and opt_tour files.
    tsp_file.write_text(
        f"NAME: {instance_name}\nTYPE: TSP\nDIMENSION: {len(coords)}\n"
        "NODE_COORD_SECTION\n1 0 0\nEOF"
    )
    opt_tour_file.write_text("TOUR_SECTION\n-1\nEOF")

    mock_coords = coords
    error_message = "Simulated CLK Failure"

    with patch('lin_kernighan_tsp_solver.main.read_tsp_file', return_value=mock_coords) as mock_read_tsp, \
         patch('lin_kernighan_tsp_solver.main.read_opt_tour', return_value=None) as mock_read_opt, \
         patch('lin_kernighan_tsp_solver.main.chained_lin_kernighan', side_effect=Exception(error_message)) as mock_clk:

        result = process_single_instance(str(tsp_file), str(opt_tour_file))

        mock_read_tsp.assert_called_once_with(str(tsp_file))
        mock_read_opt.assert_called_once_with(str(opt_tour_file))
        mock_clk.assert_called_once()

        assert result is not None, "Result dictionary should not be None."
        assert result.get('name') == instance_name, "Instance name should be correct."
        assert result.get('error') is True, "Result should indicate an error."
        # The generic except block in process_single_instance currently does not store
        # the error message string in the results dictionary.
        # assert result.get('error_message') == error_message  # This assertion would fail.
        assert result.get('heu_len') == float('inf'), "Heuristic length should be default error value."
        assert result.get('nodes') == len(mock_coords), "Nodes should be set if coords were read."
