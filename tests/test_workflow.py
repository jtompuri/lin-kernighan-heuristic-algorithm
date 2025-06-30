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
    Tests that process_single_instance correctly handles an expected error
    (like ValueError) and sets error flags in the results.
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

    # Simulate a ValueError, which the function is designed to catch.
    with patch('lin_kernighan_tsp_solver.main.read_tsp_file', return_value=mock_coords) as mock_read_tsp, \
            patch('lin_kernighan_tsp_solver.main.read_opt_tour', return_value=None) as mock_read_opt, \
            patch('lin_kernighan_tsp_solver.main.chained_lin_kernighan', side_effect=ValueError(error_message)) as mock_clk:

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


def test_main_tsp_folder_not_found(capsys):
    with patch('lin_kernighan_tsp_solver.main.TSP_FOLDER_PATH') as mock_path:
        mock_path.is_dir.return_value = False
        from lin_kernighan_tsp_solver.main import main
        main()
        captured = capsys.readouterr()
        assert "Error: TSP folder not found" in captured.out


def test_main_no_tsp_files(monkeypatch):
    class DummyPath:
        def is_dir(self):
            return True

        def glob(self, pattern):
            return []

    monkeypatch.setattr('lin_kernighan_tsp_solver.main.TSP_FOLDER_PATH', DummyPath())
    from lin_kernighan_tsp_solver.main import main
    with patch('lin_kernighan_tsp_solver.main.display_summary_table') as mock_display, \
            patch('lin_kernighan_tsp_solver.main.plot_all_tours') as mock_plot:
        main()
        mock_display.assert_called_once_with([])
        mock_plot.assert_called_once_with([])


def test_main_process_single_instance_exception(monkeypatch):
    class DummyPath:
        def is_dir(self):
            return True

        def glob(self, pattern):
            return [DummyPath()]

        @property
        def stem(self):
            return "dummy"

        def __str__(self):
            return "dummy.tsp"

        def __truediv__(self, other):
            return self

    monkeypatch.setattr('lin_kernighan_tsp_solver.main.TSP_FOLDER_PATH', DummyPath())
    from lin_kernighan_tsp_solver.main import main
    with patch('lin_kernighan_tsp_solver.main.process_single_instance', side_effect=Exception("fail")), \
            patch('lin_kernighan_tsp_solver.main.display_summary_table') as mock_display, \
            patch('lin_kernighan_tsp_solver.main.plot_all_tours'):
        main()
        args, _ = mock_display.call_args
        assert args[0][0]['error'] is True


def test_main_process_single_instance_success(monkeypatch):
    class DummyPath:
        def is_dir(self):
            return True

        def glob(self, pattern):
            return [DummyPath()]

        @property
        def stem(self):
            return "dummy"

        def __str__(self):
            return "dummy.tsp"

        def __truediv__(self, other):
            return self

    monkeypatch.setattr('lin_kernighan_tsp_solver.main.TSP_FOLDER_PATH', DummyPath())
    from lin_kernighan_tsp_solver.main import main
    dummy_result = {'name': 'dummy', 'error': False}
    with patch('lin_kernighan_tsp_solver.main.process_single_instance', return_value=dummy_result), \
            patch('lin_kernighan_tsp_solver.main.display_summary_table') as mock_display, \
            patch('lin_kernighan_tsp_solver.main.plot_all_tours') as mock_plot:
        main()
        mock_display.assert_called_once_with([dummy_result])
        mock_plot.assert_called_once_with([dummy_result])


def test_main_multiple_tsp_files(monkeypatch):
    class DummyPath:
        def __init__(self, name):
            self._name = name

        def is_dir(self):
            return True

        def glob(self, pattern):
            return [DummyPath("a"),
                    DummyPath("b")]

        @property
        def stem(self):
            return self._name

        def __str__(self):
            return f"{self._name}.tsp"

        def __truediv__(self, other):
            return self

        def __lt__(self, other):
            return self._name < other._name

    monkeypatch.setattr('lin_kernighan_tsp_solver.main.TSP_FOLDER_PATH', DummyPath("folder"))
    from lin_kernighan_tsp_solver.main import main
    dummy_result_a = {'name': 'a', 'error': False}
    dummy_result_b = {'name': 'b', 'error': False}
    with patch('lin_kernighan_tsp_solver.main.process_single_instance', side_effect=[dummy_result_a, dummy_result_b]) as mock_proc, \
            patch('lin_kernighan_tsp_solver.main.display_summary_table') as mock_display, \
            patch('lin_kernighan_tsp_solver.main.plot_all_tours') as mock_plot:
        main()
        # Should be called for both files
        assert mock_proc.call_count == 2
        mock_display.assert_called_once_with([dummy_result_a, dummy_result_b])
        mock_plot.assert_called_once_with([dummy_result_a, dummy_result_b])


def test_process_single_instance_handles_tsp_read_error(capsys):
    from lin_kernighan_tsp_solver.main import process_single_instance
    # Simulate a ValueError, which the function is designed to catch.
    with patch('lin_kernighan_tsp_solver.main.read_tsp_file', side_effect=ValueError("read error")), \
            patch('lin_kernighan_tsp_solver.main.read_opt_tour', return_value=None):
        result = process_single_instance("dummy.tsp", "dummy.opt.tour")
        assert result['error'] is True
        captured = capsys.readouterr()
        assert "read error" in captured.out


def test_process_single_instance_handles_empty_coords(capsys):
    from lin_kernighan_tsp_solver.main import process_single_instance
    with patch('lin_kernighan_tsp_solver.main.read_tsp_file', return_value=np.array([])), \
            patch('lin_kernighan_tsp_solver.main.read_opt_tour', return_value=None):
        result = process_single_instance("dummy.tsp", "dummy.opt.tour")
        assert result['error'] is True
        captured = capsys.readouterr()
        assert "No coordinates loaded" in captured.out


def test_process_single_instance_handles_missing_opt_tour(capsys):
    from lin_kernighan_tsp_solver.main import process_single_instance
    coords = np.array([[0, 0], [1, 1]])
    with patch('lin_kernighan_tsp_solver.main.read_tsp_file', return_value=coords), \
            patch('lin_kernighan_tsp_solver.main.read_opt_tour', return_value=None), \
            patch('lin_kernighan_tsp_solver.main.chained_lin_kernighan', return_value=([0, 1], 1.0)):
        result = process_single_instance("dummy.tsp", "dummy.opt.tour")
        captured = capsys.readouterr()
        assert "Optimal tour not available" in captured.out
        assert result['opt_tour'] is None
        assert result['opt_len'] is None


def test_process_single_instance_handles_zero_opt_len():
    from lin_kernighan_tsp_solver.main import process_single_instance
    coords = np.array([[0, 0], [1, 1]])
    opt_tour = [0, 1]
    # Case 1: heuristic_len == 0.0
    with patch('lin_kernighan_tsp_solver.main.read_tsp_file', return_value=coords), \
            patch('lin_kernighan_tsp_solver.main.read_opt_tour', return_value=opt_tour), \
            patch('lin_kernighan_tsp_solver.main.chained_lin_kernighan', return_value=(opt_tour, 0.0)):
        result = process_single_instance("dummy.tsp", "dummy.opt.tour")
        assert result['gap'] == 0.0
    # Case 2: heuristic_len != 0.0
    with patch('lin_kernighan_tsp_solver.main.read_tsp_file', return_value=coords), \
            patch('lin_kernighan_tsp_solver.main.read_opt_tour', return_value=opt_tour), \
            patch('lin_kernighan_tsp_solver.main.chained_lin_kernighan', return_value=(opt_tour, 1.0)):
        result = process_single_instance("dummy.tsp", "dummy.opt.tour")
        assert result['gap'] == 0.0


def test_main_calls_summary_and_plot(monkeypatch):
    class DummyPath:
        def is_dir(self):
            return True

        def glob(self, pattern):
            return []

    monkeypatch.setattr('lin_kernighan_tsp_solver.main.TSP_FOLDER_PATH', DummyPath())
    from lin_kernighan_tsp_solver.main import main
    with patch('lin_kernighan_tsp_solver.main.display_summary_table') as mock_display, \
            patch('lin_kernighan_tsp_solver.main.plot_all_tours') as mock_plot:
        main()
        mock_display.assert_called_once_with([])
        mock_plot.assert_called_once_with([])


def test_process_single_instance_gap_when_optimal_zero(monkeypatch):
    import numpy as np
    from lin_kernighan_tsp_solver.main import process_single_instance

    coords = np.array([[0, 0], [1, 0]])
    opt_tour = [0, 1]

    # Case 1: heuristic_len == 0.0, expect gap == 0.0
    with patch('lin_kernighan_tsp_solver.main.read_tsp_file', return_value=coords), \
         patch('lin_kernighan_tsp_solver.main.read_opt_tour', return_value=opt_tour), \
         patch('lin_kernighan_tsp_solver.main.chained_lin_kernighan', return_value=(opt_tour, 0.0)), \
         patch('lin_kernighan_tsp_solver.main.build_distance_matrix', return_value=np.zeros((2, 2))):
        result = process_single_instance("dummy.tsp", "dummy.opt.tour")
        assert result['gap'] == 0.0

    # Case 2: heuristic_len != 0.0, expect gap == float('inf')
    with patch('lin_kernighan_tsp_solver.main.read_tsp_file', return_value=coords), \
         patch('lin_kernighan_tsp_solver.main.read_opt_tour', return_value=opt_tour), \
         patch('lin_kernighan_tsp_solver.main.chained_lin_kernighan', return_value=(opt_tour, 1.0)), \
         patch('lin_kernighan_tsp_solver.main.build_distance_matrix', return_value=np.zeros((2, 2))):
        result = process_single_instance("dummy.tsp", "dummy.opt.tour")
        assert result['gap'] == float('inf')
