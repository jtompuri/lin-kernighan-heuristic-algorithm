import numpy as np
from pathlib import Path
from lin_kernighan_tsp_solver import main as lk_main
import pytest


def test_main_tsp_folder_not_found(monkeypatch, capsys):
    # Covers lines 192-194: TSP_FOLDER_PATH.is_dir() is False
    class DummyPath:
        def is_dir(self):
            return False

        def __str__(self):
            return "/nonexistent/folder"
    monkeypatch.setattr(lk_main, "TSP_FOLDER_PATH", DummyPath())
    lk_main.main()
    captured = capsys.readouterr()
    assert "Error: TSP folder not found" in captured.out


def test_main_no_tsp_files(monkeypatch, tmp_path, capsys):
    # Covers lines 202-204: No TSP files found in the directory
    monkeypatch.setattr(lk_main, "TSP_FOLDER_PATH", tmp_path)
    lk_main.main()
    captured = capsys.readouterr()
    assert "No TSP files found" in captured.out

# --- Tests for helper functions ---


def test_calculate_tour_length_edge_cases():
    # Covers lines 48-50: Empty and single-node tours
    D = np.array([[0, 1], [1, 0]])
    assert lk_main._calculate_tour_length([], D) == 0.0
    assert lk_main._calculate_tour_length([0], D) == 0.0

# --- Tests for process_single_instance() ---


def test_process_single_instance_catches_io_error(monkeypatch, capsys):
    # Covers the `except` block in process_single_instance (lines 189-195)
    def fake_read_tsp_file(path):
        raise IOError("Cannot read file")
    monkeypatch.setattr(lk_main, "read_tsp_file", fake_read_tsp_file)

    result = lk_main.process_single_instance("dummy.tsp", "dummy.opt.tour", verbose=True)
    assert result['error'] is True
    captured = capsys.readouterr()
    assert "Skipping dummy due to error: Cannot read file" in captured.out

# --- Robust Test for Parallel and Sequential Error Handling ---

# Define this mock at the top level of the file so it can be pickled


def mock_process_instance_with_failure(tsp_file, opt_file, time_limit=None, verbose=False):
    """
    A mock function that can be pickled.
    - Raises a RuntimeError for specific files to test the outer except block.
    - Returns a complete dictionary for successful files.
    """
    problem_name = Path(tsp_file).stem
    if "fail" in problem_name:
        raise RuntimeError("Simulated processing failure")

    # Return a complete, valid dictionary for success cases
    return {
        'name': problem_name, 'coords': np.array([[0, 0]]), 'opt_tour': [0],
        'heu_tour': [0], 'opt_len': 0.0, 'heu_len': 0.0,
        'gap': 0.0, 'time': 0.1, 'error': False, 'nodes': 1
    }


def test_parallel_processing_handles_errors(monkeypatch, tmp_path, capsys):
    # Covers the `except` block in _process_parallel (lines 283-285)
    # Create dummy files
    (tmp_path / "success.tsp").write_text("dummy")
    (tmp_path / "fail.tsp").write_text("dummy")

    # Mock the TSP folder path and the processing function
    monkeypatch.setattr(lk_main, "TSP_FOLDER_PATH", tmp_path)
    monkeypatch.setattr(lk_main, "plot_all_tours", lambda x, force_save_plot=False: None)  # Prevent plotting
    monkeypatch.setattr(lk_main, "process_single_instance", mock_process_instance_with_failure)

    # Run main, which calls _process_parallel
    lk_main.main(use_parallel=True, max_workers=2)

    output = capsys.readouterr().out

    # 1. Check that the error message for 'fail' IS present in the overall output
    assert "Error processing fail: Simulated processing failure" in output

    # 2. Check that the completion message for 'success' IS present
    assert "Completed: success" in output

    # 3. Isolate the summary table and check its contents specifically
    table_header = "Instance     OptLen   HeuLen"
    try:
        # Find where the table starts
        table_start_index = output.index(table_header)
        table_content = output[table_start_index:]

        # Assert that 'fail' is NOT in the table, but 'success' IS
        assert "fail" not in table_content
        assert "success" in table_content
    except ValueError:
        pytest.fail(f"Could not find summary table header '{table_header}' in output.")


def test_sequential_processing_handles_errors(monkeypatch, tmp_path, capsys):
    # Covers the `except` block in _process_sequential (lines 241-246)
    (tmp_path / "success.tsp").write_text("dummy")
    (tmp_path / "fail.tsp").write_text("dummy")

    monkeypatch.setattr(lk_main, "TSP_FOLDER_PATH", tmp_path)
    monkeypatch.setattr(lk_main, "plot_all_tours", lambda x, force_save_plot=False: None)
    monkeypatch.setattr(lk_main, "process_single_instance", mock_process_instance_with_failure)

    # Run main sequentially
    lk_main.main(use_parallel=False)

    captured = capsys.readouterr()
    # Check that the error was caught and reported
    assert "Critical error processing fail: Simulated processing failure" in captured.out
    # Check that the summary table still works
    assert "success" in captured.out


def test_calculate_tour_length_large_tour():
    # Covers lines 48-50: Large tour using vectorized operations (tour_len > 10)
    # Create a distance matrix for 12 nodes (> 10 to trigger vectorized path)
    D = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7],
        [5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6],
        [6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5],
        [7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4],
        [8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1],
        [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ])

    # Test tour with 12 nodes (0 -> 1 -> 2 -> ... -> 11 -> 0)
    large_tour = list(range(12))

    # Expected length: 1+1+1+1+1+1+1+1+1+1+1+11 = 21
    # (distances from each node to the next, with 11->0 being distance 11)
    expected_length = 11 + 11  # 11 edges of length 1, plus edge 11->0 of length 11

    result = lk_main._calculate_tour_length(large_tour, D)
    assert result == expected_length
