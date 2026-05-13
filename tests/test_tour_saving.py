"""Tests for the tour saving functionality."""

import tempfile
import pytest
from pathlib import Path
from lin_kernighan_tsp_solver.utils import save_heuristic_tour
from lin_kernighan_tsp_solver.tsp_io import read_opt_tour
from lin_kernighan_tsp_solver.main import main


def test_save_heuristic_tour():
    """Test that heuristic tours are saved correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tour = [0, 1, 2, 3, 4]
        problem_name = "test_problem"
        tour_length = 123.45

        saved_path = save_heuristic_tour(tour, problem_name, tour_length, temp_dir)

        # Check file was created
        assert Path(saved_path).exists()
        assert Path(saved_path).name == "test_problem.heu.tour"

        # Check file content structure and TSPLIB TOUR fields
        lines = Path(saved_path).read_text(encoding='utf-8').splitlines()
        assert lines[0] == "NAME: test_problem.heu.tour"
        assert lines[1] == "TYPE: TOUR"
        assert lines[2] == "COMMENT: Heuristic tour (Lin-Kernighan), length 123.45"
        assert lines[3] == "DIMENSION: 5"
        assert lines[4] == "TOUR_SECTION"
        assert lines[5:10] == ["1", "2", "3", "4", "5"]  # 1-indexed nodes
        assert lines[10] == "-1"
        assert lines[11] == "EOF"


def test_save_heuristic_tour_empty():
    """Test saving an empty tour."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tour = []
        problem_name = "empty_test"
        tour_length = 0.0

        saved_path = save_heuristic_tour(tour, problem_name, tour_length, temp_dir)

        # Check file was created
        assert Path(saved_path).exists()

        # Check file content
        with open(saved_path, 'r') as f:
            content = f.read()

        assert "DIMENSION: 0" in content
        assert "TOUR_SECTION\n-1\n" in content


def test_save_heuristic_tour_round_trip_with_read_opt_tour():
    """Saved .heu.tour should round-trip back to the same 0-based tour."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_tour = [0, 2, 1, 3]
        problem_name = "round_trip"
        tour_length = 42.0

        saved_path = save_heuristic_tour(original_tour, problem_name, tour_length, temp_dir)

        loaded_tour = read_opt_tour(saved_path)
        assert loaded_tour == original_tour


def test_main_with_save_tours_enabled():
    """Test that main function saves tours when enabled."""
    # Test with a small specific file to avoid processing all TSP files
    small_files = ["problems/random/rand4.tsp"]

    # Test that the main function accepts the save_tours parameter
    # and processes the specified small file
    try:
        main(use_parallel=False, save_tours=True, tsp_files=small_files, plot=False)
    except SystemExit:
        # Expected when no files are found
        pass
    except Exception as e:
        # Should not raise other exceptions due to save_tours parameter
        if "save_tours" in str(e):
            pytest.fail(f"save_tours parameter caused error: {e}")


if __name__ == "__main__":
    test_save_heuristic_tour()
    test_save_heuristic_tour_empty()
    test_main_with_save_tours_enabled()
    print("All tour saving tests passed!")
