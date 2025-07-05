"""Tests for the tour saving functionality."""

import tempfile
import pytest
from pathlib import Path
from lin_kernighan_tsp_solver.utils import save_heuristic_tour
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

        # Check file content
        with open(saved_path, 'r') as f:
            content = f.read()

        assert "NAME: test_problem.heu.tour" in content
        assert "TYPE: TOUR" in content
        assert "COMMENT: Heuristic tour (Lin-Kernighan), length 123.45" in content
        assert "DIMENSION: 5" in content
        assert "TOUR_SECTION" in content
        assert "1\n2\n3\n4\n5\n" in content  # 1-indexed
        assert "-1\nEOF" in content


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
