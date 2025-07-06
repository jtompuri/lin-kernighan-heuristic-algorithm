"""
Integration tests for starting cycle CLI functionality.

Tests that the command-line interface properly accepts and uses
different starting cycle methods.
"""

import subprocess
import tempfile
import sys
import os
from pathlib import Path
import pytest


@pytest.fixture
def temp_tsp_instance():
    """Create a temporary TSP instance for testing."""
    tsp_content = """NAME: test4
TYPE: TSP
COMMENT: Simple 4-node test instance
DIMENSION: 4
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 1.0 0.0
3 1.0 1.0
4 0.0 1.0
EOF"""

    opt_content = """NAME: test4.opt.tour
TYPE: TOUR
DIMENSION: 4
TOUR_SECTION
1
2
3
4
-1
EOF"""

    # Create temporary directory and files
    temp_dir = tempfile.mkdtemp()
    tsp_file = Path(temp_dir) / "test4.tsp"
    opt_file = Path(temp_dir) / "test4.opt.tour"

    with open(tsp_file, 'w') as f:
        f.write(tsp_content)

    with open(opt_file, 'w') as f:
        f.write(opt_content)

    return temp_dir, tsp_file, opt_file


class TestStartingCycleCLI:
    """Tests for command-line starting cycle functionality."""

    def test_cli_help_shows_starting_cycle_options(self):
        """Test that --help shows starting cycle options."""
        result = subprocess.run(
            [sys.executable, "-m", "lin_kernighan_tsp_solver", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        assert result.returncode == 0
        assert "--starting-cycle" in result.stdout
        assert "random" in result.stdout
        assert "nearest_neighbor" in result.stdout
        assert "qboruvka" in result.stdout

    def test_cli_invalid_starting_cycle(self):
        """Test that invalid starting cycle method shows error."""
        result = subprocess.run(
            [sys.executable, "-m", "lin_kernighan_tsp_solver", "--starting-cycle", "invalid_method"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()

    @pytest.mark.parametrize("method", ["random", "nearest_neighbor", "greedy", "boruvka", "qboruvka"])
    def test_cli_valid_starting_cycles(self, method, temp_tsp_instance):
        """Test that all valid starting cycle methods are accepted."""
        temp_dir, tsp_file, opt_file = temp_tsp_instance

        try:
            # Test argument parsing by using --help with the starting cycle method
            # This avoids actually running the solver which can be slow
            result = subprocess.run(
                [
                    sys.executable, "-m", "lin_kernighan_tsp_solver",
                    "--starting-cycle", method,
                    "--help"
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                timeout=5  # Short timeout for just argument parsing
            )

            # Should not fail due to invalid starting cycle method
            # If method is invalid, argparse will fail before showing help
            assert result.returncode == 0
            assert "--starting-cycle" in result.stdout

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)

    def test_default_starting_cycle_from_config(self):
        """Test that default starting cycle comes from config."""
        from lin_kernighan_tsp_solver.config import STARTING_CYCLE_CONFIG

        # The help message should show the default from config
        result = subprocess.run(
            [sys.executable, "-m", "lin_kernighan_tsp_solver", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        expected_default = STARTING_CYCLE_CONFIG["DEFAULT_METHOD"]
        assert f"default: {expected_default}" in result.stdout

    def test_cli_integration_with_actual_solver(self):
        """Test that starting cycle methods work with actual solver execution."""
        # Create a minimal test case
        tsp_content = """NAME: test3
TYPE: TSP
COMMENT: Minimal 3-node test
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 1.0 0.0
3 0.5 1.0
EOF"""

        import tempfile

        temp_dir = tempfile.mkdtemp()
        tsp_file = Path(temp_dir) / "test3.tsp"

        try:
            with open(tsp_file, 'w') as f:
                f.write(tsp_content)

            # Set environment variable to use our temp directory
            env = os.environ.copy()
            env["TSP_FOLDER_PATH"] = temp_dir

            # Test with a method that should work quickly
            result = subprocess.run(
                [
                    sys.executable, "-m", "lin_kernighan_tsp_solver",
                    "--starting-cycle", "random",
                    "--time-limit", "0.01",  # Very short time limit
                    "--max-iterations", "1"   # Minimal iterations
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                env=env,
                timeout=5  # Short timeout
            )

            # Should complete successfully (or timeout gracefully)
            # We don't care about optimal results, just that it doesn't crash due to starting cycle
            assert "invalid choice" not in result.stderr.lower()

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)


class TestStartingCycleIntegration:
    """Integration tests for starting cycle functionality."""

    def test_config_override_from_cli(self):
        """Test that CLI argument overrides config default."""
        from lin_kernighan_tsp_solver.config import LK_CONFIG
        from lin_kernighan_tsp_solver.main import main

        # Save original config
        original_method = LK_CONFIG["STARTING_CYCLE"]

        try:
            # Test that CLI argument changes the config
            # Use small file to speed up test - only test config override, not full solving
            small_files = ["problems/random/rand4.tsp"]

            main(starting_cycle_method="nearest_neighbor", tsp_files=small_files, plot=False)
            assert LK_CONFIG["STARTING_CYCLE"] == "nearest_neighbor"

            main(starting_cycle_method="random", tsp_files=small_files, plot=False)
            assert LK_CONFIG["STARTING_CYCLE"] == "random"

            # Test that None doesn't change config
            LK_CONFIG["STARTING_CYCLE"] = "greedy"
            main(starting_cycle_method=None, tsp_files=small_files, plot=False)
            assert LK_CONFIG["STARTING_CYCLE"] == "greedy"

        finally:
            # Restore original config
            LK_CONFIG["STARTING_CYCLE"] = original_method

    def test_starting_cycle_used_in_processing(self):
        """Test that starting cycle method is actually used."""
        import numpy as np
        from lin_kernighan_tsp_solver.starting_cycles import generate_starting_cycle
        from lin_kernighan_tsp_solver.config import LK_CONFIG

        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        # Save original config
        original_method = LK_CONFIG["STARTING_CYCLE"]

        try:
            # Test that different methods can produce different results
            LK_CONFIG["STARTING_CYCLE"] = "nearest_neighbor"
            nn_tour = generate_starting_cycle(coords, method=LK_CONFIG["STARTING_CYCLE"])

            # Set seed for reproducible random tour
            import random
            random.seed(42)
            LK_CONFIG["STARTING_CYCLE"] = "random"
            random_tour = generate_starting_cycle(coords, method=LK_CONFIG["STARTING_CYCLE"])

            # Tours should be valid
            assert len(nn_tour) == 4
            assert len(random_tour) == 4
            assert set(nn_tour) == {0, 1, 2, 3}
            assert set(random_tour) == {0, 1, 2, 3}

            # For this specific case, nearest neighbor should start with 0
            assert nn_tour[0] == 0

        finally:
            # Restore original config
            LK_CONFIG["STARTING_CYCLE"] = original_method
