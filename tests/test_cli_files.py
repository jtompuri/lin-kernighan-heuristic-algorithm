"""
Tests for CLI file-specific functionality.
"""

from lin_kernighan_tsp_solver.main import main
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCLIFiles:
    """Test CLI functionality with specific files."""

    def test_main_with_specific_files(self, tmp_path, monkeypatch):
        """Test main function with specific TSP files."""
        # Create temporary TSP files
        tsp_file1 = tmp_path / "test1.tsp"
        tsp_file2 = tmp_path / "test2.tsp"
        opt_file1 = tmp_path / "test1.opt.tour"
        opt_file2 = tmp_path / "test2.opt.tour"

        # Create dummy files
        tsp_file1.write_text("DUMMY TSP CONTENT")
        tsp_file2.write_text("DUMMY TSP CONTENT")
        opt_file1.write_text("DUMMY OPT CONTENT")
        opt_file2.write_text("DUMMY OPT CONTENT")

        # Mock the processing functions
        with patch('lin_kernighan_tsp_solver.main._process_sequential') as mock_seq, \
                patch('lin_kernighan_tsp_solver.main.display_summary_table'), \
                patch('lin_kernighan_tsp_solver.main.plot_all_tours'):

            mock_seq.return_value = [
                {'name': 'test1', 'error': False},
                {'name': 'test2', 'error': False}
            ]

            main(
                use_parallel=False,
                tsp_files=[str(tsp_file1), str(tsp_file2)]
            )

            # Check that sequential processing was called with correct files
            mock_seq.assert_called_once()
            args = mock_seq.call_args[0]
            tsp_pairs = args[0]

            assert len(tsp_pairs) == 2
            assert str(tsp_file1) in tsp_pairs[0][0]
            assert str(tsp_file2) in tsp_pairs[1][0]
            assert str(opt_file1) in tsp_pairs[0][1]
            assert str(opt_file2) in tsp_pairs[1][1]

    def test_main_with_missing_file(self, tmp_path, capsys):
        """Test main function with non-existent TSP file."""
        missing_file = tmp_path / "missing.tsp"

        main(
            use_parallel=False,
            tsp_files=[str(missing_file)]
        )

        captured = capsys.readouterr()
        assert "No valid TSP files found" in captured.out

    def test_main_with_missing_opt_file(self, tmp_path):
        """Test main function with missing optimal tour file."""
        tsp_file = tmp_path / "test.tsp"
        tsp_file.write_text("DUMMY TSP CONTENT")

        with patch('lin_kernighan_tsp_solver.main._process_sequential') as mock_seq, \
                patch('lin_kernighan_tsp_solver.main.display_summary_table'), \
                patch('lin_kernighan_tsp_solver.main.plot_all_tours'):

            mock_seq.return_value = [{'name': 'test', 'error': False}]

            main(
                use_parallel=False,
                tsp_files=[str(tsp_file)]
            )

            # Should still work, just with a dummy optimal file path
            mock_seq.assert_called_once()

    def test_main_file_discovery_priority(self, tmp_path):
        """Test that optimal file discovery follows correct priority."""
        tsp_file = tmp_path / "test.tsp"
        opt_tour_file = tmp_path / "test.opt.tour"
        opt_file = tmp_path / "test.opt"
        tour_file = tmp_path / "test.tour"

        # Create files
        tsp_file.write_text("DUMMY TSP CONTENT")
        opt_tour_file.write_text("DUMMY OPT TOUR")
        opt_file.write_text("DUMMY OPT")
        tour_file.write_text("DUMMY TOUR")

        with patch('lin_kernighan_tsp_solver.main._process_sequential') as mock_seq, \
                patch('lin_kernighan_tsp_solver.main.display_summary_table'), \
                patch('lin_kernighan_tsp_solver.main.plot_all_tours'):

            mock_seq.return_value = [{'name': 'test', 'error': False}]

            main(
                use_parallel=False,
                tsp_files=[str(tsp_file)]
            )

            # Should prefer .opt.tour over .opt over .tour
            args = mock_seq.call_args[0]
            tsp_pairs = args[0]
            assert str(opt_tour_file) in tsp_pairs[0][1]

    def test_main_original_behavior_preserved(self):
        """Test that original behavior (no files specified) still works."""
        with patch('lin_kernighan_tsp_solver.main.TSP_FOLDER_PATH') as mock_path, \
                patch('lin_kernighan_tsp_solver.main._process_sequential') as mock_seq, \
                patch('lin_kernighan_tsp_solver.main.display_summary_table'), \
                patch('lin_kernighan_tsp_solver.main.plot_all_tours'):

            # Mock the folder path
            mock_path.is_dir.return_value = True
            mock_path.glob.return_value = [MagicMock()]
            mock_seq.return_value = [{'name': 'test', 'error': False}]

            # Call without specific files (original behavior)
            main(use_parallel=False, tsp_files=None)

            # Should still work
            mock_seq.assert_called_once()
