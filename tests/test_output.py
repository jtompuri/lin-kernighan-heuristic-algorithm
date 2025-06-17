import numpy as np
from unittest.mock import MagicMock, patch
import matplotlib.axes
import matplotlib.figure
from lin_kernighan_tsp_solver.utils import (
    display_summary_table,
    plot_all_tours,
)
from lin_kernighan_tsp_solver.config import (
    MAX_SUBPLOTS_IN_PLOT,
)


def test_plot_all_tours_num_to_plot_actual_is_zero(capsys):
    """
    Tests plot_all_tours when num_to_plot_actual becomes 0 after limiting.
    This happens if MAX_SUBPLOTS_IN_PLOT is 0 but there are valid results.
    Covers line 1023.
    """
    results_data = [
        {'name': 'inst1', 'error': False, 'coords': np.array([[0, 0], [1, 1]]), 'heu_tour': [0, 1]}
    ]

    # Mock MAX_SUBPLOTS_IN_PLOT to be 0 for this test
    with patch('lin_kernighan_tsp_solver.config.MAX_SUBPLOTS_IN_PLOT', 0):
        # Mock matplotlib calls as they shouldn't be reached if it returns early
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.show') as mock_show:

            plot_all_tours(results_data)

    captured = capsys.readouterr()

    expected_warning = "Warning: Plotting first 0 of 1 valid results."
    assert expected_warning in captured.out
    mock_subplots.assert_not_called()  # Plotting setup should not occur
    mock_show.assert_not_called()


def test_display_summary_table_edge_cases(capsys):
    """
    Tests display_summary_table with various edge case scenarios for results_data
    to cover conditional logic in summary calculations and formatting.
    This targets lines approx. 891-903.
    """
    # Scenario 1: No valid results (all errored)
    results_all_errored = [
        {'name': 'err1', 'error': True, 'heu_len': float('inf'), 'time': 0.1, 'opt_len': None, 'gap': None},
    ]
    display_summary_table(results_all_errored)
    captured_s1 = capsys.readouterr().out
    expected_header_s1 = "Instance     OptLen   HeuLen   Gap(%)  Time(s)"
    assert expected_header_s1 in captured_s1
    assert "err1" not in captured_s1
    assert "SUMMARY" not in captured_s1
    assert "Done." in captured_s1

    # Scenario 2: Valid results, but all opt_len and gap are None
    results_no_opt_gap = [
        {'name': 'no_opt', 'error': False, 'opt_len': None, 'heu_len': 150.0, 'gap': None, 'time': 1.5},
        {'name': 'no_gp2', 'error': False, 'opt_len': None, 'heu_len': 250.0, 'gap': None, 'time': 2.5},
    ]
    display_summary_table(results_no_opt_gap)
    captured_s2 = capsys.readouterr().out

    expected_no_opt_s2 = "no_opt          N/A   150.00      N/A     1.50"
    expected_no_gp2_s2 = "no_gp2          N/A   250.00      N/A     2.50"
    assert expected_no_opt_s2 in captured_s2
    assert expected_no_gp2_s2 in captured_s2

    expected_summary_line_s2 = "SUMMARY         N/A   400.00      N/A     2.00"
    assert " ".join(expected_summary_line_s2.split()) in " ".join(captured_s2.split())

    # Scenario 3: Valid results, some gaps are 'inf', some heu_len are 'inf'
    results_inf_values = [
        {'name': 'inf_g', 'error': False, 'opt_len': 10.0, 'heu_len': 20.0, 'gap': float('inf'), 'time': 0.5},
        {'name': 'inf_h', 'error': False, 'opt_len': 20.0, 'heu_len': float('inf'), 'gap': float('inf'), 'time': 0.7},
        {'name': 'valid', 'error': False, 'opt_len': 30.0, 'heu_len': 33.0, 'gap': 10.0, 'time': 0.9},
    ]
    display_summary_table(results_inf_values)
    captured_s3 = capsys.readouterr().out

    expected_inf_g_s3 = "inf_g         10.00    20.00      inf     0.50"
    expected_inf_h_s3 = "inf_h         20.00      inf      inf     0.70"
    expected_valid_s3 = "valid         30.00    33.00    10.00     0.90"
    assert expected_inf_g_s3 in captured_s3
    assert expected_inf_h_s3 in captured_s3
    assert expected_valid_s3 in captured_s3

    expected_summary_line_s3 = "SUMMARY       60.00    53.00    10.00     0.70"
    assert " ".join(expected_summary_line_s3.split()) in " ".join(captured_s3.split())

    # Scenario 4: All valid results have heu_len = inf
    results_all_inf_heu = [
        {'name': 'all_ih1', 'error': False, 'opt_len': 100.0, 'heu_len': float('inf'), 'gap': float('inf'), 'time': 1.0},
        {'name': 'all_ih2', 'error': False, 'opt_len': 200.0, 'heu_len': float('inf'), 'gap': float('inf'), 'time': 2.0},
    ]
    display_summary_table(results_all_inf_heu)
    captured_s4 = capsys.readouterr().out

    expected_all_ih1_s4 = "all_ih1      100.00      inf      inf     1.00"
    expected_all_ih2_s4 = "all_ih2      200.00      inf      inf     2.00"
    assert expected_all_ih1_s4 in captured_s4
    assert expected_all_ih2_s4 in captured_s4

    expected_summary_all_inf_s4 = "SUMMARY      300.00     0.00      N/A     1.50"
    assert " ".join(expected_summary_all_inf_s4.split()) in " ".join(captured_s4.split())

    # Scenario 5: Mixed valid, errored, and partial data
    results_mixed = [
        {'name': 'valid1', 'error': False, 'opt_len': 100.0, 'heu_len': 110.0, 'gap': 10.0, 'time': 1.0},
        {'name': 'errored', 'error': True, 'opt_tour': None, 'heu_len': float('inf'), 'gap': None, 'time': 0.5},
        {'name': 'no_opt', 'error': False, 'opt_len': None, 'heu_len': 50.0, 'gap': None, 'time': 2.0},
        {'name': 'inf_heu', 'error': False, 'opt_len': 200.0, 'heu_len': float('inf'), 'gap': float('inf'), 'time': 3.0},
        {'name': 'zero_opt', 'error': False, 'opt_len': 0.0, 'heu_len': 10.0, 'gap': float('inf'), 'time': 0.8}
    ]
    display_summary_table(results_mixed)
    captured_s5 = capsys.readouterr().out

    expected_valid1_s5 = "valid1       100.00   110.00    10.00     1.00"
    expected_no_opt_s5 = "no_opt          N/A    50.00      N/A     2.00"
    expected_inf_heu_s5 = "inf_heu      200.00      inf      inf     3.00"
    expected_zero_opt_s5 = "zero_opt       0.00    10.00      inf     0.80"

    assert expected_valid1_s5 in captured_s5
    assert "errored" not in captured_s5
    assert expected_no_opt_s5 in captured_s5
    assert expected_inf_heu_s5 in captured_s5
    assert expected_zero_opt_s5 in captured_s5

    expected_summary_line_s5 = "SUMMARY      300.00   170.00    10.00     1.70"
    assert " ".join(expected_summary_line_s5.split()) in " ".join(captured_s5.split())


def test_plot_all_tours_no_results(capsys):
    """
    Tests plot_all_tours with an empty results_data list.
    It should print a message and return early.
    Covers lines 913-915 (adjust line numbers based on current file).
    """
    # The function plot_all_tours was updated and does not take output_dir as argument.
    # It also prints different messages for early returns.
    # The target lines are now around 919-922 for empty or no valid results.

    with patch('matplotlib.pyplot') as mock_plt:  # Mock the entire pyplot module
        plot_all_tours([])  # Corrected: removed second argument

    captured = capsys.readouterr()
    # Check for the message when no valid results are found (which includes empty input)
    assert "No valid results with coordinates to plot." in captured.out
    mock_plt.figure.assert_not_called()  # Plotting should not start
    mock_plt.show.assert_not_called()


def test_plot_all_tours_no_valid_results_for_plotting(capsys):
    """
    Tests plot_all_tours when results_data contains items, but none are valid for plotting
    (e.g., all errored, or no coordinates).
    It should print a message and return early.
    Covers lines approx. 919-922 (adjust line numbers).
    """
    results_no_valid_plot_data = [
        {'name': 'err_plot', 'error': True, 'coords': np.array([[0, 0], [1, 1]]), 'heu_tour': [0, 1]},
        {'name': 'no_coords_plot', 'error': False, 'coords': None, 'heu_tour': [0, 1]},  # coords is None
        {'name': 'empty_coords_plot', 'error': False, 'coords': np.array([]), 'heu_tour': []},  # coords is empty
    ]

    with patch('matplotlib.pyplot') as mock_plt:
        plot_all_tours(results_no_valid_plot_data)

    captured = capsys.readouterr()
    # Corrected expected message to match the actual output of plot_all_tours
    assert "No valid results with coordinates to plot." in captured.out
    mock_plt.figure.assert_not_called()  # Plotting should not start
    mock_plt.show.assert_not_called()


# Test for successful plotting path (mocking plt.show)
def test_plot_all_tours_successful_path_mocked(tmp_path, capsys):
    """
    Tests the successful plotting path of plot_all_tours by mocking matplotlib calls.
    Ensures that plotting functions are called when valid data is provided.
    The function now only shows plots, does not save them.
    """
    results_valid_data = [
        {
            'name': 'valid_instance',
            'error': False,
            'coords': np.array([[0, 0], [1, 1], [0, 1]]),
            'heu_tour': [0, 1, 2],
            'opt_tour': [0, 2, 1],
            'opt_len': 2 + np.sqrt(2),
            'heu_len': 2 + np.sqrt(2),
            'gap': 0.0
        }
    ]

    # Create a mock Axes object. This will be the object whose methods we check.
    mock_ax_instance = MagicMock(spec=matplotlib.axes.Axes)  # Changed spec
    # Ensure methods that might be called exist as mocks
    mock_ax_instance.plot = MagicMock()
    mock_ax_instance.set_title = MagicMock()
    mock_ax_instance.set_xticks = MagicMock()
    mock_ax_instance.set_yticks = MagicMock()
    mock_ax_instance.set_aspect = MagicMock()
    mock_ax_instance.set_axis_off = MagicMock()

    # Create a mock Figure object.
    mock_figure_instance = MagicMock(spec=matplotlib.figure.Figure)  # Changed spec
    mock_figure_instance.legend = MagicMock()
    mock_figure_instance.subplots_adjust = MagicMock()

    # Simulate the 2D array of Axes returned by plt.subplots(..., squeeze=False)
    # For one plot, it's a 1x1 array containing the Axes object.
    mock_axes_array = np.array([[mock_ax_instance]])

    # Patch the necessary matplotlib functions
    # Correct the patch target for Line2D
    with patch('matplotlib.pyplot.subplots', return_value=(mock_figure_instance, mock_axes_array)) as mock_subplots_call, \
         patch('lin_kernighan_tsp_solver.utils.Line2D') as mock_line2d_call, \
         patch('matplotlib.pyplot.tight_layout') as mock_tight_layout_call, \
         patch('matplotlib.pyplot.show') as mock_show_call:

        plot_all_tours(results_valid_data)

    # Assert that plt.subplots was called (it creates the figure and axes)
    mock_subplots_call.assert_called_once()

    # Assert that methods on the mock_ax_instance were called
    # For the given valid_data, ax.plot is called for heuristic and optimal tours.
    assert mock_ax_instance.plot.call_count == 2, "ax.plot should be called twice (heuristic and optimal)"
    mock_ax_instance.set_title.assert_called_once()
    mock_ax_instance.set_xticks.assert_called_once()
    mock_ax_instance.set_yticks.assert_called_once()
    mock_ax_instance.set_aspect.assert_called_once()
    # set_axis_off is not called for the first subplot if it's the only one.
    # If there were more subplots than data, it would be called on unused ones.
    # For a single plot, num_to_plot_actual (1) == len(axes_list) (1), so the loop for set_axis_off is not entered.
    mock_ax_instance.set_axis_off.assert_not_called()

    # Assert that Line2D was called for creating legend elements
    # (one for heuristic, one for optimal)
    assert mock_line2d_call.call_count == 2, "Line2D should be called twice for legend"

    # Assert that methods on the mock_figure_instance were called (for legend)
    mock_figure_instance.legend.assert_called_once()
    mock_figure_instance.subplots_adjust.assert_called_once()

    # Assert that other plt functions were called
    mock_tight_layout_call.assert_called_once()
    mock_show_call.assert_called_once()


def test_plot_all_tours_exceeds_max_subplots_prints_warning(capsys):
    """
    Tests that plot_all_tours prints a warning if num_valid_results > MAX_SUBPLOTS_IN_PLOT.
    Covers line 1018.
    """
    num_results = MAX_SUBPLOTS_IN_PLOT + 1
    results_data = []
    for i in range(num_results):
        results_data.append({
            'name': f'instance_{i}', 'error': False,
            'coords': np.array([[0, 0], [1, 1]]), 'heu_tour': [0, 1]
        })

    # Mock matplotlib calls to prevent actual plotting and control subplot return
    mock_ax_instance = MagicMock(spec=matplotlib.axes.Axes)  # Changed spec
    mock_ax_instance.plot = MagicMock()
    mock_ax_instance.set_title = MagicMock()
    mock_ax_instance.set_xticks = MagicMock()
    mock_ax_instance.set_yticks = MagicMock()
    mock_ax_instance.set_aspect = MagicMock()
    mock_ax_instance.set_axis_off = MagicMock()

    mock_figure_instance = MagicMock(spec=matplotlib.figure.Figure)  # Changed spec
    mock_figure_instance.legend = MagicMock()
    mock_figure_instance.subplots_adjust = MagicMock()

    cols_for_mock = int(np.ceil(np.sqrt(MAX_SUBPLOTS_IN_PLOT)))
    rows_for_mock = int(np.ceil(MAX_SUBPLOTS_IN_PLOT / cols_for_mock))
    mock_axes_array_2d = np.array(
        [[MagicMock(spec=matplotlib.axes.Axes) for _ in range(cols_for_mock)] for _ in range(rows_for_mock)]  # Changed spec
    )
    for r_idx in range(rows_for_mock):  # Renamed r to r_idx to avoid conflict if you use r elsewhere
        for c_idx in range(cols_for_mock):  # Renamed c to c_idx
            ax = mock_axes_array_2d[r_idx, c_idx]
            ax.plot = MagicMock()
            ax.set_title = MagicMock()
            ax.set_xticks = MagicMock()
            ax.set_yticks = MagicMock()
            ax.set_aspect = MagicMock()
            ax.set_axis_off = MagicMock()

    with patch('matplotlib.pyplot.subplots', return_value=(mock_figure_instance, mock_axes_array_2d)), \
         patch('lin_kernighan_tsp_solver.utils.Line2D'), \
         patch('matplotlib.pyplot.tight_layout'), \
         patch('matplotlib.pyplot.show'):
        plot_all_tours(results_data)

    captured = capsys.readouterr()
    # The warning message in your main code was slightly different than my previous suggestion.
    # Let's use the one from your main code:
    expected_warning = (
        f"Warning: Plotting first {MAX_SUBPLOTS_IN_PLOT} of "
        f"{num_results} valid results."
    )
    assert expected_warning in captured.out


def test_plot_all_tours_turns_off_unused_subplots():
    """
    Tests that plot_all_tours calls set_axis_off() on unused subplots.
    Covers line 1060.
    """
    num_valid_results = 3

    results_data = []
    for i in range(num_valid_results):
        results_data.append({
            'name': f'instance_{i}', 'error': False,
            'coords': np.array([[0, 0], [1, 1]]), 'heu_tour': [0, 1],
            'opt_tour': None  # Ensure opt_tour key exists if plot_all_tours expects it
        })

    mock_axes_instances = [MagicMock(spec=matplotlib.axes.Axes) for _ in range(4)]
    for ax_mock in mock_axes_instances:
        ax_mock.plot = MagicMock()
        ax_mock.set_title = MagicMock()
        ax_mock.set_xticks = MagicMock()
        ax_mock.set_yticks = MagicMock()
        ax_mock.set_aspect = MagicMock()
        ax_mock.set_axis_off = MagicMock()

    mock_axes_array_2d = np.array([
        [mock_axes_instances[0], mock_axes_instances[1]],
        [mock_axes_instances[2], mock_axes_instances[3]]
    ])
    mock_figure_instance = MagicMock(spec=matplotlib.figure.Figure)  # Changed spec
    mock_figure_instance.legend = MagicMock()
    mock_figure_instance.subplots_adjust = MagicMock()

    with patch('matplotlib.pyplot.subplots', return_value=(mock_figure_instance, mock_axes_array_2d)), \
         patch('lin_kernighan_tsp_solver.utils.Line2D'), \
         patch('matplotlib.pyplot.tight_layout'), \
         patch('matplotlib.pyplot.show'):
        plot_all_tours(results_data)

    mock_axes_instances[0].set_axis_off.assert_not_called()
    mock_axes_instances[1].set_axis_off.assert_not_called()
    mock_axes_instances[2].set_axis_off.assert_not_called()
    mock_axes_instances[3].set_axis_off.assert_called_once()
