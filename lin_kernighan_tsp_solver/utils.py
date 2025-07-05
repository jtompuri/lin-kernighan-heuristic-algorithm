"""
Utility functions for the Lin-Kernighan TSP solver.

This module provides functions to display a summary table of results
and to plot tours for all processed TSP instances.

Functions:
    display_summary_table(results_data): Prints a formatted summary table of processing results.
    plot_all_tours(results_data): Plots optimal and heuristic tours for processed instances.
"""

import math
from pathlib import Path
from typing import Any, Optional
import matplotlib

# Backend configuration function
def _configure_matplotlib_backend(interactive: bool = False):
    """Configure matplotlib backend based on plotting mode."""
    if interactive:
        # Try to use interactive backend
        try:
            import tkinter
            matplotlib.use('TkAgg')
            return True
        except ImportError:
            print("Warning: Interactive plotting requested but tkinter not available.")
            print("Install tkinter with: sudo apt-get install python3-tk")
            print("Falling back to non-interactive plotting.")
            matplotlib.use('Agg')
            return False
    else:
        matplotlib.use('Agg')
        return False

# Set non-interactive backend by default
_configure_matplotlib_backend(False)

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError as e:
    if "tkinter" in str(e).lower():
        print("\nError: tkinter is required for matplotlib but not installed.")
        print("On Ubuntu/Debian, install it with:")
        print("    sudo apt-get install python3-tk")
        print("\nAlternatively, matplotlib is configured to use non-interactive backend,")
        print("so this error shouldn't occur during normal operation.")
    raise

# Check tkinter availability (without printing info message)
def _check_tkinter_available():
    """Check if tkinter is available without printing messages."""
    try:
        import tkinter
        return True
    except ImportError:
        return False

# Store tkinter availability
_tkinter_available = _check_tkinter_available()

from . import config


def display_summary_table(
    results_data: list[dict[str, Any]],
    override_config: Optional[dict[str, Any]] = None
) -> None:
    """
    Prints a formatted summary table of processing results.

    Args:
        results_data (list[dict[str, Any]]): List of result dictionaries from instances.
        override_config (dict[str, Any], optional): Runtime config values (e.g., from CLI args).
    """
    print("\nConfiguration parameters:")
    config_to_print = config.LK_CONFIG.copy()
    if override_config:
        config_to_print.update(override_config)
    for key, value in config_to_print.items():
        if isinstance(value, float):
            print(f"  {key:<11s} = {value:.2f}")
        else:
            print(f"  {key:<11s} = {value}")
    print("")

    header = f"{'Instance':<10s} {'OptLen':>8s} {'HeuLen':>8s} " \
        f"{'Gap(%)':>8s} {'Time(s)':>8s}"
    print(header)
    print("-" * len(header))

    valid_results_for_table = [r for r in results_data if not r.get('error')]
    for r_item in valid_results_for_table:
        opt_len_str = (f"{r_item['opt_len']:>8.2f}"
                       if r_item['opt_len'] is not None else f"{'N/A':>8s}")
        gap_str = (f"{r_item['gap']:>8.2f}"
                   if r_item['gap'] is not None else f"{'N/A':>8s}")
        print(
            f"{r_item['name']:<10s} {opt_len_str} "
            f"{r_item['heu_len']:>8.2f} {gap_str} "
            f"{r_item['time']:>8.2f}"
        )

    if valid_results_for_table:  # Calculate summary only if there are valid results
        print("-" * len(header))
        num_valid_items = len(valid_results_for_table)

        valid_opt_lens = [r['opt_len'] for r in valid_results_for_table
                          if r['opt_len'] is not None]
        # Filter out inf gaps for average calculation
        valid_gaps = [r['gap'] for r in valid_results_for_table
                      if r['gap'] is not None and r['gap'] != float('inf')]

        total_opt_len_sum = sum(valid_opt_lens) if valid_opt_lens else None
        total_heu_len_sum = sum(r['heu_len'] for r in valid_results_for_table
                                if r['heu_len'] != float('inf'))  # Sum finite heuristic lengths
        avg_gap_val = sum(valid_gaps) / len(valid_gaps) if valid_gaps else None
        avg_time_val = (sum(r['time'] for r in valid_results_for_table) / num_valid_items
                        if num_valid_items > 0 else 0.0)

        total_opt_str = (f"{total_opt_len_sum:>8.2f}"
                         if total_opt_len_sum is not None else f"{'N/A':>8s}")
        avg_gap_str = (f"{avg_gap_val:>8.2f}"
                       if avg_gap_val is not None else f"{'N/A':>8s}")
        total_heu_str = (f"{total_heu_len_sum:>8.2f}"
                         if total_heu_len_sum != float('inf') else f"{'N/A':>8s}")

        print(
            f"{'SUMMARY':<10s} {total_opt_str} {total_heu_str} "
            f"{avg_gap_str} {avg_time_val:>8.2f}"
        )
    print("Done.")


def plot_all_tours(results_data: list[dict[str, Any]], force_save_plot: bool = False) -> None:
    """
    Plots optimal and heuristic tours for processed instances.
    
    Automatically uses interactive plotting if tkinter is available, otherwise saves to file.
    This behavior can be overridden with force_save_plot.

    Args:
        results_data (list[dict[str, Any]]): List of result dictionaries.
        force_save_plot (bool): If True, always save to file instead of showing interactively.
    """
    # Determine plotting mode: interactive if tkinter available and not forced to save
    use_interactive = _tkinter_available and not force_save_plot
    
    # Configure matplotlib backend based on plotting mode
    if use_interactive:
        backend_available = _configure_matplotlib_backend(True)
        if not backend_available:
            use_interactive = False  # Fall back to non-interactive
            print("Falling back to saving plot to file.")
    else:
        _configure_matplotlib_backend(False)
    
    # Filter for results that are not errored and have coordinates
    valid_results_to_plot = [
        r for r in results_data
        if not r.get('error') and r.get('coords') is not None and r['coords'].size > 0
    ]
    num_valid_results = len(valid_results_to_plot)

    if num_valid_results == 0:
        print("No valid results with coordinates to plot.")
        return

    # Limit number of plots
    results_to_plot_limited = (valid_results_to_plot[:config.MAX_SUBPLOTS_IN_PLOT] if num_valid_results > config.MAX_SUBPLOTS_IN_PLOT else valid_results_to_plot)
    if num_valid_results > config.MAX_SUBPLOTS_IN_PLOT:
        print(f"Warning: Plotting first {config.MAX_SUBPLOTS_IN_PLOT} of {num_valid_results} valid results.")

    num_to_plot_actual = len(results_to_plot_limited)
    if num_to_plot_actual == 0:
        print("No tours to plot.")
        return

    cols = int(math.ceil(math.sqrt(num_to_plot_actual)))
    rows = int(math.ceil(num_to_plot_actual / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)  # Ensure axes is always 2D
    axes_list = axes.flatten()
    plotted_heuristic_legend, plotted_optimal_legend = False, False

    for i, r_item in enumerate(results_to_plot_limited):
        ax = axes_list[i]
        coords = r_item['coords']

        # Plot heuristic tour if available
        if r_item['heu_tour']:
            heu_plot_nodes = r_item['heu_tour'] + [r_item['heu_tour'][0]]  # Close the loop
            ax.plot(coords[heu_plot_nodes, 0], coords[heu_plot_nodes, 1],
                    '-', label='Heuristic', zorder=1, color='black')
            plotted_heuristic_legend = True

        # Plot optimal tour if available
        opt_tour_data = r_item.get('opt_tour')
        if opt_tour_data:  # Check if opt_tour_data is not None and not empty
            opt_plot_nodes = opt_tour_data + [opt_tour_data[0]]  # Close the loop
            ax.plot(coords[opt_plot_nodes, 0], coords[opt_plot_nodes, 1],
                    ':', label='Optimal', zorder=2, color='red')
            plotted_optimal_legend = True

        title = f"{r_item['name']}"
        # Safely get gap, opt_len, heu_len for the title
        gap_val = r_item.get('gap')

        if gap_val is not None and gap_val != float('inf'):
            title += f" gap={gap_val:.2f}%"
        ax.set_title(title)
        ax.set_xticks([])  # Hide ticks and labels
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')  # Square aspect ratio

    # Turn off unused subplots
    for i in range(num_to_plot_actual, len(axes_list)):
        axes_list[i].set_axis_off()

    # Create legend for the figure
    legend_elements = []
    if plotted_heuristic_legend:
        legend_elements.append(Line2D([0], [0], color='black', ls='-', label='Heuristic'))
    if plotted_optimal_legend:
        legend_elements.append(Line2D([0], [0], color='red', ls=':', label='Optimal'))
    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.0))
        # Adjust top margin if legend is present
        fig.subplots_adjust(top=(0.95 if num_to_plot_actual > cols else 0.90))

    plt.tight_layout(rect=(0, 0, 1, 0.96 if legend_elements else 1.0))  # Adjust for legend
    
    # Show interactive plot or save to file based on availability and preference
    if use_interactive:
        plt.show()
        print("Interactive plot displayed. Close the plot window to continue.")
    else:
        # Save plot to solutions/plots directory with descriptive filename
        plots_dir = Path("solutions/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename based on problem names
        if num_to_plot_actual == 1:
            # Single problem
            plot_filename = f"{results_to_plot_limited[0]['name']}_tour.png"
        else:
            # Multiple problems - use first and last names or count
            if num_to_plot_actual <= 3:
                names = "_".join(r['name'] for r in results_to_plot_limited)
                plot_filename = f"{names}_tours.png"
            else:
                first_name = results_to_plot_limited[0]['name']
                last_name = results_to_plot_limited[-1]['name']
                plot_filename = f"{first_name}_to_{last_name}_{num_to_plot_actual}_tours.png"
        
        plot_file = plots_dir / plot_filename
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
        if not _tkinter_available:
            print("Note: Install tkinter for interactive plots: sudo apt-get install python3-tk")
    
    plt.close()


def save_heuristic_tour(tour: list[int], problem_name: str, tour_length: float,
                        solutions_folder: str | None = None) -> str:
    """Save a heuristic tour to a file in TSPLIB format.

    Args:
        tour (list[int]): The tour as a list of node indices.
        problem_name (str): Name of the problem (used for filename).
        tour_length (float): Length/cost of the tour.
        solutions_folder (str | None, optional): Folder to save to. If None, uses config default.

    Returns:
        str: Path to the saved file.

    Raises:
        IOError: If the file cannot be written.
    """
    from pathlib import Path
    from .config import SOLUTIONS_FOLDER_PATH

    if solutions_folder is None:
        folder_path = Path(SOLUTIONS_FOLDER_PATH)
    else:
        folder_path = Path(solutions_folder)

    # Ensure the solutions folder exists
    folder_path.mkdir(parents=True, exist_ok=True)

    # Create filename: problem_name.heu.tour
    filename = f"{problem_name}.heu.tour"
    filepath = folder_path / filename

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"NAME: {problem_name}.heu.tour\n")
            f.write("TYPE: TOUR\n")
            f.write(f"COMMENT: Heuristic tour (Lin-Kernighan), length {tour_length:.2f}\n")
            f.write(f"DIMENSION: {len(tour)}\n")
            f.write("TOUR_SECTION\n")

            # Write tour nodes (TSPLIB uses 1-indexed)
            for node in tour:
                f.write(f"{node + 1}\n")

            f.write("-1\n")  # End marker
            f.write("EOF\n")

        return str(filepath)

    except IOError as e:
        raise IOError(f"Failed to save tour to {filepath}: {e}")
