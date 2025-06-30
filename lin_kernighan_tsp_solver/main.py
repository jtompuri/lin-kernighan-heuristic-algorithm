"""
Main entry point for the Lin-Kernighan TSP solver.

This script processes TSPLIB format instances, computes heuristic solutions
using chained Lin-Kernighan, compares against optimal tours if available,
and displays results in both tabular and graphical form.

Usage:
    python -m lin_kernighan_tsp_solver
"""

import time
import math
from pathlib import Path
from typing import Any
import numpy as np

from .config import TSP_FOLDER_PATH, FLOAT_COMPARISON_TOLERANCE
from .lk_algorithm import (
    build_distance_matrix,
    chained_lin_kernighan
)
from .tsp_io import read_tsp_file, read_opt_tour
from .utils import display_summary_table, plot_all_tours


def _calculate_tour_length(tour_nodes: list[int], D: np.ndarray) -> float:
    """Calculates the total length of a given tour.

    Args:
        tour_nodes (list[int]): A list of node indices representing the tour.
        D (np.ndarray): The distance matrix.

    Returns:
        float: The total length of the tour.
    """
    if not tour_nodes:
        return 0.0
    length = 0.0
    tour_len = len(tour_nodes)
    for i, a in enumerate(tour_nodes):
        b = tour_nodes[(i + 1) % tour_len]
        length += D[a, b]
    return float(length)


def _calculate_gap(heuristic_len: float, opt_len: float | None) -> float | None:
    """Calculates the percentage gap between heuristic and optimal lengths.

    Args:
        heuristic_len (float): The length of the tour found by the heuristic.
        opt_len (float | None): The known optimal tour length.

    Returns:
        float | None: The percentage gap, or None if the optimal length is not
            provided or too small for a meaningful calculation.
    """
    if opt_len is None:
        return None
    if opt_len > FLOAT_COMPARISON_TOLERANCE * 10:
        gap = 100.0 * (heuristic_len - opt_len) / opt_len
        return max(0.0, gap)
    if math.isclose(opt_len, 0.0):
        return 0.0 if math.isclose(heuristic_len, 0.0) else float('inf')
    return None  # Gap is undefined for very small, non-zero optimal lengths


def process_single_instance(
        tsp_file_path_str: str, opt_tour_file_path_str: str
) -> dict[str, Any]:
    """Processes a single TSP instance.

    Loads a TSP problem and its optimal tour, runs the chained Lin-Kernighan
    heuristic, and calculates performance statistics.

    Args:
        tsp_file_path_str (str): The file path to the .tsp problem file.
        opt_tour_file_path_str (str): The file path to the .opt.tour solution file.

    Returns:
        dict[str, Any]: A dictionary containing the results, including problem name,
            tour lengths, gap, and execution time.
    """
    problem_name = Path(tsp_file_path_str).stem
    print(f"Processing {problem_name} (EUC_2D)...")
    # Initialize results dictionary
    results: dict[str, Any] = {
        'name': problem_name, 'coords': np.array([]), 'opt_tour': None,
        'heu_tour': [], 'opt_len': None, 'heu_len': float('inf'),
        'gap': None, 'time': 0.0, 'error': False,
        'nodes': 0  # Initialize 'nodes' key
    }
    try:
        coords = read_tsp_file(tsp_file_path_str)
        results['coords'] = coords
        if coords.size == 0:  # Check if read_tsp_file returned empty
            raise ValueError("No coordinates loaded from TSP file.")
        results['nodes'] = coords.shape[0]  # Set the number of nodes
        D = build_distance_matrix(coords)

        opt_tour_nodes = read_opt_tour(opt_tour_file_path_str)
        results['opt_tour'] = opt_tour_nodes
        opt_len = _calculate_tour_length(opt_tour_nodes, D) if opt_tour_nodes else None

        if opt_len is not None:
            results['opt_len'] = opt_len
            print(f"  Optimal length: {opt_len:.2f}")
        else:
            print(f"  Optimal tour not available for {problem_name}.")

        initial_tour = list(range(len(coords)))  # Simple initial tour: 0,1,2...
        start_time = time.time()

        heuristic_tour, heuristic_len = chained_lin_kernighan(
            coords, initial_tour, known_optimal_length=opt_len
        )
        elapsed_time = time.time() - start_time
        results['heu_tour'], results['heu_len'] = heuristic_tour, heuristic_len
        results['time'] = elapsed_time

        # Calculate percentage gap if optimal length is known
        results['gap'] = _calculate_gap(heuristic_len, opt_len)

        gap_str = f"Gap: {results['gap']:.2f}%  " if results['gap'] is not None else ""
        print(f"  Heuristic length: {heuristic_len:.2f}  {gap_str}Time: {elapsed_time:.2f}s")

    except (IOError, ValueError) as e:
        print(f"  Skipping {problem_name} due to error: {e}")
        results['error'] = True  # Mark instance as errored
        # Ensure essential keys exist for summary, even on error
        results['heu_len'] = float('inf')
        results['time'] = results.get('time', 0.0)  # Keep time if partially run
    return results


def main():
    """Main function to process all TSP instances in the configured folder.

    Finds all .tsp files in the target directory, processes each instance
    sequentially, and then displays a summary table and plots all tours.
    """
    all_instance_results_list = []
    if not TSP_FOLDER_PATH.is_dir():
        print(f"Error: TSP folder not found at {TSP_FOLDER_PATH}")
    else:
        # Iterate over .tsp files in the specified folder
        for tsp_file_path_obj in sorted(TSP_FOLDER_PATH.glob('*.tsp')):
            base_name = tsp_file_path_obj.stem
            # Corresponding .opt.tour file path
            opt_tour_path_obj = TSP_FOLDER_PATH / (base_name + '.opt.tour')

            try:
                result_dict = process_single_instance(
                    str(tsp_file_path_obj), str(opt_tour_path_obj)
                )
                all_instance_results_list.append(result_dict)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Critical error processing {base_name}: {e}")
                # Append a basic error entry for summary purposes
                all_instance_results_list.append({
                    'name': base_name, 'coords': np.array([]),
                    'opt_tour': None, 'heu_tour': [], 'opt_len': None,
                    'heu_len': float('inf'), 'gap': None, 'time': 0.0,
                    'error': True
                })

    display_summary_table(all_instance_results_list)
    plot_all_tours(all_instance_results_list)
