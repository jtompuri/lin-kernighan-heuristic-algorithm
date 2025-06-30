"""
main.py

Main entry point for the Lin-Kernighan TSP solver.

This script processes TSPLIB format instances, computes heuristic solutions using chained LK,
compares against optimal tours if available, and displays results in both tabular and graphical form.

Usage:
    python -m lin_kernighan_tsp_solver.main
"""

import time
import math
from pathlib import Path
from typing import Dict, Any
import numpy as np

from .config import TSP_FOLDER_PATH, FLOAT_COMPARISON_TOLERANCE
from .lk_algorithm import (
    build_distance_matrix,
    chained_lin_kernighan
)
from .tsp_io import read_tsp_file, read_opt_tour
from .utils import display_summary_table, plot_all_tours


def process_single_instance(
        tsp_file_path_str: str, opt_tour_file_path_str: str
) -> Dict[str, Any]:
    """
    Processes one TSP instance: loads, runs LK, calculates stats.

    Args:
        tsp_file_path_str (str): Path to .tsp file.
        opt_tour_file_path_str (str): Path to .opt.tour file.

    Returns:
        Dict[str, Any]: Dictionary with results: name, coords, tours, lengths, gap, time, error.
    """
    problem_name = Path(tsp_file_path_str).stem
    print(f"Processing {problem_name} (EUC_2D)...")
    # Initialize results dictionary
    results: Dict[str, Any] = {
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
        opt_len: float | None = None

        if opt_tour_nodes:  # If an optimal tour was successfully read
            current_opt_len = 0.0
            for i in range(len(opt_tour_nodes)):
                a = opt_tour_nodes[i]
                b = opt_tour_nodes[(i + 1) % len(opt_tour_nodes)]
                current_opt_len += D[a, b]
            opt_len = current_opt_len
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

        # Calculate percentage gap if optimal length is known and positive
        if opt_len is not None:
            if opt_len > FLOAT_COMPARISON_TOLERANCE * 10:  # Avoid division by zero/small
                gap_percentage = 100.0 * (heuristic_len - opt_len) / opt_len
                results['gap'] = max(0.0, gap_percentage)  # Gap cannot be negative
            elif math.isclose(opt_len, 0.0):  # Optimal is zero
                results['gap'] = 0.0 if math.isclose(heuristic_len, 0.0) else float('inf')
        # If opt_len is None, results['gap'] remains None

        gap_str = f"Gap: {results['gap']:.2f}%  " if results['gap'] is not None else ""
        print(f"  Heuristic length: {heuristic_len:.2f}  {gap_str}Time: {elapsed_time:.2f}s")

    except Exception as e:
        print(f"  Skipping {problem_name} due to error: {e}")
        results['error'] = True  # Mark instance as errored
        # Ensure essential keys exist for summary, even on error
        results['heu_len'] = float('inf')
        results['time'] = results.get('time', 0.0)  # Keep time if partially run
    return results


def main():
    """
    Main function to process all TSP instances in the configured folder.

    Finds all .tsp files, processes each instance, displays a summary table,
    and plots all tours.
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
            except Exception as e:  # Catch any unexpected error during processing
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
