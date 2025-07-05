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
import concurrent.futures
from multiprocessing import cpu_count
import numpy as np

from .config import TSP_FOLDER_PATH, FLOAT_COMPARISON_TOLERANCE, LK_CONFIG
from .lk_algorithm import (
    build_distance_matrix,
    chained_lin_kernighan
)
from .tsp_io import read_tsp_file, read_opt_tour
from .utils import display_summary_table, plot_all_tours, save_heuristic_tour
from .starting_cycles import generate_starting_cycle


def _calculate_tour_length(tour_nodes: list[int], D: np.ndarray) -> float:
    """Calculates the total length of a given tour using vectorized operations.

    Args:
        tour_nodes (list[int]): A list of node indices representing the tour.
        D (np.ndarray): The distance matrix.

    Returns:
        float: The total length of the tour.
    """
    if not tour_nodes:
        return 0.0

    tour_len = len(tour_nodes)
    if tour_len == 1:
        return 0.0

    # Use vectorized operations for better performance on larger tours
    if tour_len > 10:
        tour_array = np.array(tour_nodes, dtype=np.int32)
        next_nodes = np.roll(tour_array, -1)  # Shift by one position
        return float(np.sum(D[tour_array, next_nodes]))

    # Use loop for small tours where overhead isn't worth it
    length = 0.0
    for i in range(tour_len):
        a = tour_nodes[i]
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
        tsp_file_path_str: str,
        opt_tour_file_path_str: str,
        time_limit: float | None = None,
        verbose: bool = True
) -> dict[str, Any]:
    """Processes a single TSP instance.

    Loads a TSP problem and its optimal tour, runs the chained Lin-Kernighan
    heuristic, and calculates performance statistics.

    Args:
        tsp_file_path_str (str): The file path to the .tsp problem file.
        opt_tour_file_path_str (str): The file path to the .opt.tour solution file.
        time_limit (float | None, optional): Time limit for the algorithm.
            Defaults to None (uses algorithm default).
        verbose (bool, optional): Whether to print progress messages.
            Defaults to True.

    Returns:
        dict[str, Any]: A dictionary containing the results, including problem name,
            tour lengths, gap, and execution time.
    """
    problem_name = Path(tsp_file_path_str).stem
    if verbose:
        print(f"Processing {problem_name} (EUC_2D)...")

    # Initialize results dictionary
    results: dict[str, Any] = {
        'name': problem_name, 'coords': np.array([]), 'opt_tour': None,
        'heu_tour': [], 'opt_len': None, 'heu_len': float('inf'),
        'gap': None, 'time': 0.0, 'error': False,
        'nodes': 0
    }

    try:
        coords = read_tsp_file(tsp_file_path_str)
        results['coords'] = coords
        if coords.size == 0:
            raise ValueError("No coordinates loaded from TSP file.")
        results['nodes'] = coords.shape[0]
        D = build_distance_matrix(coords)

        opt_tour_nodes = read_opt_tour(opt_tour_file_path_str)
        results['opt_tour'] = opt_tour_nodes
        opt_len = _calculate_tour_length(opt_tour_nodes, D) if opt_tour_nodes else None

        if opt_len is not None:
            results['opt_len'] = opt_len
            if verbose:
                print(f"  Optimal length: {opt_len:.2f}")
        elif verbose:
            print(f"  Optimal tour not available for {problem_name}.")

        initial_tour = generate_starting_cycle(coords, method=LK_CONFIG["STARTING_CYCLE"])
        start_time = time.time()

        heuristic_tour, heuristic_len = chained_lin_kernighan(
            coords, initial_tour,
            known_optimal_length=opt_len,
            time_limit_seconds=time_limit
        )
        elapsed_time = time.time() - start_time
        results['heu_tour'], results['heu_len'] = heuristic_tour, heuristic_len
        results['time'] = elapsed_time

        # Calculate percentage gap if optimal length is known
        results['gap'] = _calculate_gap(heuristic_len, opt_len)

        # Save heuristic tour if enabled in configuration
        if LK_CONFIG.get("SAVE_TOURS", False) and heuristic_tour:
            try:
                saved_path = save_heuristic_tour(heuristic_tour, problem_name, heuristic_len)
                if verbose:
                    print(f"  Saved heuristic tour to: {saved_path}")
            except IOError as save_error:
                if verbose:
                    print(f"  Warning: Failed to save tour: {save_error}")

        if verbose:
            gap_str = f"Gap: {results['gap']:.2f}%  " if results['gap'] is not None else ""
            print(f"  Heuristic length: {heuristic_len:.2f}  {gap_str}Time: {elapsed_time:.2f}s")

    except (IOError, ValueError) as e:
        if verbose:
            print(f"  Skipping {problem_name} due to error: {e}")
        results['error'] = True
        results['heu_len'] = float('inf')
        results['time'] = results.get('time', 0.0)

    return results


def main(
    use_parallel: bool = True,
    max_workers: int | None = None,
    time_limit: float | None = None,
    starting_cycle_method: str | None = None,
    tsp_files: list[str] | None = None,
    save_tours: bool | None = None,
    plot: bool = True,
    force_save_plot: bool = False
):
    """Main function with configurable options.

    Args:
        use_parallel: Whether to use parallel processing.
        max_workers: Maximum number of parallel workers.
        time_limit: Time limit per instance in seconds.
        starting_cycle_method: Starting cycle algorithm to use.
        tsp_files: List of specific TSP files to process. If None, processes all files in TSP_FOLDER_PATH.
        save_tours: Whether to save heuristic tours. If None, uses config default.
    """
    # Update LK_CONFIG with starting cycle method if provided
    if starting_cycle_method is not None:
        LK_CONFIG["STARTING_CYCLE"] = starting_cycle_method

    # Update LK_CONFIG with save tours setting if provided
    if save_tours is not None:
        LK_CONFIG["SAVE_TOURS"] = save_tours

    all_instance_results_list = []

    # Collect TSP file pairs based on input
    if tsp_files:
        # Process specific files provided via command line
        tsp_file_pairs = []
        for tsp_file in tsp_files:
            tsp_path = Path(tsp_file)
            if not tsp_path.exists():
                print(f"Error: TSP file not found: {tsp_file}")
                continue

            # Look for corresponding .opt.tour file
            base_name = tsp_path.stem
            opt_file_candidates = [
                tsp_path.parent / f"{base_name}.opt.tour",
                tsp_path.parent / f"{base_name}.opt",
                tsp_path.parent / f"{base_name}.tour"
            ]

            opt_file = None
            for candidate in opt_file_candidates:
                if candidate.exists():
                    opt_file = str(candidate)
                    break

            if opt_file is None:
                print(f"Warning: No optimal tour file found for {tsp_file}, using dummy path")
                opt_file = str(tsp_path.parent / f"{base_name}.opt.tour")

            tsp_file_pairs.append((str(tsp_path), opt_file))
    else:
        # Process all files in TSP_FOLDER_PATH (original behavior)
        if not TSP_FOLDER_PATH.is_dir():
            print(f"Error: TSP folder not found at {TSP_FOLDER_PATH}")
            return

        # Collect all TSP file pairs
        tsp_file_pairs = []
        for tsp_file_path_obj in sorted(TSP_FOLDER_PATH.glob('*.tsp')):
            base_name = tsp_file_path_obj.stem
            opt_tour_path_obj = TSP_FOLDER_PATH / (base_name + '.opt.tour')
            tsp_file_pairs.append((str(tsp_file_path_obj), str(opt_tour_path_obj)))

    if not tsp_file_pairs:
        if tsp_files:
            print("No valid TSP files found from the provided list.")
        else:
            print("No TSP files found in the specified directory.")
        return

    print(f"Found {len(tsp_file_pairs)} TSP instances.")

    if use_parallel and len(tsp_file_pairs) > 1:
        # Parallel processing
        effective_workers = min(
            max_workers or cpu_count(),
            len(tsp_file_pairs)
        )
        print(f"Processing using {effective_workers} parallel workers...")

        all_instance_results_list = _process_parallel(
            tsp_file_pairs, effective_workers, time_limit  # Add time_limit here
        )
    else:
        # Sequential processing (original behavior)
        print("Processing sequentially...")
        all_instance_results_list = _process_sequential(
            tsp_file_pairs, time_limit  # Add time_limit here
        )

    # Sort results by name for consistent output
    all_instance_results_list.sort(key=lambda x: x['name'])

    # Display results
    display_summary_table(all_instance_results_list, override_config={'TIME_LIMIT': time_limit})
    
    # Generate plots if requested
    if plot and all_instance_results_list:
        plot_all_tours(all_instance_results_list, force_save_plot=force_save_plot)


def _process_sequential(
    tsp_file_pairs: list[tuple[str, str]],
    time_limit: float | None = None
) -> list[dict[str, Any]]:
    """Process TSP instances sequentially (original behavior).

    Args:
        tsp_file_pairs (list[tuple[str, str]]): List of (tsp_file, opt_file) pairs.
        time_limit (float | None, optional): Time limit per instance. Defaults to None.

    Returns:
        list[dict[str, Any]]: List of result dictionaries.
    """
    results = []
    for tsp_file_path_str, opt_tour_path_str in tsp_file_pairs:
        try:
            result_dict = process_single_instance(
                tsp_file_path_str, opt_tour_path_str, time_limit=time_limit, verbose=True  # Enable verbose output
            )
            results.append(result_dict)
        except (IOError, ValueError, OSError, RuntimeError, MemoryError) as e:
            base_name = Path(tsp_file_path_str).stem
            print(f"Critical error processing {base_name}: {e}")
            # Append a basic error entry for summary purposes
            results.append(_create_error_result(base_name))
    return results


def _process_parallel(
    tsp_file_pairs: list[tuple[str, str]],
    max_workers: int,
    time_limit: float | None = None
) -> list[dict[str, Any]]:
    """Process TSP instances in parallel using ProcessPoolExecutor.

    Args:
        tsp_file_pairs (list[tuple[str, str]]): List of (tsp_file, opt_file) pairs.
        max_workers (int): Maximum number of worker processes.
        time_limit (float | None, optional): Time limit per instance. Defaults to None.

    Returns:
        list[dict[str, Any]]: List of result dictionaries.
    """
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs with time_limit and verbose=False for parallel processing
        future_to_files = {
            executor.submit(process_single_instance, tsp_file, opt_file, time_limit, False): (tsp_file, opt_file)
            for tsp_file, opt_file in tsp_file_pairs
        }

        # Collect results as they complete
        completed_count = 0
        total_count = len(future_to_files)

        for future in concurrent.futures.as_completed(future_to_files):
            tsp_file, opt_file = future_to_files[future]
            base_name = Path(tsp_file).stem
            completed_count += 1

            try:
                result_dict = future.result()
                results.append(result_dict)
                print(f"[{completed_count}/{total_count}] Completed: {result_dict['name']}")
            except (IOError, ValueError, OSError, RuntimeError, MemoryError) as e:
                print(f"[{completed_count}/{total_count}] Error processing {base_name}: {e}")
                results.append(_create_error_result(base_name))

    return results


def _create_error_result(problem_name: str) -> dict[str, Any]:
    """Creates a standardized error result dictionary.

    Args:
        problem_name (str): Name of the problem that failed.

    Returns:
        dict[str, Any]: Standardized error result dictionary.
    """
    return {
        'name': problem_name,
        'coords': np.array([]),
        'opt_tour': None,
        'heu_tour': [],
        'opt_len': None,
        'heu_len': float('inf'),
        'gap': None,
        'time': 0.0,
        'error': True,
        'nodes': 0
    }
