"""
A simplified Traveling Salesperson Problem (TSP) solver.

This module implements a basic k-opt based heuristic for solving the TSP.
It focuses on the core k-opt move and a recursive improvement strategy.

Key features:
- Computes pairwise Euclidean distances.
- Implements 2-opt swaps.
- Uses a recursive k-opt approach for tour improvement.
- Includes a simple TSP solver function with a time limit.
- Provides functionality for parsing TSP files (EUC_2D) and optimal tour files.
- Visualizes tours using matplotlib.

This solver is a simplified version and does not include many advanced
features of more sophisticated algorithms like the full Lin-Kernighan heuristic,
such as:
- Comprehensive flip tracking and rollback mechanisms.
- Specialized alternate first steps (alternate_step).
- Lin-Kernighan specific neighborhood ordering.
- Configurable breadth and depth parameters for the search.
- Restart mechanisms like "kick" or double-bridge perturbations.
- An abstracted Tour object with methods like `next` or `flip`.
- Delta-based cost updates for tentative improvements.
- Chained Lin-Kernighan algorithm structure.
"""
import os
import time
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Constants ---
# Define path relative to this script file for robustness
TSP_FOLDER_PATH = Path(__file__).resolve().parent.parent / "verifications" / "tsplib95"
MAX_SUBPLOTS = 25  # Maximum number of subplots in the tour visualization


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances for all city coordinates.

    Args:
        coords: A numpy array of shape (n, 2) where n is the number of cities,
                and each row contains the [x, y] coordinates of a city.

    Returns:
        A numpy array D of shape (n, n) where D[i, j] is the Euclidean
        distance between city i and city j.
    """
    diff = coords[:, None] - coords[None, :]  # shape (n, n, 2)
    return np.linalg.norm(diff, axis=2)       # shape (n, n)


def total_distance(tour: list[int], dist_matrix: np.ndarray) -> float:
    """
    Compute the total length of a tour given a distance matrix.

    Args:
        tour: A list of integers representing the order of cities visited.
        dist_matrix: A square numpy array where dist_matrix[i, j] is the
                     distance between city i and city j.

    Returns:
        The total length of the tour.
    """
    return sum(dist_matrix[tour[i], tour[(i + 1) % len(tour)]]
               for i in range(len(tour)))


def compute_gain(dist_matrix: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    """
    Compute the gain from swapping two edges (a-b) and (c-d) to (a-c) and (b-d).
    A positive gain means the new tour (with edges a-c, b-d) is shorter.

    Args:
        dist_matrix: The distance matrix.
        a, b: Nodes defining the first edge (a-b).
        c, d: Nodes defining the second edge (c-d).

    Returns:
        The calculated gain.
    """
    return dist_matrix[a][b] + dist_matrix[c][d] - \
        dist_matrix[a][c] - dist_matrix[b][d]


def two_opt_swap(tour: list[int], i: int, j: int) -> list[int]:
    """
    Perform a 2-opt swap by reversing the segment of the tour between
    indices i+1 and j (inclusive).

    Args:
        tour: The current tour (list of node indices).
        i: The index of the node before the start of the segment to reverse.
        j: The index of the last node in the segment to reverse.

    Returns:
        A new tour list with the segment reversed.
    """
    return tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]


def recursive_k_opt(
    tour: list[int],
    dist_matrix: np.ndarray,
    start_index: int,
    visited: set[int],
    gain_sum: float,
    max_k: int,
    deadline: float
) -> bool:
    """
    Recursive search function that attempts to improve the tour by applying
    k-opt moves.

    Args:
        tour: The current tour (list of node indices). Modified in-place if
              an improvement is found.
        dist_matrix: The distance matrix.
        start_index: The starting index in the tour for the current move.
        visited: A set of indices already involved in the current k-opt sequence.
        gain_sum: The accumulated gain from previous steps in this k-opt sequence.
        max_k: The maximum depth (number of edges to swap) for the k-opt move.
        deadline: The wall-clock time by which the search must conclude.

    Returns:
        True if an improvement was made to the tour, False otherwise.
    """
    # Abort recursion if depth exceeded or time limit reached
    if max_k == 0 or time.time() >= deadline:
        return False

    n = len(tour)
    t1 = tour[start_index]
    t2 = tour[(start_index + 1) % n]

    for j_idx in range(n):
        t3 = tour[j_idx]
        t4 = tour[(j_idx + 1) % n]

        # Skip invalid pairs (same node, adjacent edge, or already visited)
        if t3 == t1 or t4 == t2 or j_idx == start_index or j_idx in visited:
            continue

        current_gain = compute_gain(dist_matrix, t1, t2, t3, t4)
        # Only consider moves that improve or might lead to improvement
        if current_gain + gain_sum <= 0:
            continue

        new_tour_candidate = two_opt_swap(tour, start_index, j_idx)
        # Check if this direct 2-opt swap is an improvement
        if total_distance(new_tour_candidate, dist_matrix) < total_distance(tour, dist_matrix):
            tour[:] = new_tour_candidate  # Commit the improved tour in-place
            return True

        # Try extending the move recursively to a deeper k-opt move
        if recursive_k_opt(new_tour_candidate, dist_matrix, start_index,
                           visited | {j_idx}, gain_sum + current_gain,
                           max_k - 1, deadline):
            tour[:] = new_tour_candidate  # Commit if recursive call improved
            return True

        if time.time() >= deadline:  # Time guard in deep recursion
            return False

    return False  # No improving move found at this level


def simple_tsp_solver(
    coords: np.ndarray,
    max_k: int = 4,
    time_limit: float = 10.0  # Function's own default time limit
) -> tuple[list[int], float]:
    """
    Runs a simplified k-opt search until no further improvement is found
    or the wall-clock time limit is reached.

    Args:
        coords: A numpy array of city coordinates.
        max_k: The maximum k for k-opt moves.
        time_limit: The maximum time in seconds allowed for the search.

    Returns:
        A tuple containing:
            - The best tour found (list of node indices).
            - The length of the best tour.
    """
    deadline = time.time() + time_limit

    n = len(coords)
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0

    coords_arr = np.asarray(coords, float)
    dist_matrix = compute_distance_matrix(coords_arr)

    current_tour = list(range(n))
    best_tour = current_tour.copy()
    best_tour_length = total_distance(current_tour, dist_matrix)
    has_improved_in_pass = True

    while has_improved_in_pass and time.time() < deadline:
        has_improved_in_pass = False
        for i in range(n):
            if time.time() >= deadline:
                break  # Stop outer loop if time is up

            visited_indices = set()
            # Pass a copy of current_tour to recursive_k_opt if it modifies it
            # and we only want to commit if it's better than best_tour_length
            temp_tour_for_recursion = current_tour.copy()
            if recursive_k_opt(temp_tour_for_recursion, dist_matrix, i,
                               visited_indices, 0.0, max_k, deadline):
                current_tour_length = total_distance(temp_tour_for_recursion, dist_matrix)
                if current_tour_length < best_tour_length:
                    best_tour = temp_tour_for_recursion.copy()
                    best_tour_length = current_tour_length
                    current_tour = temp_tour_for_recursion.copy()  # Update current for next iterations
                    has_improved_in_pass = True
                    break  # Restart outer loop from beginning with improved tour

    return best_tour, best_tour_length


def plot_tour(coords: np.ndarray, tour: list[int], length: float | None = None):
    """
    Visualize the TSP tour in 2D using matplotlib.

    Args:
        coords: Numpy array of city coordinates.
        tour: List of node indices representing the tour.
        length: Optional total length of the tour to display in the title.
    """
    if not tour:
        print("Cannot plot an empty tour.")
        return

    coords_arr = np.asarray(coords)
    tour_cycle = tour + [tour[0]]  # Close the tour into a cycle
    x_coords, y_coords = coords_arr[tour_cycle, 0], coords_arr[tour_cycle, 1]

    plt.figure(figsize=(6, 6))
    plt.plot(x_coords, y_coords, 'o-', markersize=2)
    title = "TSP Path"
    if length is not None:
        title = f"TSP Path (Length: {length:.2f})"
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def parse_tsp_file(path: str) -> tuple[np.ndarray, str | None]:
    """
    Reads a TSPLIB formatted TSP file.

    Returns a numpy array of coordinates (sorted by node ID, effectively
    0-indexed if nodes are 1...N) and the edge weight type.

    Args:
        path: The file path to the TSPLIB .tsp file.

    Returns:
        A tuple containing:
            - A numpy array of coordinates.
            - The edge weight type as a string, or None if not found/error.
    """
    coords_dict: dict[int, list[float]] = {}
    edge_weight_type: str | None = None
    reading_nodes = False

    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_content in f:
                line = line_content.strip()
                if not line:
                    continue

                if ":" in line and not reading_nodes:
                    key, value = [part.strip() for part in line.split(":", 1)]
                    if key.upper() == "EDGE_WEIGHT_TYPE":
                        edge_weight_type = value.upper()

                if line.upper().startswith("NODE_COORD_SECTION"):
                    reading_nodes = True
                    continue
                if line.upper() == "EOF":
                    break

                if reading_nodes:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            # TSPLIB nodes are usually 1-based
                            node_id = int(parts[0])
                            x_coord = float(parts[1])
                            y_coord = float(parts[2])
                            coords_dict[node_id] = [x_coord, y_coord]
                        except ValueError:
                            print(f"Warning: Could not parse node coord line: "
                                  f"'{line_content.strip()}' in {path}")
        if not coords_dict:
            return np.array([], dtype=float), edge_weight_type

        sorted_node_ids = sorted(coords_dict.keys())
        if not sorted_node_ids or sorted_node_ids[0] != 1 or \
           sorted_node_ids[-1] != len(sorted_node_ids):
            print(f"Warning: Node IDs in {path} may not be a contiguous "
                  f"1-N sequence. Coords ordered by sorted node ID.")

        coords_list = [coords_dict[node_id] for node_id in sorted_node_ids]
        return np.array(coords_list, dtype=float), edge_weight_type

    except FileNotFoundError:
        print(f"Error: TSP file not found at {path}")
        return np.array([], dtype=float), None
    except Exception as e:
        print(f"Error reading TSP file {path}: {e}")
        return np.array([], dtype=float), None


def read_opt_tour(path: str) -> list[int] | None:
    """
    Reads an optimal tour from a .opt.tour file (TSPLIB format).

    Args:
        path: The file path to the .opt.tour file.

    Returns:
        A list of 0-indexed node IDs representing the optimal tour,
        or None if the file cannot be read or is malformed.
    """
    tour: list[int] = []
    reading = False
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.strip()
                if token.upper().startswith('TOUR_SECTION'):
                    reading = True
                    continue
                if not reading:
                    continue
                for part in token.split():
                    if part in ('-1', 'EOF'):
                        reading = False
                        break
                    if part.isdigit() and int(part) > 0:
                        # TSPLIB .opt.tour files are 1-indexed
                        tour.append(int(part) - 1)
                if not reading:
                    break
        return tour if tour else None
    except FileNotFoundError:
        print(f"Error: Optimal tour file not found at {path}")
        return None
    except Exception as e:
        print(f"Error reading optimal tour file {path}: {e}")
        return None


if __name__ == '__main__':
    results_data = []
    tsp_folder = TSP_FOLDER_PATH
    # Define the time limit for the main script's solver runs directly
    SOLVER_TIME_LIMIT_FOR_MAIN = 5.0

    for filename in sorted(os.listdir(tsp_folder)):
        if not filename.lower().endswith('.tsp'):
            continue

        base_name = filename[:-4]
        tsp_file_path = os.path.join(tsp_folder, filename)
        opt_tour_file_path = os.path.join(tsp_folder, base_name + '.opt.tour')

        if not os.path.exists(opt_tour_file_path):
            print(f"Skipping {base_name}: .opt.tour file not found at {opt_tour_file_path}")
            continue

        coords_data, problem_edge_weight_type = parse_tsp_file(tsp_file_path)

        if not coords_data.size:
            print(f"Skipping {base_name}: Error parsing TSP or no coordinates.")
            continue
        if problem_edge_weight_type != 'EUC_2D':
            print(f"Skipping {base_name}: Not EUC_2D (type: {problem_edge_weight_type}).")
            continue

        print(f"\nProcessing {base_name} (EUC_2D)...")
        distance_matrix = compute_distance_matrix(coords_data)
        optimal_tour_nodes = read_opt_tour(opt_tour_file_path)

        if optimal_tour_nodes is None:
            print(f"Skipping {base_name}: Optimal tour could not be read.")
            continue

        optimal_length = total_distance(optimal_tour_nodes, distance_matrix)
        print(f"  Optimal length: {optimal_length:.2f}")

        start_time = time.time()
        heuristic_tour, heuristic_length = simple_tsp_solver(
            coords_data, max_k=4, time_limit=SOLVER_TIME_LIMIT_FOR_MAIN
        )
        elapsed_time = time.time() - start_time
        optimality_gap = float('inf')
        if optimal_length > 0:
            optimality_gap = max(0.0, 100.0 * (heuristic_length - optimal_length) / optimal_length)
        elif heuristic_length == 0:  # Both opt and heu are 0
            optimality_gap = 0.0

        print(f"  Heuristic length: {heuristic_length:.2f}  "
              f"Gap: {optimality_gap:.2f}%  Time: {elapsed_time:.2f}s")

        results_data.append({
            'name': base_name, 'coords': coords_data,
            'opt_tour': optimal_tour_nodes, 'heu_tour': heuristic_tour,
            'opt_len': optimal_length, 'heu_len': heuristic_length,
            'gap': optimality_gap, 'time': elapsed_time
        })

    print("\n" + "-" * 50)
    print(f"{'Instance':<10s} {'OptLen':>8s} {'HeuLen':>8s} "
          f"{'Gap(%)':>8s} {'Time(s)':>8s}")
    print("-" * 50)

    for r_item in results_data:
        opt_len_str = f"{r_item['opt_len']:>8.2f}"
        gap_str = f"{r_item['gap']:>8.2f}"
        print(
            f"{r_item['name']:<10s} {opt_len_str} "
            f"{r_item['heu_len']:>8.2f} {gap_str} "
            f"{r_item['time']:>8.2f}"
        )

    if results_data:
        print("-" * 50)
        num_items = len(results_data)
        total_opt_len = sum(r['opt_len'] for r in results_data)
        total_heu_len = sum(r['heu_len'] for r in results_data)
        avg_gap = sum(r['gap'] for r in results_data if r['gap'] != float('inf')) / \
            len([r for r in results_data if r['gap'] != float('inf')]) \
            if any(r['gap'] != float('inf') for r in results_data) else float('nan')
        avg_time = sum(r['time'] for r in results_data) / num_items

        print(
            f"{'SUMMARY':<10s} {total_opt_len:>8.2f} {total_heu_len:>8.2f} "
            f"{avg_gap:>8.2f} {avg_time:>8.2f}"
        )
    print("Done.")

    if results_data:
        num_total = len(results_data)
        if num_total == 0:
            print("No results to plot.")
        else:
            results_to_plot_list = results_data
            if num_total > MAX_SUBPLOTS:
                print(f"Warning: Plotting first {MAX_SUBPLOTS} of {num_total} results.")
                results_to_plot_list = results_data[:MAX_SUBPLOTS]

            num_to_plot = len(results_to_plot_list)
            if num_to_plot > 0:
                cols = int(math.ceil(math.sqrt(num_to_plot)))
                rows = int(math.ceil(num_to_plot / cols))
                cols = max(1, cols)  # Ensure at least 1 column
                rows = max(1, rows)  # Ensure at least 1 row

                fig, axes_array = plt.subplots(
                    rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False
                )
                flat_axes = axes_array.flatten()
                has_plotted_heuristic, has_plotted_optimal = False, False

                for idx, res_item in enumerate(results_to_plot_list):
                    ax = flat_axes[idx]
                    item_coords = res_item['coords']
                    if 'heu_tour' in res_item and res_item['heu_tour']:
                        h_tour = res_item['heu_tour'] + [res_item['heu_tour'][0]]
                        ax.plot(item_coords[h_tour, 0], item_coords[h_tour, 1],
                                '-', label='Heuristic', zorder=1, color='C0')
                        has_plotted_heuristic = True
                    if 'opt_tour' in res_item and res_item['opt_tour']:
                        o_tour = res_item['opt_tour'] + [res_item['opt_tour'][0]]
                        ax.plot(item_coords[o_tour, 0], item_coords[o_tour, 1],
                                ':', label='Optimal', zorder=2, color='C1')
                        has_plotted_optimal = True

                    title_str_parts = [res_item['name']]
                    if 'gap' in res_item and res_item['gap'] != float('inf'):
                        title_str_parts.append(f"gap={res_item['gap']:.2f}%")
                    ax.set_title(" ".join(title_str_parts))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect('equal', adjustable='box')

                for idx_off in range(num_to_plot, len(flat_axes)):
                    flat_axes[idx_off].set_axis_off()

                legend_handles = []
                if has_plotted_heuristic:
                    legend_handles.append(Line2D([0], [0], color='C0', ls='-', label='Heuristic'))
                if has_plotted_optimal:
                    legend_handles.append(Line2D([0], [0], color='C1', ls=':', label='Optimal'))

                if legend_handles:
                    fig.legend(handles=legend_handles, loc='upper center',
                               ncol=len(legend_handles), bbox_to_anchor=(0.5, 1.0))
                    fig.subplots_adjust(top=(0.95 if num_to_plot > cols else 0.90))

                plt.tight_layout(rect=(0, 0, 1, 0.96 if legend_handles else 1.0))
                plt.show()
