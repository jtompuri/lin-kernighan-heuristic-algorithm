"""
This is a simplified TSP solver that implements only the core features:
    - Basic k-opt moves
    - Recursive improvement (step)

Missing features of Lin-Kernighan heuristic algorithm:
    - Recursive improvement is only partially implemented
    - Flip tracking & rollback
    - Alternate first step (alternate_step)
    - Neighborhood ordering (lk-ordering)
    - Breadth & depth parameters
    - Kick / double-bridge restarts
    - Tour object abstraction (next, flip)
    - Delta-based tentative improvements
    - Chained Lin-Kernighan algorithm
"""
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # ADDED

# --- Constants ---
TSP_FOLDER_PATH = '../verifications/tsplib95'  # Path to the folder containing TSPLIB .tsp files
TIME_LIMIT = 1.0  # Default time limit for the solver in seconds
MAX_SUBPLOTS = 25  # ADDED: Maximum number of subplots in the tour visualization


# Compute pairwise Euclidean distances for all city coordinates.
# Returns a distance matrix D where D[i, j] = distance between city i and j.
def compute_distance_matrix(coords):
    diff = coords[:, None] - coords[None, :]        # shape (n, n, 2)
    return np.linalg.norm(diff, axis=2)             # shape (n, n)


# Compute the total length of a tour given a distance matrix.
def total_distance(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[(i + 1) % len(tour)]]  # wrap around
               for i in range(len(tour)))


# Compute the gain from swapping two edges: (a-b) and (c-d) -> (a-c) and (b-d).
def compute_gain(dist_matrix, a, b, c, d):
    return dist_matrix[a][b] + dist_matrix[c][d] - dist_matrix[a][c] - dist_matrix[b][d]


# Reverse a segment between two indices [i+1, j] to simulate a 2-opt move.
def two_opt_swap(tour, i, j):
    return tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]


# Recursive search that attempts to improve the tour incrementally.
def recursive_k_opt(
    tour,
    dist_matrix,
    start_index,
    visited,
    gain_sum,
    max_k,
    deadline
):
    # Abort recursion if depth exceeded or time limit reached
    if max_k == 0 or time.time() >= deadline:
        return False

    n = len(tour)
    t1 = tour[start_index]
    t2 = tour[(start_index + 1) % n]

    for j in range(n):
        t3 = tour[j]
        t4 = tour[(j + 1) % n]

        # Skip invalid pairs (same node, adjacent edge, or already visited)
        if t3 == t1 or t4 == t2 or j == start_index or j in visited:
            continue

        gain = compute_gain(dist_matrix, t1, t2, t3, t4)
        if gain + gain_sum <= 0:
            continue  # Only consider moves that improve or might improve total cost

        new_tour = two_opt_swap(tour, start_index, j)
        new_length = total_distance(new_tour, dist_matrix)
        old_length = total_distance(tour, dist_matrix)

        if new_length < old_length:
            tour[:] = new_tour  # Commit the improved tour in-place
            return True

        # Try extending the move recursively to a deeper k-opt move
        if recursive_k_opt(new_tour, dist_matrix, start_index,
                           visited | {j}, gain + gain_sum, max_k - 1, deadline):
            tour[:] = new_tour
            return True

        if time.time() >= deadline:      # Time guard in deep recursion
            return False

    return False  # No improving move found at this level


# Simplified TSP solver with a wall-clock time limit.
def simple_tsp_solver(
    coords,
    max_k=4,
    time_limit=10.0          # Seconds allowed for search
):
    """
    Runs the simplified k-opt search until no improvement
    OR the wall-clock time limit is reached.
    Returns the best tour found so far.
    """
    deadline = time.time() + time_limit

    n = len(coords)
    coords = np.asarray(coords, float)
    dist_matrix = compute_distance_matrix(coords)

    tour = list(range(n))            # Initial tour: sequential node order
    best = tour.copy()
    best_length = total_distance(tour, dist_matrix)
    improved = True

    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break                       # Stop outer loop if time is up
            visited = set()
            if recursive_k_opt(tour, dist_matrix, i, visited, 0.0, max_k, deadline):
                new_length = total_distance(tour, dist_matrix)
                if new_length < best_length:
                    best = tour.copy()
                    best_length = new_length
                    improved = True
                    break  # Restart outer loop to allow global improvement

    return best, best_length


# Visualize the tour in 2D using matplotlib.
def plot_tour(coords, tour, length=None):
    coords = np.asarray(coords)
    tour_cycle = tour + [tour[0]]             # Close the tour into a cycle
    x, y = coords[tour_cycle, 0], coords[tour_cycle, 1]

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'o-', markersize=2)
    plt.title(f"TSP Path (Length: {length:.2f})" if length is not None else "TSP Path")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# NEW custom TSP parsing function
def parse_tsp_file(path: str) -> tuple[np.ndarray, str | None]:
    """
    Reads a TSPLIB formatted TSP file.
    Returns a numpy array of coordinates (sorted by node ID, effectively 0-indexed
    if nodes are 1...N) and the edge weight type.
    """
    coords_dict: dict[int, list[float]] = {}
    edge_weight_type: str | None = None
    reading_nodes = False

    try:
        with open(path, 'r') as f:
            for line_content in f:
                line = line_content.strip()
                if not line:
                    continue

                # Parse headers like EDGE_WEIGHT_TYPE before NODE_COORD_SECTION
                if ":" in line and not reading_nodes:
                    key, value = [part.strip() for part in line.split(":", 1)]
                    if key.upper() == "EDGE_WEIGHT_TYPE":
                        edge_weight_type = value.upper()
                    # Add other header parsing here if needed (e.g., DIMENSION)

                if line.upper().startswith("NODE_COORD_SECTION"):
                    reading_nodes = True
                    continue

                if line.upper() == "EOF":
                    break

                if reading_nodes:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            node_id = int(parts[0])  # TSPLIB nodes are usually 1-based
                            x = float(parts[1])
                            y = float(parts[2])
                            coords_dict[node_id] = [x, y]
                        except ValueError:
                            print(f"Warning: Could not parse node coord line: '{line_content.strip()}' in {path}")

        if not coords_dict:
            # print(f"Warning: No coordinates found in {path}")  # Can be noisy if file is invalid
            return np.array([], dtype=float), edge_weight_type

        # Sort by node ID to ensure consistent order for the output array.
        # This effectively maps 1-based node IDs to 0-based array indices
        # if the original node IDs are 1, 2, ..., N.
        sorted_node_ids = sorted(coords_dict.keys())

        # Optional: Add a check if node IDs are as expected (e.g., 1 to N)
        if not sorted_node_ids or sorted_node_ids[0] != 1 or sorted_node_ids[-1] != len(sorted_node_ids):
            print(f"Warning: Node IDs in {path} may not be a contiguous 1-N sequence. Coordinates will be ordered by sorted node ID.")

        coords_list = [coords_dict[node_id] for node_id in sorted_node_ids]
        return np.array(coords_list, dtype=float), edge_weight_type

    except FileNotFoundError:
        print(f"Error: TSP file not found at {path}")
        return np.array([], dtype=float), None  # Return empty array and None type
    except Exception as e:
        print(f"Error reading TSP file {path}: {e}")
        return np.array([], dtype=float), None  # Return empty array and None type


def read_opt_tour(path):
    tour, reading = [], False
    with open(path) as f:
        for line in f:
            tok = line.strip()
            if tok.upper().startswith('TOUR_SECTION'):
                reading = True
                continue
            if not reading:
                continue
            for p in tok.split():
                if p in ('-1', 'EOF'):
                    reading = False
                    break
                if p.isdigit() and int(p) > 0:
                    tour.append(int(p) - 1)
            if not reading:
                break
    return tour


# Batch run and plot
if __name__ == '__main__':

    # Change to your TSPLIB path
    folder = TSP_FOLDER_PATH  # MODIFIED: Use constant
    results = []

    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith('.tsp'):
            continue
        base = fn[:-4]
        tsp_path = os.path.join(folder, fn)
        opt_path = os.path.join(folder, base + '.opt.tour')
        if not os.path.exists(opt_path):
            # If opt_path doesn't exist, we might still want to process the TSP file
            # For now, the original logic was to skip if opt_path doesn't exist.
            # Let's keep that, but it could be changed.
            print(f"Skipping {base}: .opt.tour file not found at {opt_path}")
            continue

        # problem = tsplib95.load(tsp_path)  # REMOVED
        # if getattr(problem, 'edge_weight_type', '').upper() != 'EUC_2D':  # OLD CHECK
        #     continue

        # print(f"\nProcessing {base} (EUC_2D)...")  # MOVED
        # coords = read_tsp(tsp_path)  # REMOVED

        coords, edge_weight_type = parse_tsp_file(tsp_path)  # NEW

        if not coords.size:  # Check if coordinates were loaded successfully
            print(f"Skipping {base}: Error parsing TSP file or no coordinates found.")
            continue

        if edge_weight_type != 'EUC_2D':
            print(f"Skipping {base}: Not an EUC_2D problem (type: {edge_weight_type}).")
            continue

        print(f"\nProcessing {base} (EUC_2D)...")  # MOVED HERE

        D = compute_distance_matrix(coords)
        opt_tour = read_opt_tour(opt_path)

        if opt_tour is None:  # If read_opt_tour failed or returned empty
            print(f"Skipping {base}: Optimal tour could not be read from {opt_path}.")
            continue

        opt_len = total_distance(opt_tour, D)
        print(f"  Optimal length: {opt_len:.2f}")

        start = time.time()
        heu_tour, heu_len = simple_tsp_solver(coords, max_k=4, time_limit=TIME_LIMIT)  # MODIFIED: Use constant
        elapsed = time.time() - start
        gap = max(0.0, 100.0 * (heu_len - opt_len) / opt_len) if opt_len > 0 else float('inf') if heu_len > 0 else 0.0
        print(f"  Heuristic length: {heu_len:.2f}  Gap: {gap:.2f}%  Time: {elapsed:.2f}s")

        results.append({'name': base, 'coords': coords,
                        'opt_tour': opt_tour, 'heu_tour': heu_tour,
                        'opt_len': opt_len, 'heu_len': heu_len,
                        'gap': gap, 'time': elapsed})

    # MODIFIED Summary Printing Block
    print("\n" + "-" * 50)  # Separator
    header = "Instance   OptLen   HeuLen   Gap(%)   Time(s)"
    print(header)
    print("\n" + "-" * 50)  # Separator

    for r_item in results:
        opt_len_str = f"{r_item['opt_len']:>8.2f}" if r_item['opt_len'] is not None else "   N/A  "
        gap_str = f"{r_item['gap']:>8.2f}" if r_item['gap'] is not None else "   N/A  "
        # heu_len and time are assumed to always exist
        print(
            f"{r_item['name']:<10s} {opt_len_str} "
            f"{r_item['heu_len']:>8.2f} {gap_str} "
            f"{r_item['time']:>8.2f}"
        )

    if results:
        print("\n" + "-" * 50)  # Separator
        num_total_items = len(results)

        valid_opt_lens = [r['opt_len']
                          for r in results if r['opt_len'] is not None]
        valid_gaps = [r['gap'] for r in results if r['gap'] is not None]

        total_opt_len_sum = sum(valid_opt_lens) if valid_opt_lens else None
        # heu_len should always exist and be a float
        total_heu_len_sum = sum(r_item['heu_len'] for r_item in results)

        avg_gap_val = sum(valid_gaps) / len(valid_gaps) if valid_gaps else None
        # time should always exist and be a float
        avg_time_val = sum(r_item['time'] for r_item in results) / \
            num_total_items if num_total_items > 0 else 0.0

        total_opt_len_str = f"{total_opt_len_sum:>8.2f}" if total_opt_len_sum is not None else "   N/A  "
        avg_gap_str = f"{avg_gap_val:>8.2f}" if avg_gap_val is not None else "   N/A  "

        print(
            f"{'SUMMARY':<10s} {total_opt_len_str} {total_heu_len_sum:>8.2f} "
            f"{avg_gap_str} {avg_time_val:>8.2f}"
        )
    print("Done.")

    # Plot tours
    if results:
        num_results_total = len(results)
        if num_results_total == 0:
            print("No results to plot.")
        else:
            results_to_plot = results
            if num_results_total > MAX_SUBPLOTS:
                print(f"Warning: Plotting only the first {MAX_SUBPLOTS} of {num_results_total} results due to limit.")
                results_to_plot = results[:MAX_SUBPLOTS]

            num_plot_items = len(results_to_plot)

            if num_plot_items > 0:
                cols = int(math.ceil(math.sqrt(num_plot_items)))
                rows = int(math.ceil(num_plot_items / cols))

                if num_plot_items == 1:  # Ensure cols and rows are at least 1 for a single plot
                    cols = 1
                    rows = 1

                fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
                axes_list = axes.flatten()

                plotted_heuristic = False
                plotted_optimal = False

                for i, r_item in enumerate(results_to_plot):
                    ax = axes_list[i]
                    coords = r_item['coords']

                    if 'heu_tour' in r_item and r_item['heu_tour']:
                        heu_plot_tour = r_item['heu_tour'] + [r_item['heu_tour'][0]]
                        ax.plot(coords[heu_plot_tour, 0], coords[heu_plot_tour, 1], '-', label='Heuristic', zorder=1, color='C0')
                        plotted_heuristic = True

                    if 'opt_tour' in r_item and r_item['opt_tour'] is not None:
                        opt_plot_tour = r_item['opt_tour'] + [r_item['opt_tour'][0]]
                        ax.plot(coords[opt_plot_tour, 0], coords[opt_plot_tour, 1], ':', label='Optimal', zorder=2, color='C1')
                        plotted_optimal = True

                    title_parts = [r_item['name']]
                    if 'gap' in r_item and r_item['gap'] is not None:
                        title_parts.append(f"gap={r_item['gap']:.2f}%")
                    ax.set_title(" ".join(title_parts))

                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect('equal', adjustable='box')
                    # ax.grid(False)  # Grid is off by default if not specified, or ensure it's False

                for i in range(num_plot_items, len(axes_list)):
                    axes_list[i].set_axis_off()

                legend_elements = []
                if plotted_heuristic:
                    legend_elements.append(Line2D([0], [0], color='C0', linestyle='-', label='Heuristic'))
                if plotted_optimal:
                    legend_elements.append(Line2D([0], [0], color='C1', linestyle=':', label='Optimal'))

                if legend_elements:
                    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.0))
                    # Adjust layout to make space for the legend
                    fig.subplots_adjust(top=(0.95 if num_plot_items > cols else 0.90))

                plt.tight_layout(rect=(0, 0, 1, 0.96 if legend_elements else 1.0))  # MODIFIED: Changed list to tuple
                plt.show()
