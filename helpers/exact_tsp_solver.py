"""
exact_tsp_solver.py

Brute-force exact solver for the Traveling Salesperson Problem (TSP).

This script generates a random TSP instance, solves it exactly using a brute-force
algorithm (by checking all permutations of cities, fixing the starting city), and
saves the instance and optimal tour in TSPLIB format. The solution and its length
are printed and optionally plotted. Intended for small instances due to factorial time complexity.

Usage:
    python exact_tsp_solver.py [options]

Options:
    --nodes, -n NODES       Number of cities to generate (default: 10)
    --seed SEED             Random seed for reproducible results (default: 42)
    --output-dir DIR        Output directory for .tsp files (default: problems/random/)
    --name NAME             Base name for files (default: rand{nodes})
    --plot                  Show plot of the optimal tour (default: False)
    --max-coord COORD       Maximum coordinate value (default: 1000)

Note: .tsp files are saved to problems/random/, .opt.tour files to solutions/exact/

Examples:
    python exact_tsp_solver.py
    python exact_tsp_solver.py --nodes 8 --plot
    python exact_tsp_solver.py --nodes 12 --seed 123 --name custom
"""

import argparse
import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np


def compute_distance_matrix(coords):
    """Compute the pairwise Euclidean distance matrix for city coordinates.

    Args:
        coords (np.ndarray): Array of shape (n, 2) with city coordinates.

    Returns:
        np.ndarray: Distance matrix of shape (n, n).
    """
    diff = coords[:, np.newaxis] - coords[np.newaxis, :]
    return np.linalg.norm(diff, axis=2)


def total_distance(tour, D, best_so_far=float('inf')):
    """Calculate the total length of a tour, with optional early abandoning.

    Args:
        tour (list[int]): Tour as a list of city indices.
        D (np.ndarray): Distance matrix.
        best_so_far (float, optional): If partial sum exceeds this, abandon. Defaults to inf.

    Returns:
        float: Total tour length, or inf if abandoned.
    """
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += D[tour[i], tour[(i + 1) % n]]
        if total > best_so_far:
            return float('inf')
    return total


def tsp_brute_force(coords):
    """Solve the TSP exactly by brute-force (fixing city 0).

    Args:
        coords (np.ndarray): Array of shape (n, 2) with city coordinates.

    Returns:
        tuple: (best_tour, best_length)
            best_tour (list[int]): Optimal tour as list of indices.
            best_length (float): Length of the optimal tour.
    """
    n = len(coords)
    D = compute_distance_matrix(coords)
    cities = list(range(n))
    best_tour = []
    best_length = float('inf')
    for perm in itertools.permutations(cities[1:]):
        tour = [0] + list(perm)
        length = total_distance(tour, D, best_length)
        if length < best_length:
            best_tour, best_length = tour, length
    return best_tour, best_length


def plot_tour(coords, tour, length=None):
    """Plot the TSP tour using Matplotlib.

    Args:
        coords (np.ndarray): Array of city coordinates.
        tour (list[int]): Tour as a list of city indices.
        length (float, optional): Tour length for title. Defaults to None.
    """
    path = tour + [tour[0]]
    coords = np.array(coords)
    x, y = coords[path, 0], coords[path, 1]
    plt.figure(figsize=(4, 4))
    plt.plot(x, y, '-')
    if length is not None:
        plt.title(f"TSP tour (length: {length:.2f})")
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def save_tour(tour, filepath, name):
    """Save a tour in TSPLIB .tour format.

    Args:
        tour (list[int]): Tour as a list of city indices.
        filepath (str): Output file path.
        name (str): Name for the tour.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TOUR\n")
        f.write(f"DIMENSION: {len(tour)}\n")
        f.write("TOUR_SECTION\n")
        for node in tour:
            f.write(f"{node + 1}\n")
        f.write("-1\n")
        f.write("EOF\n")


def main():
    """Parse arguments and solve TSP instance."""
    parser = argparse.ArgumentParser(
        description="Generate and solve a random TSP instance exactly using brute-force.",
        epilog="Examples:\n"
               "  python exact_tsp_solver.py\n"
               "  python exact_tsp_solver.py --nodes 8 --plot\n"
               "  python exact_tsp_solver.py --nodes 12 --seed 123 --name custom",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--nodes", "-n", type=int, default=10,
                        help="Number of cities to generate (default: 10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible results (default: 42).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for .tsp files (default: problems/random/).")
    parser.add_argument("--name", type=str, default=None,
                        help="Base name for files (default: rand{nodes}).")
    parser.add_argument("--plot", action="store_true",
                        help="Show plot of the optimal tour (default: False).")
    parser.add_argument("--max-coord", type=int, default=1000,
                        help="Maximum coordinate value (default: 1000).")

    args = parser.parse_args()

    # Validate number of nodes
    if args.nodes <= 0:
        print("Error: Number of nodes must be positive.")
        return

    if args.nodes > 12:
        print(f"Warning: {args.nodes} nodes will take a very long time due to factorial complexity!")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Set random seed
    np.random.seed(args.seed)

    # Smart defaults
    if args.name is None:
        args.name = f"rand{args.nodes}"

    if args.output_dir is None:
        # Default to problems/random/ directory for TSP files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.output_dir = os.path.join(project_root, "problems", "random")
        solutions_dir = os.path.join(project_root, "solutions", "exact")
    else:
        # Custom output directory provided
        if not os.path.isabs(args.output_dir):
            # Make relative paths relative to project root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            args.output_dir = os.path.join(project_root, args.output_dir)

        # Try to map problems directory to corresponding solutions directory
        if "problems" in args.output_dir:
            solutions_dir = args.output_dir.replace("problems", "solutions")
        else:
            # Fallback to solutions/exact/ for non-standard paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            solutions_dir = os.path.join(project_root, "solutions", "exact")

    # Create both directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(solutions_dir, exist_ok=True)

    # Generate random coordinates
    coords = np.random.randint(1, args.max_coord + 1, size=(args.nodes, 2))

    # File paths
    tsp_fname = f"{args.name}.tsp"
    tour_fname = f"{args.name}.opt.tour"
    tsp_path = os.path.join(args.output_dir, tsp_fname)
    tour_path = os.path.join(solutions_dir, tour_fname)

    print("Generating and solving TSP instance:")
    print(f"  Nodes: {args.nodes}")
    print(f"  Random seed: {args.seed}")
    print(f"  Coordinate range: 1-{args.max_coord}")
    print(f"  Base name: {args.name}")

    # Show relative paths from project root for cleaner output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    try:
        relative_problems_dir = os.path.relpath(args.output_dir, project_root)
        relative_solutions_dir = os.path.relpath(solutions_dir, project_root)
        print(f"  Problems directory: {relative_problems_dir}/")
        print(f"  Solutions directory: {relative_solutions_dir}/")
    except ValueError:
        print(f"  Problems directory: {args.output_dir}/")
        print(f"  Solutions directory: {solutions_dir}/")

    # Save TSP instance
    with open(tsp_path, 'w', encoding='utf-8') as f:
        f.write(f"NAME: {args.name}\n")
        f.write("TYPE: TSP\n")
        f.write(f"COMMENT: Random instance with {args.nodes} nodes, seed {args.seed}\n")
        f.write(f"DIMENSION: {args.nodes}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, start=1):
            f.write(f"{i} {x} {y}\n")
        f.write("EOF\n")

    try:
        relative_tsp_path = os.path.relpath(tsp_path, project_root)
        print(f"Saved TSP instance: {relative_tsp_path}")
    except ValueError:
        print(f"Saved TSP instance: {tsp_path}")

    # Solve TSP
    print(f"\nSolving TSP with {args.nodes} cities using brute-force...")
    start = time.time()
    tour, length = tsp_brute_force(coords)
    duration = time.time() - start

    print(f"Optimal tour: {tour}")
    print(f"Optimal length: {length:.2f}")
    print(f"Solve time: {duration:.2f} seconds")

    # Save optimal tour
    save_tour(tour, tour_path, args.name)
    try:
        relative_tour_path = os.path.relpath(tour_path, project_root)
        print(f"Saved optimal tour: {relative_tour_path}")
    except ValueError:
        print(f"Saved optimal tour: {tour_path}")

    # Plot if requested
    if args.plot:
        print("\nDisplaying plot...")
        plot_tour(coords, tour, length)


if __name__ == "__main__":
    main()
