"""
create_tsp_problem.py

Random TSP Problem Generator.

Generates random Traveling Salesperson Problem (TSP) instances with 2D Euclidean
coordinates and saves them in TSPLIB format. The user specifies the number of nodes,
with smart defaults for output filename, coordinate range, and problem name.

Usage:
    python create_tsp_problem.py <n_nodes> [options]

Arguments:
    n_nodes                 Number of nodes (cities) to generate

Options:
    --output, -o FILENAME   Output TSP filename (default: problems/random/random{n_nodes}.tsp)
    --max-coord COORD       Maximum coordinate value (default: 10 * sqrt(n_nodes), min 100)
    --name NAME             Problem name (default: Random{n_nodes})
    --seed SEED             Random seed for reproducible results (default: random)

Examples:
    python create_tsp_problem.py 20
    python create_tsp_problem.py 50 --output custom50.tsp
    python create_tsp_problem.py 100 --max-coord 500 --name Custom100
    python create_tsp_problem.py 30 --seed 42
"""

import argparse
import random
import os


def generate_random_coordinates(n_nodes: int, max_coord: int = 1000) -> dict:
    """Generate random 2D coordinates for TSP nodes.

    Args:
        n_nodes (int): Number of nodes to generate.
        max_coord (int): Maximum value for x and y coordinates.

    Returns:
        dict: Dictionary mapping node indices (1-based) to [x, y] coordinates.
    """
    coords = {}
    for i in range(1, n_nodes + 1):
        x = random.randint(0, max_coord)
        y = random.randint(0, max_coord)
        coords[i] = [x, y]
    return coords


def save_tsp_manually(
    filepath: str,
    name: str,
    dimension: int,
    node_coords: dict,
    comment: str = "Randomly generated TSP instance"
):
    """Save a TSP instance to a file in TSPLIB format.

    Args:
        filepath (str): Output file path.
        name (str): Problem name.
        dimension (int): Number of nodes.
        node_coords (dict): Node coordinates {id: [x, y], ...}, 1-based.
        comment (str): Problem comment.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"NAME: {name}\n")
        f.write(f"COMMENT: {comment}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {dimension}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for node_id in range(1, dimension + 1):
            if node_id in node_coords:
                x, y = node_coords[node_id]
                f.write(f"{node_id} {x} {y}\n")
            else:
                print(f"Warning: Node ID {node_id} not found in node_coords during manual save.")
        f.write("EOF\n")


def main():
    """Parse arguments and generate a random TSP instance."""
    parser = argparse.ArgumentParser(
        description="Generate a random TSP file.",
        epilog="Examples:\n"
               "  python create_tsp_problem.py 20\n"
               "  python create_tsp_problem.py 50 --output custom50.tsp\n"
               "  python create_tsp_problem.py 100 --max-coord 500 --name Custom100",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("n_nodes", type=int, help="Number of nodes (cities) to generate.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output TSP filename (default: problems/random/random{n_nodes}.tsp).")
    parser.add_argument("--max-coord", type=int, default=None,
                        help="Maximum coordinate value (default: 10 * sqrt(n_nodes), min 100).")
    parser.add_argument("--name", type=str, default=None,
                        help="Name of the TSP problem (default: Random{n_nodes}).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible results (default: random).")

    args = parser.parse_args()

    if args.n_nodes <= 0:
        print("Error: Number of nodes must be positive.")
        return

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Smart default for output filename
    if args.output is None:
        args.output = f"random{args.n_nodes}.tsp"

    # Smart default for max-coord based on problem size
    if args.max_coord is None:
        # Use 10 * sqrt(n_nodes) with minimum of 100 for reasonable coordinate range
        import math
        args.max_coord = max(100, int(10 * math.sqrt(args.n_nodes)))

    # Smart default for problem name
    if args.name is None:
        args.name = f"Random{args.n_nodes}"

    # Determine output path - default to problems/random/ directory
    if os.path.isabs(args.output):
        # User provided absolute path
        output_file_path = args.output
    elif os.sep in args.output or "/" in args.output:
        # User provided relative path with directory (handle both / and \ separators)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_file_path = os.path.join(project_root, args.output)
    else:
        # Just filename provided - use problems/random/ as default directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        problems_random_dir = os.path.join(project_root, "problems", "random")

        # Create the directory if it doesn't exist
        os.makedirs(problems_random_dir, exist_ok=True)

        output_file_path = os.path.join(problems_random_dir, args.output)

    print("Generating TSP instance:")
    print(f"  Nodes: {args.n_nodes}")
    print(f"  Coordinate range: 0-{args.max_coord}")

    # Show relative path from project root for cleaner output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    try:
        relative_path = os.path.relpath(output_file_path, project_root)
        print(f"  Output file: {relative_path}")
    except ValueError:
        # If relative path can't be computed (different drives on Windows), show absolute
        print(f"  Output file: {output_file_path}")

    print(f"  Problem name: {args.name}")
    if args.seed is not None:
        print(f"  Random seed: {args.seed}")

    coordinates = generate_random_coordinates(args.n_nodes, args.max_coord)

    try:
        problem_comment = f"Randomly generated TSP instance with {args.n_nodes} nodes."
        if args.seed is not None:
            problem_comment += f" Random seed: {args.seed}."

        save_tsp_manually(
            filepath=output_file_path,
            name=args.name,
            dimension=args.n_nodes,
            node_coords=coordinates,
            comment=problem_comment
        )
        print(f"Successfully created TSP file: {output_file_path}")
    except Exception as e:
        print(f"Error creating TSP file: {e}")


if __name__ == "__main__":
    main()
