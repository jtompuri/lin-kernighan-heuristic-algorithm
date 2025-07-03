"""
create_tsp_problem.py

Random TSP Problem Generator.

Generates random Traveling Salesperson Problem (TSP) instances with 2D Euclidean
coordinates and saves them in TSPLIB format. The user specifies the number of nodes,
output filename, and optional parameters for coordinate range and problem name.

Usage:
    python create_tsp_problem.py <n_nodes> <output_filename> [--max_coord MAX_COORD] [--name PROBLEM_NAME]

Example:
    python create_tsp_problem.py 20 my_tsp20.tsp
    python create_tsp_problem.py 50 random50.tsp --max_coord 500 --name Random50
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
    with open(filepath, 'w') as f:
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
    parser = argparse.ArgumentParser(description="Generate a random TSP file.")
    parser.add_argument("n_nodes", type=int, help="Number of nodes (cities) to generate.")
    parser.add_argument("output_filename", type=str, help="Name of the output TSP file (e.g., random_tsp.tsp).")
    parser.add_argument("--max_coord", type=int, default=1000, help="Maximum coordinate value (default: 1000).")
    parser.add_argument("--name", type=str, default=None, help="Name of the TSP problem (default: derived from filename).")

    args = parser.parse_args()

    if args.n_nodes <= 0:
        print("Error: Number of nodes must be positive.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(script_dir, os.path.basename(args.output_filename))

    problem_name_arg = args.name
    if not problem_name_arg:
        problem_name_arg = os.path.splitext(os.path.basename(args.output_filename))[0]

    print(f"Generating {args.n_nodes} random coordinates (0-{args.max_coord})...")
    coordinates = generate_random_coordinates(args.n_nodes, args.max_coord)

    try:
        print(f"Manually saving TSP file to '{output_file_path}'...")
        problem_comment = f"Randomly generated TSP instance with {args.n_nodes} nodes."
        save_tsp_manually(
            filepath=output_file_path,
            name=problem_name_arg,
            dimension=args.n_nodes,
            node_coords=coordinates,
            comment=problem_comment
        )
        print(f"Successfully saved {output_file_path} manually.")
    except Exception as e:
        print(f"Error during manual save of TSP file: {e}")


if __name__ == "__main__":
    main()
