"""
TSP Problem Generator.

This script generates random Traveling Salesperson Problem (TSP) instances
and saves them in the TSPLIB format. It creates a specified number of nodes
with 2D Euclidean coordinates (EUC_2D) within a defined range.
The generated .tsp files can then be used as input for TSP solvers.

The output .tsp file will be saved in the same directory where this script is located.

Usage:
  python create_tsp_problem.py <n_nodes> <output_filename> [options]

Arguments:
  n_nodes             The number of nodes (cities) to generate for the TSP instance.
                      Must be a positive integer.
  output_filename     The name for the output .tsp file (e.g., 'random_problem.tsp').
                      The file will be saved in the script's directory.

Optional Arguments:
  --max_coord MAX_COORD
                        The maximum integer value for the x and y coordinates
                        of the generated nodes. Coordinates will be in the range
                        [0, MAX_COORD]. (Default: 1000)
  --name PROBLEM_NAME
                        The internal name for the TSP problem instance to be written
                        into the .tsp file. If not provided, the name will be
                        derived from the output_filename (without the extension).
  -h, --help            Show this help message and exit.

Example:
  To generate a TSP instance with 20 nodes and save it as 'my_tsp20.tsp':
    python create_tsp_problem.py 20 my_tsp20.tsp

  To generate a TSP instance with 50 nodes, coordinates up to 500,
  and an internal problem name 'Random50':
    python create_tsp_problem.py 50 random50.tsp --max_coord 500 --name Random50
"""
import argparse
import random
import tsplib95
import os


def generate_random_coordinates(n_nodes: int, max_coord: int = 1000) -> dict:
    """
    Generates n_nodes random 2D coordinates.

    Args:
        n_nodes (int): The number of nodes (coordinates) to generate.
        max_coord (int): The maximum value for x and y coordinates.

    Returns:
        dict: A dictionary where keys are node indices (1 to n_nodes)
              and values are [x, y] coordinate lists.
    """
    coords = {}
    for i in range(1, n_nodes + 1):
        x = random.randint(0, max_coord)
        y = random.randint(0, max_coord)
        coords[i] = [x, y]
    return coords


def create_tsp_problem_object(name: str, n_nodes: int, coords: dict, comment: str = "Randomly generated TSP instance") -> tsplib95.models.Problem:
    """
    Creates a tsplib95.models.Problem object from given coordinates.
    This object can then be used for manual saving or other operations.
    """
    problem = tsplib95.models.Problem(
        name=name,
        comment=comment,
        type="TSP",
        dimension=n_nodes,
        edge_weight_type="EUC_2D",
        node_coords=coords,
        display_data_type='COORD_DISPLAY'
    )
    return problem


def save_tsp_manually(filepath: str, name: str, dimension: int, node_coords: dict, comment: str = "Randomly generated TSP instance"):
    """
    Saves a TSP instance to a file manually in TSPLIB format.

    Args:
        filepath (str): The path to save the .tsp file.
        name (str): The NAME field for the TSP problem.
        dimension (int): The DIMENSION field (number of nodes).
        node_coords (dict): Dictionary of node coordinates {id: [x, y], ...}.
                            Node IDs are expected to be 1-indexed.
        comment (str): The COMMENT field for the TSP problem.
    """
    with open(filepath, 'w') as f:
        f.write(f"NAME: {name}\n")
        f.write(f"COMMENT: {comment}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {dimension}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for node_id in range(1, dimension + 1):  # Ensure correct iteration if node_coords keys might not be perfectly sequential
            if node_id in node_coords:
                x, y = node_coords[node_id]
                f.write(f"{node_id} {x} {y}\n")
            else:
                # This case should ideally not happen if node_coords is well-formed
                print(f"Warning: Node ID {node_id} not found in node_coords during manual save.")
        f.write("EOF\n")


def main():
    parser = argparse.ArgumentParser(description="Generate a random TSP file.")
    parser.add_argument("n_nodes", type=int, help="Number of nodes (cities) to generate.")
    parser.add_argument("output_filename", type=str, help="Name of the output TSP file (e.g., random_tsp.tsp). This will be saved in the script's directory.")
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
