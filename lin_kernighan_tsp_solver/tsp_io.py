"""
Input/output utilities for TSPLIB TSP instances.

This module provides functions to read TSP instance files (.tsp) and
optimal tour files (.opt.tour).
"""

from typing import List, Optional, Dict
import numpy as np


def read_opt_tour(path: str) -> Optional[List[int]]:
    """
    Reads an optimal tour from a .opt.tour file (TSPLIB format).

    Args:
        path (str): Path to .opt.tour file.

    Returns:
        Optional[List[int]]: List of 0-indexed node IDs, or None on error/malformed.
    """
    tour: List[int] = []
    in_tour_section = False
    found_minus_one_terminator = False
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                tok = line.strip()
                if tok.upper().startswith('TOUR_SECTION'):
                    in_tour_section = True
                    continue
                if not in_tour_section:
                    continue

                for p_token in tok.split():
                    if p_token == '-1':
                        found_minus_one_terminator = True
                        in_tour_section = False  # Stop reading nodes
                        break  # From token loop
                    if p_token == 'EOF':  # EOF before -1
                        in_tour_section = False
                        break  # From token loop
                    try:
                        node_val = int(p_token)
                        tour.append(node_val - 1)  # TSPLIB is 1-indexed
                    except ValueError:
                        # print(f"Warning: Invalid token '{p_token}' in {path}")
                        return None  # Invalid token in tour section
                if not in_tour_section:  # Broke from token loop
                    break  # From line loop
        # Valid tour must have nodes and be terminated by -1
        if not tour or not found_minus_one_terminator:
            return None
    except FileNotFoundError:
        return None
    except IOError:  # Catch other potential file I/O errors
        # print(f"Error reading optimal tour file {path}: {e}")
        return None
    return tour


def read_tsp_file(path: str) -> np.ndarray:
    """
    Reads TSPLIB .tsp file (EUC_2D only) and returns coordinates.

    Args:
        path (str): Path to .tsp file.

    Returns:
        np.ndarray: Numpy array (n, 2) of coordinates.

    Raises:
        FileNotFoundError: If the TSP file is not found.
        ValueError: If EDGE_WEIGHT_TYPE is not "EUC_2D".
        Exception: For other parsing errors.
    """
    coords_dict: Dict[int, List[float]] = {}
    reading_nodes = False
    edge_weight_type = None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_content in f:
                line = line_content.strip()
                if not line:
                    continue  # Skip empty lines

                # Read metadata before NODE_COORD_SECTION
                if ":" in line and not reading_nodes:
                    key, value = [part.strip() for part in line.split(":", 1)]
                    if key.upper() == "EDGE_WEIGHT_TYPE":
                        edge_weight_type = value.upper()

                if line.upper().startswith("NODE_COORD_SECTION"):
                    if edge_weight_type != "EUC_2D":
                        raise ValueError(
                            f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}."
                            f" Only EUC_2D is supported."
                        )
                    reading_nodes = True
                    continue
                if line.upper() == "EOF":  # End of file marker
                    break

                if reading_nodes:  # Parse node coordinates
                    parts = line.split()
                    if len(parts) >= 3:  # node_id x_coord y_coord
                        try:
                            node_id = int(parts[0])
                            x_coord = float(parts[1])
                            y_coord = float(parts[2])
                            coords_dict[node_id] = [x_coord, y_coord]
                        except ValueError:
                            # Silently skip unparsable node lines, or add warning
                            # print(f"Warning: Skipping unparsable node line: '{line}' in {path}")
                            pass
        if not coords_dict:  # No coordinates found
            # print(f"Warning: No coordinates found in {path}")
            return np.array([], dtype=float)

        # Sort by node ID to ensure consistent order before creating array
        sorted_node_ids = sorted(coords_dict.keys())
        coords_list = [coords_dict[node_id] for node_id in sorted_node_ids]
        return np.array(coords_list, dtype=float)

    except (FileNotFoundError, ValueError) as e:
        # Let specific, expected errors propagate up to be handled by the caller.
        # The docstring correctly indicates that these can be raised.
        raise e
