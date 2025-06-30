"""
Input/output utilities for TSPLIB TSP instances.

This module provides functions to read TSP instance files (.tsp) and
optimal tour files (.opt.tour).
"""

from typing import List, Optional
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


def _parse_tsp_header(file_handle) -> dict:
    """Parses the header of a TSPLIB file until NODE_COORD_SECTION."""
    metadata = {}
    for line in file_handle:
        line = line.strip()
        if not line:
            continue

        if 'NODE_COORD_SECTION' in line:
            break  # Header parsing is complete

        parts = line.split(':')
        if len(parts) == 2:
            key, value = parts[0].strip(), parts[1].strip()
            if key == 'DIMENSION':
                metadata['dimension'] = int(value)
            elif key == 'EDGE_WEIGHT_TYPE':
                if value != 'EUC_2D':
                    raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {value}. Only EUC_2D is supported.")
    return metadata


def read_tsp_file(path: str) -> np.ndarray:
    """
    Reads TSPLIB .tsp file (EUC_2D only) and returns coordinates.

    Args:
        path (str): Path to .tsp file.

    Returns:
        np.ndarray: Numpy array (n, 2) of coordinates.

    Raises:
        FileNotFoundError: If the TSP file is not found.
        ValueError: If EDGE_WEIGHT_TYPE is not "EUC_2D" or for parsing errors.
    """
    coords_dict = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            metadata = _parse_tsp_header(f)
            dimension = metadata.get('dimension', -1)

            # At this point, the file handle is at the start of the node section
            for line in f:
                line = line.strip()
                if not line or line == 'EOF':
                    break

                tokens = line.split()
                if len(tokens) < 3:
                    continue  # Skip malformed lines

                node_id, x, y = int(tokens[0]), float(tokens[1]), float(tokens[2])
                coords_dict[node_id] = (x, y)

        if dimension != -1 and len(coords_dict) != dimension:
            print(f"Warning: Dimension mismatch in {path}. "
                  f"Header said {dimension}, but found {len(coords_dict)} nodes.")

        if not coords_dict:
            return np.array([])

        # Sort nodes by their ID to ensure consistent order
        sorted_node_ids = sorted(coords_dict.keys())
        coords_list = [coords_dict[node_id] for node_id in sorted_node_ids]
        return np.array(coords_list, dtype=float)

    except (FileNotFoundError, ValueError) as e:
        # Let specific, expected errors propagate up to be handled by the caller.
        # The docstring correctly indicates that these can be raised.
        raise e
