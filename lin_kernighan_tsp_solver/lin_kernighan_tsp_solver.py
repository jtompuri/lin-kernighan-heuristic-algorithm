"""
Lin-Kernighan Heuristic for the Traveling Salesperson Problem (TSP).

This script implements the Lin-Kernighan (LK) heuristic, a powerful local
search algorithm for finding high-quality approximate solutions to the TSP.
The implementation is based on the descriptions and algorithms presented in
"The Traveling Salesman Problem: A Computational Study" by Applegate, Bixby,
Chvátal & Cook, and "An Effective Heuristic Algorithm for the
Traveling-Salesman Problem" by Lin & Kernighan.

The script processes TSP instances from the TSPLIB format. It computes
heuristic solutions using a chained version of the LK algorithm. If a
corresponding optimal tour file (e.g., problem_name.opt.tour) is found,
the script compares the heuristic solution against the known optimal solution
and calculates the percentage gap. If no optimal tour file is available, the
instance is still processed, but no gap calculation is performed for it.
The script displays a summary table and plots of the tours.

Usage:
  1. Ensure all dependencies are installed:
     pip install numpy matplotlib scipy

  2. Place your TSPLIB .tsp files in a designated folder.
     Optionally, place corresponding .opt.tour files (if available) in the
     same folder.

  3. Update the `TSP_FOLDER_PATH` constant at the top of this script
     (in the "--- Constants ---" section) to point to your TSPLIB folder.

  4. Run the script from the command line:
     python lin_kernighan_tsp_solver.py

The script will then process each EUC_2D TSP instance found. It prints
progress and results to the console. For instances with an optimal tour, the
gap is shown. For instances without an optimal tour, nothing is displayed for
optimal length and gap. Finally, a plot of all processed tours is displayed
(showing both optimal and heuristic tours if the optimal is available,
otherwise just the heuristic tour). Configuration parameters for the LK
algorithm can be adjusted in the `LK_CONFIG` dictionary within this script.
"""
import time
import math
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.lines import Line2D

# --- Constants ---
# Path to the folder containing TSPLIB .tsp files and optional .opt.tour files
TSP_FOLDER_PATH = (
    Path(__file__).resolve().parent.parent / "verifications" / "tsplib95"
)

# Tolerance for floating point comparisons
FLOAT_COMPARISON_TOLERANCE = 1e-12

# Maximum number of subplots in the tour visualization
MAX_SUBPLOTS_IN_PLOT = 25

# --- Configuration Parameters ---
# This dictionary holds parameters that control the behavior of the
# Lin-Kernighan heuristic, such as search depth, breadth at various stages,
# and time limits.
LK_CONFIG = {
    "MAX_LEVEL": 12,  # Max recursion depth for k-opt moves in step()
    "BREADTH": [5, 5] + [1] * 20,  # Search breadth at each level in step()
    "BREADTH_A": 5,  # Search breadth for t3 in alternate_step()
    "BREADTH_B": 5,  # Search breadth for t5 in alternate_step()
    "BREADTH_D": 1,  # Search breadth for t7 in alternate_step()
    "TIME_LIMIT": 1.0,  # Default time limit for chained_lin_kernighan
}


class Tour:
    """
    Abstract tour data structure for the Lin-Kernighan heuristic.

    Maintains a permutation of the vertices and supports efficient flip
    operations, as well as access to next/previous neighbors. This structure
    follows the requirements set in Applegate et al., Section 15.2.

    Attributes:
        n (int): Number of vertices in the tour.
        order (np.ndarray): Current permutation representing the tour.
        pos (np.ndarray): Inverse mapping: pos[v] gives index of vertex v
                          in order[].
        cost (float or None): Cost of the tour under current permutation,
                              if initialized.
    """

    def __init__(self, order: Iterable[int],
                 D: Optional[np.ndarray] = None) -> None:
        """
        Initializes the tour data structure from a given vertex ordering.

        Args:
            order: Sequence of vertices defining the tour.
                   If it's an iterator, it will be consumed.
            D: Distance/cost matrix to initialize tour cost.
               If provided, init_cost() will be called.
        """
        order_list = list(order)
        self.n: int = len(order_list)

        if self.n == 0:
            self.order: np.ndarray = np.array([], dtype=np.int32)
            self.pos: np.ndarray = np.array([], dtype=np.int32)
            self.cost: Optional[float] = 0.0
        else:
            self.order: np.ndarray = np.array(order_list, dtype=np.int32)
            # Ensure self.pos is large enough for all actual node labels.
            max_node_label = np.max(self.order)
            self.pos: np.ndarray = np.empty(max_node_label + 1,
                                            dtype=np.int32)
            for i, v_node in enumerate(self.order):
                self.pos[v_node] = i

            self.cost: Optional[float] = None
            if D is not None:
                self.init_cost(D)

    def init_cost(self, D: np.ndarray) -> None:
        """
        Computes and stores the total tour cost using the provided distance
        matrix. This method is typically called during tour initialization if a
        distance matrix is available. It iterates through the tour segments,
        sums their costs, and updates `self.cost`.

        Args:
            D: The distance (or cost) matrix where D[i, j] is the
               cost of the edge between vertex i and vertex j.
        """
        if self.n == 0:
            self.cost = 0.0
            return

        current_total_cost = 0.0
        for i in range(self.n):
            node1 = self.order[i]
            node2 = self.order[(i + 1) % self.n]
            current_total_cost += float(D[node1, node2])

        self.cost = current_total_cost

    def next(self, v: int) -> int:
        """
        Returns the vertex immediately after v in the current tour.

        Args:
            v: Vertex label.

        Returns:
            Next vertex after v.

        Raises:
            IndexError: If the tour is empty (self.n == 0) or if v is not
                        a valid node label for the current tour's pos array.
        """
        if self.n == 0:
            raise IndexError("Cannot get next node from an empty tour.")
        # self.pos[v] gives the index of vertex v in the self.order array.
        # Adding 1 and taking modulo self.n gives the index of the next vertex.
        # self.order[...] then retrieves the label of that next vertex.
        return int(self.order[(self.pos[v] + 1) % self.n])

    def prev(self, v: int) -> int:
        """
        Returns the vertex immediately before v in the current tour.

        Args:
            v: Vertex label.

        Returns:
            Previous vertex before v.

        Raises:
            IndexError: If the tour is empty (self.n == 0) or if v is not
                        a valid node label for the current tour's pos array.
        """
        if self.n == 0:
            raise IndexError("Cannot get previous node from an empty tour.")
        # self.pos[v] gives the index of vertex v in the self.order array.
        # Subtr. 1 and modulo self.n handles wrap-around for previous vertex.
        # self.order[...] then retrieves the label of that previous vertex.
        return int(self.order[(self.pos[v] - 1 + self.n) % self.n])

    def sequence(self, node_a: int, node_b: int, node_c: int) -> bool:
        """
        Determines if vertex `node_b` is on the path from `node_a` to `node_c`
        (inclusive of `node_a` and `node_c`) when traversing the tour in its
        current orientation.

        Args:
            node_a: Start vertex of the segment.
            node_b: Vertex to check.
            node_c: End vertex of the segment.

        Returns:
            True if `node_b` is on the segment from `node_a` to `node_c`
            (inclusive), otherwise False. Returns False for an empty tour.
        """
        if self.n == 0:
            return False

        # Get the positions (indices) of the nodes in the self.order array
        idx_a = self.pos[node_a]
        idx_b = self.pos[node_b]
        idx_c = self.pos[node_c]

        # Case 1: Segment from node_a to node_c does not wrap around.
        if idx_a <= idx_c:
            return idx_a <= idx_b <= idx_c
        # Case 2: Segment from node_a to node_c wraps around.
        else:  # idx_a > idx_c
            return idx_a <= idx_b or idx_b <= idx_c

    def flip(self, segment_start_node: int, segment_end_node: int) -> None:
        """
        Inverts (reverses) the segment of the tour from vertex
        segment_start_node to vertex segment_end_node (inclusive).
        Updates self.order and self.pos efficiently.

        Args:
            segment_start_node: Start vertex of the segment to flip.
            segment_end_node: End vertex of the segment to flip.
        """
        idx_a_in_order, idx_b_in_order = (
            self.pos[segment_start_node], self.pos[segment_end_node]
        )

        # Collect array indices in self.order forming the segment
        # from segment_start_node to segment_end_node.
        current_segment_indices_in_order = []
        current_idx = idx_a_in_order
        while True:
            current_segment_indices_in_order.append(current_idx)
            if current_idx == idx_b_in_order:
                break
            current_idx = (current_idx + 1) % self.n

        # Extract vertex values (nodes) in the segment
        segment_nodes = [
            self.order[i] for i in current_segment_indices_in_order
        ]

        # Reverse the segment nodes
        reversed_segment_nodes = segment_nodes[::-1]

        # Place reversed segment back into self.order and update self.pos.
        for i in range(len(current_segment_indices_in_order)):
            array_idx_to_update = current_segment_indices_in_order[i]
            node_to_place_in_order = reversed_segment_nodes[i]

            self.order[array_idx_to_update] = node_to_place_in_order
            self.pos[node_to_place_in_order] = array_idx_to_update

    def get_tour(self) -> List[int]:
        """
        Returns the current tour as a list of vertex indices,
        normalized to start from vertex 0 if present in the tour.
        If vertex 0 is not in the tour, the tour is returned as is
        (starting from its current first element).

        Returns:
            Ordered list of vertex indices representing the tour.
            Returns an empty list if the tour is empty (self.n == 0).
        """
        if self.n == 0:
            return []

        node_zero_present_and_valid = False
        # Check if 0 is a potential index for self.pos (i.e. 0 <= max_node_label)
        # and if it's actually in the tour.
        if 0 <= self.pos.shape[0] - 1:
            idx_of_0_in_order_array = self.pos[0]
            # Check if the pos[0] value is a valid index in self.order
            # and if the node at that position is indeed 0.
            if (0 <= idx_of_0_in_order_array < self.n and
                    self.order[idx_of_0_in_order_array] == 0):
                node_zero_present_and_valid = True

        if node_zero_present_and_valid:
            # Node 0 is in the tour, normalize to start with it.
            position_of_vertex_0 = self.pos[0]
            if position_of_vertex_0 == 0:
                # Already starts with 0.
                return list(self.order)
            else:
                # Rotate the order array.
                rotated_order = np.concatenate(
                    (self.order[position_of_vertex_0:],
                     self.order[:position_of_vertex_0])
                )
                return list(rotated_order)
        else:
            # Node 0 not in tour or self.pos not configured for it.
            return list(self.order)

    def flip_and_update_cost(self, node_a: int, node_b: int,
                             D: np.ndarray) -> float:
        """
        Performs a segment flip from node_a to node_b (inclusive, following
        tour order) and updates the tour's cost. This is a 2-opt move.

        Args:
            node_a: Start node of the segment to flip.
            node_b: End node of the segment to flip.
            D: Distance matrix.

        Returns:
            The change in cost (delta_cost). Positive if cost increased.
        """
        if self.n == 0:  # Should not happen with valid tours
            return 0.0

        # If node_a and node_b are the same, no change in order or cost
        if node_a == node_b:
            delta_cost = 0.0
        else:
            pos_a = self.pos[node_a]
            pos_b = self.pos[node_b]

            # Identify the nodes defining the edges to be broken and formed.
            # Edge 1 broken: (prev_node_of_a, node_a)
            # Edge 2 broken: (node_b, next_node_of_b)
            # Edge 1 added: (prev_node_of_a, node_b)
            # Edge 2 added: (node_a, next_node_of_b)
            prev_node_of_a = self.order[(pos_a - 1 + self.n) % self.n]
            next_node_of_b = self.order[(pos_b + 1) % self.n]

            # Handle case where the "segment" is the entire tour.
            # This occurs if node_a is next_node_of_b AND node_b is prev_node_of_a.
            # Flipping the entire tour results in 0 cost change for symmetric TSP.
            if prev_node_of_a == node_b and next_node_of_b == node_a:
                 delta_cost = 0.0
            else:
                term_removed = (D[prev_node_of_a, node_a] +
                                D[node_b, next_node_of_b])
                term_added = (D[prev_node_of_a, node_b] +
                              D[node_a, next_node_of_b])
                delta_cost = term_added - term_removed

        # Perform the flip operation on the tour structure
        self.flip(node_a, node_b)

        # Update the tour's cost
        if self.cost is None:
            # This case should ideally not be hit if tour cost is initialized.
            # If it is, the delta_cost is the new cost relative to an assumed 0.
            self.cost = delta_cost  # Or recompute full cost for safety
        else:
            self.cost += delta_cost
        return delta_cost


def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Computes the full distance (cost) matrix for the given coordinates.

    Args:
        coords: Array of vertex coordinates.
                Expected shape (n, d) where n is number of points,
                d is dimension.

    Returns:
        Symmetric matrix of pairwise Euclidean distances. Shape (n, n).
        Returns an empty array of shape (0,0) if input coords is empty.
    """
    if coords.ndim == 1 and coords.shape[0] == 0:
        # Handles np.array([]) which has shape (0,)
        return np.empty((0, 0), dtype=float)

    if coords.shape[0] == 0:
        # Handles np.array([[]]) or similar (0, d)
        return np.empty((0, 0), dtype=float)

    if coords.shape[0] == 1:
        # For a single point, the distance matrix should be [[0.]]
        return np.array([[0.0]], dtype=float)

    # Using broadcasting to compute all pairwise differences
    return np.linalg.norm(coords[:, None] - coords[None, :], axis=2)


def delaunay_neighbors(coords: np.ndarray) -> List[List[int]]:
    """
    Builds a list of neighbors for each vertex using Delaunay triangulation.
    The neighbors for each vertex are sorted. This list is used to restrict
    candidate moves in the Lin-Kernighan algorithm.

    Args:
        coords: Array of vertex coordinates, shape (n, 2).

    Returns:
        A list where the i-th element is a sorted list
        of neighbor vertex indices for vertex i.
    """
    num_vertices = len(coords)
    if num_vertices < 3:
        # Delaunay requires at least 3 points. For < 3 points,
        # all points are neighbors of each other (excluding self).
        all_neighbors: List[List[int]] = []
        for i in range(num_vertices):
            vertex_neighbors = [j for j in range(num_vertices) if i != j]
            all_neighbors.append(sorted(vertex_neighbors))
        return all_neighbors

    triangulation = Delaunay(coords)

    # Store sets of neighbors to auto-handle duplicate edges.
    neighbor_sets: Dict[int, set[int]] = {
        i: set() for i in range(num_vertices)
    }

    # Iterate over each simplex (triangle) in the triangulation
    for simplex_indices in triangulation.simplices:
        # For each pair of vertices in the simplex, they are neighbors
        for u_vertex, v_vertex in combinations(simplex_indices, 2):
            neighbor_sets[u_vertex].add(v_vertex)
            neighbor_sets[v_vertex].add(u_vertex)

    # Convert sets of neighbors to sorted lists
    neighbor_lists: List[List[int]] = [
        sorted(list(neighbor_sets[i])) for i in range(num_vertices)
    ]
    return neighbor_lists


def step(level: int, delta: float, base: int, tour: Tour, D: np.ndarray,
         neigh: List[List[int]], flip_seq: List[Tuple[int, int]],
         start_cost: float, best_cost: float,
         deadline: float) -> Tuple[bool, Optional[List[Tuple[int, int]]]]:
    """
    Recursively explores possible sequences of flips to find improved tours,
    using both standard and Mak-Morton moves as per Algorithm 15.1
    in Applegate et al.

    Args:
        level: Current recursion level (k in variable k-opt).
        delta: Current accumulated gain (total reduction in tour cost).
        base: The current base vertex for moves.
        tour: The current tour object.
        D: Distance/cost matrix.
        neigh: List of neighbor lists for each vertex.
        flip_seq: Current sequence of flip operations.
        start_cost: Cost of the original tour at start of this search pass.
        best_cost: Best tour cost found so far globally.
        deadline: Timestamp for time limit.

    Returns:
        (True, flip_seq) if an improved tour is found, else (False, None).
    """
    if time.time() >= deadline or level > LK_CONFIG["MAX_LEVEL"]:
        return False, None

    if not LK_CONFIG["BREADTH"]:
        breadth_limit = 1
    else:
        breadth_limit = LK_CONFIG["BREADTH"][
            min(level - 1, len(LK_CONFIG["BREADTH"]) - 1)
        ]
    s1 = tour.next(base)
    candidates = []

    # Standard flips (u-steps):
    for candidate_y1 in neigh[s1]:  # y1 is t_2i in Helsgaun's notation
        if time.time() >= deadline:
            return False, None
        if candidate_y1 in (base, s1):
            continue

        # G1 = c(t1,t2) - c(t2,y1)
        gain_G1 = D[base, s1] - D[s1, candidate_y1]
        if gain_G1 <= FLOAT_COMPARISON_TOLERANCE:  # Must be strictly positive gain
            continue

        # t3 is prev(y1)
        t3_node = tour.prev(candidate_y1)

        # G2 = c(t3,y1) - c(t3,t1)
        gain_G2 = D[t3_node, candidate_y1] - D[t3_node, base]
        total_gain_for_2_opt_part = gain_G1 + gain_G2

        # Pruning: accumulated gain (delta) + G1 must be positive.
        if delta + gain_G1 > FLOAT_COMPARISON_TOLERANCE:
            candidates.append(
                ('flip', candidate_y1, t3_node, total_gain_for_2_opt_part)
            )

    # Mak-Morton flips:
    for candidate_a_mm in neigh[base]:
        if time.time() >= deadline:
            return False, None
        if candidate_a_mm in (tour.next(base), tour.prev(base), base):
            continue

        # Gain for Mak-Morton move
        gain_mak_morton = (
            (D[base, s1] - D[base, candidate_a_mm]) +
            (D[candidate_a_mm, tour.next(candidate_a_mm)] -
             D[tour.next(candidate_a_mm), s1])
        )

        # Pruning condition similar to standard flips
        if delta + (D[base, s1] - D[base, candidate_a_mm]) \
           > FLOAT_COMPARISON_TOLERANCE:
            candidates.append(
                ('makmorton', candidate_a_mm, None, gain_mak_morton)
            )

    candidates.sort(key=lambda x: -x[3])  # Sort by gain, descending
    count = 0
    for move_type, node1_param, node2_param, gain_val in candidates:
        if time.time() >= deadline or count >= breadth_limit:
            break
        new_accumulated_gain = delta + gain_val
        if move_type == 'flip':
            # node1_param is y1, node2_param is t3
            # Flip segment between s1 (t2) and t3 (prev(y1))
            flip_start_node, flip_end_node = s1, node2_param
            tour.flip(flip_start_node, flip_end_node)
            flip_seq.append((flip_start_node, flip_end_node))

            if start_cost - new_accumulated_gain < \
               best_cost - FLOAT_COMPARISON_TOLERANCE:
                return True, flip_seq.copy()  # Found globally improving tour

            if level < LK_CONFIG["MAX_LEVEL"]:
                # Recursive call with base (t1)
                improved, final_seq = step(
                    level + 1, new_accumulated_gain, base, tour, D,
                    neigh, flip_seq, start_cost, best_cost, deadline
                )
                if improved:
                    return True, final_seq
            # Backtrack
            tour.flip(flip_end_node, flip_start_node)
            flip_seq.pop()
        else:  # 'makmorton'
            # node1_param is candidate_a_mm
            # Flip segment between next(candidate_a_mm) and base (t1)
            flip_start_node = tour.next(node1_param)
            flip_end_node = base
            tour.flip(flip_start_node, flip_end_node)
            flip_seq.append((flip_start_node, flip_end_node))

            if start_cost - new_accumulated_gain < \
               best_cost - FLOAT_COMPARISON_TOLERANCE:
                return True, flip_seq.copy()

            new_base_for_recursion = tour.next(node1_param)
            if level < LK_CONFIG["MAX_LEVEL"]:
                improved, final_seq = step(
                    level + 1, new_accumulated_gain, new_base_for_recursion,
                    tour, D, neigh, flip_seq, start_cost, best_cost, deadline
                )
                if improved:
                    return True, final_seq
            # Backtrack
            tour.flip(flip_end_node, flip_start_node)
            flip_seq.pop()
        count += 1
    return False, None


def alternate_step(
    base_node: int, tour: Tour, D: np.ndarray, neigh: List[List[int]],
    deadline: float
) -> Tuple[bool, Optional[List[Tuple[int, int]]]]:
    """
    Implements the alternative first step of LK (Algorithm 15.2, Applegate).
    Provides extra breadth for specific 3-opt or 5-opt moves.

    Args:
        base_node: The current base vertex (t1).
        tour: The tour object.
        D: Distance/cost matrix.
        neigh: List of neighbor lists for each vertex.
        deadline: Timestamp for time limit.

    Returns:
        (True, flip_seq) if an improved tour is found, else (False, None).
        flip_seq is a list of (segment_start, segment_end) for tour.flip().
    """
    if time.time() >= deadline:
        return False, None

    t1 = base_node
    t2 = tour.next(t1)

    # Stage 1: Find candidate y1
    candidates_for_y1 = []
    for y1_candidate in neigh[t2]:
        if y1_candidate in (t1, t2):
            continue
        gain_G1 = D[t1, t2] - D[t2, y1_candidate]  # G1 = c(t1,t2) - c(t2,y1)
        if gain_G1 <= FLOAT_COMPARISON_TOLERANCE:  # Must be G1 > 0
            continue
        t3 = tour.prev(y1_candidate)
        sort_metric_y1 = D[t3, y1_candidate] - D[t2, y1_candidate]
        candidates_for_y1.append((sort_metric_y1, y1_candidate, t3))
    candidates_for_y1.sort(reverse=True)

    for _, y1_chosen, t3_of_y1 in candidates_for_y1[:LK_CONFIG["BREADTH_A"]]:
        if time.time() >= deadline: return False, None
        t4 = tour.next(y1_chosen)

        # Stage 2: Find candidate y2
        candidates_for_y2 = []
        for y2_candidate in neigh[t4]:
            if y2_candidate in (t1, t2, y1_chosen): continue
            t6 = tour.next(y2_candidate)
            sort_metric_y2 = D[t6, y2_candidate] - D[t4, y2_candidate]
            candidates_for_y2.append((sort_metric_y2, y2_candidate, t6))
        candidates_for_y2.sort(reverse=True)

        for _, y2_chosen, t6_of_y2 in candidates_for_y2[:LK_CONFIG["BREADTH_B"]]:
            if time.time() >= deadline: return False, None
            if tour.sequence(t2, y2_chosen, y1_chosen):  # Specific 3-opt
                return True, [(t2, y2_chosen), (y2_chosen, y1_chosen)]

            # Stage 3: Find candidate y3 for a 5-opt move
            candidates_for_y3 = []
            for y3_candidate in neigh[t6_of_y2]:
                if y3_candidate in (t1, t2, y1_chosen, t4, y2_chosen): continue
                node_after_y3 = tour.next(y3_candidate)
                sort_metric_y3 = (D[node_after_y3, y3_candidate] -
                                  D[t6_of_y2, y3_candidate])
                candidates_for_y3.append(
                    (sort_metric_y3, y3_candidate, node_after_y3)
                )
            candidates_for_y3.sort(reverse=True)

            for _, y3_chosen, node_after_y3_chosen in \
                    candidates_for_y3[:LK_CONFIG["BREADTH_D"]]:
                if time.time() >= deadline: return False, None
                # Specific 5-opt move
                return True, [(t2, y3_chosen), (y3_chosen, y1_chosen),
                               (t4, node_after_y3_chosen)]
    return False, None


def lk_search(start_node_for_search: int, current_tour_obj: Tour,
              D: np.ndarray, neigh: List[List[int]],
              deadline: float) -> Optional[List[Tuple[int, int]]]:
    """
    Top-level Lin-Kernighan search (Algorithm 15.3, Applegate et al.).
    Attempts to find an improving flip sequence starting at `start_node`.

    Args:
        start_node_for_search: Vertex to initiate the search from.
        current_tour_obj: The current tour object.
        D: Distance/cost matrix.
        neigh: List of neighbor lists for each vertex.
        deadline: Timestamp for time limit.

    Returns:
        A list of (segment_start, segment_end) tuples for the flip
        sequence if an improved tour is found, else None.
    """
    if time.time() >= deadline:
        return None

    search_tour_copy = Tour(current_tour_obj.get_tour(), D)
    cost_at_search_start = search_tour_copy.cost
    assert cost_at_search_start is not None, "Tour cost must be initialized."

    found_step, seq_step = step(
        level=1, delta=0.0, base=start_node_for_search,
        tour=search_tour_copy, D=D, neigh=neigh, flip_seq=[],
        start_cost=cost_at_search_start, best_cost=cost_at_search_start,
        deadline=deadline
    )
    if found_step and seq_step:
        return seq_step

    if time.time() >= deadline:
        return None

    found_alt, seq_alt = alternate_step(
        base_node=start_node_for_search, tour=current_tour_obj,
        D=D, neigh=neigh, deadline=deadline
    )
    if found_alt and seq_alt:
        cost_before_alt_check = current_tour_obj.cost
        if cost_before_alt_check is None: return None  # Should be initialized

        temp_check_tour = Tour(current_tour_obj.get_tour(), D)
        for f_start, f_end in seq_alt:
            temp_check_tour.flip_and_update_cost(f_start, f_end, D)

        if temp_check_tour.cost is not None and \
           temp_check_tour.cost < cost_before_alt_check - \
           FLOAT_COMPARISON_TOLERANCE:
            return seq_alt
    return None


def lin_kernighan(coords: np.ndarray, init: List[int], D: np.ndarray,
                  neigh: List[List[int]],
                  deadline: float) -> Tuple[Tour, float]:
    """
    The main Lin-Kernighan heuristic (Algorithm 15.4, Applegate et al.).

    Iteratively applies lk_search() to all vertices. If an improving
    sequence of flips is found, it's applied, and all vertices are
    marked for re-checking. Continues until no marked vertices remain or
    time limit is reached.

    Args:
        coords: Vertex coordinates.
        init: Initial tour permutation.
        D: Distance/cost matrix.
        neigh: List of neighbor lists for each vertex.
        deadline: Timestamp for time limit.

    Returns:
        The best tour object found and its cost.
    """
    n = len(coords)
    current_best_tour_obj = Tour(init, D)
    current_best_tour_cost = current_best_tour_obj.cost
    assert current_best_tour_cost is not None, "Initial tour cost missing."

    marked_nodes = set(range(n))

    while marked_nodes:
        if time.time() >= deadline:
            break
        start_node_for_lk = marked_nodes.pop()

        tour_order_before_lk = current_best_tour_obj.get_tour()
        cost_before_lk = current_best_tour_obj.cost
        assert cost_before_lk is not None, "Cost before lk_search missing."

        improving_sequence = lk_search(
            start_node_for_lk, current_best_tour_obj, D, neigh, deadline
        )

        if improving_sequence:
            candidate_tour_obj = Tour(tour_order_before_lk, D)
            for x_flip, y_flip in improving_sequence:
                candidate_tour_obj.flip_and_update_cost(x_flip, y_flip, D)

            cost_of_candidate_tour = candidate_tour_obj.cost
            assert cost_of_candidate_tour is not None, "Candidate cost missing."

            if cost_of_candidate_tour < cost_before_lk - \
               FLOAT_COMPARISON_TOLERANCE:
                current_best_tour_obj = candidate_tour_obj
                current_best_tour_cost = cost_of_candidate_tour
                marked_nodes = set(range(n))  # Re-mark all nodes
    return current_best_tour_obj, current_best_tour_cost


def double_bridge(order: List[int]) -> List[int]:
    """
    Applies a "double-bridge" style perturbation to the tour.
    This performs a 4-opt move by selecting four cut points to define five
    segments (S0, S1, S2, S3, S4) and reorders them as S0-S3-S2-S1-S4.
    (Note: Original comment said S0-S2-S1-S3-S4, but standard double bridge
    is often S0-S3-S2-S1-S4 or similar non-sequential reordering).
    The implementation below does S0-S2-S1-S3-S4.

    Args:
        order: Current tour order.

    Returns:
        New tour order after applying the perturbation.
    """
    n = len(order)
    if n <= 4:  # Perturbation is trivial or not possible for small tours
        return list(order)

    # Choose 4 distinct random indices for cut points.
    # Indices are for slicing, so they can range from 1 to n-1.
    # Ensure cut_points are sorted to define segments correctly.
    cut_points = sorted(np.random.choice(range(1, n), 4, replace=False))
    p1, p2, p3, p4 = cut_points[0], cut_points[1], cut_points[2], cut_points[3]

    s0 = order[:p1]
    s1 = order[p1:p2]
    s2 = order[p2:p3]
    s3 = order[p3:p4]
    s4 = order[p4:]

    # Reassemble: S0-S2-S1-S3-S4 (swaps S1 and S2)
    return s0 + s2 + s1 + s3 + s4


def chained_lin_kernighan(
    coords: np.ndarray, initial_tour_order: List[int],
    known_optimal_length: Optional[float] = None,
    time_limit_seconds: Optional[float] = None
) -> Tuple[List[int], float]:
    """
    Chained Lin-Kernighan metaheuristic (Algorithm 15.5, Applegate et al.).

    Repeatedly applies Lin-Kernighan with double-bridge kicks
    to escape local minima, stopping either at the time limit or
    if the known optimum is found.

    Args:
        coords: Vertex coordinates.
        initial_tour_order: Initial tour order.
        known_optimal_length: Known optimal tour length for early stopping.
        time_limit_seconds: Maximum time (seconds) to run.
                            Defaults to LK_CONFIG["TIME_LIMIT"].

    Returns:
        The best tour order found and its cost.
    """
    effective_time_limit = (time_limit_seconds
                            if time_limit_seconds is not None
                            else LK_CONFIG["TIME_LIMIT"])
    assert effective_time_limit is not None

    start_time = time.time()
    deadline = start_time + effective_time_limit

    distance_matrix = build_distance_matrix(coords)
    neighbor_list = delaunay_neighbors(coords)

    current_best_tour_obj, current_best_cost = lin_kernighan(
        coords, initial_tour_order, distance_matrix, neighbor_list, deadline
    )
    # Ensure tour object's cost is consistent
    if current_best_tour_obj.cost is None or \
       not math.isclose(current_best_tour_obj.cost, current_best_cost):
        current_best_tour_obj.init_cost(distance_matrix)
        current_best_cost = current_best_tour_obj.cost
    assert current_best_cost is not None

    if known_optimal_length is not None and \
       math.isclose(current_best_cost, known_optimal_length,
                    rel_tol=1e-7, abs_tol=FLOAT_COMPARISON_TOLERANCE * 10):
        return current_best_tour_obj.get_tour(), current_best_cost

    while time.time() < deadline:
        kicked_tour_order = double_bridge(current_best_tour_obj.get_tour())
        lk_result_tour_obj, lk_result_cost = lin_kernighan(
            coords, kicked_tour_order, distance_matrix, neighbor_list, deadline
        )
        if lk_result_tour_obj.cost is None or \
           not math.isclose(lk_result_tour_obj.cost, lk_result_cost):
            lk_result_tour_obj.init_cost(distance_matrix)
            lk_result_cost = lk_result_tour_obj.cost
        assert lk_result_cost is not None

        if lk_result_cost < current_best_cost - FLOAT_COMPARISON_TOLERANCE:
            current_best_tour_obj = lk_result_tour_obj
            current_best_cost = lk_result_cost
            if known_optimal_length is not None and \
               math.isclose(current_best_cost, known_optimal_length,
                            rel_tol=1e-7, abs_tol=FLOAT_COMPARISON_TOLERANCE * 10):
                break
    # Final cost consistency check
    final_tour_order = current_best_tour_obj.get_tour()
    final_recomputed_cost = 0.0
    for i in range(current_best_tour_obj.n):
        node1 = final_tour_order[i]
        node2 = final_tour_order[(i + 1) % current_best_tour_obj.n]
        final_recomputed_cost += float(distance_matrix[node1, node2])
    current_best_tour_obj.cost = final_recomputed_cost

    return final_tour_order, final_recomputed_cost


def read_opt_tour(path: str) -> Optional[List[int]]:
    """
    Reads an optimal tour from a .opt.tour file in TSPLIB format.

    Args:
        path: Path to the .opt.tour file.

    Returns:
        List of 0-indexed node IDs for the optimal tour, or None if
        file not found, malformed, or TOUR_SECTION not properly terminated.
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
                        in_tour_section = False
                        break
                    if p_token == 'EOF':
                        in_tour_section = False
                        break
                    try:
                        node_val = int(p_token)
                        tour.append(node_val - 1)  # TSPLIB is 1-indexed
                    except ValueError:
                        print(f"Warning: Invalid token '{p_token}' in {path}")
                        return None
                if not in_tour_section:
                    break
        if not tour or not found_minus_one_terminator:
            return None  # No tour or not properly terminated
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading optimal tour file {path}: {e}")
        return None
    return tour


def read_tsp_file(path: str) -> np.ndarray:
    """
    Reads a TSPLIB formatted TSP file (EUC_2D) and returns coordinates.

    Args:
        path: Path to the .tsp file.

    Returns:
        Numpy array of shape (n, 2) containing coordinates.

    Raises:
        FileNotFoundError: If the TSP file is not found.
        Exception: For other parsing errors.
    """
    coords_dict: Dict[int, List[float]] = {}
    reading_nodes = False
    edge_weight_type = None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_content in f:
                line = line_content.strip()
                if not line: continue

                if ":" in line and not reading_nodes:
                    key, value = [part.strip() for part in line.split(":",1)]
                    if key.upper() == "EDGE_WEIGHT_TYPE":
                        edge_weight_type = value.upper()

                if line.upper().startswith("NODE_COORD_SECTION"):
                    if edge_weight_type != "EUC_2D":
                        raise ValueError(
                            f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}."
                            f" Only EUC_2D is supported by this reader."
                        )
                    reading_nodes = True
                    continue
                if line.upper() == "EOF":
                    break

                if reading_nodes:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            node_id = int(parts[0])
                            x_coord = float(parts[1])
                            y_coord = float(parts[2])
                            coords_dict[node_id] = [x_coord, y_coord]
                        except ValueError:
                            print(f"Warning: Skipping unparsable node line: "
                                  f"'{line_content.strip()}' in {path}")
        if not coords_dict:
            print(f"Warning: No coordinates found in {path}")
            return np.array([], dtype=float)

        sorted_node_ids = sorted(coords_dict.keys())
        coords_list = [coords_dict[node_id] for node_id in sorted_node_ids]
        return np.array(coords_list, dtype=float)

    except FileNotFoundError:
        print(f"Error: TSP file not found at {path}")
        raise
    except Exception as e:
        print(f"Error reading TSP file {path}: {e}")
        raise


def process_single_instance(
        tsp_file_path_str: str, opt_tour_file_path_str: str
) -> Dict[str, Any]:
    """
    Processes a single TSP instance: loads data, runs LK, calculates stats.

    Args:
        tsp_file_path_str: Path to the .tsp file.
        opt_tour_file_path_str: Path to the .opt.tour file.

    Returns:
        Dictionary containing results for the instance.
    """
    problem_name = Path(tsp_file_path_str).stem
    print(f"Processing {problem_name} (EUC_2D)...")

    coords = read_tsp_file(tsp_file_path_str)
    if coords.size == 0:  # Check if read_tsp_file returned empty
        print(f"  Skipping {problem_name} due to coordinate reading error.")
        # Return a structure indicating failure or skip
        return {
            'name': problem_name, 'coords': coords, 'opt_tour': None,
            'heu_tour': [], 'opt_len': None, 'heu_len': float('inf'),
            'gap': None, 'time': 0.0, 'error': True
        }
    D_matrix = build_distance_matrix(coords)

    opt_tour_nodes = read_opt_tour(opt_tour_file_path_str)
    opt_len: Optional[float] = None
    gap: Optional[float] = None

    if opt_tour_nodes:
        current_opt_len = 0.0
        for i in range(len(opt_tour_nodes)):
            a = opt_tour_nodes[i]
            b = opt_tour_nodes[(i + 1) % len(opt_tour_nodes)]
            current_opt_len += D_matrix[a, b]
        opt_len = current_opt_len
        print(f"  Optimal length: {opt_len:.2f}")
    else:
        print(f"  Optimal tour not available for {problem_name}.")

    initial_tour = list(range(len(coords)))
    start_time = time.time()

    heuristic_tour, heuristic_len = chained_lin_kernighan(
        coords, initial_tour, known_optimal_length=opt_len
    )
    elapsed_time = time.time() - start_time

    if opt_len is not None:
        if opt_len > FLOAT_COMPARISON_TOLERANCE * 10:
            gap_percentage = 100.0 * (heuristic_len - opt_len) / opt_len
            gap = max(0.0, gap_percentage)
        elif math.isclose(opt_len, 0.0):
            gap = 0.0 if math.isclose(heuristic_len, 0.0) else float('inf')

    print(
        f"  Heuristic length: {heuristic_len:.2f}  "
        f"Gap: {gap:.2f}%  Time: {elapsed_time:.2f}s" if gap is not None
        else f"  Heuristic length: {heuristic_len:.2f}  Time: {elapsed_time:.2f}s"
    )
    return {
        'name': problem_name, 'coords': coords, 'opt_tour': opt_tour_nodes,
        'heu_tour': heuristic_tour, 'opt_len': opt_len,
        'heu_len': heuristic_len, 'gap': gap, 'time': elapsed_time
    }


def display_summary_table(results_data: List[Dict[str, Any]]) -> None:
    """
    Prints a formatted summary table of the processing results.

    Args:
        results_data: List of result dictionaries from instances.
    """
    print("\nConfiguration parameters:")
    for key, value in LK_CONFIG.items():
        if isinstance(value, float):
            print(f"  {key:<11s} = {value:.2f}")
        else:
            print(f"  {key:<11s} = {value}")
    print("")

    header = f"{'Instance':<10s} {'OptLen':>8s} {'HeuLen':>8s} " \
             f"{'Gap(%)':>8s} {'Time(s)':>8s}"
    print(header)
    print("-" * len(header))

    for r_item in results_data:
        if r_item.get('error'): continue  # Skip errored instances in summary table
        opt_len_str = (f"{r_item['opt_len']:>8.2f}"
                       if r_item['opt_len'] is not None else f"{'N/A':>8s}")
        gap_str = (f"{r_item['gap']:>8.2f}"
                   if r_item['gap'] is not None else f"{'N/A':>8s}")
        print(
            f"{r_item['name']:<10s} {opt_len_str} "
            f"{r_item['heu_len']:>8.2f} {gap_str} "
            f"{r_item['time']:>8.2f}"
        )

    valid_results = [r for r in results_data if not r.get('error')]
    if valid_results:
        print("-" * len(header))
        num_valid_items = len(valid_results)

        valid_opt_lens = [r['opt_len'] for r in valid_results
                          if r['opt_len'] is not None]
        valid_gaps = [r['gap'] for r in valid_results
                      if r['gap'] is not None and r['gap'] != float('inf')]

        total_opt_len_sum = sum(valid_opt_lens) if valid_opt_lens else None
        total_heu_len_sum = sum(r['heu_len'] for r in valid_results)
        avg_gap_val = sum(valid_gaps) / len(valid_gaps) if valid_gaps else None
        avg_time_val = (sum(r['time'] for r in valid_results) / num_valid_items
                        if num_valid_items > 0 else 0.0)

        total_opt_str = (f"{total_opt_len_sum:>8.2f}"
                         if total_opt_len_sum is not None else f"{'N/A':>8s}")
        avg_gap_str = (f"{avg_gap_val:>8.2f}"
                       if avg_gap_val is not None else f"{'N/A':>8s}")

        print(
            f"{'SUMMARY':<10s} {total_opt_str} {total_heu_len_sum:>8.2f} "
            f"{avg_gap_str} {avg_time_val:>8.2f}"
        )
    print("Done.")


def plot_all_tours(results_data: List[Dict[str, Any]]) -> None:
    """
    Plots optimal and heuristic tours for processed instances.

    Args:
        results_data: List of result dictionaries.
    """
    valid_results = [r for r in results_data if not r.get('error') and r['coords'].size > 0]
    num_results_total = len(valid_results)

    if num_results_total == 0:
        print("No valid results with coordinates to plot.")
        return

    results_to_plot = (valid_results[:MAX_SUBPLOTS_IN_PLOT]
                       if num_results_total > MAX_SUBPLOTS_IN_PLOT
                       else valid_results)
    if num_results_total > MAX_SUBPLOTS_IN_PLOT:
        print(f"Warning: Plotting first {MAX_SUBPLOTS_IN_PLOT} of "
              f"{num_results_total} valid results.")

    num_to_plot = len(results_to_plot)
    if num_to_plot == 0: return

    cols = int(math.ceil(math.sqrt(num_to_plot)))
    rows = int(math.ceil(num_to_plot / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows),
                             squeeze=False)
    axes_list = axes.flatten()
    plotted_heuristic, plotted_optimal = False, False

    for i, r_item in enumerate(results_to_plot):
        ax = axes_list[i]
        coords = r_item['coords']

        if r_item['heu_tour']:
            heu_plot = r_item['heu_tour'] + [r_item['heu_tour'][0]]
            ax.plot(coords[heu_plot, 0], coords[heu_plot, 1],
                    '-', label='Heuristic', zorder=1, color='C0')
            plotted_heuristic = True
        if r_item['opt_tour']:
            opt_plot = r_item['opt_tour'] + [r_item['opt_tour'][0]]
            ax.plot(coords[opt_plot, 0], coords[opt_plot, 1],
                    ':', label='Optimal', zorder=2, color='C1')
            plotted_optimal = True

        title = f"{r_item['name']}"
        if r_item['gap'] is not None and r_item['gap'] != float('inf'):
            title += f" gap={r_item['gap']:.2f}%"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')

    for i in range(num_to_plot, len(axes_list)):
        axes_list[i].set_axis_off()

    legend_elements = []
    if plotted_heuristic:
        legend_elements.append(Line2D([0], [0], color='C0', ls='-',
                                      label='Heuristic'))
    if plotted_optimal:
        legend_elements.append(Line2D([0], [0], color='C1', ls=':',
                                      label='Optimal'))
    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper center',
                   ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.0))
        fig.subplots_adjust(top=(0.95 if num_to_plot > cols else 0.90))

    plt.tight_layout(rect=(0, 0, 1, 0.96 if legend_elements else 1.0))
    plt.show()


if __name__ == '__main__':
    all_instance_results = []
    if not TSP_FOLDER_PATH.is_dir():
        print(f"Error: TSP folder not found at {TSP_FOLDER_PATH}")
    else:
        for tsp_file_path_obj in sorted(TSP_FOLDER_PATH.iterdir()):
            if tsp_file_path_obj.suffix.lower() != '.tsp':
                continue

            base_name = tsp_file_path_obj.stem
            opt_tour_path_obj = TSP_FOLDER_PATH / (base_name + '.opt.tour')

            try:
                result = process_single_instance(
                    str(tsp_file_path_obj), str(opt_tour_path_obj)
                )
                all_instance_results.append(result)
            except Exception as e:
                print(f"Critical error processing {base_name}: {e}")
                all_instance_results.append({
                    'name': base_name, 'coords': np.array([]),
                    'opt_tour': None, 'heu_tour': [], 'opt_len': None,
                    'heu_len': float('inf'), 'gap': None, 'time': 0.0,
                    'error': True
                })

    display_summary_table(all_instance_results)
    plot_all_tours(all_instance_results)
