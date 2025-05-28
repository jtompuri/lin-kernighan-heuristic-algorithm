"""
Lin-Kernighan Heuristic for the Traveling Salesperson Problem (TSP).

Implements the Lin-Kernighan (LK) heuristic, a local search algorithm for
finding approximate solutions to the TSP. Based on Applegate et al. and
Lin & Kernighan's original paper.

Processes TSPLIB format instances, computes solutions using chained LK,
compares against optimal tours if available, and displays results.

Usage:
  1. Install dependencies: pip install numpy matplotlib scipy
  2. Place .tsp files (and optional .opt.tour files) in a folder.
  3. Update `TSP_FOLDER_PATH` constant below.
  4. Run: python lin_kernighan_tsp_solver.py

Adjust LK algorithm parameters in `LK_CONFIG`.
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
# Lin-Kernighan heuristic.
LK_CONFIG = {
    "MAX_LEVEL": 12,  # Max recursion depth for k-opt moves in step()
    "BREADTH": [5, 5] + [1] * 20,  # Search breadth at each level in step()
    "BREADTH_A": 5,  # Search breadth for y1 in alternate_step()
    "BREADTH_B": 5,  # Search breadth for y2 in alternate_step()
    "BREADTH_D": 1,  # Search breadth for y3 in alternate_step()
    "TIME_LIMIT": 20.0,  # Default time limit for chained_lin_kernighan (s)
}


class Tour:
    """
    Represents a tour (permutation of vertices) for the LK heuristic.

    Supports efficient queries (next/prev), segment flips, and cost tracking.

    Attributes:
        n: Number of vertices.
        order: Numpy array of vertex indices in tour order.
        pos: Numpy array mapping vertex index to its position in `order`.
        cost: Total cost (length) of the tour.
    """

    def __init__(self, order: Iterable[int],
                 D: Optional[np.ndarray] = None) -> None:
        """
        Initializes Tour from a vertex sequence.

        Args:
            order: Sequence of vertex indices.
            D: Optional distance matrix to calculate initial tour cost.
        """
        order_list = list(order)  # Convert iterable to list
        self.n: int = len(order_list)

        if self.n == 0:
            # Handle empty tour initialization
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
        Calculates and stores the tour's total cost using distance matrix D.

        Args:
            D: Distance matrix (D[i, j] = distance between i and j).
        """
        if self.n == 0:
            self.cost = 0.0  # Cost of an empty tour is 0
            return
        current_total_cost = sum(
            # Calculate cost for each segment in the tour
            float(D[self.order[i], self.order[(i + 1) % self.n]])
            for i in range(self.n)
        )
        self.cost = current_total_cost

    def next(self, v: int) -> int:
        """Returns the vertex immediately following v in the tour."""
        if self.n == 0:
            raise IndexError("Cannot get next node from an empty tour.")
        # self.pos[v] is index of v in self.order; +1 % n for next index
        return int(self.order[(self.pos[v] + 1) % self.n])

    def prev(self, v: int) -> int:
        """Returns the vertex immediately preceding v in the tour."""
        if self.n == 0:
            raise IndexError("Cannot get previous node from an empty tour.")
        # self.pos[v] is index of v in self.order; -1 + n % n for prev index
        return int(self.order[(self.pos[v] - 1 + self.n) % self.n])

    def sequence(self, node_a: int, node_b: int, node_c: int) -> bool:
        """
        Checks if node_b is on the path from node_a to node_c (inclusive).

        Args:
            node_a: Start vertex of the segment.
            node_b: Vertex to check.
            node_c: End vertex of the segment.

        Returns:
            True if node_b is on the segment [node_a, ..., node_c].
        """
        if self.n == 0:
            return False
        # Get positions (indices in self.order)
        idx_a, idx_b, idx_c = self.pos[node_a], self.pos[node_b], self.pos[node_c]

        if idx_a <= idx_c:  # Segment does not wrap around
            return idx_a <= idx_b <= idx_c
        else:  # Segment wraps around (e.g., a=N-1, c=1, b can be N-1,0,1)
            return idx_a <= idx_b or idx_b <= idx_c

    def flip(self, segment_start_node: int, segment_end_node: int) -> None:
        """
        Reverses the tour segment from segment_start_node to segment_end_node.

        Args:
            segment_start_node: First vertex of the segment to flip.
            segment_end_node: Last vertex of the segment to flip.
        """
        idx_a, idx_b = self.pos[segment_start_node], self.pos[segment_end_node]

        # Collect array indices in self.order forming the segment
        segment_indices_in_order = []
        current_idx = idx_a
        while True:
            segment_indices_in_order.append(current_idx)
            if current_idx == idx_b:
                break
            current_idx = (current_idx + 1) % self.n

        # Extract vertex values (nodes) in the segment
        segment_nodes = self.order[segment_indices_in_order]
        # Reverse the segment nodes
        reversed_segment_nodes = segment_nodes[::-1]

        # Place reversed segment back into self.order and update self.pos
        for i, order_idx in enumerate(segment_indices_in_order):
            node_val_to_place = reversed_segment_nodes[i]
            self.order[order_idx] = node_val_to_place
            self.pos[node_val_to_place] = order_idx  # Update position map

    def get_tour(self) -> List[int]:
        """
        Returns tour as a list, normalized to start with node 0 if present.
        If node 0 is not in the tour, returns the tour as is.

        Returns:
            List of vertex indices. Empty if tour is empty.
        """
        if self.n == 0:
            return []

        node_zero_present_and_valid = False
        # Check if 0 is a potential index for self.pos and actually in tour
        if 0 <= self.pos.shape[0] - 1:
            idx_of_0_in_order = self.pos[0]
            if (0 <= idx_of_0_in_order < self.n and self.order[idx_of_0_in_order] == 0):
                node_zero_present_and_valid = True

        if node_zero_present_and_valid:
            # Node 0 is in the tour, normalize to start with it.
            position_of_vertex_0 = self.pos[0]
            return list(np.roll(self.order, -position_of_vertex_0))
        else:
            # Node 0 not in tour or self.pos not configured for it.
            return list(self.order)

    def flip_and_update_cost(self, node_a: int, node_b: int,
                             D: np.ndarray) -> float:
        """
        Flips segment [node_a,...,node_b], updates cost, returns cost change.

        Args:
            node_a: Start node of the segment to flip.
            node_b: End node of the segment to flip.
            D: Distance matrix.

        Returns:
            Change in tour cost (delta_cost). Positive if cost increased.
        """
        if self.n == 0:  # Should not occur with valid tours
            return 0.0

        if node_a == node_b:  # No change if segment is a single node
            delta_cost = 0.0
        else:
            pos_a = self.pos[node_a]
            pos_b = self.pos[node_b]

            # Identify nodes defining edges to be broken and formed for 2-opt
            prev_node_of_a = self.order[(pos_a - 1 + self.n) % self.n]
            next_node_of_b = self.order[(pos_b + 1) % self.n]

            # Handle case where the "segment" is the entire tour (flip is identity)
            if prev_node_of_a == node_b and next_node_of_b == node_a:
                delta_cost = 0.0
            else:
                # Cost change for a 2-opt move
                term_removed = (D[prev_node_of_a, node_a] + D[node_b, next_node_of_b])
                term_added = (D[prev_node_of_a, node_b] + D[node_a, next_node_of_b])
                delta_cost = term_added - term_removed

        self.flip(node_a, node_b)  # Perform the segment reversal

        if self.cost is None:  # Should be initialized
            self.init_cost(D)  # Fallback: recompute full cost
        else:
            self.cost += delta_cost
        return delta_cost


def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Euclidean distances for coordinates.

    Args:
        coords: Numpy array of shape (n, d) for n points in d dimensions.

    Returns:
        Numpy array of shape (n, n) with distances D[i,j].
    """
    # Handle empty or single point cases
    if coords.shape[0] == 0:
        return np.empty((0, 0), dtype=float)
    if coords.shape[0] == 1:
        return np.array([[0.0]], dtype=float)

    # Efficiently compute pairwise distances using broadcasting and linalg.norm
    return np.linalg.norm(coords[:, None] - coords[None, :], axis=2)


def delaunay_neighbors(coords: np.ndarray) -> List[List[int]]:
    """
    Generates sorted neighbor lists using Delaunay triangulation.

    Used to restrict candidate moves in LK. For <3 points, all are neighbors.

    Args:
        coords: Numpy array of shape (n, 2) for n points.

    Returns:
        List where `List[i]` contains sorted neighbors of vertex i.
    """
    num_vertices = len(coords)
    if num_vertices < 3:
        # Delaunay requires >= 3 points. For fewer, all nodes are neighbors.
        return [[j for j in range(num_vertices) if i != j]
                for i in range(num_vertices)]

    triangulation = Delaunay(coords)
    # Use sets to automatically handle duplicate edges from shared simplices
    neighbor_sets: Dict[int, set[int]] = {
        i: set() for i in range(num_vertices)
    }
    # Populate neighbor sets from Delaunay simplices
    for simplex_indices in triangulation.simplices:
        for u_vertex, v_vertex in combinations(simplex_indices, 2):
            neighbor_sets[u_vertex].add(v_vertex)
            neighbor_sets[v_vertex].add(u_vertex)
    # Convert sets to sorted lists
    return [sorted(list(neighbor_sets[i])) for i in range(num_vertices)]


def step(level: int, delta: float, base: int, tour: Tour, D: np.ndarray,
         neigh: List[List[int]], flip_seq: List[Tuple[int, int]],
         start_cost: float, best_cost: float,
         deadline: float) -> Tuple[bool, Optional[List[Tuple[int, int]]]]:
    """
    Recursive step of Lin-Kernighan (Algorithm 15.1, Applegate et al.).

    Explores k-opt moves (standard and Mak-Morton) to improve the tour.

    Args:
        level: Current recursion depth (k in k-opt).
        delta: Accumulated gain from previous flips.
        base: Current base node (t1) for moves.
        tour: Tour object (modified and reverted during search).
        D: Distance matrix.
        neigh: Neighbor lists for candidate selection.
        flip_seq: Current sequence of (start_node, end_node) flips.
        start_cost: Cost of tour at the beginning of the current lk_search pass.
        best_cost: Globally best tour cost found so far.
        deadline: Time limit for search.

    Returns:
        (True, improving_flip_sequence) if a tour better than `best_cost`
        is found, else (False, None).
    """
    if time.time() >= deadline or level > LK_CONFIG["MAX_LEVEL"]:
        return False, None

    breadth_limit = (LK_CONFIG["BREADTH"][min(level - 1, len(LK_CONFIG["BREADTH"]) - 1)]
                     if LK_CONFIG["BREADTH"] else 1)
    s1 = tour.next(base)  # s1 is t2 in common LK notation (node after base)
    candidates = []

    # Standard flips (u-steps):
    # Try to break (base, s1) and (t3_node, y1_cand), add (s1, y1_cand) and (base, t3_node)
    for y1_cand in neigh[s1]:  # y1_cand is a candidate for y1 (t_2i in Helsgaun)
        if time.time() >= deadline:
            return False, None
        if y1_cand in (base, s1):
            continue  # y1 cannot be base or s1

        gain_G1 = D[base, s1] - D[s1, y1_cand]  # Cost diff: c(t1,t2) - c(t2,y1)
        if gain_G1 <= FLOAT_COMPARISON_TOLERANCE:
            continue  # Must be strictly positive gain

        t3_node = tour.prev(y1_cand)  # t3_node is t_2i+1 in notation
        gain_G2 = D[t3_node, y1_cand] - D[t3_node, base]  # Cost diff: c(t3,y1) - c(t3,t1)
        total_gain_for_2_opt_part = gain_G1 + gain_G2

        # Pruning: accumulated gain (delta) + G1 must be positive.
        if delta + gain_G1 > FLOAT_COMPARISON_TOLERANCE:
            candidates.append(
                ('flip', y1_cand, t3_node, total_gain_for_2_opt_part)
            )

    # Mak-Morton flips:
    for candidate_a_mm in neigh[base]:
        if time.time() >= deadline:
            return False, None
        if candidate_a_mm in (s1, tour.prev(base), base):
            continue

        gain_mak_morton = (
            (D[base, s1] - D[base, candidate_a_mm]) + (D[candidate_a_mm, tour.next(candidate_a_mm)] - D[tour.next(candidate_a_mm), s1])
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
        current_flips_this_step = []  # Flips made in this specific call/level

        if move_type == 'flip':
            # node1_param is y1_cand, node2_param is t3_node
            # Flip segment between s1 (t2) and t3_node (prev(y1_cand))
            flip_start_node, flip_end_node = s1, node2_param
            tour.flip(flip_start_node, flip_end_node)
            current_flips_this_step.append((flip_start_node, flip_end_node))
            next_base_for_recursion = base  # For standard flips, base remains t1
        else:  # 'makmorton'
            # node1_param is candidate_a_mm
            # Flip segment between next(candidate_a_mm) and base (t1)
            flip_start_node = tour.next(node1_param)
            flip_end_node = base
            tour.flip(flip_start_node, flip_end_node)
            current_flips_this_step.append((flip_start_node, flip_end_node))
            # New base for Mak-Morton is next(candidate_a_mm) after flip
            next_base_for_recursion = tour.next(node1_param)

        flip_seq.extend(current_flips_this_step)  # Add current flip(s) to sequence

        if start_cost - new_accumulated_gain < \
           best_cost - FLOAT_COMPARISON_TOLERANCE:  # Check for global improvement
            return True, flip_seq.copy()

        if level < LK_CONFIG["MAX_LEVEL"]:  # Recurse if max depth not reached
            improved, final_seq = step(
                level + 1, new_accumulated_gain, next_base_for_recursion,
                tour, D, neigh, flip_seq, start_cost, best_cost, deadline
            )
            if improved:
                return True, final_seq
        # Backtrack: undo flips made in this step, in reverse order
        for f_start, f_end in reversed(current_flips_this_step):
            tour.flip(f_end, f_start)  # Re-flip to revert
            flip_seq.pop()  # Remove from overall sequence
        count += 1
    return False, None


def alternate_step(
    base_node: int, tour: Tour, D: np.ndarray, neigh: List[List[int]],
    deadline: float
) -> Tuple[bool, Optional[List[Tuple[int, int]]]]:
    """
    Alternative first step of LK (Algorithm 15.2, Applegate et al.).

    Explores specific 3-opt or 5-opt moves for additional breadth.
    This function identifies candidate flip sequences; the caller (`lk_search`)
    verifies if these sequences result in an actual tour cost improvement.

    Args:
        base_node: Current base vertex (t1) for the LK move.
        tour: The current Tour object.
        D: Distance matrix.
        neigh: Neighbor lists for candidate selection.
        deadline: Time limit for the search.

    Returns:
        (True, flip_sequence) if a potentially improving 3-opt or 5-opt
        sequence is identified, else (False, None). The flip_sequence
        contains (start_node, end_node) tuples.
    """
    if time.time() >= deadline:
        return False, None

    t1 = base_node
    t2 = tour.next(t1)  # Edge (t1,t2) is the first edge considered for breaking.

    # --- Stage 1: Find candidate y1 ---
    # y1 is chosen from neighbors of t2.
    candidates_for_y1 = []
    for y1_candidate in neigh[t2]:
        if y1_candidate == t1 or y1_candidate == t2:
            continue
        # G1 = c(t1,t2) - c(t2,y1_candidate). Must be G1 > 0.
        gain_G1 = D[t1, t2] - D[t2, y1_candidate]
        if gain_G1 <= FLOAT_COMPARISON_TOLERANCE:
            continue
        t3_of_y1_candidate = tour.prev(y1_candidate)  # t3 is node before y1
        # Sort metric for y1 (Applegate et al. p. 376): D[t3,y1] - D[t2,y1]
        sort_metric_y1 = D[t3_of_y1_candidate, y1_candidate] - D[t2, y1_candidate]
        candidates_for_y1.append((sort_metric_y1, y1_candidate, t3_of_y1_candidate))
    candidates_for_y1.sort(reverse=True)  # Best first

    for _, chosen_y1, chosen_t3 in candidates_for_y1[:LK_CONFIG["BREADTH_A"]]:
        if time.time() >= deadline:
            return False, None
        t4 = tour.next(chosen_y1)  # t4 follows chosen_y1

        # --- Stage 2: Find candidate y2 ---
        candidates_for_y2 = []
        for y2_candidate in neigh[t4]:
            if y2_candidate == t1 or y2_candidate == t2 or y2_candidate == chosen_y1:
                continue
            t6_of_y2_candidate = tour.next(y2_candidate)  # t6 follows y2
            # Sort metric for y2: D[t6,y2] - D[t4,y2]
            sort_metric_y2 = D[t6_of_y2_candidate, y2_candidate] - D[t4, y2_candidate]
            candidates_for_y2.append((sort_metric_y2, y2_candidate, t6_of_y2_candidate))
        candidates_for_y2.sort(reverse=True)

        for _, chosen_y2, chosen_t6 in candidates_for_y2[:LK_CONFIG["BREADTH_B"]]:
            if time.time() >= deadline:
                return False, None
            # Check for a specific 3-opt move (Type R in Applegate et al. Fig 15.4)
            if tour.sequence(t2, chosen_y2, chosen_y1):
                flip_sequence_3_opt = [(t2, chosen_y2), (chosen_y2, chosen_y1)]
                return True, flip_sequence_3_opt

            # --- Stage 3: Find candidate y3 for a 5-opt move (Type Q in Applegate et al. Fig 15.5) ---
            candidates_for_y3 = []
            for y3_candidate in neigh[chosen_t6]:  # y3 from neighbors of t6
                if (y3_candidate == t1 or y3_candidate == t2 or y3_candidate == chosen_y1 or y3_candidate == t4 or y3_candidate == chosen_y2):
                    continue
                node_after_y3_candidate = tour.next(y3_candidate)  # t8
                # Sort metric for y3: D[node_after_y3,y3] - D[chosen_t6,y3]
                sort_metric_y3 = D[node_after_y3_candidate, y3_candidate] - D[chosen_t6, y3_candidate]
                candidates_for_y3.append((sort_metric_y3, y3_candidate, node_after_y3_candidate))
            candidates_for_y3.sort(reverse=True)

            for _, chosen_y3, chosen_node_after_y3 in candidates_for_y3[:LK_CONFIG["BREADTH_D"]]:
                if time.time() >= deadline:
                    return False, None
                # Identified a specific 5-opt move.
                flip_sequence_5_opt = [(t2, chosen_y3), (chosen_y3, chosen_y1), (t4, chosen_node_after_y3)]
                return True, flip_sequence_5_opt
    return False, None


def lk_search(start_node_for_search: int, current_tour_obj: Tour,
              D: np.ndarray, neigh: List[List[int]],
              deadline: float) -> Optional[List[Tuple[int, int]]]:
    """
    Single Lin-Kernighan search pass (Algorithm 15.3, Applegate et al.).

    Tries `step` then `alternate_step` to find an improving flip sequence.

    Args:
        start_node_for_search: Vertex to initiate search from (t1).
        current_tour_obj: Current tour.
        D: Distance matrix.
        neigh: Neighbor lists.
        deadline: Time limit.

    Returns:
        List of (start, end) flips if improvement found, else None.
    """
    if time.time() >= deadline:
        return None

    # Attempt 1: Standard Recursive Step Search on a copy of the tour
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
        return seq_step  # `step` found a sequence improving on its `best_cost`
    if time.time() >= deadline:
        return None

    # Attempt 2: Alternate Step Search
    # `alternate_step` identifies candidate sequences based on `current_tour_obj`.
    found_alt, seq_alt = alternate_step(
        base_node=start_node_for_search, tour=current_tour_obj,
        D=D, neigh=neigh, deadline=deadline
    )
    if found_alt and seq_alt:
        # Verify if the sequence from alternate_step is strictly improving
        cost_before_alt_check = current_tour_obj.cost
        if cost_before_alt_check is None:
            return None

        temp_check_tour = Tour(current_tour_obj.get_tour(), D)
        for f_start, f_end in seq_alt:
            temp_check_tour.flip_and_update_cost(f_start, f_end, D)

        if temp_check_tour.cost is not None and \
           temp_check_tour.cost < cost_before_alt_check - FLOAT_COMPARISON_TOLERANCE:
            return seq_alt  # Sequence is strictly improving
    return None


def lin_kernighan(coords: np.ndarray, init: List[int], D: np.ndarray,
                  neigh: List[List[int]],
                  deadline: float) -> Tuple[Tour, float]:
    """
    Main Lin-Kernighan heuristic (Algorithm 15.4, Applegate et al.).

    Iteratively applies `lk_search` from marked nodes until no improvement
    or time limit.

    Args:
        coords: Vertex coordinates (for `n`).
        init: Initial tour permutation.
        D: Distance matrix.
        neigh: Neighbor lists.
        deadline: Time limit.

    Returns:
        (Best Tour object, its cost).
    """
    n = len(coords)
    current_best_tour_obj = Tour(init, D)
    current_best_tour_cost = current_best_tour_obj.cost
    assert current_best_tour_cost is not None, "Initial tour cost missing."

    marked_nodes = set(range(n))  # Nodes to start lk_search from

    while marked_nodes:
        if time.time() >= deadline:
            break
        start_node_for_lk = marked_nodes.pop()

        # Preserve state before lk_search to compare against
        tour_order_before_lk = current_best_tour_obj.get_tour()
        cost_before_lk = current_best_tour_obj.cost
        assert cost_before_lk is not None

        # lk_search attempts to find an improving sequence for current_best_tour_obj
        improving_sequence = lk_search(
            start_node_for_lk, current_best_tour_obj, D, neigh, deadline
        )

        if improving_sequence:
            # Apply the found sequence to a tour based on pre-search state
            candidate_tour_obj = Tour(tour_order_before_lk, D)
            for x_flip, y_flip in improving_sequence:
                candidate_tour_obj.flip_and_update_cost(x_flip, y_flip, D)
            cost_of_candidate_tour = candidate_tour_obj.cost
            assert cost_of_candidate_tour is not None

            # If strictly better than the tour before this lk_search call
            if cost_of_candidate_tour < cost_before_lk - FLOAT_COMPARISON_TOLERANCE:
                current_best_tour_obj = candidate_tour_obj
                current_best_tour_cost = cost_of_candidate_tour
                marked_nodes = set(range(n))  # Improvement found, re-mark all
    return current_best_tour_obj, current_best_tour_cost


def double_bridge(order: List[int]) -> List[int]:
    """
    Applies a double-bridge 4-opt move to perturb the tour.

    Selects 4 random cut points, defining 5 segments S0-S4.
    This implementation reorders to S0-S2-S1-S3-S4 (swaps S1 and S2).

    Args:
        order: Current tour order.

    Returns:
        New perturbed tour order. Returns original if n <= 4.
    """
    n = len(order)
    if n <= 4:  # Perturbation is trivial or not possible for small tours
        return list(order)

    # Choose 4 distinct random indices for cut points.
    cut_points = sorted(np.random.choice(range(1, n), 4, replace=False))
    p1, p2, p3, p4 = cut_points[0], cut_points[1], cut_points[2], cut_points[3]

    s0 = order[:p1]
    s1 = order[p1:p2]
    s2 = order[p2:p3]
    s3 = order[p3:p4]
    s4 = order[p4:]

    # Reassemble: S0-S2-S1-S3-S4
    return s0 + s2 + s1 + s3 + s4


def chained_lin_kernighan(
    coords: np.ndarray, initial_tour_order: List[int],
    known_optimal_length: Optional[float] = None,
    time_limit_seconds: Optional[float] = None
) -> Tuple[List[int], float]:
    """
    Chained Lin-Kernighan metaheuristic (Algorithm 15.5, Applegate et al.).

    Repeatedly applies LK, with double-bridge kicks to escape local optima.
    Stops at time limit or if known optimum is found.

    Args:
        coords: Vertex coordinates.
        initial_tour_order: Initial tour.
        known_optimal_length: Optional known optimal length for early stop.
        time_limit_seconds: Max run time. Uses LK_CONFIG if None.

    Returns:
        (Best tour order found, its cost).
    """
    effective_time_limit = (time_limit_seconds
                            if time_limit_seconds is not None
                            else LK_CONFIG["TIME_LIMIT"])
    deadline = time.time() + effective_time_limit

    distance_matrix = build_distance_matrix(coords)
    neighbor_list = delaunay_neighbors(coords)

    # Initial LK run
    current_best_tour_obj, current_best_cost = lin_kernighan(
        coords, initial_tour_order, distance_matrix, neighbor_list, deadline
    )
    assert current_best_tour_obj.cost is not None and \
           math.isclose(current_best_tour_obj.cost, current_best_cost)

    if known_optimal_length is not None and \
       math.isclose(current_best_cost, known_optimal_length,
                    rel_tol=1e-7, abs_tol=FLOAT_COMPARISON_TOLERANCE * 10):
        return current_best_tour_obj.get_tour(), current_best_cost

    # Main chain loop: kick and re-run LK
    while time.time() < deadline:
        kicked_tour_order = double_bridge(current_best_tour_obj.get_tour())
        lk_result_tour_obj, lk_result_cost = lin_kernighan(
            coords, kicked_tour_order, distance_matrix, neighbor_list, deadline
        )
        assert lk_result_tour_obj.cost is not None and \
               math.isclose(lk_result_tour_obj.cost, lk_result_cost)

        if lk_result_cost < current_best_cost - FLOAT_COMPARISON_TOLERANCE:
            current_best_tour_obj = lk_result_tour_obj
            current_best_cost = lk_result_cost
            if known_optimal_length is not None and \
               math.isclose(current_best_cost, known_optimal_length,
                            rel_tol=1e-7, abs_tol=FLOAT_COMPARISON_TOLERANCE * 10):
                break  # Optimum found
    # Final cost consistency check before returning
    final_tour_order = current_best_tour_obj.get_tour()
    final_recomputed_cost = 0.0
    if current_best_tour_obj.n > 0:  # Ensure tour is not empty
        for i in range(current_best_tour_obj.n):
            node1 = final_tour_order[i]
            node2 = final_tour_order[(i + 1) % current_best_tour_obj.n]
            final_recomputed_cost += float(distance_matrix[node1, node2])
    # current_best_tour_obj.cost = final_recomputed_cost  # Update object if needed
    return final_tour_order, final_recomputed_cost


def read_opt_tour(path: str) -> Optional[List[int]]:
    """
    Reads an optimal tour from a .opt.tour file (TSPLIB format).

    Expects TOUR_SECTION, 1-indexed nodes, terminated by -1.

    Args:
        path: Path to .opt.tour file.

    Returns:
        List of 0-indexed node IDs, or None on error/malformed.
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
    except Exception:  # Catch other potential errors during file processing
        # print(f"Error reading optimal tour file {path}: {e}")
        return None
    return tour


def read_tsp_file(path: str) -> np.ndarray:
    """
    Reads TSPLIB .tsp file (EUC_2D only) and returns coordinates.

    Args:
        path: Path to .tsp file.

    Returns:
        Numpy array (n, 2) of coordinates.

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

    except FileNotFoundError:
        # print(f"Error: TSP file not found at {path}")
        raise
    except Exception:  # Catch other errors like ValueError from unsupported EWT
        # print(f"Error reading TSP file {path}: {e}")
        raise


def process_single_instance(
        tsp_file_path_str: str, opt_tour_file_path_str: str
) -> Dict[str, Any]:
    """
    Processes one TSP instance: loads, runs LK, calculates stats.

    Args:
        tsp_file_path_str: Path to .tsp file.
        opt_tour_file_path_str: Path to .opt.tour file.

    Returns:
        Dictionary with results: name, coords, tours, lengths, gap, time.
        Includes 'error': True on failure.
    """
    problem_name = Path(tsp_file_path_str).stem
    print(f"Processing {problem_name} (EUC_2D)...")
    # Initialize results dictionary
    results: Dict[str, Any] = {
        'name': problem_name, 'coords': np.array([]), 'opt_tour': None,
        'heu_tour': [], 'opt_len': None, 'heu_len': float('inf'),
        'gap': None, 'time': 0.0, 'error': False,
        'nodes': 0  # Initialize 'nodes' key
    }
    try:
        coords = read_tsp_file(tsp_file_path_str)
        results['coords'] = coords
        if coords.size == 0:  # Check if read_tsp_file returned empty
            raise ValueError("No coordinates loaded from TSP file.")
        results['nodes'] = coords.shape[0]  # Set the number of nodes
        D_matrix = build_distance_matrix(coords)

        opt_tour_nodes = read_opt_tour(opt_tour_file_path_str)
        results['opt_tour'] = opt_tour_nodes
        opt_len: Optional[float] = None

        if opt_tour_nodes:  # If an optimal tour was successfully read
            current_opt_len = 0.0
            for i in range(len(opt_tour_nodes)):
                a = opt_tour_nodes[i]
                b = opt_tour_nodes[(i + 1) % len(opt_tour_nodes)]
                current_opt_len += D_matrix[a, b]
            opt_len = current_opt_len
            results['opt_len'] = opt_len
            print(f"  Optimal length: {opt_len:.2f}")
        else:
            print(f"  Optimal tour not available for {problem_name}.")

        initial_tour = list(range(len(coords)))  # Simple initial tour: 0,1,2...
        start_time = time.time()

        heuristic_tour, heuristic_len = chained_lin_kernighan(
            coords, initial_tour, known_optimal_length=opt_len
        )
        elapsed_time = time.time() - start_time
        results['heu_tour'], results['heu_len'] = heuristic_tour, heuristic_len
        results['time'] = elapsed_time

        # Calculate percentage gap if optimal length is known and positive
        if opt_len is not None:
            if opt_len > FLOAT_COMPARISON_TOLERANCE * 10:  # Avoid division by zero/small
                gap_percentage = 100.0 * (heuristic_len - opt_len) / opt_len
                results['gap'] = max(0.0, gap_percentage)  # Gap cannot be negative
            elif math.isclose(opt_len, 0.0):  # Optimal is zero
                results['gap'] = 0.0 if math.isclose(heuristic_len, 0.0) else float('inf')
        # If opt_len is None, results['gap'] remains None

        gap_str = f"Gap: {results['gap']:.2f}%  " if results['gap'] is not None else ""
        print(f"  Heuristic length: {heuristic_len:.2f}  {gap_str}Time: {elapsed_time:.2f}s")

    except Exception as e:
        print(f"  Skipping {problem_name} due to error: {e}")
        results['error'] = True  # Mark instance as errored
        # Ensure essential keys exist for summary, even on error
        results['heu_len'] = float('inf')
        results['time'] = results.get('time', 0.0)  # Keep time if partially run
    return results


def display_summary_table(results_data: List[Dict[str, Any]]) -> None:
    """
    Prints a formatted summary table of processing results.

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

    valid_results_for_table = [r for r in results_data if not r.get('error')]
    for r_item in valid_results_for_table:
        opt_len_str = (f"{r_item['opt_len']:>8.2f}"
                       if r_item['opt_len'] is not None else f"{'N/A':>8s}")
        gap_str = (f"{r_item['gap']:>8.2f}"
                   if r_item['gap'] is not None else f"{'N/A':>8s}")
        print(
            f"{r_item['name']:<10s} {opt_len_str} "
            f"{r_item['heu_len']:>8.2f} {gap_str} "
            f"{r_item['time']:>8.2f}"
        )

    if valid_results_for_table:  # Calculate summary only if there are valid results
        print("-" * len(header))
        num_valid_items = len(valid_results_for_table)

        valid_opt_lens = [r['opt_len'] for r in valid_results_for_table
                          if r['opt_len'] is not None]
        # Filter out inf gaps for average calculation
        valid_gaps = [r['gap'] for r in valid_results_for_table
                      if r['gap'] is not None and r['gap'] != float('inf')]

        total_opt_len_sum = sum(valid_opt_lens) if valid_opt_lens else None
        total_heu_len_sum = sum(r['heu_len'] for r in valid_results_for_table
                                if r['heu_len'] != float('inf'))  # Sum finite heuristic lengths
        avg_gap_val = sum(valid_gaps) / len(valid_gaps) if valid_gaps else None
        avg_time_val = (sum(r['time'] for r in valid_results_for_table) / num_valid_items
                        if num_valid_items > 0 else 0.0)

        total_opt_str = (f"{total_opt_len_sum:>8.2f}"
                         if total_opt_len_sum is not None else f"{'N/A':>8s}")
        avg_gap_str = (f"{avg_gap_val:>8.2f}"
                       if avg_gap_val is not None else f"{'N/A':>8s}")
        total_heu_str = (f"{total_heu_len_sum:>8.2f}"
                         if total_heu_len_sum != float('inf') else f"{'N/A':>8s}")

        print(
            f"{'SUMMARY':<10s} {total_opt_str} {total_heu_str} "
            f"{avg_gap_str} {avg_time_val:>8.2f}"
        )
    print("Done.")


def plot_all_tours(results_data: List[Dict[str, Any]]) -> None:
    """
    Plots optimal and heuristic tours for processed instances.

    Args:
        results_data: List of result dictionaries.
    """
    # Filter for results that are not errored and have coordinates
    valid_results_to_plot = [
        r for r in results_data
        if not r.get('error') and r.get('coords') is not None and r['coords'].size > 0
    ]
    num_valid_results = len(valid_results_to_plot)

    if num_valid_results == 0:
        print("No valid results with coordinates to plot.")
        return

    # Limit number of plots
    results_to_plot_limited = (valid_results_to_plot[:MAX_SUBPLOTS_IN_PLOT] if num_valid_results > MAX_SUBPLOTS_IN_PLOT else valid_results_to_plot)
    if num_valid_results > MAX_SUBPLOTS_IN_PLOT:
        print(f"Warning: Plotting first {MAX_SUBPLOTS_IN_PLOT} of {num_valid_results} valid results.")

    num_to_plot_actual = len(results_to_plot_limited)
    if num_to_plot_actual == 0:
        return  # Should not happen if num_valid_results > 0

    cols = int(math.ceil(math.sqrt(num_to_plot_actual)))
    rows = int(math.ceil(num_to_plot_actual / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)  # Ensure axes is always 2D
    axes_list = axes.flatten()
    plotted_heuristic_legend, plotted_optimal_legend = False, False

    for i, r_item in enumerate(results_to_plot_limited):
        ax = axes_list[i]
        coords = r_item['coords']

        # Plot heuristic tour if available
        if r_item['heu_tour']:
            heu_plot_nodes = r_item['heu_tour'] + [r_item['heu_tour'][0]]  # Close the loop
            ax.plot(coords[heu_plot_nodes, 0], coords[heu_plot_nodes, 1],
                    '-', label='Heuristic', zorder=1, color='C0')
            plotted_heuristic_legend = True
        # Plot optimal tour if available
        opt_tour_data = r_item.get('opt_tour')
        if opt_tour_data:  # Check if opt_tour_data is not None and not empty
            opt_plot_nodes = opt_tour_data + [opt_tour_data[0]]  # Close the loop
            ax.plot(coords[opt_plot_nodes, 0], coords[opt_plot_nodes, 1],
                    ':', label='Optimal', zorder=2, color='C1')
            plotted_optimal_legend = True

            title = f"{r_item['name']}"
            # Safely get gap, opt_len, heu_len for the title
            gap_val = r_item.get('gap')
            # opt_len_val = r_item.get('opt_len')
            # heu_len_val = r_item.get('heu_len')

            if gap_val is not None and gap_val != float('inf'):
                title += f" gap={gap_val:.2f}%"
            # if opt_len_val is not None:
            #     title += f" OptLen={opt_len_val:.2f}"
            # if heu_len_val is not None:
            #     title += f" HeuLen={heu_len_val:.2f}"
            ax.set_title(title)
            ax.set_xticks([])  # Hide ticks and labels
            ax.set_yticks([])
            ax.set_aspect('equal', adjustable='box')  # Square aspect ratio

    # Turn off unused subplots
    for i in range(num_to_plot_actual, len(axes_list)):
        axes_list[i].set_axis_off()

    # Create legend for the figure
    legend_elements = []
    if plotted_heuristic_legend:
        legend_elements.append(Line2D([0], [0], color='C0', ls='-', label='Heuristic'))
    if plotted_optimal_legend:
        legend_elements.append(Line2D([0], [0], color='C1', ls=':', label='Optimal'))
    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.0))
        # Adjust top margin if legend is present
        fig.subplots_adjust(top=(0.95 if num_to_plot_actual > cols else 0.90))

    plt.tight_layout(rect=(0, 0, 1, 0.96 if legend_elements else 1.0))  # Adjust for legend
    plt.show()


if __name__ == '__main__':
    all_instance_results_list = []
    if not TSP_FOLDER_PATH.is_dir():
        print(f"Error: TSP folder not found at {TSP_FOLDER_PATH}")
    else:
        # Iterate over .tsp files in the specified folder
        for tsp_file_path_obj in sorted(TSP_FOLDER_PATH.glob('*.tsp')):
            base_name = tsp_file_path_obj.stem
            # Corresponding .opt.tour file path
            opt_tour_path_obj = TSP_FOLDER_PATH / (base_name + '.opt.tour')

            try:
                result_dict = process_single_instance(
                    str(tsp_file_path_obj), str(opt_tour_path_obj)
                )
                all_instance_results_list.append(result_dict)
            except Exception as e:  # Catch any unexpected error during processing
                print(f"Critical error processing {base_name}: {e}")
                # Append a basic error entry for summary purposes
                all_instance_results_list.append({
                    'name': base_name, 'coords': np.array([]),
                    'opt_tour': None, 'heu_tour': [], 'opt_len': None,
                    'heu_len': float('inf'), 'gap': None, 'time': 0.0,
                    'error': True
                })

    display_summary_table(all_instance_results_list)
    plot_all_tours(all_instance_results_list)
