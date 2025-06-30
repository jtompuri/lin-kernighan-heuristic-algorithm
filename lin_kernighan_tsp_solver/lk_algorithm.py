"""
Lin-Kernighan algorithm core for the Traveling Salesperson Problem (TSP).

This module implements the Lin-Kernighan (LK) heuristic and related metaheuristics
for solving TSP instances. It provides the core Tour class, LK search routines,
and utilities for distance and neighbor calculations.

Functions and classes:
    Tour: Represents a TSP tour and supports efficient manipulation.
    build_distance_matrix(coords): Compute the Euclidean distance matrix.
    delaunay_neighbors(coords): Compute Delaunay triangulation neighbors.
    step(...): Recursive step of the Lin-Kernighan algorithm.
    alternate_step(...): Alternative first step for additional breadth.
    lk_search(...): Single Lin-Kernighan search pass.
    lin_kernighan(...): Main Lin-Kernighan heuristic.
    double_bridge(order): Apply a double-bridge 4-opt move.
    chained_lin_kernighan(...): Chained LK metaheuristic with double-bridge kicks.
"""

import time
import math
from itertools import combinations
from typing import List, Dict, Tuple, Optional, Iterable

import numpy as np
from scipy.spatial import Delaunay

from .config import FLOAT_COMPARISON_TOLERANCE, LK_CONFIG


class Tour:
    """
    Represents a tour (permutation of vertices) for the LK heuristic.

    Supports efficient queries (next/prev), segment flips, and cost tracking.

    Attributes:
        n (int): Number of vertices.
        order (np.ndarray): Vertex indices in tour order.
        pos (np.ndarray): Mapping vertex index to its position in `order`.
        cost (float or None): Total cost (length) of the tour.
    """

    def __init__(self, order: Iterable[int],
                 D: Optional[np.ndarray] = None) -> None:
        """
        Calculates and stores the tour's total cost using distance matrix D.

        Args:
            D (np.ndarray): Distance matrix (D[i, j] = distance between i and j).
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
        """
        Returns the vertex immediately following v in the tour.

        Args:
            v (int): Vertex index.

        Returns:
            int: Next vertex in the tour.
        """
        if self.n == 0:
            raise IndexError("Cannot get next node from an empty tour.")
        return int(self.order[(self.pos[v] + 1) % self.n])

    def prev(self, v: int) -> int:
        """
        Returns the vertex immediately preceding v in the tour.

        Args:
            v (int): Vertex index.

        Returns:
            int: Previous vertex in the tour.
        """
        if self.n == 0:
            raise IndexError("Cannot get previous node from an empty tour.")
        return int(self.order[(self.pos[v] - 1 + self.n) % self.n])

    def sequence(self, node_a: int, node_b: int, node_c: int) -> bool:
        """
        Checks if node_b is on the path from node_a to node_c (inclusive).

        Args:
            node_a (int): Start vertex of the segment.
            node_b (int): Vertex to check.
            node_c (int): End vertex of the segment.

        Returns:
            bool: True if node_b is on the segment [node_a, ..., node_c].
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
        Reverses the tour segment from segment_start_node to segment_end_node
        in-place.

        Args:
            segment_start_node (int): First vertex of the segment to flip.
            segment_end_node (int): Last vertex of the segment to flip.
        """
        idx_a, idx_b = self.pos[segment_start_node], self.pos[segment_end_node]

        if idx_a == idx_b:  # Segment is a single node, no change needed.
            return

        # Determine the number of elements in the segment to be flipped.
        # This handles both non-wrapping (idx_a <= idx_b) and wrapping (idx_a > idx_b) segments.
        if idx_a <= idx_b:
            segment_len = idx_b - idx_a + 1
        else:  # Segment wraps around the end of the tour array
            segment_len = (self.n - idx_a) + (idx_b + 1)

        # Perform in-place reversal by swapping pairs of nodes.
        # We iterate for half the length of the segment.
        for i in range(segment_len // 2):
            # Calculate current left and right indices in self.order array
            # The left index starts at idx_a and moves forward along the segment.
            # The right index starts at idx_b and moves backward along the segment.
            # Modulo self.n handles the circular nature of the tour array.
            current_left_order_idx = (idx_a + i) % self.n
            current_right_order_idx = (idx_b - i + self.n) % self.n  # Ensure positive before modulo

            # Swap the nodes at these positions in self.order
            node_at_left = self.order[current_left_order_idx]
            node_at_right = self.order[current_right_order_idx]

            self.order[current_left_order_idx] = node_at_right
            self.order[current_right_order_idx] = node_at_left

            # Update their positions in the self.pos mapping
            self.pos[node_at_right] = current_left_order_idx
            self.pos[node_at_left] = current_right_order_idx

    def get_tour(self) -> List[int]:
        """
        Returns tour as a list, normalized to start with node 0 if present.

        Returns:
            List[int]: List of vertex indices. Empty if tour is empty.
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
            node_a (int): Start node of the segment to flip.
            node_b (int): End node of the segment to flip.
            D (np.ndarray): Distance matrix.

        Returns:
            float: Change in tour cost (delta_cost). Positive if cost increased.
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
        coords (np.ndarray): Array of shape (n, d) for n points in d dimensions.

    Returns:
        np.ndarray: Array of shape (n, n) with distances D[i,j].
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

    Args:
        coords (np.ndarray): Array of shape (n, 2) for n points.

    Returns:
        List[List[int]]: List where List[i] contains sorted neighbors of vertex i.
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
        for u_vertex, v_vertex in combinations(simplex_indices, 2):  # type: ignore[arg-type]
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
        level (int): Current recursion depth (k in k-opt).
        delta (float): Accumulated gain from previous flips.
        base (int): Current base node (t1) for moves.
        tour (Tour): Tour object (modified and reverted during search).
        D (np.ndarray): Distance matrix.
        neigh (List[List[int]]): Neighbor lists for candidate selection.
        flip_seq (List[Tuple[int, int]]): Current sequence of flips.
        start_cost (float): Cost of tour at the beginning of the current lk_search pass.
        best_cost (float): Globally best tour cost found so far.
        deadline (float): Time limit for search.

    Returns:
        Tuple[bool, Optional[List[Tuple[int, int]]]]: (True, improving_flip_sequence) if a tour
        better than best_cost is found, else (False, None).
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
            (D[base, s1] - D[base, candidate_a_mm]) + (D[candidate_a_mm, tour.next(candidate_a_mm)]
                                                       - D[tour.next(candidate_a_mm), s1])
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

    Args:
        base_node (int): Current base vertex (t1) for the LK move.
        tour (Tour): The current Tour object.
        D (np.ndarray): Distance matrix.
        neigh (List[List[int]]): Neighbor lists for candidate selection.
        deadline (float): Time limit for the search.

    Returns:
        Tuple[bool, Optional[List[Tuple[int, int]]]]: (True, flip_sequence) if a potentially
        improving sequence is identified, else (False, None).
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

            # --- Stage 3: Find candidate y3 for a 5-opt move (Type Q in Applegate et al.) ---
            candidates_for_y3 = []
            for y3_candidate in neigh[chosen_t6]:  # y3 from neighbors of t6
                if (y3_candidate == t1 or y3_candidate == t2 or y3_candidate == chosen_y1
                   or y3_candidate == t4 or y3_candidate == chosen_y2):
                    continue
                node_after_y3_candidate = tour.next(y3_candidate)  # t8
                # Sort metric for y3: D[node_after_y3,y3] - D[chosen_t6,y3]
                sort_metric_y3 = (D[node_after_y3_candidate, y3_candidate]
                                  - D[chosen_t6, y3_candidate])
                candidates_for_y3.append((sort_metric_y3, y3_candidate, node_after_y3_candidate))
            candidates_for_y3.sort(reverse=True)

            for _, chosen_y3, chosen_node_after_y3 in candidates_for_y3[:LK_CONFIG["BREADTH_D"]]:
                if time.time() >= deadline:
                    return False, None
                # Identified a specific 5-opt move.
                flip_sequence_5_opt = [(t2, chosen_y3), (chosen_y3, chosen_y1),
                                       (t4, chosen_node_after_y3)]
                return True, flip_sequence_5_opt
    return False, None


def lk_search(start_node_for_search: int, current_tour_obj: Tour,
              D: np.ndarray, neigh: List[List[int]],
              deadline: float) -> Optional[List[Tuple[int, int]]]:
    """
    Single Lin-Kernighan search pass (Algorithm 15.3, Applegate et al.).

    Tries step then alternate_step to find an improving flip sequence.

    Args:
        start_node_for_search (int): Vertex to initiate search from (t1).
        current_tour_obj (Tour): Current tour.
        D (np.ndarray): Distance matrix.
        neigh (List[List[int]]): Neighbor lists.
        deadline (float): Time limit.

    Returns:
        Optional[List[Tuple[int, int]]]: List of (start, end) flips if improvement found, else None.
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
        if cost_before_alt_check is None:  # pragma: no cover
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

    Iteratively applies lk_search from marked nodes until no improvement or time limit.

    Args:
        coords (np.ndarray): Vertex coordinates.
        init (List[int]): Initial tour permutation.
        D (np.ndarray): Distance matrix.
        neigh (List[List[int]]): Neighbor lists.
        deadline (float): Time limit.

    Returns:
        Tuple[Tour, float]: (Best Tour object, its cost).
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

    Args:
        order (List[int]): Current tour order.

    Returns:
        List[int]: New perturbed tour order. Returns original if n <= 4.
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

    Args:
        coords (np.ndarray): Vertex coordinates.
        initial_tour_order (List[int]): Initial tour.
        known_optimal_length (float, optional): Known optimal length for early stop.
        time_limit_seconds (float, optional): Max run time.

    Returns:
        Tuple[List[int], float]: (Best tour order found, its cost).
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
