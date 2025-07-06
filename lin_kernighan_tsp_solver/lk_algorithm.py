"""Lin-Kernighan algorithm core for the Traveling Salesperson Problem (TSP).

This module implements the Lin-Kernighan (LK) heuristic and related metaheuristics
for solving TSP instances. It provides the core Tour class, LK search routines,
and utilities for distance and neighbor calculations.
"""

import time
import math
from itertools import combinations
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay

from .config import FLOAT_COMPARISON_TOLERANCE, LK_CONFIG


@dataclass
class SearchContext:
    """Holds the static data and search parameters for an LK search pass."""
    D: np.ndarray
    neigh: list[list[int]]
    start_cost: float
    best_cost: float
    deadline: float


class Tour:
    """Represents a tour (permutation of vertices) for the LK heuristic.

    Supports efficient queries (next/prev), segment flips, and cost tracking.

    Attributes:
        n (int): Number of vertices.
        order (np.ndarray): Vertex indices in tour order.
        pos (np.ndarray): Mapping vertex index to its position in `order`.
        cost (float | None): Total cost (length) of the tour.
        n_minus_1 (int): Cached value of n-1 for fast modulo operations.
    """
    __slots__ = ('n', 'order', 'pos', 'cost', 'n_minus_1')

    def __init__(self, order: Iterable[int],
                 D: np.ndarray | None = None) -> None:
        """Initializes a Tour object.

        Args:
            order (Iterable[int]): An iterable of vertex indices defining the tour.
            D (np.ndarray | None, optional): The distance matrix. If provided,
                the initial tour cost is calculated. Defaults to None.
        """
        order_list = list(order)  # Convert iterable to list
        self.n: int = len(order_list)
        self.n_minus_1: int = self.n - 1  # Cache for fast modulo operations

        if self.n == 0:
            # Handle empty tour initialization
            self.order: np.ndarray = np.array([], dtype=np.int32)
            self.pos: np.ndarray = np.array([], dtype=np.int32)
            self.cost: float | None = 0.0
        else:
            self.order: np.ndarray = np.array(order_list, dtype=np.int32)
            # Ensure self.pos is large enough for all actual node labels.
            max_node_label = np.max(self.order)
            self.pos: np.ndarray = np.empty(max_node_label + 1,
                                            dtype=np.int32)
            for i, v_node in enumerate(self.order):
                self.pos[v_node] = i

            self.cost: float | None = None
            if D is not None:
                self.init_cost(D)

    def init_cost(self, D: np.ndarray) -> None:
        """Calculates and stores the tour's total cost using distance matrix D."""
        if self.n == 0:
            self.cost = 0.0
            return

        # Use vectorized operations for larger tours where the overhead is justified
        # Benchmark shows vectorization becomes beneficial around n > 30-50
        if self.n >= 50:  # Threshold based on empirical benchmarking
            indices = np.arange(self.n)
            current_nodes = self.order[indices]
            next_nodes = self.order[(indices + 1) % self.n]
            self.cost = float(np.sum(D[current_nodes, next_nodes]))
        else:
            # Use simple loop for small tours to avoid vectorization overhead
            self.cost = sum(
                float(D[self.order[i], self.order[(i + 1) % self.n]])
                for i in range(self.n)
            )

    def next(self, v: int) -> int:
        """Returns the vertex immediately following v in the tour.

        Args:
            v (int): Vertex index.

        Returns:
            int: Next vertex in the tour.

        Raises:
            IndexError: If the tour is empty.
        """
        if self.n == 0:
            raise IndexError("Cannot get next node from an empty tour.")

        # Fast modulo replacement: avoid expensive modulo operation
        idx = self.pos[v] + 1
        return int(self.order[idx if idx < self.n else 0])

    def prev(self, v: int) -> int:
        """Returns the vertex immediately preceding v in the tour.

        Args:
            v (int): Vertex index.

        Returns:
            int: Previous vertex in the tour.

        Raises:
            IndexError: If the tour is empty.
        """
        if self.n == 0:
            raise IndexError("Cannot get previous node from an empty tour.")

        # Fast modulo replacement: avoid expensive modulo operation
        idx = self.pos[v] - 1
        return int(self.order[idx if idx >= 0 else self.n_minus_1])

    def sequence(self, node_a: int, node_b: int, node_c: int) -> bool:
        """Checks if node_b is on the path from node_a to node_c (inclusive).

        Args:
            node_a (int): Start vertex of the segment.
            node_b (int): Vertex to check.
            node_c (int): End vertex of the segment.

        Returns:
            bool: True if node_b is on the segment [node_a, ..., node_c].
        """
        if self.n == 0:
            return False
        # Get positions (indices in self.order) - these are direct array lookups
        idx_a, idx_b, idx_c = self.pos[node_a], self.pos[node_b], self.pos[node_c]

        if idx_a <= idx_c:  # Segment does not wrap around
            return idx_a <= idx_b <= idx_c
        # Segment wraps around (e.g., a=N-1, c=1, b can be N-1,0,1)
        return idx_a <= idx_b or idx_b <= idx_c

    def flip(self, segment_start_node: int, segment_end_node: int) -> None:
        """Reverses the tour segment from segment_start_node to segment_end_node
        in-place.

        Args:
            segment_start_node (int): First vertex of the segment to flip.
            segment_end_node (int): Last vertex of the segment to flip.
        """
        idx_a, idx_b = self.pos[segment_start_node], self.pos[segment_end_node]

        if idx_a == idx_b:  # Segment is a single node, no change needed.
            return

        # Determine the number of elements in the segment to be flipped.
        if idx_a <= idx_b:
            segment_len = idx_b - idx_a + 1
        else:  # Segment wraps around the end of the tour array
            segment_len = (self.n - idx_a) + (idx_b + 1)

        # Perform in-place reversal by swapping pairs of nodes.
        # Cache n for fast modulo replacement
        n = self.n

        for i in range(segment_len // 2):
            # Fast modulo replacement for left index
            left_idx = idx_a + i
            current_left_order_idx = left_idx if left_idx < n else left_idx - n

            # Fast modulo replacement for right index
            right_idx = idx_b - i
            current_right_order_idx = right_idx if right_idx >= 0 else right_idx + n

            # Swap the nodes at these positions in self.order
            node_at_left = self.order[current_left_order_idx]
            node_at_right = self.order[current_right_order_idx]

            self.order[current_left_order_idx] = node_at_right
            self.order[current_right_order_idx] = node_at_left

            # Update their positions in the self.pos mapping
            self.pos[node_at_right] = current_left_order_idx
            self.pos[node_at_left] = current_right_order_idx

    def get_tour(self) -> list[int]:
        """Returns tour as a list, normalized to start with node 0 if present."""
        if self.n == 0:
            return []

        # Check if node 0 is in the tour
        if 0 < self.pos.shape[0]:  # Check bounds first
            if 0 <= self.pos[0] < self.n and self.order[self.pos[0]] == 0:
                position_of_vertex_0 = self.pos[0]
                return list(np.roll(self.order, -position_of_vertex_0))

        return self.order.tolist()

    def flip_and_update_cost(self, node_a: int, node_b: int, D: np.ndarray) -> float:
        """Flips segment [node_a,...,node_b], updates cost, and returns cost change.

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

        self.flip(node_a, node_b)

        if self.cost is None:
            self.init_cost(D)
            return 0.0  # Return 0 since we can't calculate delta without initial cost
        self.cost += delta_cost
        return delta_cost


def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Computes pairwise Euclidean distances for coordinates.

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


def delaunay_neighbors(coords: np.ndarray) -> list[list[int]]:
    """Generates sorted neighbor lists using Delaunay triangulation.

    Args:
        coords (np.ndarray): Array of shape (n, 2) for n points.

    Returns:
        list[list[int]]: List where list[i] contains sorted neighbors of vertex i.
    """
    num_vertices = len(coords)
    if num_vertices < 3:
        # Delaunay requires >= 3 points. For fewer, all nodes are neighbors.
        return [[j for j in range(num_vertices) if i != j]
                for i in range(num_vertices)]

    triangulation = Delaunay(coords)
    # Use sets to automatically handle duplicate edges from shared simplices
    neighbor_sets: dict[int, set[int]] = {
        i: set() for i in range(num_vertices)
    }
    # Populate neighbor sets from Delaunay simplices
    for simplex_indices in triangulation.simplices:
        for u_vertex, v_vertex in combinations(simplex_indices, 2):  # type: ignore[arg-type]
            neighbor_sets[u_vertex].add(v_vertex)
            neighbor_sets[v_vertex].add(u_vertex)
    # Convert sets to sorted lists
    return [sorted(list(neighbor_sets[i])) for i in range(num_vertices)]


def _generate_standard_flip_candidates(
    base: int, s1: int, tour: Tour, ctx: SearchContext, delta: float
) -> list[tuple[str, int, int | None, float]]:
    """Generates candidates for standard LK flips (u-steps).

    Args:
        base (int): The current base node (t1).
        s1 (int): The node following the base node in the tour (t2).
        tour (Tour): The current tour object.
        ctx (SearchContext): The search context.
        delta (float): The accumulated gain from previous flips.

    Returns:
        list[tuple[str, int, int | None, float]]: A list of candidate moves.
            Each tuple contains ('flip', y1_cand, t3_node, total_gain).
    """
    candidates = []

    # Cache frequently accessed attributes in local variables
    D = ctx.D
    neigh = ctx.neigh
    tour_prev = tour.prev  # Cache method reference

    for y1_cand in neigh[s1]:
        if y1_cand in (base, s1):
            continue

        gain_G1 = D[base, s1] - D[s1, y1_cand]
        if gain_G1 <= FLOAT_COMPARISON_TOLERANCE:
            continue

        t3_node = tour_prev(y1_cand)  # Use cached method reference
        gain_G2 = D[t3_node, y1_cand] - D[t3_node, base]
        total_gain = gain_G1 + gain_G2

        if delta + gain_G1 > FLOAT_COMPARISON_TOLERANCE:
            candidates.append(('flip', y1_cand, t3_node, total_gain))
    return candidates


def _generate_mak_morton_flip_candidates(
    base: int, s1: int, tour: Tour, ctx: SearchContext, delta: float
) -> list[tuple[str, int, int | None, float]]:
    """Generates candidates for Mak-Morton flips.

    Args:
        base (int): The current base node (t1).
        s1 (int): The node following the base node in the tour (t2).
        tour (Tour): The current tour object.
        ctx (SearchContext): The search context.
        delta (float): The accumulated gain from previous flips.

    Returns:
        list[tuple[str, int, int | None, float]]: A list of candidate moves.
            Each tuple contains ('makmorton', candidate_a_mm, None, gain).
    """
    candidates = []

    # Cache frequently accessed attributes
    D = ctx.D
    neigh = ctx.neigh
    tour_prev = tour.prev
    tour_next = tour.next

    for candidate_a_mm in neigh[base]:
        if candidate_a_mm in (s1, tour_prev(base), base):
            continue

        next_candidate = tour_next(candidate_a_mm)
        gain_mak_morton = (
            (D[base, s1] - D[base, candidate_a_mm]) + (D[candidate_a_mm, next_candidate] - D[next_candidate, s1])
        )
        if delta + (D[base, s1] - D[base, candidate_a_mm]) > FLOAT_COMPARISON_TOLERANCE:
            candidates.append(('makmorton', candidate_a_mm, None, gain_mak_morton))
    return candidates


def step(level: int, delta: float, base: int, tour: Tour,
         ctx: SearchContext, flip_seq: list[tuple[int, int]]
         ) -> tuple[bool, list[tuple[int, int]] | None]:
    """Recursive step of Lin-Kernighan (Algorithm 15.1, Applegate et al.).

    Explores k-opt moves (standard and Mak-Morton) to improve the tour.

    Args:
        level (int): Current recursion depth (k in k-opt).
        delta (float): Accumulated gain from previous flips.
        base (int): Current base node (t1) for moves.
        tour (Tour): Tour object (modified and reverted during search).
        ctx (SearchContext): The search context, containing D, neighbors, and costs.
        flip_seq (list[tuple[int, int]]): Current sequence of flips.

    Returns:
        tuple[bool, list[tuple[int, int]] | None]: A tuple containing:
            - True if a tour better than the global best cost is found, False otherwise.
            - The improving flip sequence, or None if no improvement found.
    """    # Check deadline once per recursion level
    current_time = time.time()
    if current_time >= ctx.deadline or level > LK_CONFIG["MAX_LEVEL"]:
        return False, None

    # Cache frequently accessed values
    max_level = LK_CONFIG["MAX_LEVEL"]
    breadth_config = LK_CONFIG["BREADTH"]
    start_cost = ctx.start_cost
    best_cost = ctx.best_cost

    breadth_limit = (breadth_config[min(level - 1, len(breadth_config) - 1)]
                     if breadth_config else 1)

    s1 = tour.next(base)  # s1 is t2 in common LK notation

    # Generate candidates from both standard and Mak-Morton moves
    candidates: list[tuple[str, int, int | None, float]] = _generate_standard_flip_candidates(base, s1, tour, ctx, delta)
    candidates.extend(_generate_mak_morton_flip_candidates(base, s1, tour, ctx, delta))

    candidates.sort(key=lambda x: -x[3])  # Sort by gain, descending
    count = 0

    # Check time periodically, not every iteration
    time_check_interval = 32
    tour_flip = tour.flip  # Cache method reference
    tour_next = tour.next  # Cache method reference

    for move_type, node1_param, node2_param, gain_val in candidates:
        if count % time_check_interval == 0:
            current_time = time.time()
            if current_time >= ctx.deadline:
                break

        if count >= breadth_limit:
            break

        new_accumulated_gain = delta + gain_val
        current_flips_this_step = []

        if move_type == 'flip':
            assert node2_param is not None, "Standard flip move must have a valid t3_node."
            flip_start_node, flip_end_node = s1, node2_param
            tour_flip(flip_start_node, flip_end_node)
            current_flips_this_step.append((flip_start_node, flip_end_node))
            next_base_for_recursion = base
        else:  # 'makmorton'
            flip_start_node = tour_next(node1_param)
            flip_end_node = base
            tour_flip(flip_start_node, flip_end_node)
            current_flips_this_step.append((flip_start_node, flip_end_node))
            next_base_for_recursion = tour_next(node1_param)

        flip_seq.extend(current_flips_this_step)

        if start_cost - new_accumulated_gain < best_cost - FLOAT_COMPARISON_TOLERANCE:
            return True, flip_seq.copy()

        if level < max_level:
            improved, final_seq = step(
                level + 1, new_accumulated_gain, next_base_for_recursion,
                tour, ctx, flip_seq
            )
            if improved:
                return True, final_seq

        # Backtrack: undo flips made in this step, in reverse order
        for f_start, f_end in reversed(current_flips_this_step):
            tour_flip(f_end, f_start)  # Re-flip to revert
            flip_seq.pop()
        count += 1
    return False, None


def _find_y1_candidates(t1: int, t2: int, D: np.ndarray, neigh: list[list[int]], tour: Tour) -> list[tuple[float, int, int]]:
    """Finds and sorts candidate nodes for y1 in the alternate_step.

    Args:
        t1 (int): The base node of the search.
        t2 (int): The node following t1 in the tour.
        D (np.ndarray): The distance matrix.
        neigh (list[list[int]]): The neighbor lists.
        tour (Tour): The current tour object.

    Returns:
        list[tuple[float, int, int]]: A list of sorted candidates. Each tuple
            contains (sort_metric, y1_candidate, t3_node).
    """
    candidates = []
    for y1_candidate in neigh[t2]:
        if y1_candidate in (t1, t2):
            continue
        gain_G1 = D[t1, t2] - D[t2, y1_candidate]
        if gain_G1 <= FLOAT_COMPARISON_TOLERANCE:
            continue
        t3 = tour.prev(y1_candidate)
        sort_metric = D[t3, y1_candidate] - D[t2, y1_candidate]
        candidates.append((sort_metric, y1_candidate, t3))
    candidates.sort(reverse=True)
    return candidates


def _find_y2_candidates(t1: int, t2: int, chosen_y1: int, t4: int, D: np.ndarray, neigh: list[list[int]], tour: Tour) -> list[tuple[float, int, int]]:
    """Finds and sorts candidate nodes for y2 in the alternate_step.

    Args:
        t1 (int): The base node of the search.
        t2 (int): The node following t1 in the tour.
        chosen_y1 (int): The selected y1 candidate from the previous stage.
        t4 (int): The node following chosen_y1 in the tour.
        D (np.ndarray): The distance matrix.
        neigh (list[list[int]]): The neighbor lists.
        tour (Tour): The current tour object.

    Returns:
        list[tuple[float, int, int]]: A list of sorted candidates for y2.
            Each tuple contains (sort_metric, y2_candidate, t6_of_y2_candidate).
    """
    candidates = []
    for y2_candidate in neigh[t4]:
        if y2_candidate in (t1, t2, chosen_y1):
            continue
        t6_of_y2_candidate = tour.next(y2_candidate)
        sort_metric = D[t6_of_y2_candidate, y2_candidate] - D[t4, y2_candidate]
        candidates.append((sort_metric, y2_candidate, t6_of_y2_candidate))
    candidates.sort(reverse=True)
    return candidates


def _find_y3_candidates(t1: int, t2: int, chosen_y1: int, t4: int, chosen_y2: int, chosen_t6: int, D: np.ndarray, neigh: list[list[int]], tour: Tour) -> list[tuple[float, int, int]]:
    """Finds and sorts candidate nodes for y3 in the alternate_step.

    Args:
        t1 (int): The base node of the search.
        t2 (int): The node following t1 in the tour.
        chosen_y1 (int): The selected y1 candidate.
        t4 (int): The node following chosen_y1.
        chosen_y2 (int): The selected y2 candidate.
        chosen_t6 (int): The node following chosen_y2.
        D (np.ndarray): The distance matrix.
        neigh (list[list[int]]): The neighbor lists.
        tour (Tour): The current tour object.

    Returns:
        list[tuple[float, int, int]]: A list of sorted candidates for y3.
            Each tuple contains (sort_metric, y3_candidate, node_after_y3).
    """
    candidates = []
    for y3_candidate in neigh[chosen_t6]:
        if y3_candidate in (t1, t2, chosen_y1, t4, chosen_y2):
            continue
        node_after_y3 = tour.next(y3_candidate)
        sort_metric = (D[node_after_y3, y3_candidate] - D[chosen_t6, y3_candidate])
        candidates.append((sort_metric, y3_candidate, node_after_y3))
    candidates.sort(reverse=True)
    return candidates


def alternate_step(
    base_node: int, tour: Tour, D: np.ndarray, neigh: list[list[int]],
    deadline: float
) -> tuple[bool, list[tuple[int, int]] | None]:
    """Alternative first step of LK (Algorithm 15.2, Applegate et al.).

    Explores specific 3-opt or 5-opt moves for additional breadth.

    Args:
        base_node (int): Current base vertex (t1) for the LK move.
        tour (Tour): The current Tour object.
        D (np.ndarray): Distance matrix.
        neigh (list[list[int]]): Neighbor lists for candidate selection.
        deadline (float): Time limit for the search.

    Returns:
        tuple[bool, list[tuple[int, int]] | None]: A tuple containing:
            - True if a potentially improving sequence is identified, False otherwise.
            - The flip sequence, or None if no sequence found.
    """
    if time.time() >= deadline:
        return False, None

    t1 = base_node
    t2 = tour.next(t1)

    y1_candidates = _find_y1_candidates(t1, t2, D, neigh, tour)

    for _, chosen_y1, _ in y1_candidates[:LK_CONFIG["BREADTH_A"]]:
        if time.time() >= deadline:
            return False, None
        t4 = tour.next(chosen_y1)

        # --- Stage 2: Find candidate y2 ---
        candidates_for_y2 = _find_y2_candidates(t1, t2, chosen_y1, t4, D, neigh, tour)

        for _, chosen_y2, chosen_t6 in candidates_for_y2[:LK_CONFIG["BREADTH_B"]]:
            if time.time() >= deadline:
                return False, None
            # Check for a specific 3-opt move (Type R in Applegate et al. Fig 15.4)
            if tour.sequence(t2, chosen_y2, chosen_y1):
                return True, [(t2, chosen_y2), (chosen_y2, chosen_y1)]

            # --- Stage 3: Find candidate y3 for a 5-opt move (Type Q in Applegate et al.) ---
            candidates_for_y3 = _find_y3_candidates(t1, t2, chosen_y1, t4, chosen_y2, chosen_t6, D, neigh, tour)

            for _, chosen_y3, chosen_node_after_y3 in candidates_for_y3[:LK_CONFIG["BREADTH_D"]]:
                if time.time() >= deadline:
                    return False, None
                # Identified a specific 5-opt move.
                return True, [(t2, chosen_y3), (chosen_y3, chosen_y1), (t4, chosen_node_after_y3)]
    return False, None


def lk_search(start_node_for_search: int, current_tour_obj: Tour,
              D: np.ndarray, neigh: list[list[int]],
              deadline: float) -> list[tuple[int, int]] | None:
    """Single Lin-Kernighan search pass (Algorithm 15.3, Applegate et al.).

    Tries `step` then `alternate_step` to find an improving flip sequence.

    Args:
        start_node_for_search (int): Vertex to initiate search from (t1).
        current_tour_obj (Tour): Current tour.
        D (np.ndarray): Distance matrix.
        neigh (list[list[int]]): Neighbor lists.
        deadline (float): Time limit.

    Returns:
        list[tuple[int, int]] | None: List of (start, end) flips if improvement
            found, None otherwise.
    """

    if time.time() >= deadline:
        return None

    # Attempt 1: Standard Recursive Step Search on a copy of the tour
    search_tour_copy = Tour(current_tour_obj.get_tour(), D)
    cost_at_search_start = search_tour_copy.cost
    assert cost_at_search_start is not None, "Tour cost must be initialized."

    ctx = SearchContext(D=D, neigh=neigh, start_cost=cost_at_search_start,
                        best_cost=cost_at_search_start, deadline=deadline)

    found_step, seq_step = step(
        level=1, delta=0.0, base=start_node_for_search,
        tour=search_tour_copy, ctx=ctx, flip_seq=[]
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


def _apply_and_update_best_tour(
    improving_sequence: list[tuple[int, int]],
    tour_order_before_lk: list[int],
    cost_before_lk: float,
    D: np.ndarray
) -> tuple[Tour, float] | None:
    """Applies a flip sequence and returns a new tour if it's an improvement.

    Args:
        improving_sequence (list[tuple[int, int]]): The sequence of flips to apply.
        tour_order_before_lk (list[int]): The tour order before the LK search.
        cost_before_lk (float): The tour cost before the LK search.
        D (np.ndarray): The distance matrix.

    Returns:
        tuple[Tour, float] | None: A tuple containing the new Tour and its cost
            if the new tour is an improvement, None otherwise.
    """
    candidate_tour = Tour(tour_order_before_lk, D)
    for x_flip, y_flip in improving_sequence:
        candidate_tour.flip_and_update_cost(x_flip, y_flip, D)

    new_cost = candidate_tour.cost
    assert new_cost is not None

    if new_cost < cost_before_lk - FLOAT_COMPARISON_TOLERANCE:
        return candidate_tour, new_cost
    return None


def lin_kernighan(coords: np.ndarray, init: list[int], D: np.ndarray,
                  neigh: list[list[int]],
                  deadline: float) -> tuple[Tour, float]:
    """Main Lin-Kernighan heuristic (Algorithm 15.4, Applegate et al.).

    Iteratively applies `lk_search` from marked nodes until no improvement
    or the time limit is reached.

    Args:
        coords (np.ndarray): Vertex coordinates.
        init (list[int]): Initial tour permutation.
        D (np.ndarray): Distance matrix.
        neigh (list[list[int]]): Neighbor lists.
        deadline (float): Time limit.

    Returns:
        tuple[Tour, float]: A tuple containing:
            - The best Tour object found.
            - The cost of the best tour.
    """
    n = len(coords)
    best_tour = Tour(init, D)
    best_cost = best_tour.cost
    assert best_cost is not None, "Initial tour cost missing."

    marked_nodes = set(range(n))  # Nodes to start lk_search from

    while marked_nodes:
        if time.time() >= deadline:
            break
        start_node = marked_nodes.pop()

        # lk_search attempts to find an improving sequence for the current best tour.
        # It operates on a temporary copy, so `best_tour` is not modified.
        improving_sequence = lk_search(
            start_node, best_tour, D, neigh, deadline
        )

        if improving_sequence:
            # An improving sequence was found. Apply it to the state *before* the search.
            update_result = _apply_and_update_best_tour(
                improving_sequence, best_tour.get_tour(), best_cost, D
            )
            if update_result:
                best_tour, best_cost = update_result
                marked_nodes = set(range(n))  # Improvement found, re-mark all

    return best_tour, best_cost


def double_bridge(order: list[int]) -> list[int]:
    """Applies a double-bridge 4-opt move to perturb the tour.

    Args:
        order (list[int]): Current tour order.

    Returns:
        list[int]: New perturbed tour order. Returns original if n <= 4.
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


def _perform_kick_and_lk_run(
    current_best_tour: Tour,
    coords: np.ndarray,
    D: np.ndarray,
    neigh: list[list[int]],
    deadline: float
) -> tuple[Tour, float] | None:
    """Performs a double-bridge kick and runs LK on the new tour.

    Args:
        current_best_tour (Tour): The current best tour to perturb.
        coords (np.ndarray): Vertex coordinates.
        D (np.ndarray): The distance matrix.
        neigh (list[list[int]]): The neighbor lists.
        deadline (float): The time limit for the search.

    Returns:
        tuple[Tour, float] | None: A tuple containing the improved Tour and its
            cost if an improved tour is found, None otherwise.
    """
    kicked_order = double_bridge(current_best_tour.get_tour())
    candidate_tour, candidate_cost = lin_kernighan(
        coords, kicked_order, D, neigh, deadline
    )
    # The cost of the current best tour must be initialized at this point.
    assert current_best_tour.cost is not None, "Current best tour cost cannot be None."
    if candidate_cost < current_best_tour.cost - FLOAT_COMPARISON_TOLERANCE:
        return candidate_tour, candidate_cost
    return None


def _check_for_optimality(cost: float, optimal_len: float | None) -> bool:
    """Checks if the current cost is close enough to the known optimum.

    Args:
        cost (float): The tour cost to check.
        optimal_len (float | None): The known optimal length.

    Returns:
        bool: True if the cost is close to the optimal length.
    """
    if optimal_len is None:
        return False
    # Use relative and absolute tolerance for floating-point comparison
    return math.isclose(cost, optimal_len, rel_tol=1e-7, abs_tol=FLOAT_COMPARISON_TOLERANCE * 10)


def chained_lin_kernighan(
    coords: np.ndarray, initial_tour_order: list[int],
    known_optimal_length: float | None = None,
    time_limit_seconds: float | None = None
) -> tuple[list[int], float]:
    """Chained Lin-Kernighan metaheuristic (Algorithm 15.5, Applegate et al.).

    Repeatedly applies LK, with double-bridge kicks to escape local optima.

    Args:
        coords (np.ndarray): Vertex coordinates.
        initial_tour_order (list[int]): Initial tour.
        known_optimal_length (float | None, optional): Known optimal length for
            early stop. Defaults to None.
        time_limit_seconds (float | None, optional): Max run time. Defaults to
            None, which uses the value from LK_CONFIG.

    Returns:
        tuple[list[int], float]: A tuple containing:
            - The best tour order found.
            - The cost of the best tour.
    """
    effective_time_limit = (time_limit_seconds
                            if time_limit_seconds is not None
                            else LK_CONFIG["TIME_LIMIT"])
    deadline = time.time() + effective_time_limit

    distance_matrix = build_distance_matrix(coords)
    neighbor_list = delaunay_neighbors(coords)

    # Initial LK run
    best_tour, best_cost = lin_kernighan(
        coords, initial_tour_order, distance_matrix, neighbor_list, deadline
    )

    if _check_for_optimality(best_cost, known_optimal_length):
        return best_tour.get_tour(), best_cost

    # Main chain loop: kick and re-run LK
    while time.time() < deadline:
        update_result = _perform_kick_and_lk_run(
            best_tour, coords, distance_matrix, neighbor_list, deadline
        )

        if update_result:
            best_tour, best_cost = update_result
            if _check_for_optimality(best_cost, known_optimal_length):
                break  # Optimum found

    # Final cost consistency check before returning
    final_tour_order = best_tour.get_tour()
    final_recomputed_cost = 0.0
    if best_tour.n > 0:  # Ensure tour is not empty
        for i in range(best_tour.n):
            node1 = final_tour_order[i]
            node2 = final_tour_order[(i + 1) % best_tour.n]
            final_recomputed_cost += float(distance_matrix[node1, node2])
    # best_tour.cost = final_recomputed_cost  # Update object if needed
    return final_tour_order, final_recomputed_cost
