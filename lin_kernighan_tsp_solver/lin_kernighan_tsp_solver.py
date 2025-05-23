"""
Lin-Kernighan Heuristic for the Traveling Salesperson Problem (TSP).

This script implements the Lin-Kernighan (LK) heuristic, a powerful local search
algorithm for finding high-quality approximate solutions to the TSP. The implementation
is based on the descriptions and algorithms presented in _The Traveling Salesman Problem:
A Computational Study_"_ by Applegate, Bixby, Chvátal & Cook, and "An Effective Heuristic
Algorithm for the Traveling-Salesman Problem" by Lin & Kernighan.

The script processes TSP instances from the TSPLIB format. It computes heuristic solutions
using a chained version of the LK algorithm. If a corresponding optimal tour file
(e.g., problem_name.opt.tour) is found, the script compares the heuristic solution
against the known optimal solution and calculates the percentage gap. If no optimal
tour file is available, the instance is still processed, but no gap calculation is
performed for it. The script displays a summary table and plots of the tours.

Usage:
  1. Ensure all dependencies are installed:
     pip install numpy matplotlib scipy

  2. Place your TSPLIB .tsp files in a designated folder.
     Optionally, place corresponding .opt.tour files (if available) in the same
       folder.

  3. Update the `TSP_FOLDER_PATH` constant at the top of this script
     (in the "--- Constants ---" section) to point to your TSPLIB folder.

  4. Run the script from the command line:
     python lin_kernighan_tsp_solver.py

The script will then process each EUC_2D TSP instance found. It prints progress
and results to the console. For instances with an optimal tour, the gap is shown.
For instances without an optimal tour, nothing is displayed for optimal length and gap.
Finally, a plot of all processed tours is displayed (showing both optimal and heuristic
tours if the optimal is available, otherwise just the heuristic tour). Configuration
parameters for the LK algorithm can be adjusted in the `LK_CONFIG` dictionary
within this script.
"""
import time
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.lines import Line2D

# --- Constants ---
# Path to the folder containing TSPLIB .tsp files and optional .opt.tour files
TSP_FOLDER_PATH = Path('../verifications/tsplib95')

# Tolerance for floating point comparisons
FLOAT_COMPARISON_TOLERANCE = 1e-12

# Maximum number of subplots in the tour visualization
MAX_SUBPLOTS_IN_PLOT = 25

# --- Configuration Parameters ---
# This dictionary holds parameters that control the behavior of the Lin-Kernighan heuristic,
# such as search depth, breadth at various stages, and time limits.
LK_CONFIG = {
    "MAX_LEVEL": 12,  # Max recursion depth for k-opt moves in step()
    "BREADTH": [5, 5] + [1] * 20,  # Search breadth at each level in step()
    "BREADTH_A": 5,  # Search breadth for t3 in alternate_step()
    "BREADTH_B": 5,  # Search breadth for t5 in alternate_step()
    "BREADTH_D": 1,  # Search breadth for t7 in alternate_step()
    "TIME_LIMIT": 1.0,  # Default time limit for chained_lin_kernighan in seconds
}


class Tour:
    """
    Abstract tour data structure for the Lin-Kernighan heuristic.

    Maintains a permutation of the vertices and supports efficient flip operations,
    as well as access to next/previous neighbors. This structure follows the
    requirements set in Applegate et al., Section 15.2.

    Attributes:
        n (int): Number of vertices in the tour.
        order (np.ndarray): Current permutation representing the tour.
        pos (np.ndarray): Inverse mapping: pos[v] gives index of vertex v in order[].
        cost (float or None): Cost of the tour under current permutation, if initialized.
    """

    def __init__(self, order: Iterable[int], D: Optional[np.ndarray] = None) -> None:
        """
        Initializes the tour data structure from a given vertex ordering.

        Args:
            order (Iterable[int]): Sequence of vertices defining the tour.
            D (Optional[np.ndarray]): Distance/cost matrix to initialize tour cost.
        """
        self.n: int = len(
            list(order))  # Ensure order can be sized, convert to list if iterator
        self.order: np.ndarray = np.array(list(order), dtype=np.int32)
        self.pos: np.ndarray = np.empty(self.n, dtype=np.int32)
        # Renamed v to v_node for clarity
        for i, v_node in enumerate(self.order):
            self.pos[v_node] = i
        self.cost: Optional[float] = None
        if D is not None:
            self.init_cost(D)

    def init_cost(self, D: np.ndarray) -> None:
        """
        Computes and stores the total tour cost from the cost matrix D.

        Args:
            D (np.ndarray): Distance/cost matrix.
        """
        c = 0.0
        for i in range(self.n):
            a = self.order[i]
            b = self.order[(i + 1) % self.n]
            c += float(D[a, b])  # Explicit cast
        self.cost = c

    def next(self, v: int) -> int:
        """
        Returns the vertex immediately after v in the current tour.

        Args:
            v (int): Vertex label.

        Returns:
            int: Next vertex after v.
        """
        return int(self.order[(self.pos[v] + 1) % self.n])

    def prev(self, v: int) -> int:
        """
        Returns the vertex immediately before v in the current tour.

        Args:
            v (int): Vertex label.

        Returns:
            int: Previous vertex before v.
        """
        return int(self.order[(self.pos[v] - 1) % self.n])

    def sequence(self, a: int, b: int, c: int) -> bool:
        """
        Determines if vertex b appears between a and c (inclusive of endpoints)
        on the current tour, following the orientation.

        Args:
            a (int): Start vertex.
            b (int): Test vertex.
            c (int): End vertex.

        Returns:
            bool: True if b is in the segment from a to c, otherwise False.
        """
        ia, ib, ic = self.pos[a], self.pos[b], self.pos[c]
        if ia <= ic:
            return ia < ib <= ic
        return ia < ib or ib <= ic

    def flip(self, a: int, b: int) -> None:
        """
        Inverts (reverses) the segment of the tour from a to b (inclusive).

        Args:
            a (int): Start vertex.
            b (int): End vertex.
        """
        ia, ib = self.pos[a], self.pos[b]
        indices = []
        i = ia
        while True:
            indices.append(i)
            if i == ib:
                break
            i = (i + 1) % self.n
        segment = [self.order[i] for i in indices][::-1]
        for i, idx in enumerate(indices):
            self.order[idx] = segment[i]
        for i, v in enumerate(self.order):
            self.pos[v] = i

    def get_tour(self) -> List[int]:
        """
        Returns the current tour as a list, starting from vertex 0.

        Returns:
            list: Ordered list of vertex indices representing the tour.
        """
        zero_pos = self.pos[0]
        if zero_pos == 0:
            return list(self.order)
        else:
            return list(np.concatenate((self.order[zero_pos:], self.order[:zero_pos])))

    def flip_and_update_cost(self, a: int, b: int, D: np.ndarray) -> float:
        """
        Performs flip(a, b) and efficiently updates the tour cost using the change in edge weights.

        Args:
            a (int): Start vertex.
            b (int): End vertex.
            D (np.ndarray): Distance/cost matrix.

        Returns:
            float: The cost delta resulting from the flip.
        """
        pa = self.prev(a)
        nb = self.next(b)
        removed = D[pa, a] + D[b, nb]
        added = D[pa, b] + D[a, nb]
        delta = added - removed
        self.flip(a, b)
        self.cost += delta
        return delta


def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Computes the full distance (cost) matrix for the given coordinates.

    Args:
        coords (np.ndarray): Array of vertex coordinates.

    Returns:
        np.ndarray: Symmetric matrix of pairwise Euclidean distances.
    """
    return np.linalg.norm(coords[:, None] - coords[None, :], axis=2)


def delaunay_neighbors(coords: np.ndarray) -> List[List[int]]:
    """
    Builds a list of neighbors for each vertex using Delaunay triangulation.
    This is used to restrict candidate moves as in the LK algorithm.

    Args:
        coords (np.ndarray): Array of vertex coordinates.

    Returns:
        list of list: For each vertex, a list of neighbor vertex indices.
    """
    tri = Delaunay(coords)
    neigh = {i: set() for i in range(len(coords))}
    for simplex in tri.simplices:
        for u, v in combinations(simplex, 2):
            neigh[u].add(v)
            neigh[v].add(u)
    return [sorted(neigh[i]) for i in range(len(coords))]


def step(level: int, delta: float, base: int, tour: Tour, D: np.ndarray,
         neigh: List[List[int]], flip_seq: List[Tuple[int, int]],
         start_cost: float, best_cost: float, deadline: float) -> Tuple[bool, Optional[List[Tuple[int, int]]]]:
    """
    Recursively explores possible sequences of flips to find improved tours,
    using both standard and Mak-Morton moves as per Algorithm 15.1 in the book.

    Args:
        level (int): Current recursion level (corresponds to k in variable k-opt).
        delta (float): Current accumulated gain (total reduction in tour cost).
        base (int): The current base vertex for moves.
        tour (Tour): The current tour object.
        D (np.ndarray): Distance/cost matrix.
        neigh (list): List of neighbor lists for each vertex.
        flip_seq (list): Current sequence of flip operations.
        start_cost (float): Cost of the original tour at start of search.
        best_cost (float): Best cost found so far.
        deadline (float): Timestamp for time limit.

    Returns:
        (bool, list): (True, flip_seq) if an improved tour is found, else (False, None).
    """
    if time.time() >= deadline:
        return False, None
    b = LK_CONFIG["BREADTH"][min(level - 1, len(LK_CONFIG["BREADTH"]) - 1)]
    s1 = tour.next(base)
    candidates = []

    # Standard flips (u-steps):
    for a_candidate_node in neigh[s1]:
        is_invalid_node = a_candidate_node in (base, s1)

        # Gain from breaking edge (base,s1) and making edge (s1,a_candidate_node)
        gain_first_exchange = D[base, s1] - D[s1, a_candidate_node]

        # Pruning: if the first exchange doesn't offer positive gain, skip.
        # (Helsgaun's paper suggests G_i > 0, so D[t_2i-1, t_2i] - D[t_2i, x_i] > 0)
        if is_invalid_node or gain_first_exchange <= FLOAT_COMPARISON_TOLERANCE:  # Check for strictly positive
            continue

        # This is t_2i+1 in some notations
        probe_node = tour.prev(a_candidate_node)

        # Gain from breaking edge (probe_node, base) and making edge (probe_node, a_candidate_node)
        # This completes a 2-opt or is part of a 3-opt.
        gain_second_exchange = D[probe_node, a_candidate_node] - D[probe_node, base]

        total_gain_for_this_move = gain_first_exchange + gain_second_exchange

        # Pruning condition: current accumulated gain (delta) + gain from this first new edge must be positive.
        # This ensures that the sequence of choices so far maintains a positive cumulative gain.
        if delta + gain_first_exchange > FLOAT_COMPARISON_TOLERANCE:  # Check for strictly positive
            candidates.append(('flip', a_candidate_node, probe_node, total_gain_for_this_move))

    # Mak-Morton flips:
    for a_candidate_node in neigh[base]:  # Renamed 'a' to 'a_candidate_node'
        if a_candidate_node in (tour.next(base), tour.prev(base), base):
            continue

        # Gain calculation for Mak-Morton move
        gain_mak_morton = (D[base, s1] - D[base, a_candidate_node]) + \
                          (D[a_candidate_node, tour.next(a_candidate_node)
                             ] - D[tour.next(a_candidate_node), s1])

        # Pruning condition similar to standard flips
        # Check for strictly positive
        if delta + (D[base, s1] - D[base, a_candidate_node]) > FLOAT_COMPARISON_TOLERANCE:
            candidates.append(
                ('makmorton', a_candidate_node, None, gain_mak_morton))

    candidates.sort(key=lambda x: -x[3])
    count = 0
    for typ, a, probe, g in candidates:
        if time.time() >= deadline or count >= b:  # Enforce time and search breadth limits
            break
        new_delta = delta + g
        if typ == 'flip':
            x, y = s1, probe
            tour.flip(x, y)
            flip_seq.append((x, y))
            if start_cost - new_delta < best_cost:
                return True, flip_seq.copy()
            if level < LK_CONFIG["MAX_LEVEL"]:
                ok, seq = step(level + 1, new_delta, base, tour, D,
                               neigh, flip_seq, start_cost, best_cost, deadline)
                if ok:
                    return True, seq
            tour.flip(y, x)  # Backtrack: undo the flip
            flip_seq.pop()
        else:  # 'makmorton'
            x, y = tour.next(a), base
            tour.flip(x, y)
            flip_seq.append((x, y))
            if start_cost - new_delta < best_cost:
                return True, flip_seq.copy()
            new_base = tour.next(a)
            if level < LK_CONFIG["MAX_LEVEL"]:
                ok, seq = step(level + 1, new_delta, new_base, tour,
                               D, neigh, flip_seq, start_cost, best_cost, deadline)
                if ok:
                    return True, seq
            tour.flip(y, x)
            flip_seq.pop()
        count += 1
    return False, None


def alternate_step(base_node: int, tour: Tour, D: np.ndarray, neigh: List[List[int]],
                   deadline: float) -> Tuple[bool, Optional[List[Tuple[int, int]]]]:
    """
    Implements the alternative first step of LK (similar to Algorithm 15.2 in
    Applegate et al.), providing extra breadth in the initial search for specific
    3-opt or 5-opt moves.

    Notation guide (approximate mapping to Applegate et al. for context):
    - base_node (int): The current base vertex (t1).
    - t2 (int): tour.next(base_node).
    - y1, y2, y3 (int): Candidates for nodes to connect to, chosen from neighbors.
    - t3 (int): tour.prev(y1).
    - t4 (int): tour.next(y1).
    - t6 (int): tour.next(y2).
    - node_after_y3 (int): tour.next(y3).

    Args:
        base_node (int): The current base vertex (t1).
        tour (Tour): The tour object.
        D (np.ndarray): Distance/cost matrix.
        neigh (list): List of neighbor lists for each vertex.
        deadline (float): Timestamp for time limit.

    Returns:
        (bool, list): (True, flip_seq) if an improved tour is found, else (False, None).
                      flip_seq is a list of (segment_start, segment_end) tuples for tour.flip().
    """
    if time.time() >= deadline:
        return False, None

    t1 = base_node
    t2 = tour.next(t1)

    # --- Stage 1: Find candidate y1 (originally 'a') ---
    candidates_for_y1 = []
    for y1_candidate in neigh[t2]:
        # Ensure y1_candidate is a valid choice
        if y1_candidate in (t1, t2):
            continue

        # Check G1 = c(t1,t2) - c(t2,y1_candidate). We need G1 > 0.
        gain_G1 = D[t1, t2] - D[t2, y1_candidate]
        if gain_G1 <= FLOAT_COMPARISON_TOLERANCE:  # Must be a strict improvement potential
            continue

        t3 = tour.prev(y1_candidate)
        # Heuristic metric for sorting y1 candidates: D[t3,y1] - D[t2,y1]
        # Prioritizes y1 where t3 is "far" from y1 and t2 is "close" to y1 (if D are distances)
        # or where the edge (t3,y1) is more costly than (t2,y1).
        sort_metric_y1 = D[t3, y1_candidate] - D[t2, y1_candidate]
        candidates_for_y1.append((sort_metric_y1, y1_candidate, t3))

    candidates_for_y1.sort(reverse=True)  # Sort by metric, descending

    for _, y1_chosen, t3_of_y1 in candidates_for_y1[:LK_CONFIG["BREADTH_A"]]:
        if time.time() >= deadline:
            return False, None

        t4 = tour.next(y1_chosen)

        # --- Stage 2: Find candidate y2 (originally 'b') ---
        candidates_for_y2 = []
        for y2_candidate in neigh[t4]:
            if y2_candidate in (t1, t2, y1_chosen):
                continue

            t6 = tour.next(y2_candidate)
            # Heuristic metric for sorting y2 candidates: D[t6,y2] - D[t4,y2]
            sort_metric_y2 = D[t6, y2_candidate] - D[t4, y2_candidate]
            candidates_for_y2.append((sort_metric_y2, y2_candidate, t6))

        candidates_for_y2.sort(reverse=True)

        for _, y2_chosen, t6_of_y2 in candidates_for_y2[:LK_CONFIG["BREADTH_B"]]:
            if time.time() >= deadline:
                return False, None

            # Check for a specific 3-opt move (Type R in Applegate et al.)
            # If t2 -> y2_chosen -> y1_chosen is a segment in the current tour.
            if tour.sequence(t2, y2_chosen, y1_chosen):
                # This 3-opt move breaks (t1,t2), (y1_chosen,t4), (y2_chosen, tour.prev(y2_chosen))
                # and adds (t1,y1_chosen), (t2,y2_chosen), (t4, tour.prev(y2_chosen)).
                # The sequence of flips to achieve this is [(t2, y2_chosen), (y2_chosen, y1_chosen)].
                return True, [(t2, y2_chosen), (y2_chosen, y1_chosen)]

            # --- Stage 3: Find candidate y3 (originally 'd') for a 5-opt move ---
            candidates_for_y3 = []
            # y3_candidate is chosen from neighbors of t6_of_y2
            for y3_candidate in neigh[t6_of_y2]:
                if y3_candidate in (t1, t2, y1_chosen, t4, y2_chosen):
                    continue

                node_after_y3 = tour.next(y3_candidate)
                # Heuristic metric for sorting y3 candidates: D[node_after_y3,y3] - D[t6_of_y2,y3]
                sort_metric_y3 = D[node_after_y3, y3_candidate] - D[t6_of_y2, y3_candidate]
                candidates_for_y3.append((sort_metric_y3, y3_candidate, node_after_y3))

            candidates_for_y3.sort(reverse=True)

            for _, y3_chosen, node_after_y3_chosen in candidates_for_y3[:LK_CONFIG["BREADTH_D"]]:
                if time.time() >= deadline:
                    return False, None

                # This is a specific 5-opt move.
                # The sequence of flips to achieve this is:
                # [(t2, y3_chosen), (y3_chosen, y1_chosen), (t4, node_after_y3_chosen)]
                return True, [(t2, y3_chosen), (y3_chosen, y1_chosen), (t4, node_after_y3_chosen)]

    return False, None


def lk_search(start_node_for_search: int, current_tour_obj: Tour, D: np.ndarray,
              neigh: List[List[int]], deadline: float) -> Optional[List[Tuple[int, int]]]:
    """
    Top-level Lin-Kernighan search (Algorithm 15.3 in Applegate et al.).
    Attempts to find an improving flip sequence starting at `start_node_for_search`.

    This function first attempts the standard recursive `step` search.
    If no improvement is found, it then tries the `alternate_step` search,
    which looks for specific 3-opt or 5-opt moves.

    Args:
        start_node_for_search (int): The vertex from which to initiate the search.
        current_tour_obj (Tour): The current tour object. The `step` search will operate
                                 on a copy, while `alternate_step` may operate on this
                                 instance directly (as per its design).
        D (np.ndarray): Distance/cost matrix.
        neigh (list): List of neighbor lists for each vertex.
        deadline (float): Timestamp for time limit.

    Returns:
        Optional[List[Tuple[int, int]]]: A list of (segment_start, segment_end)
                                         tuples representing the flip sequence if an
                                         improved tour is found, else None.
    """
    if time.time() >= deadline:
        return None

    # --- Attempt 1: Standard Recursive Step Search ---
    # Create a temporary copy of the tour for the 'step' search to explore flips
    # without modifying the 'current_tour_obj' prematurely.
    search_tour_copy = Tour(current_tour_obj.get_tour(), D)
    cost_at_search_start = search_tour_copy.cost
    assert cost_at_search_start is not None, "Tour cost must be initialized for search_tour_copy."

    # Call the recursive 'step' procedure.
    # 'delta' starts at 0, 'level' at 1.
    # 'best_cost' for 'step' is initialized to 'cost_at_search_start', meaning 'step'
    # aims to find a sequence of flips that results in a tour cost strictly less than this.
    found_improvement_step, improving_sequence_step = step(
        level=1,
        delta=0.0,
        base=start_node_for_search,
        tour=search_tour_copy,  # Operates on the copy
        D=D,
        neigh=neigh,
        flip_seq=[],  # Initial empty flip sequence
        start_cost=cost_at_search_start,  # Cost of the tour before this 'step' call
        best_cost=cost_at_search_start,  # Target to beat
        deadline=deadline
    )

    if found_improvement_step and improving_sequence_step:
        # If 'step' found an improving sequence, return it.
        return improving_sequence_step

    if time.time() >= deadline:  # Check time limit again before alternate_step
        return None

    # --- Attempt 2: Alternate Step Search ---
    # If the standard 'step' search did not find an improvement,
    # try the 'alternate_step' search.
    # 'alternate_step' looks for specific 3-opt or 5-opt moves from the
    # state of 'current_tour_obj'.
    found_improvement_alt, improving_sequence_alt = alternate_step(
        base_node=start_node_for_search,
        tour=current_tour_obj,  # Operates on the original tour object passed to lk_search
        D=D,
        neigh=neigh,
        deadline=deadline
    )

    if found_improvement_alt and improving_sequence_alt:
        return improving_sequence_alt

    # No improvement found by either method
    return None


def lin_kernighan(coords: np.ndarray, init: List[int], D: np.ndarray,
                  neigh: List[List[int]], deadline: float) -> Tuple[Tour, float]:
    """
    The main Lin-Kernighan heuristic (Algorithm 15.4 in Applegate et al.).

    Iteratively applies lk_search() to all vertices. If an improving sequence
    of flips is found, it's applied, and all vertices are marked for re-checking.
    The process continues until no marked vertices remain or the time limit is reached.

    Args:
        coords (np.ndarray): Vertex coordinates. Used to determine 'n'.
        init (list): Initial tour permutation.
        D (np.ndarray): Distance/cost matrix.
        neigh (list): List of neighbor lists for each vertex.
        deadline (float): Timestamp for time limit.

    Returns:
        (Tour, float): The best tour object found and its cost.
    """
    n = len(coords)
    current_best_tour_obj = Tour(init, D)
    current_best_tour_cost = current_best_tour_obj.cost
    assert current_best_tour_cost is not None, "Initial tour cost should be set."

    # 'marked' contains nodes from which an lk_search might lead to an improvement.
    # Initially, all nodes are marked.
    marked_nodes = set(range(n))

    while marked_nodes:  # Continue as long as there are nodes to explore for improvements
        if time.time() >= deadline:
            break

        # Select a node from which to start the search for an improving sequence.
        # The order of selection from 'marked_nodes' can vary; here, pop() is used.
        start_node_for_lk = marked_nodes.pop()

        # Store the state of the tour before calling lk_search.
        # lk_search might explore modifications, but we only commit if an actual
        # improvement is found relative to this pre-search state.
        tour_order_before_lk = current_best_tour_obj.get_tour()
        cost_before_lk = current_best_tour_obj.cost
        assert cost_before_lk is not None, "Cost before lk_search must be defined."

        # Call lk_search. Note: lk_search itself uses a copy for its 'step' part,
        # and 'alternate_step' operates on the passed tour.
        # The 'current_best_tour_obj' is passed here, but its state is preserved
        # by 'tour_order_before_lk' and 'cost_before_lk' for comparison.
        improving_sequence = lk_search(
            start_node_for_lk, current_best_tour_obj, D, neigh, deadline)

        if improving_sequence:
            # An improving sequence was found by lk_search.
            # Apply this sequence to a fresh tour object based on the state *before* lk_search.
            candidate_tour_obj = Tour(tour_order_before_lk, D)
            for x_flip, y_flip in improving_sequence:
                candidate_tour_obj.flip_and_update_cost(x_flip, y_flip, D)

            cost_of_candidate_tour = candidate_tour_obj.cost
            assert cost_of_candidate_tour is not None, "Cost of candidate tour must be defined."

            # Check if this candidate tour is genuinely better than the tour before lk_search.
            if cost_of_candidate_tour < cost_before_lk - FLOAT_COMPARISON_TOLERANCE:
                # Yes, an improvement was made. Update the current best tour and cost.
                current_best_tour_obj = candidate_tour_obj
                current_best_tour_cost = cost_of_candidate_tour
                # Since an improvement was made, all nodes are marked again for potential further improvements.
                marked_nodes = set(range(n))
            # else: The sequence returned by lk_search, when applied, did not result in a
            #       better tour than 'cost_before_lk'. No change to 'current_best_tour_obj'
            #       or 'marked_nodes' (beyond the pop() at the start of the loop).
            #       The 'current_best_tour_obj' effectively remains as it was before this lk_search call.

        # else: No improving sequence was found by lk_search starting from 'start_node_for_lk'.
        #       'current_best_tour_obj' and 'marked_nodes' remain as they are (except for the pop).

    # After the loop finishes (no more marked nodes or time limit reached),
    # return the best tour found.
    return current_best_tour_obj, current_best_tour_cost


def double_bridge(order: List[int]) -> List[int]:
    """
    Applies the "double-bridge" 4-opt move (see Figure 15.7),
    used for generating kicks in Chained Lin-Kernighan.

    Args:
        order (list): Current tour order.

    Returns:
        list: New tour after applying the double-bridge move.
    """
    n = len(order)
    if n <= 4:
        return list(order)
    # Apply a 4-opt double-bridge move to perturb the tour
    a, b, c, d = sorted(np.random.choice(range(1, n), 4, replace=False))
    s0, s1 = order[:a], order[a:b]
    s2, s3 = order[b:c], order[c:d]
    s4 = order[d:]
    return s0 + s2 + s1 + s3 + s4


def chained_lin_kernighan(coords: np.ndarray, init: List[int],
                          opt_len: Optional[float] = None,
                          time_limit: Optional[float] = None) -> Tuple[List[int], float]:
    """
    Chained Lin-Kernighan metaheuristic (Algorithm 15.5).

    Repeatedly applies Lin-Kernighan with double-bridge kicks
    to escape local minima, stopping either at the time limit or
    as soon as the known optimum is found.

    Args:
        coords (np.ndarray): Vertex coordinates.
        init (list): Initial tour order.
        opt_len (float, optional): Known optimal tour length (for early stopping).
        time_limit (float, optional): Maximum time (seconds) to run the algorithm.

    Returns:
        (list, float): The best tour found and its length.
    """
    if time_limit is None:
        time_limit = LK_CONFIG["TIME_LIMIT"]
    assert time_limit is not None, "time_limit should be a float at this point."
    t_start = time.time()
    deadline = t_start + time_limit
    D = build_distance_matrix(coords)
    neigh = delaunay_neighbors(coords)
    tour_obj, best_cost = lin_kernighan(coords, init, D, neigh, deadline)
    while time.time() < deadline:
        cand = double_bridge(tour_obj.get_tour())
        t2_obj, l2 = lin_kernighan(coords, cand, D, neigh, deadline)
        if l2 < best_cost:
            tour_obj, best_cost = t2_obj, l2
            # Use named constant
            if opt_len is not None and abs(best_cost - opt_len) < FLOAT_COMPARISON_TOLERANCE:
                # Early exit if optimal solution found
                break
    # Final cost recompute
    true_cost = 0.0
    for i in range(tour_obj.n):
        x = tour_obj.order[i]
        y = tour_obj.order[(i + 1) % tour_obj.n]
        true_cost += float(D[x, y])  # Explicit cast
    tour_obj.cost = true_cost  # Now true_cost is float
    best_cost = true_cost     # And best_cost is float
    return tour_obj.get_tour(), best_cost


def read_opt_tour(path: str) -> Optional[List[int]]:
    """Reads an optimal tour from a .opt.tour file in TSPLIB format.
    Returns None if the file is not found or cannot be parsed.
    """
    tour: List[int] = []
    reading = False
    try:
        with open(path) as f:
            for line in f:
                tok = line.strip()
                if tok.upper().startswith('TOUR_SECTION'):
                    reading = True
                    continue
                if not reading:
                    continue
                for p in tok.split():
                    if p in ('-1', 'EOF'):
                        reading = False
                        break
                    try:
                        idx = int(p)
                    except ValueError:
                        print(
                            f"Warning: Could not parse token '{p}' as integer in {path}")
                        continue
                    if idx > 0:
                        tour.append(idx - 1)
                if not reading:
                    break
        if not tour:  # If tour section was found but no nodes, or section not found
            return None
    except FileNotFoundError:
        # print(f"Info: Optimal tour file not found at {path}")  # Optional: less verbose
        return None
    except Exception as e:
        print(f"Error reading optimal tour file {path}: {e}")
        return None
    return tour


def read_tsp(path: str) -> np.ndarray:
    """
    Reads a TSPLIB formatted TSP file and returns the coordinates as a numpy array.
    Only supports EUC_2D instances.

    Args:
        path (str): Path to the .tsp file

    Returns:
        np.ndarray: Array of shape (n, 2) containing the coordinates
    """
    coords_dict = {}  # {node_id: [x, y]}
    reading_nodes = False

    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Check if we're starting to read node coordinates
                if line.startswith("NODE_COORD_SECTION"):
                    reading_nodes = True
                    continue

                # Stop reading when we reach EOF
                if line == "EOF":
                    break

                # Read node coordinates
                if reading_nodes:
                    parts = line.split()
                    if len(parts) >= 3:  # node_id x y
                        try:
                            node_id = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            coords_dict[node_id] = [x, y]
                        except ValueError:
                            print(f"Warning: Could not parse line: '{line}' in {path}")

        if not coords_dict:
            print(f"Warning: No coordinates found in {path}")
            return np.array([], dtype=float)

        # Sort by node ID to ensure consistent order
        sorted_nodes = sorted(coords_dict.keys())
        coords_list = [coords_dict[node_id] for node_id in sorted_nodes]

        return np.array(coords_list, dtype=float)

    except FileNotFoundError:
        print(f"Error: TSP file not found at {path}")
        raise
    except Exception as e:
        print(f"Error reading TSP file {path}: {e}")
        raise


def process_single_instance(tsp_file_path: str, opt_tour_file_path: str) -> Dict[str, Any]:
    """
    Processes a single TSP instance by loading its data, optionally its optimal tour,
    running the Lin-Kernighan heuristic, and calculating performance statistics.
    """
    problem_name = Path(tsp_file_path).stem  # Use Path.stem for consistency
    print(f"Processing {problem_name} (EUC_2D)...")

    coords = read_tsp(tsp_file_path)
    D_matrix = build_distance_matrix(coords)

    opt_tour_nodes = read_opt_tour(opt_tour_file_path)
    opt_len: Optional[float] = None  # Initialize opt_len as Optional
    gap: Optional[float] = None     # Initialize gap as Optional

    if opt_tour_nodes is not None:  # Check if opt_tour_nodes is a list
        current_opt_len = 0.0
        # These lines (642-644 in your error report) are now inside the 'if' block
        for i in range(len(opt_tour_nodes)):
            a = opt_tour_nodes[i]
            b = opt_tour_nodes[(i + 1) % len(opt_tour_nodes)]
            current_opt_len += D_matrix[a, b]
        opt_len = current_opt_len
        print(f"  Optimal length: {opt_len:.2f}")
    else:
        print(f"  Optimal tour not available or not found for {problem_name}.")

    initial_tour = list(range(len(coords)))
    start_time = time.time()

    # Pass opt_len (which might be None) to chained_lin_kernighan
    heuristic_tour, heuristic_len = chained_lin_kernighan(
        coords, initial_tour, opt_len=opt_len
    )
    elapsed_time = time.time() - start_time

    if opt_len is not None and opt_len > FLOAT_COMPARISON_TOLERANCE:
        gap_percentage = 100.0 * (heuristic_len - opt_len) / opt_len
        gap = max(0.0, gap_percentage)
    elif opt_len is not None and abs(opt_len) <= FLOAT_COMPARISON_TOLERANCE:
        gap = float(
            'inf') if heuristic_len > FLOAT_COMPARISON_TOLERANCE else 0.0
    # If opt_len is None, gap remains None

    print(
        f"  Heuristic length: {heuristic_len:.2f}  "
        f"Gap: {gap:.2f}%  Time: {elapsed_time:.2f}s" if gap is not None
        else f"  Heuristic length: {heuristic_len:.2f}  Time: {elapsed_time:.2f}s"
    )

    return {
        'name': problem_name,
        'coords': coords,
        'opt_tour': opt_tour_nodes,
        'heu_tour': heuristic_tour,
        'opt_len': opt_len,
        'heu_len': heuristic_len,
        'gap': gap,
        'time': elapsed_time
    }


def display_summary_table(results_data: List[Dict[str, Any]]) -> None:
    """
    Prints a formatted summary table of the results.
    """
    print("\nConfiguration parameters:")
    print(f"  MAX_LEVEL   = {LK_CONFIG['MAX_LEVEL']}")
    print(f"  BREADTH     = {LK_CONFIG['BREADTH']}")
    print(f"  BREADTH_A   = {LK_CONFIG['BREADTH_A']}")
    print(f"  BREADTH_B   = {LK_CONFIG['BREADTH_B']}")
    print(f"  BREADTH_D   = {LK_CONFIG['BREADTH_D']}")
    print(f"  TIME_LIMIT  = {LK_CONFIG['TIME_LIMIT']:.2f}s\n")

    header = "Instance   OptLen   HeuLen   Gap(%)   Time(s)"
    print(header)
    print("-" * len(header))

    for r_item in results_data:
        opt_len_str = f"{r_item['opt_len']:>8.2f}" if r_item['opt_len'] is not None else "   N/A  "
        gap_str = f"{r_item['gap']:>8.2f}" if r_item['gap'] is not None else "   N/A  "
        print(
            f"{r_item['name']:<10s} {opt_len_str} "
            f"{r_item['heu_len']:>8.2f} {gap_str} "
            f"{r_item['time']:>8.2f}"
        )

    if results_data:
        print("-" * len(header))
        num_total_items = len(results_data)

        # Calculate sums and averages carefully, considering None values
        valid_opt_lens = [r['opt_len']
                          for r in results_data if r['opt_len'] is not None]
        valid_gaps = [r['gap'] for r in results_data if r['gap'] is not None]

        total_opt_len_sum = sum(valid_opt_lens) if valid_opt_lens else None
        # Heu_len should always exist
        total_heu_len_sum = sum(r_item['heu_len'] for r_item in results_data)

        avg_gap_val = sum(valid_gaps) / len(valid_gaps) if valid_gaps else None
        avg_time_val = sum(r_item['time'] for r_item in results_data) / \
            num_total_items if num_total_items > 0 else 0.0

        total_opt_len_str = f"{total_opt_len_sum:>8.2f}" if total_opt_len_sum is not None else "   N/A  "
        avg_gap_str = f"{avg_gap_val:>8.2f}" if avg_gap_val is not None else "   N/A  "

        print(
            f"{'SUMMARY':<10s} {total_opt_len_str} {total_heu_len_sum:>8.2f} "
            f"{avg_gap_str} {avg_time_val:>8.2f}"
        )
    print("Done.")


def plot_all_tours(results_data: List[Dict[str, Any]]) -> None:
    """
    Plots the optimal and heuristic tours for all processed instances.
    Shows subplot borders, hides grid, ticks, and coordinate numbers.
    Subplots are square. A single legend is displayed for the entire figure.
    Limits the number of subplots to a maximum of MAX_SUBPLOTS_IN_PLOT.
    """
    num_results_total = len(results_data)

    if num_results_total == 0:
        print("No results to plot.")
        return

    if num_results_total > MAX_SUBPLOTS_IN_PLOT:  # Use the global constant
        print(
            f"Warning: Plotting only the first {MAX_SUBPLOTS_IN_PLOT} of {num_results_total} results due to limit.")
        results_data_to_plot = results_data[:MAX_SUBPLOTS_IN_PLOT]
    else:
        results_data_to_plot = results_data

    num_results = len(results_data_to_plot)

    import math
    cols = int(math.ceil(math.sqrt(num_results)))
    rows = int(math.ceil(num_results / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(
        4 * cols, 4 * rows), squeeze=False)
    axes_list = axes.flatten()

    plotted_heuristic = False
    plotted_optimal = False

    for i, r_item in enumerate(results_data_to_plot):
        ax = axes_list[i]
        coords = r_item['coords']

        if r_item['heu_tour']:
            heu_plot_tour = r_item['heu_tour'] + [r_item['heu_tour'][0]]
            ax.plot(coords[heu_plot_tour, 0], coords[heu_plot_tour,
                    1], '-', label='Heuristic', zorder=1, color='C0')
            plotted_heuristic = True

        if r_item['opt_tour'] is not None:
            opt_plot_tour = r_item['opt_tour'] + [r_item['opt_tour'][0]]
            ax.plot(coords[opt_plot_tour, 0], coords[opt_plot_tour,
                    1], ':', label='Optimal', zorder=2, color='C1')
            plotted_optimal = True

        title = f"{r_item['name']}"
        if r_item['gap'] is not None:
            title += f" gap={r_item['gap']:.2f}%"
        # If gap is None, nothing is added to the title regarding "Opt N/A"
        ax.set_title(title)

        # Keep borders, remove ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')  # Make subplot square

    # Turn off unused subplots
    for i in range(num_results, len(axes_list)):
        axes_list[i].set_axis_off()  # Turn off unused subplots completely

    plt.tight_layout()

    legend_elements = []
    if plotted_heuristic:
        legend_elements.append(
            Line2D([0], [0], color='C0', linestyle='-', label='Heuristic'))
    if plotted_optimal:
        legend_elements.append(
            Line2D([0], [0], color='C1', linestyle=':', label='Optimal'))

    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper center',
                   ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.0))
        fig.subplots_adjust(top=(0.95 if num_results > cols else 0.90))  # Adjust for legend

    plt.show()


if __name__ == '__main__':
    all_results = []

    # Iterate over Path objects
    for tsp_file_candidate in sorted(TSP_FOLDER_PATH.iterdir()):
        if tsp_file_candidate.suffix.lower() != '.tsp':
            continue

        problem_base_name = tsp_file_candidate.stem

        # Construct path for optional .opt.tour file
        opt_tour_file_path = TSP_FOLDER_PATH / \
            (problem_base_name + '.opt.tour')
        try:
            instance_result = process_single_instance(
                str(tsp_file_candidate), str(opt_tour_file_path)
            )
            if instance_result:  # Ensure instance_result is not None if process_single_instance can return None
                all_results.append(instance_result)
        except Exception as e:  # Catch errors from process_single_instance or file operations
            print(f"Error processing {problem_base_name}: {e}")

    display_summary_table(all_results)
    plot_all_tours(all_results)
