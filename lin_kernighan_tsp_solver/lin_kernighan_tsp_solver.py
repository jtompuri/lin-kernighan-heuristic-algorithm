import os
import time
from itertools import combinations 
from pathlib import Path         

import matplotlib.pyplot as plt
import numpy as np             
import tsplib95                
from scipy.spatial import Delaunay
from typing import List, Dict, Tuple, Optional, Any, Iterable

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

# --- Constants ---
FLOAT_COMPARISON_TOLERANCE = 1e-12 # Tolerance for floating point comparisons


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
        self.n: int = len(list(order)) # Ensure order can be sized, convert to list if iterator
        self.order: np.ndarray = np.array(list(order), dtype=np.int32)
        self.pos: np.ndarray = np.empty(self.n, dtype=np.int32)
        for i, v_node in enumerate(self.order): # Renamed v to v_node for clarity
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
            c += float(D[a, b]) # Explicit cast
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
    for a_candidate_node in neigh[s1]: # Renamed 'a' to 'a_candidate_node' for clarity
        is_invalid_node = a_candidate_node in (base, s1)
        
        # Gain from breaking edge (base,s1) and making edge (s1,a_candidate_node)
        # This is G_i in some notations if this is the i-th edge pair considered.
        gain_first_exchange = D[base, s1] - D[s1, a_candidate_node]
        
        # Pruning: if the first exchange doesn't offer positive gain, skip.
        # (Helsgaun's paper suggests G_i > 0, so D[t_2i-1, t_2i] - D[t_2i, x_i] > 0)
        if is_invalid_node or gain_first_exchange <= FLOAT_COMPARISON_TOLERANCE: # Check for strictly positive
            continue

        probe_node = tour.prev(a_candidate_node) # This is t_2i+1 in some notations
        
        # Gain from breaking edge (probe_node, base) and making edge (probe_node, a_candidate_node)
        # This completes a 2-opt or is part of a 3-opt.
        gain_second_exchange = D[probe_node, a_candidate_node] - D[probe_node, base]
        
        total_gain_for_this_move = gain_first_exchange + gain_second_exchange
        
        # Pruning condition: current accumulated gain (delta) + gain from this first new edge must be positive.
        # This ensures that the sequence of choices so far maintains a positive cumulative gain.
        if delta + gain_first_exchange > FLOAT_COMPARISON_TOLERANCE: # Check for strictly positive
            candidates.append(('flip', a_candidate_node, probe_node, total_gain_for_this_move))

    # Mak-Morton flips:
    for a_candidate_node in neigh[base]: # Renamed 'a' to 'a_candidate_node'
        if a_candidate_node in (tour.next(base), tour.prev(base), base): continue
        
        # Gain calculation for Mak-Morton move
        gain_mak_morton = (D[base, s1] - D[base, a_candidate_node]) + \
                          (D[a_candidate_node, tour.next(a_candidate_node)] - D[tour.next(a_candidate_node), s1])
        
        # Pruning condition similar to standard flips
        if delta + (D[base, s1] - D[base, a_candidate_node]) > FLOAT_COMPARISON_TOLERANCE: # Check for strictly positive
            candidates.append(('makmorton', a_candidate_node, None, gain_mak_morton))

    candidates.sort(key=lambda x: -x[3])
    count = 0
    for typ, a, probe, g in candidates:
        if time.time() >= deadline or count >= b: # Enforce time and search breadth limits
            break
        new_delta = delta + g
        if typ == 'flip':
            x, y = s1, probe
            tour.flip(x, y)
            flip_seq.append((x, y))
            if start_cost - new_delta < best_cost:
                return True, flip_seq.copy()
            if level < LK_CONFIG["MAX_LEVEL"]:
                ok, seq = step(level + 1, new_delta, base, tour, D, neigh, flip_seq, start_cost, best_cost, deadline)
                if ok: return True, seq
            tour.flip(y, x) # Backtrack: undo the flip
            flip_seq.pop()
        else: # 'makmorton'
            x, y = tour.next(a), base
            tour.flip(x, y)
            flip_seq.append((x, y))
            if start_cost - new_delta < best_cost:
                return True, flip_seq.copy()
            new_base = tour.next(a)
            if level < LK_CONFIG["MAX_LEVEL"]:
                ok, seq = step(level + 1, new_delta, new_base, tour, D, neigh, flip_seq, start_cost, best_cost, deadline)
                if ok: return True, seq
            tour.flip(y, x)
            flip_seq.pop()
        count += 1
    return False, None

def alternate_step(base: int, tour: Tour, D: np.ndarray, neigh: List[List[int]], 
                   deadline: float) -> Tuple[bool, Optional[List[Tuple[int, int]]]]:
    """
    Implements the alternative first step of LK (Algorithm 15.2), providing extra
    breadth in the initial search for improved moves, as described in the book.

    Args:
        base (int): The current base vertex.
        tour (Tour): The tour object.
        D (np.ndarray): Distance/cost matrix.
        neigh (list): List of neighbor lists for each vertex.
        deadline (float): Timestamp for time limit.

    Returns:
        (bool, list): (True, flip_seq) if an improved tour is found, else (False, None).
    """
    if time.time() >= deadline:
        return False, None
    s1 = tour.next(base)
    A = []
    for a in neigh[s1]:
        is_invalid_node = a in (base, s1)
        # Check if swapping edge (base,s1) for (s1,a) offers non-positive immediate gain
        offers_no_gain = (D[base, s1] - D[s1, a]) <= 0 
        if is_invalid_node or offers_no_gain:
            continue
        probe = tour.prev(a)
        A.append((D[probe, a] - D[s1, a], a, probe))
    A.sort(reverse=True)
    for _, a, probe in A[:LK_CONFIG["BREADTH_A"]]:
        if time.time() >= deadline: return False, None
        a1 = tour.next(a)
        B = []
        for b in neigh[a1]:
            if b in (base, s1, a): continue
            b1 = tour.next(b)
            B.append((D[b1, b] - D[a1, b], b, b1))
        B.sort(reverse=True)
        for _, b, b1 in B[:LK_CONFIG["BREADTH_B"]]:
            if time.time() >= deadline: return False, None
            if tour.sequence(s1, b, a):
                return True, [(s1, b), (b, a)]
            C = []
            for d in neigh[b1]:
                if d in (base, s1, a, a1, b): continue
                d1 = tour.next(d)
                C.append((D[d1, d] - D[b1, d], d, d1))
            C.sort(reverse=True)
            for _, d, d1 in C[:LK_CONFIG["BREADTH_D"]]:
                if time.time() >= deadline: return False, None
                return True, [(s1, d), (d, a), (a1, d1)]
    return False, None

def lk_search(v: int, tour: Tour, D: np.ndarray, neigh: List[List[int]], 
              deadline: float) -> Optional[List[Tuple[int, int]]]:
    """
    Top-level Lin-Kernighan search (Algorithm 15.3).
    Attempts to find an improving flip sequence starting at vertex v.

    Args:
        v (int): Starting vertex for the search.
        tour (Tour): The current tour object.
        D (np.ndarray): Distance/cost matrix.
        neigh (list): List of neighbor lists for each vertex.
        deadline (float): Timestamp for time limit.

    Returns:
        list or None: List of flip operations if an improved tour is found, else None.
    """
    if time.time() >= deadline:
        return None
    temp_tour = Tour(tour.get_tour(), D)
    current_start_cost = temp_tour.cost
    assert current_start_cost is not None, "Tour cost should be initialized when D is provided."
    ok, seq = step(1, 0, v, temp_tour, D, neigh, [], current_start_cost, current_start_cost, deadline)
    if ok:
        return seq
    if time.time() >= deadline:
        return None
    ok, seq = alternate_step(v, tour, D, neigh, deadline)
    return seq if ok else None

def lin_kernighan(coords: np.ndarray, init: List[int], D: np.ndarray, 
                  neigh: List[List[int]], deadline: float) -> Tuple[Tour, float]:
    """
    The main Lin-Kernighan heuristic (Algorithm 15.4).

    Iteratively applies lk_search() to all vertices, making
    improvements and marking/unmarking as described in the book.

    Args:
        coords (np.ndarray): Vertex coordinates (not directly used, for interface consistency).
        init (list): Initial tour permutation.
        D (np.ndarray): Distance/cost matrix.
        neigh (list): List of neighbor lists for each vertex.
        deadline (float): Timestamp for time limit.

    Returns:
        (Tour, float): Improved tour object and its cost.
    """
    n = len(coords)
    current_tour_obj = Tour(init, D) 
    current_global_best_cost = current_tour_obj.cost
    assert current_global_best_cost is not None, "Initial tour cost should be set."
    
    marked = set(range(n))
    while marked:
        if time.time() >= deadline:
            break
        
        v_start_node = marked.pop()
        
        tour_order_before_lk_call = current_tour_obj.get_tour()
        cost_before_lk_call = current_tour_obj.cost 
        assert cost_before_lk_call is not None

        improving_sequence = lk_search(v_start_node, current_tour_obj, D, neigh, deadline) 
        
        if improving_sequence:
            made_improvement_this_iteration = False
            cost_after_lk_call = current_tour_obj.cost
            assert cost_after_lk_call is not None

            if cost_after_lk_call < cost_before_lk_call - FLOAT_COMPARISON_TOLERANCE:
                current_global_best_cost = cost_after_lk_call
                marked = set(range(n)) 
                made_improvement_this_iteration = True
            else:
                current_tour_obj = Tour(tour_order_before_lk_call, D)
                for x_flip, y_flip in improving_sequence:
                    current_tour_obj.flip_and_update_cost(x_flip, y_flip, D)
                
                new_cost_after_step_flips = current_tour_obj.cost
                assert new_cost_after_step_flips is not None

                if new_cost_after_step_flips < cost_before_lk_call - FLOAT_COMPARISON_TOLERANCE:
                    current_global_best_cost = new_cost_after_step_flips
                    marked = set(range(n)) 
                    made_improvement_this_iteration = True

            if not made_improvement_this_iteration:
                # This is where errors for lines 456, 457 occur.
                # cost_before_lk_call is already asserted not None.
                assert current_tour_obj.cost is not None, "Cost must be defined for this check (improving_sequence path)"
                if abs(current_tour_obj.cost - cost_before_lk_call) > FLOAT_COMPARISON_TOLERANCE and \
                   current_tour_obj.cost >= cost_before_lk_call - FLOAT_COMPARISON_TOLERANCE :
                    current_tour_obj = Tour(tour_order_before_lk_call, D) 
        
        else: # This replaces 'elif' and handles 'not improving_sequence'. Error for line 465 was in the original elif condition.
            # cost_before_lk_call is asserted not None earlier in the loop.
            assert current_tour_obj.cost is not None, "Cost must be defined for this check (no improving_sequence path)"
            if abs(current_tour_obj.cost - cost_before_lk_call) > FLOAT_COMPARISON_TOLERANCE:
                 current_tour_obj = Tour(tour_order_before_lk_call, D)

    return current_tour_obj, current_global_best_cost
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
            if opt_len is not None and abs(best_cost - opt_len) < FLOAT_COMPARISON_TOLERANCE: # Use named constant
                # Early exit if optimal solution found
                break
    # Final cost recompute
    true_cost = 0.0
    for i in range(tour_obj.n):
        x = tour_obj.order[i]
        y = tour_obj.order[(i + 1) % tour_obj.n]
        true_cost += float(D[x, y]) # Explicit cast
    tour_obj.cost = true_cost # Now true_cost is float
    best_cost = true_cost     # And best_cost is float
    return tour_obj.get_tour(), best_cost


def read_tsp(path: str) -> np.ndarray:
    """Reads TSPLIB instance from file and returns the coordinates array."""
    try:
        prob = tsplib95.load(path)
    except FileNotFoundError:
        print(f"Error: TSP file not found at {path}")
        raise # Or return None / handle as appropriate
    except Exception as e: # Catch other tsplib95 load errors
        print(f"Error loading TSP file {path}: {e}")
        raise # Or return None

    coords_map = dict(prob.node_coords) # type: ignore # If prob.node_coords is problematic for Pylance
    nodes = sorted(coords_map.keys())
    
    # Explicitly create a list of lists of floats
    coords_list: List[List[float]] = []
    for i in nodes:
        node_coord_data = coords_map[i]
        coords_list.append([float(val) for val in node_coord_data])
    
    return np.array(coords_list, dtype=float)

def read_opt_tour(path: str) -> List[int]:
    """Reads an optimal tour from a .opt.tour file in TSPLIB format."""
    tour, reading = [], False
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
                    except ValueError: # More specific exception
                        print(f"Warning: Could not parse token '{p}' as integer in {path}")
                        continue
                    if idx > 0:
                        tour.append(idx - 1)
                if not reading:
                    break
    except FileNotFoundError:
        print(f"Error: Optimal tour file not found at {path}")
        raise # Or return empty list / handle as appropriate
    return tour


def process_single_instance(tsp_file_path: str, opt_tour_path: str) -> Dict[str, Any]:
    """
    Processes a single TSP instance by loading its data and optimal tour,
    running the Lin-Kernighan heuristic, and calculating performance statistics.

    Args:
        tsp_file_path (str): The file path to the .tsp problem instance.
        opt_tour_path (str): The file path to the .opt.tour file containing the optimal tour.

    Returns:
        dict: A dictionary containing the results for the instance, including:
            'name' (str): The name of the problem instance.
            'coords' (np.ndarray): The coordinates of the cities.
            'opt_tour' (list): The optimal tour as a list of node indices.
            'heu_tour' (list): The heuristic tour found by LK.
            'opt_len' (float): The length of the optimal tour.
            'heu_len' (float): The length of the heuristic tour.
            'gap' (float): The percentage gap between heuristic and optimal length.
            'time' (float): The time taken by the heuristic in seconds.
    """
    problem_name = os.path.basename(tsp_file_path)[:-4]
    print(f"Processing {problem_name} (EUC_2D)...")

    coords = read_tsp(tsp_file_path)
    D_matrix = build_distance_matrix(coords)
    
    opt_tour_nodes = read_opt_tour(opt_tour_path)
    opt_len = 0.0
    for i in range(len(opt_tour_nodes)):
        a = opt_tour_nodes[i]
        b = opt_tour_nodes[(i + 1) % len(opt_tour_nodes)]
        opt_len += D_matrix[a, b]
    print(f"  Optimal length: {opt_len:.2f}")

    initial_tour = list(range(len(coords)))
    start_time = time.time()
    
    heuristic_tour, heuristic_len = chained_lin_kernighan(
        coords, initial_tour, opt_len=opt_len
    )
    elapsed_time = time.time() - start_time
    
    if opt_len > 0:
        gap_percentage = 100.0 * (heuristic_len - opt_len) / opt_len
    else:
        # Handle cases where opt_len is zero (e.g., single node problem or undefined)
        gap_percentage = float('inf') if heuristic_len > 0 else 0.0 
    gap = max(0.0, gap_percentage)
    
    print(
        f"  Heuristic length: {heuristic_len:.2f}  "
        f"Gap: {gap:.2f}%  Time: {elapsed_time:.2f}s"
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
        print(
            f"{r_item['name']:<10s} {r_item['opt_len']:>8.2f} "
            f"{r_item['heu_len']:>8.2f} {r_item['gap']:>8.2f} "
            f"{r_item['time']:>8.2f}"
        )
    
    if results_data:
        print("-" * len(header))
        num_items = len(results_data)
        
        # Calculate sums for lengths
        total_opt_len = sum(r_item['opt_len'] for r_item in results_data)
        total_heu_len = sum(r_item['heu_len'] for r_item in results_data)
        
        # Keep averages for gap and time
        avg_gap = sum(r_item['gap'] for r_item in results_data) / num_items
        avg_time = sum(r_item['time'] for r_item in results_data) / num_items
        
        # Update the label and print statement for the summary line
        # The label "SUMMARY" can encompass both sums and averages
        print(
            f"{'SUMMARY':<10s} {total_opt_len:>8.2f} {total_heu_len:>8.2f} "
            f"{avg_gap:>8.2f} {avg_time:>8.2f}"
        )
    print("Done.")


def plot_all_tours(results_data: List[Dict[str, Any]]) -> None:
    """
    Plots the optimal and heuristic tours for all processed instances.
    """
    num_results = len(results_data) # Renamed n to num_results
    if num_results == 0:
        return

    import math # Keep import local if only used here
    cols = int(math.ceil(math.sqrt(num_results)))
    rows = int(math.ceil(num_results / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_list = axes.flatten() if hasattr(axes, 'flatten') else [axes] # Handle single plot case

    for ax, r_item in zip(axes_list, results_data): # Renamed r to r_item
        coords = r_item['coords']
        # Ensure tour lists are closed for plotting
        opt_plot_tour = r_item['opt_tour'] + [r_item['opt_tour'][0]]
        heu_plot_tour = r_item['heu_tour'] + [r_item['heu_tour'][0]]
        
        ax.plot(coords[heu_plot_tour, 0], coords[heu_plot_tour, 1], '-', label='Heuristic')
        ax.plot(coords[opt_plot_tour, 0], coords[opt_plot_tour, 1], ':', label='Optimal')
        ax.set_title(f"{r_item['name']} gap={r_item['gap']:.2f}%")
        ax.axis('equal')
        ax.grid(True)
        # Simplified legend call: matplotlib handles it well if labels are provided.
        ax.legend() 
    # Turn off unused subplots
    for i in range(num_results, len(axes_list)):
        axes_list[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Set your TSPLIB path here
    tsp_folder_path = Path('../TSPLIB95/tsp') # Use Path object
    all_results = []

    for tsp_file_candidate in sorted(tsp_folder_path.iterdir()): # Iterate over Path objects
        if tsp_file_candidate.suffix.lower() != '.tsp':
            continue
        
        problem_base_name = tsp_file_candidate.stem # .stem gets filename without suffix
        # tsp_file_full_path is tsp_file_candidate itself
        
        opt_tour_file_path = tsp_folder_path / (problem_base_name + '.opt.tour') # Path concatenation

        if not opt_tour_file_path.exists(): # Use Path.exists()
            print(
                f"Optimal tour file not found for {problem_base_name}, skipping."
            )
            continue
        
        try:
            # tsplib95.load can take a Path object directly or its string representation
            problem_instance = tsplib95.load(tsp_file_candidate) 
        except Exception as e:
            print(f"Error loading TSP file {tsp_file_candidate}: {e}. Skipping.")
            continue

        edge_weight_type = getattr(problem_instance, 'edge_weight_type', '')
        if edge_weight_type.upper() != 'EUC_2D':
            print(
                f"Skipping non-EUC_2D instance: {problem_base_name} "
                f"(type: {edge_weight_type})"
            )
            continue
        
        try:
            instance_result = process_single_instance(
                str(tsp_file_candidate), str(opt_tour_file_path) # Pass strings if functions expect them
            )
            if instance_result: 
                all_results.append(instance_result)
        except Exception as e:
            print(f"Error processing {problem_base_name}: {e}")

    display_summary_table(all_results)
    plot_all_tours(all_results)