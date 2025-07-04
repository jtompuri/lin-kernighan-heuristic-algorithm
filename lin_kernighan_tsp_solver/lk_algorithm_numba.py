"""Numba-optimized components for Lin-Kernighan TSP solver.

This module provides JIT-compiled versions of performance-critical functions
from the original LK algorithm implementation. It features automatic fallback
to pure Python implementations when Numba is not available.

The module exports optimized implementations of:
- Tour operations (next, prev, sequence, flip)
- Distance matrix computation
- Candidate generation for Lin-Kernighan moves
- Tour cost calculation

Note: This module intentionally defines functions twice (with and without Numba decorators)
to provide fallback implementations when Numba is not available.

Example:
    >>> import numpy as np
    >>> from lin_kernighan_tsp_solver.lk_algorithm_numba import TourNumba, build_distance_matrix_numba
    >>> coords = np.array([[0, 0], [1, 1], [2, 0]], dtype=float)
    >>> D = build_distance_matrix_numba(coords)
    >>> tour = TourNumba([0, 1, 2], D)
    >>> next_node = tour.next(0)
"""
# pyright: reportOptionalMemberAccess=false
# pyright: reportGeneralTypeIssues=false

import math
import time
from typing import Tuple, List

import numpy as np

try:
    import numba  # type: ignore[import-untyped]
    from numba import jit, prange  # type: ignore[import-untyped]
    NUMBA_AVAILABLE = True
    print("Numba JIT compilation available")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available - falling back to pure Python")

    # Create dummy decorator for when Numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    # Create dummy prange for fallback
    def prange(*args):  # type: ignore[misc]
        return range(*args)

    # Create dummy numba module for type annotations
    class _DummyNumba:  # type: ignore[misc]
        boolean = bool

    numba = _DummyNumba()  # type: ignore[misc]


# =============================================================================
# Core Tour Operations (Numba-optimized)
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def tour_next_numba(order: np.ndarray, pos: np.ndarray, v: int) -> int:  # type: ignore[misc] # Numba-decorated version
        """Numba-optimized tour.next() operation."""
        n = len(order)
        idx = pos[v] + 1
        return order[idx if idx < n else 0]

    @jit(nopython=True, cache=True)
    def tour_prev_numba(order: np.ndarray, pos: np.ndarray, v: int) -> int:  # type: ignore[misc] # Numba-decorated version
        """Numba-optimized tour.prev() operation."""
        n = len(order)
        idx = pos[v] - 1
        return order[idx if idx >= 0 else n - 1]

    @jit(nopython=True, cache=True)
    def tour_sequence_numba(pos: np.ndarray, node_a: int, node_b: int, node_c: int) -> bool:  # type: ignore[misc] # Numba-decorated version
        """Numba-optimized tour.sequence() operation."""
        idx_a, idx_b, idx_c = pos[node_a], pos[node_b], pos[node_c]
        if idx_a <= idx_c:
            return idx_a <= idx_b <= idx_c
        return idx_a <= idx_b or idx_b <= idx_c

    @jit(nopython=True, cache=True)
    def tour_flip_numba(order: np.ndarray, pos: np.ndarray, start_node: int, end_node: int):  # type: ignore[misc] # Numba-decorated version
        """Numba-optimized tour.flip() operation."""
        n = len(order)
        if n == 0:
            return

        idx_a, idx_b = pos[start_node], pos[end_node]

        if idx_a == idx_b:
            return

        # Calculate segment length
        if idx_a <= idx_b:
            segment_len = idx_b - idx_a + 1
        else:
            segment_len = (n - idx_a) + (idx_b + 1)

        # Perform in-place reversal
        for i in range(segment_len // 2):
            left_idx = idx_a + i
            if left_idx >= n:
                left_idx -= n

            right_idx = idx_b - i
            if right_idx < 0:
                right_idx += n

            # Swap nodes in order array
            node_left = order[left_idx]
            node_right = order[right_idx]

            order[left_idx] = node_right
            order[right_idx] = node_left

            # Update position mapping
            pos[node_left] = right_idx
            pos[node_right] = left_idx

    @jit(nopython=True, cache=True)
    def tour_init_cost_numba(order: np.ndarray, D: np.ndarray) -> float:  # type: ignore[misc] # Numba-decorated version
        """Numba-optimized tour cost calculation."""
        n = len(order)
        if n == 0:
            return 0.0

        total_cost = 0.0
        for i in range(n):
            current_node = order[i]
            next_node = order[(i + 1) % n]
            total_cost += D[current_node, next_node]

        return float(total_cost)  # type: ignore[return-value] # Ensure float return type

    @jit(nopython=True, parallel=True, cache=True)
    def tour_init_cost_numba_parallel(order: np.ndarray, D: np.ndarray) -> float:
        """Parallel Numba-optimized tour cost calculation for large tours."""
        n = len(order)
        if n == 0:
            return 0.0

        # Create array for parallel cost computation
        costs = np.zeros(n, dtype=np.float64)

        # Parallel computation of edge costs
        for i in prange(n):  # type: ignore[possibly-unbound]
            current_node = order[i]
            next_node = order[(i + 1) % n]
            costs[i] = D[current_node, next_node]

        # Sum the costs
        return float(np.sum(costs))  # type: ignore[return-value] # Ensure float return type

    @jit(nopython=True, cache=True)
    def distance_matrix_numba(coords: np.ndarray) -> np.ndarray:  # type: ignore[misc] # Numba-decorated version
        """Numba-optimized distance matrix computation."""
        n = coords.shape[0]
        if n == 0:
            return np.empty((0, 0), dtype=np.float64)
        if n == 1:
            return np.array([[0.0]], dtype=np.float64)

        D = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                D[i, j] = dist
                D[j, i] = dist

        return D

    @jit(nopython=True, parallel=True, cache=True)
    def distance_matrix_numba_parallel(coords: np.ndarray) -> np.ndarray:
        """Parallel Numba-optimized distance matrix computation for large problems."""
        n = coords.shape[0]
        if n == 0:
            return np.empty((0, 0), dtype=np.float64)
        if n == 1:
            return np.array([[0.0]], dtype=np.float64)

        D = np.zeros((n, n), dtype=np.float64)

        # Parallel computation of distance matrix
        for i in prange(n):  # type: ignore[possibly-unbound]
            for j in range(i + 1, n):
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                D[i, j] = dist
                D[j, i] = dist

        return D

    @jit(nopython=True, cache=True)
    def generate_standard_candidates_numba(  # type: ignore[misc] # Numba-decorated version
        base: int, s1: int, order: np.ndarray, pos: np.ndarray,
        D: np.ndarray, neigh_s1: np.ndarray, tolerance: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba-optimized standard flip candidate generation."""
        max_candidates = len(neigh_s1)
        y1_candidates = np.empty(max_candidates, dtype=np.int32)
        t3_candidates = np.empty(max_candidates, dtype=np.int32)
        gains = np.empty(max_candidates, dtype=np.float64)

        count = 0
        for i in range(len(neigh_s1)):
            y1_cand = neigh_s1[i]
            if y1_cand == base or y1_cand == s1:
                continue

            gain_G1 = D[base, s1] - D[s1, y1_cand]
            if gain_G1 <= tolerance:
                continue

            t3_node = tour_prev_numba(order, pos, y1_cand)
            gain_G2 = D[t3_node, y1_cand] - D[t3_node, base]
            total_gain = gain_G1 + gain_G2

            y1_candidates[count] = y1_cand
            t3_candidates[count] = t3_node
            gains[count] = total_gain
            count += 1

        return y1_candidates[:count], t3_candidates[:count], gains[:count]

    @jit(nopython=True, cache=True)
    def generate_mak_morton_candidates_numba(  # type: ignore[misc] # Numba-decorated version
        base: int, s1: int, order: np.ndarray, pos: np.ndarray,
        D: np.ndarray, neigh_base: np.ndarray, tolerance: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Numba-optimized Mak-Morton flip candidate generation."""
        max_candidates = len(neigh_base)
        candidates = np.empty(max_candidates, dtype=np.int32)
        gains = np.empty(max_candidates, dtype=np.float64)

        prev_base = tour_prev_numba(order, pos, base)
        count = 0

        for i in range(len(neigh_base)):
            candidate = neigh_base[i]
            if candidate == s1 or candidate == prev_base or candidate == base:
                continue

            next_candidate = tour_next_numba(order, pos, candidate)
            gain = (D[base, s1] - D[base, candidate]) + (D[candidate, next_candidate] - D[next_candidate, s1])

            candidates[count] = candidate
            gains[count] = gain
            count += 1

        return candidates[:count], gains[:count]

    @jit(nopython=True, parallel=True, cache=True)
    def generate_standard_candidates_numba_parallel(
        base: int, s1: int, order: np.ndarray, pos: np.ndarray,
        D: np.ndarray, neigh_s1: np.ndarray, tolerance: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parallel Numba-optimized standard flip candidate generation."""
        max_candidates = len(neigh_s1)
        y1_candidates = np.empty(max_candidates, dtype=np.int32)
        t3_candidates = np.empty(max_candidates, dtype=np.int32)
        gains = np.empty(max_candidates, dtype=np.float64)
        valid = np.empty(max_candidates, dtype=numba.boolean)  # type: ignore[possibly-unbound]

        # Parallel candidate evaluation
        for i in prange(len(neigh_s1)):  # type: ignore[possibly-unbound]
            y1_cand = neigh_s1[i]
            valid[i] = False

            if y1_cand == base or y1_cand == s1:
                continue

            gain_G1 = D[base, s1] - D[s1, y1_cand]
            if gain_G1 <= tolerance:
                continue

            t3_node = tour_prev_numba(order, pos, y1_cand)
            gain_G2 = D[t3_node, y1_cand] - D[t3_node, base]
            total_gain = gain_G1 + gain_G2

            y1_candidates[i] = y1_cand
            t3_candidates[i] = t3_node
            gains[i] = total_gain
            valid[i] = True

        # Serial compaction of valid candidates
        count = 0
        for i in range(len(neigh_s1)):
            if valid[i]:
                y1_candidates[count] = y1_candidates[i]
                t3_candidates[count] = t3_candidates[i]
                gains[count] = gains[i]
                count += 1

        return y1_candidates[:count], t3_candidates[:count], gains[:count]

    @jit(nopython=True, parallel=True, cache=True)
    def generate_mak_morton_candidates_numba_parallel(
        base: int, s1: int, order: np.ndarray, pos: np.ndarray,
        D: np.ndarray, neigh_base: np.ndarray, tolerance: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parallel Numba-optimized Mak-Morton flip candidate generation."""
        max_candidates = len(neigh_base)
        candidates = np.empty(max_candidates, dtype=np.int32)
        gains = np.empty(max_candidates, dtype=np.float64)
        valid = np.empty(max_candidates, dtype=numba.boolean)  # type: ignore[possibly-unbound]

        prev_base = tour_prev_numba(order, pos, base)

        # Parallel candidate evaluation
        for i in prange(len(neigh_base)):  # type: ignore[possibly-unbound]
            t3_candidate = neigh_base[i]
            valid[i] = False

            if t3_candidate == base or t3_candidate == s1:
                continue

            t4_node = tour_next_numba(order, pos, t3_candidate)
            if t4_node == base:
                continue

            # Check feasibility and calculate gain
            if not tour_sequence_numba(pos, base, t3_candidate, t4_node):
                continue

            gain_G2 = D[t3_candidate, t4_node] - D[prev_base, base]
            if gain_G2 <= tolerance:
                continue

            candidates[i] = t3_candidate
            gains[i] = gain_G2
            valid[i] = True

        # Serial compaction of valid candidates
        count = 0
        for i in range(len(neigh_base)):
            if valid[i]:
                candidates[count] = candidates[i]
                gains[count] = gains[i]
                count += 1

        return candidates[:count], gains[:count]

else:
    # Fallback implementations when Numba is not available
    def tour_next_numba(order, pos, v):  # type: ignore[misc] # Intentional redefinition for fallback
        """Get next vertex in tour (fallback implementation).
        
        Args:
            order: Array of tour vertex order.
            pos: Position mapping for vertices.
            v: Current vertex.
            
        Returns:
            Next vertex in the tour.
        """
        n = len(order)
        idx = pos[v] + 1
        return order[idx if idx < n else 0]

    def tour_prev_numba(order, pos, v):  # type: ignore[misc] # Intentional redefinition for fallback
        """Get previous vertex in tour (fallback implementation).
        
        Args:
            order: Array of tour vertex order.
            pos: Position mapping for vertices.
            v: Current vertex.
            
        Returns:
            Previous vertex in the tour.
        """
        n = len(order)
        idx = pos[v] - 1
        return order[idx if idx >= 0 else n - 1]

    def tour_sequence_numba(pos, node_a, node_b, node_c):  # type: ignore[misc] # Intentional redefinition for fallback
        """Check if node_b is in sequence from node_a to node_c (fallback implementation).
        
        Args:
            pos: Position mapping for vertices.
            node_a: First node in sequence.
            node_b: Node to check.
            node_c: Last node in sequence.
            
        Returns:
            True if node_b is between node_a and node_c in tour order.
        """
        idx_a, idx_b, idx_c = pos[node_a], pos[node_b], pos[node_c]
        if idx_a <= idx_c:
            return idx_a <= idx_b <= idx_c
        return idx_a <= idx_b or idx_b <= idx_c

    def tour_flip_numba(order, pos, start_node, end_node):  # type: ignore[misc] # Intentional redefinition for fallback
        """Flip tour segment between start_node and end_node (fallback implementation).
        
        Args:
            order: Array of tour vertex order (modified in-place).
            pos: Position mapping for vertices (modified in-place).
            start_node: Starting node of segment to flip.
            end_node: Ending node of segment to flip.
        """
        # Pure Python fallback - use original implementation
        from .lk_algorithm import Tour
        tour = Tour(order.tolist())
        tour.pos = pos
        tour.flip(start_node, end_node)
        order[:] = tour.order
        pos[:] = tour.pos

    def tour_init_cost_numba(order, D):  # type: ignore[misc] # Intentional redefinition for fallback
        """Calculate total tour cost (fallback implementation).
        
        Args:
            order: Array of tour vertex order.
            D: Distance matrix.
            
        Returns:
            Total cost of the tour.
        """
        n = len(order)
        if n == 0:
            return 0.0
        return sum(float(D[order[i], order[(i + 1) % n]]) for i in range(n))

    def distance_matrix_numba(coords):  # type: ignore[misc] # Intentional redefinition for fallback
        """Build distance matrix from coordinates (fallback implementation).
        
        Args:
            coords: Array of (x, y) coordinates.
            
        Returns:
            Symmetric distance matrix.
        """
        return np.linalg.norm(coords[:, None] - coords[None, :], axis=2)

    def generate_standard_candidates_numba(base, s1, order, pos, D, neigh_s1, tolerance):  # type: ignore[misc] # Intentional redefinition for fallback
        """Generate standard flip candidates (fallback implementation).
        
        Args:
            base: Base node for the move.
            s1: First node in current move.
            order: Tour order array.
            pos: Position mapping.
            D: Distance matrix.
            neigh_s1: Neighbors of s1.
            tolerance: Minimum gain tolerance.
            
        Returns:
            Tuple of (y1_candidates, t3_candidates, gains).
        """
        # Fallback to lists
        y1_candidates = []
        t3_candidates = []
        gains = []

        for y1_cand in neigh_s1:
            if y1_cand == base or y1_cand == s1:
                continue
            gain_G1 = D[base, s1] - D[s1, y1_cand]
            if gain_G1 <= tolerance:
                continue
            t3_node = tour_prev_numba(order, pos, y1_cand)
            gain_G2 = D[t3_node, y1_cand] - D[t3_node, base]
            total_gain = gain_G1 + gain_G2

            y1_candidates.append(y1_cand)
            t3_candidates.append(t3_node)
            gains.append(total_gain)

        return np.array(y1_candidates), np.array(t3_candidates), np.array(gains)

    def generate_mak_morton_candidates_numba(base, s1, order, pos, D, neigh_base, tolerance):  # type: ignore[misc] # Intentional redefinition for fallback
        """Generate Mak-Morton flip candidates (fallback implementation).
        
        Args:
            base: Base node for the move.
            s1: First node in current move.
            order: Tour order array.
            pos: Position mapping.
            D: Distance matrix.
            neigh_base: Neighbors of base node.
            tolerance: Minimum gain tolerance.
            
        Returns:
            Tuple of (candidates, gains).
        """
        candidates = []
        gains = []
        prev_base = tour_prev_numba(order, pos, base)

        for candidate in neigh_base:
            if candidate in (s1, prev_base, base):
                continue
            next_candidate = tour_next_numba(order, pos, candidate)
            gain = (D[base, s1] - D[base, candidate]) + (D[candidate, next_candidate] - D[next_candidate, s1])
            candidates.append(candidate)
            gains.append(gain)

        return np.array(candidates), np.array(gains)


# =============================================================================
# Optimized Tour Class
# =============================================================================

class TourNumba:
    """Tour class with Numba-optimized operations for better performance.
    
    This class provides a high-performance tour representation using Numba-compiled
    operations for critical tour manipulations in the Lin-Kernighan algorithm.
    
    Attributes:
        n: Number of nodes in the tour.
        order: NumPy array representing the tour order.
        pos: Position mapping array for O(1) position lookups.
        cost: Current tour cost (None if not calculated).
    
    Example:
        >>> import numpy as np
        >>> coords = np.array([[0, 0], [1, 1], [2, 0]], dtype=float)
        >>> D = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
        >>> tour = TourNumba([0, 1, 2], D)
        >>> next_node = tour.next(0)  # Get next node after 0
        >>> tour.flip(0, 2)  # Flip segment between nodes 0 and 2
    """

    def __init__(self, order, D=None):
        """Initialize tour with optional distance matrix for cost calculation.
        
        Args:
            order: Initial tour order as list or array.
            D: Optional distance matrix for cost calculation.
            
        Raises:
            ValueError: If order contains invalid node indices.
        """
        order_list = list(order) if not isinstance(order, list) else order
        self.n = len(order_list)

        if self.n == 0:
            self.order = np.array([], dtype=np.int32)
            self.pos = np.array([], dtype=np.int32)
            self.cost = 0.0
        else:
            self.order = np.array(order_list, dtype=np.int32)
            max_node = int(np.max(self.order))
            self.pos = np.empty(max_node + 1, dtype=np.int32)

            # Initialize position mapping
            for i in range(self.n):
                self.pos[self.order[i]] = i

            self.cost = None
            if D is not None:
                self.init_cost(D)

    def next(self, v: int) -> int:
        """Get next vertex in tour.
        
        Args:
            v: Current vertex.
            
        Returns:
            Next vertex in the tour order.
            
        Raises:
            IndexError: If tour is empty.
        """
        if self.n == 0:
            raise IndexError("Cannot get next node from an empty tour.")
        return int(tour_next_numba(self.order, self.pos, v))

    def prev(self, v: int) -> int:
        """Get previous vertex in tour.
        
        Args:
            v: Current vertex.
            
        Returns:
            Previous vertex in the tour order.
            
        Raises:
            IndexError: If tour is empty.
        """
        if self.n == 0:
            raise IndexError("Cannot get previous node from an empty tour.")
        return int(tour_prev_numba(self.order, self.pos, v))

    def sequence(self, node_a: int, node_b: int, node_c: int) -> bool:
        """Check if node_b is in sequence from node_a to node_c.
        
        This determines if node_b appears between node_a and node_c in the
        tour order, handling wraparound at tour boundaries.
        
        Args:
            node_a: Starting node of the sequence.
            node_b: Node to check for presence in sequence.
            node_c: Ending node of the sequence.
            
        Returns:
            True if node_b is between node_a and node_c in tour order.
        """
        if self.n == 0:
            return False
        return tour_sequence_numba(self.pos, node_a, node_b, node_c)

    def flip(self, start_node: int, end_node: int):
        """Flip tour segment between start_node and end_node.
        
        Reverses the order of nodes in the tour segment from start_node
        to end_node, inclusive. Updates both order and position arrays.
        
        Args:
            start_node: Starting node of segment to flip.
            end_node: Ending node of segment to flip.
        """
        tour_flip_numba(self.order, self.pos, start_node, end_node)

    def init_cost(self, D: np.ndarray):
        """Initialize tour cost using distance matrix.
        
        Args:
            D: Distance matrix where D[i,j] is distance from node i to j.
        """
        self.cost = tour_init_cost_numba(self.order, D)

    def flip_and_update_cost(self, node_a: int, node_b: int, D: np.ndarray) -> float:
        """Flip segment and update cost, returning cost change.
        
        Efficiently calculates the cost change from flipping a segment
        and updates the tour cost accordingly.
        
        Args:
            node_a: Starting node of segment to flip.
            node_b: Ending node of segment to flip.
            D: Distance matrix.
            
        Returns:
            Change in tour cost (positive for increase, negative for decrease).
        """
        if self.n == 0:
            return 0.0

        if node_a == node_b:
            return 0.0

        # Calculate cost change before flipping
        pos_a = self.pos[node_a]
        pos_b = self.pos[node_b]

        prev_node_of_a = self.order[(pos_a - 1 + self.n) % self.n]
        next_node_of_b = self.order[(pos_b + 1) % self.n]

        if prev_node_of_a == node_b and next_node_of_b == node_a:
            delta_cost = 0.0
        else:
            term_removed = D[prev_node_of_a, node_a] + D[node_b, next_node_of_b]
            term_added = D[prev_node_of_a, node_b] + D[node_a, next_node_of_b]
            delta_cost = term_added - term_removed

        # Perform flip
        self.flip(node_a, node_b)

        # Update cost
        if self.cost is None:
            self.init_cost(D)
            return 0.0

        self.cost += delta_cost
        return delta_cost

    def get_tour(self) -> List[int]:
        """Get tour as list, normalized to start with node 0 if present.
        
        Returns the tour as a list of node indices, rotated so that node 0
        appears first if it exists in the tour.
        
        Returns:
            List of node indices representing the tour order.
        """
        if self.n == 0:
            return []

        # Check if node 0 is in tour and normalize
        if 0 < len(self.pos) and 0 <= self.pos[0] < self.n and self.order[self.pos[0]] == 0:
            position_of_vertex_0 = self.pos[0]
            return list(np.roll(self.order, -position_of_vertex_0))

        return self.order.tolist()


# =============================================================================
# Utility Functions
# =============================================================================

def build_distance_matrix_numba(coords: np.ndarray) -> np.ndarray:
    """Build distance matrix using Numba optimization if available.
    
    Computes the Euclidean distance matrix from a set of 2D coordinates.
    Uses Numba JIT compilation for improved performance when available.
    
    Args:
        coords: Array of shape (n, 2) containing (x, y) coordinates.
        
    Returns:
        Symmetric distance matrix of shape (n, n) where D[i,j] is the
        Euclidean distance between points i and j.
        
    Example:
        >>> coords = np.array([[0, 0], [1, 1], [2, 0]], dtype=float)
        >>> D = build_distance_matrix_numba(coords)
        >>> print(D.shape)
        (3, 3)
    """
    return distance_matrix_numba(coords)


def benchmark_numba_speedup(n_nodes: int = 50, n_iterations: int = 1000) -> dict:
    """Benchmark Numba vs original implementation speedup.
    
    Compares the performance of Numba-optimized tour operations against
    the original pure Python implementation.
    
    Args:
        n_nodes: Number of nodes in test tour.
        n_iterations: Number of flip operations to benchmark.
        
    Returns:
        Dictionary containing benchmark results:
            - n_nodes: Number of nodes tested
            - n_iterations: Number of iterations performed
            - original_time: Time for original implementation (seconds)
            - numba_time: Time for Numba implementation (seconds)
            - speedup: Performance ratio (original_time / numba_time)
            - numba_available: Whether Numba is available
            
    Example:
        >>> result = benchmark_numba_speedup(50, 1000)
        >>> print(f"Speedup: {result['speedup']:.1f}x")
        Speedup: 2.3x
    """
    np.random.seed(42)
    coords = np.random.uniform(0, 1000, (n_nodes, 2))

    # Build distance matrix
    D = build_distance_matrix_numba(coords)

    # Create test tour
    order = np.arange(n_nodes, dtype=np.int32)
    np.random.shuffle(order)

    # Benchmark Numba implementation
    tour_numba = TourNumba(order.tolist(), D)

    start_time = time.time()
    for _ in range(n_iterations):
        if n_nodes > 4:
            tour_numba.flip(0, 2)
            tour_numba.flip(2, 0)  # Undo
    numba_time = time.time() - start_time

    # Benchmark original implementation
    from .lk_algorithm import Tour
    tour_original = Tour(order.tolist(), D)

    start_time = time.time()
    for _ in range(n_iterations):
        if n_nodes > 4:
            tour_original.flip(0, 2)
            tour_original.flip(2, 0)  # Undo
    original_time = time.time() - start_time

    speedup = original_time / numba_time if numba_time > 0 else float('inf')

    return {
        'n_nodes': n_nodes,
        'n_iterations': n_iterations,
        'original_time': original_time,
        'numba_time': numba_time,
        'speedup': speedup,
        'numba_available': NUMBA_AVAILABLE
    }


if __name__ == "__main__":
    # Quick benchmark when run as main module
    print("Running Numba optimization benchmark...")

    for n in [20, 50, 100]:
        result = benchmark_numba_speedup(n, 1000)
        print(f"n={n:3d}: {result['speedup']:.2f}x speedup "
              f"({result['original_time']:.4f}s → {result['numba_time']:.4f}s)")
