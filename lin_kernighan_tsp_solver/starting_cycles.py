"""
Starting cycle algorithms for the Lin-Kernighan TSP solver.

This module provides different algorithms for generating initial tours:
- Random permutation
- Nearest neighbor heuristic
- Greedy edge selection
- Borůvka's MST-based construction
- Quick Borůvka (QBorůvka) - Concorde default

Usage:
    from .starting_cycles import generate_starting_cycle
    initial_tour = generate_starting_cycle(coords, method="nearest_neighbor")
"""

import random
from typing import Any
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from .config import STARTING_CYCLE_CONFIG


def generate_starting_cycle(
    coords: np.ndarray,
    method: str | None = None,
    **kwargs: Any
) -> list[int]:
    """Generate an initial tour using the specified method.

    Args:
        coords (np.ndarray): Vertex coordinates (n x 2 array).
        method (str | None, optional): Starting cycle method. If None, uses
            the default method from config. Defaults to None.
        **kwargs: Additional parameters for specific methods.

    Returns:
        list[int]: Initial tour as a list of node indices.

    Raises:
        ValueError: If method is not supported or coordinates are invalid.
    """
    if coords.size == 0:
        return []

    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError("Coordinates must be a 2D array with shape (n, 2)")

    n_nodes = coords.shape[0]
    if n_nodes <= 1:
        return list(range(n_nodes))

    # Use default method from config if not specified
    if method is None:
        method = STARTING_CYCLE_CONFIG["DEFAULT_METHOD"]

    # Validate method
    if method not in STARTING_CYCLE_CONFIG["AVAILABLE_METHODS"]:
        raise ValueError(
            f"Unknown starting cycle method: {method}. "
            f"Available methods: {STARTING_CYCLE_CONFIG['AVAILABLE_METHODS']}"
        )

    # Dispatch to appropriate algorithm
    if method == "natural":
        return _natural_tour(n_nodes, **kwargs)
    elif method == "random":
        return _random_tour(n_nodes, **kwargs)
    elif method == "nearest_neighbor":
        return _nearest_neighbor_tour(coords, **kwargs)
    elif method == "greedy":
        return _greedy_tour(coords, **kwargs)
    elif method == "boruvka":
        return _boruvka_tour(coords, **kwargs)
    elif method == "qboruvka":
        return _qboruvka_tour(coords, **kwargs)
    else:
        # This shouldn't happen due to validation above, but safety first
        raise ValueError(f"Method '{method}' not implemented")


def _natural_tour(n_nodes: int, **kwargs: Any) -> list[int]:
    """Generate a natural order tour (fastest, original behavior).

    Creates a tour in natural node order: [0, 1, 2, ..., n-1].
    This is the fastest possible starting cycle and matches the original
    behavior before starting cycles were added.

    Args:
        n_nodes (int): Number of nodes.
        **kwargs: Additional parameters (unused).

    Returns:
        list[int]: Natural order tour.
    """
    return list(range(n_nodes))


def _random_tour(n_nodes: int, **kwargs: Any) -> list[int]:
    """Generate a random permutation tour.

    Args:
        n_nodes (int): Number of nodes.
        **kwargs: Additional parameters (unused).

    Returns:
        list[int]: Random tour.
    """
    tour = list(range(n_nodes))
    random.shuffle(tour)
    return tour


def _nearest_neighbor_tour(coords: np.ndarray, **kwargs: Any) -> list[int]:
    """Generate a tour using the nearest neighbor heuristic.

    Args:
        coords (np.ndarray): Vertex coordinates.
        **kwargs: Additional parameters. Supports:
            - start_node (int): Starting node (default: 0)

    Returns:
        list[int]: Nearest neighbor tour.
    """
    n_nodes = coords.shape[0]
    start_node = kwargs.get("start_node", STARTING_CYCLE_CONFIG["NEAREST_NEIGHBOR_START"])

    # Ensure start_node is valid
    if start_node >= n_nodes:
        start_node = 0

    # Calculate distance matrix (upper triangle only for efficiency)
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)

    tour = [start_node]
    unvisited = set(range(n_nodes))
    unvisited.remove(start_node)

    current = start_node
    while unvisited:
        # Find nearest unvisited node
        nearest_dist = float('inf')
        nearest_node = -1

        for node in unvisited:
            dist = distances[current, node]
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_node = node

        tour.append(nearest_node)
        unvisited.remove(nearest_node)
        current = nearest_node

    return tour


def _edges_to_tour(edges: list[tuple[int, int]], n_nodes: int) -> list[int]:
    """Convert edge list to tour using iterative DFS to avoid stack overflow."""
    # Build adjacency list
    adj = [[] for _ in range(n_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Iterative DFS to avoid recursion depth issues
    visited = [False] * n_nodes
    tour = []
    stack = [0]  # Start from node 0

    while stack:
        node = stack.pop()
        if not visited[node]:
            visited[node] = True
            tour.append(node)

            # Add unvisited neighbors to stack (in reverse order to maintain order)
            for neighbor in reversed(adj[node]):
                if not visited[neighbor]:
                    stack.append(neighbor)

    return tour


def _greedy_tour(coords: np.ndarray, **kwargs: Any) -> list[int]:
    """Generate tour using greedy edge selection (shortest edges first).

    Falls back to nearest neighbor for very large instances to avoid performance issues.
    """
    n_nodes = len(coords)

    # For very large instances, fall back to nearest neighbor
    if n_nodes > 1000:
        return _nearest_neighbor_tour(coords, **kwargs)

    if n_nodes <= 2:
        return list(range(n_nodes))

    # Compute all pairwise distances
    distances = squareform(pdist(coords))

    # Get all edges with their distances
    edges_with_dist = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            edges_with_dist.append((distances[i, j], i, j))

    # Sort edges by distance (greedy: shortest first)
    edges_with_dist.sort()

    # Greedily select edges to form a Hamiltonian cycle
    selected_edges = []
    degree = [0] * n_nodes

    # Union-Find for cycle detection
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for dist, u, v in edges_with_dist:
        # Check if adding this edge would violate constraints
        if degree[u] >= 2 or degree[v] >= 2:
            continue

        # Check if adding this edge would create a cycle (unless it completes the tour)
        if find(u) == find(v) and len(selected_edges) < n_nodes - 1:
            continue

        # Add the edge
        selected_edges.append((u, v))
        degree[u] += 1
        degree[v] += 1
        union(u, v)

        # Stop when we have n edges (complete tour)
        if len(selected_edges) == n_nodes:
            break

    # Convert edges to tour
    return _edges_to_tour(selected_edges, n_nodes)


def _boruvka_tour(coords: np.ndarray, **kwargs: Any) -> list[int]:
    """Generate a tour using Borůvka's MST-based construction.

    First builds a minimum spanning tree using Borůvka's algorithm,
    then converts it to a tour using DFS and shortcuts.

    Args:
        coords (np.ndarray): Vertex coordinates.
        **kwargs: Additional parameters (unused).

    Returns:
        list[int]: Borůvka-based tour.
    """
    n_nodes = coords.shape[0]

    if n_nodes <= 2:
        return list(range(n_nodes))

    # Calculate distance matrix
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)

    # Borůvka's MST algorithm
    mst_edges = _boruvka_mst(distances, n_nodes)

    # Convert MST to tour using DFS traversal
    return _mst_to_tour(mst_edges, n_nodes)


def _qboruvka_tour(coords: np.ndarray, **kwargs: Any) -> list[int]:
    """Generate a tour using Quick Borůvka (Concorde's default method).

    This is a refined version of Borůvka that includes additional
    optimizations and refinement steps.

    Args:
        coords (np.ndarray): Vertex coordinates.
        **kwargs: Additional parameters. Supports:
            - iterations (int): Number of refinement iterations.

    Returns:
        list[int]: QBorůvka tour.
    """
    n_nodes = coords.shape[0]
    iterations = kwargs.get("iterations", STARTING_CYCLE_CONFIG["QBORUVKA_ITERATIONS"])

    if n_nodes <= 2:
        return list(range(n_nodes))

    # Start with Borůvka MST
    tour = _boruvka_tour(coords)

    # Apply refinement iterations with time limit
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)

    # Scale time limit based on problem size (larger problems get more time)
    max_time_per_iteration = min(0.5, 0.001 * n_nodes)

    for _ in range(iterations):
        tour = _improve_tour_2opt(tour, distances, max_time=max_time_per_iteration)

    return tour


# Helper functions

def _boruvka_mst(distances: np.ndarray, n_nodes: int) -> list[tuple[int, int]]:
    """Compute minimum spanning tree using Borůvka's algorithm.

    Args:
        distances (np.ndarray): Distance matrix.
        n_nodes (int): Number of nodes.

    Returns:
        list[tuple[int, int]]: MST edges.
    """
    # Union-find data structure
    parent = list(range(n_nodes))
    rank = [0] * n_nodes

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> bool:
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    mst_edges = []

    while len(mst_edges) < n_nodes - 1:
        # Find minimum edge for each component
        cheapest = [-1] * n_nodes

        for u in range(n_nodes):
            for v in range(u + 1, n_nodes):
                if find(u) != find(v):
                    comp_u, comp_v = find(u), find(v)

                    no_cheapest_u = cheapest[comp_u] == -1
                    current_cheapest_u_dist = distances[cheapest[comp_u] // n_nodes,
                                                        cheapest[comp_u] % n_nodes]
                    is_cheaper_u = distances[u, v] < current_cheapest_u_dist

                    if no_cheapest_u or is_cheaper_u:
                        cheapest[comp_u] = u * n_nodes + v

                    no_cheapest_v = cheapest[comp_v] == -1
                    current_cheapest_v_dist = distances[cheapest[comp_v] // n_nodes,
                                                        cheapest[comp_v] % n_nodes]
                    is_cheaper_v = distances[u, v] < current_cheapest_v_dist

                    if no_cheapest_v or is_cheaper_v:
                        cheapest[comp_v] = u * n_nodes + v

        # Add cheapest edges
        for comp in range(n_nodes):
            if cheapest[comp] != -1:
                u = cheapest[comp] // n_nodes
                v = cheapest[comp] % n_nodes
                if union(u, v):
                    mst_edges.append((u, v))

    return mst_edges


def _mst_to_tour(mst_edges: list[tuple[int, int]], n_nodes: int) -> list[int]:
    """Convert MST to tour using DFS traversal and shortcuts.

    Args:
        mst_edges (list[tuple[int, int]]): MST edges.
        n_nodes (int): Number of nodes.

    Returns:
        list[int]: Tour order.
    """
    if not mst_edges:
        return list(range(n_nodes))

    # Build adjacency list for MST
    adj = [[] for _ in range(n_nodes)]
    for u, v in mst_edges:
        adj[u].append(v)
        adj[v].append(u)

    # DFS preorder traversal
    visited = [False] * n_nodes
    tour = []

    def dfs(node: int) -> None:
        visited[node] = True
        tour.append(node)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                dfs(neighbor)

    # Start from node 0
    dfs(0)

    # Add unvisited nodes (shouldn't happen with connected MST)
    for i in range(n_nodes):
        if not visited[i]:
            tour.append(i)

    return tour


def _improve_tour_2opt(tour: list[int], distances: np.ndarray, max_time: float = 1.0) -> list[int]:
    """Improve tour using 2-opt local search with time limit.

    Args:
        tour (list[int]): Current tour.
        distances (np.ndarray): Distance matrix.
        max_time (float): Maximum time to spend on 2-opt improvement in seconds.

    Returns:
        list[int]: Improved tour.
    """
    import time

    n = len(tour)
    if n <= 3:
        return tour[:]

    improved_tour = tour[:]
    improved = True
    start_time = time.time()

    while improved:
        improved = False

        # Check time limit
        if time.time() - start_time > max_time:
            break

        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue  # Skip if it would reverse entire tour

                # Calculate current distance
                dist1 = distances[improved_tour[i], improved_tour[i + 1]]
                dist2 = distances[improved_tour[j], improved_tour[(j + 1) % n]]
                current_dist = dist1 + dist2

                # Calculate new distance after 2-opt swap
                new_dist1 = distances[improved_tour[i], improved_tour[j]]
                new_dist2 = distances[improved_tour[i + 1], improved_tour[(j + 1) % n]]
                new_dist = new_dist1 + new_dist2

                if new_dist < current_dist:
                    # Perform 2-opt swap: reverse tour[i+1:j+1]
                    improved_tour[i + 1:j + 1] = reversed(improved_tour[i + 1:j + 1])
                    improved = True
                    break
            if improved:
                break

            # Check time limit in inner loop for large instances
            if time.time() - start_time > max_time:
                break

    return improved_tour
