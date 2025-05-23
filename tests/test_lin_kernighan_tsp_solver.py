"""
Unit tests for the Tour class and other functions in the Lin-Kernighan TSP solver.

This module contains a suite of pytest tests designed to verify the correctness
of the Tour data structure and other utility and algorithmic functions
within the Lin-Kernighan heuristic implementation.
"""
import time  # Add this if you plan to test deadlines
import numpy as np
import pytest
from lin_kernighan_tsp_solver.lin_kernighan_tsp_solver import (
    Tour,
    build_distance_matrix,
    delaunay_neighbors,
    double_bridge,
    read_opt_tour,
    read_tsp_file,
    step,             # Added
    # alternate_step,  # Add when testing this
    # lk_search,      # Add when testing this
    # lin_kernighan,  # Add when testing this
    LK_CONFIG         # Added
)
from pathlib import Path
# --- Tour class tests ---


def test_init_and_get_tour():
    # Basic initialization and round-trip
    order = [0, 1, 2, 3, 4]
    t = Tour(order)
    assert t.get_tour() == order


def test_next_prev():
    # Check next() and prev() in a simple 4-node cycle
    order = [2, 0, 3, 1]
    # Cycle: 2->0->3->1->2
    t = Tour(order)
    assert t.next(2) == 0
    assert t.next(0) == 3
    assert t.prev(2) == 1
    assert t.prev(0) == 2


def test_sequence_wrap_and_nonwrap():
    # Using order [0, 1, 2, 3, 4]
    order = [0, 1, 2, 3, 4]
    t = Tour(order)
    # Non-wrap segment: from 1 to 3 includes 2
    assert t.sequence(1, 2, 3)
    assert not t.sequence(1, 4, 3)
    # Wrap segment: from 3 to 1 wraps through 4->0->1
    assert t.sequence(3, 4, 1)
    assert t.sequence(3, 0, 1)
    assert not t.sequence(3, 2, 1)


def test_flip_no_wrap():
    # Flip the segment [1, 2, 3] in 0-4 cycle
    order = [0, 1, 2, 3, 4]
    t = Tour(order)
    t.flip(1, 3)
    # Expect: [0, 3, 2, 1, 4]
    assert t.get_tour() == [0, 3, 2, 1, 4]


def test_flip_wrap():
    # Flip the segment 3->4->0->1
    order = [0, 1, 2, 3, 4]
    t = Tour(order)
    t.flip(3, 1)
    # Expect: [0, 4, 3, 2, 1]
    assert t.get_tour() == [0, 4, 3, 2, 1]


# --- Tests for other solver functions ---

def test_build_distance_matrix():
    coords = np.array([[0, 0], [3, 0], [0, 4]])
    # Expected distances:
    # d(0, 0) = 0, d(0, 1) = 3, d(0, 2) = 4
    # d(1, 0) = 3, d(1, 1) = 0, d(1, 2) = 5 (3-4-5 triangle)
    # d(2, 0) = 4, d(2, 1) = 5, d(2, 2) = 0
    expected_D = np.array([
        [0, 3, 4],
        [3, 0, 5],
        [4, 5, 0]
    ])
    D = build_distance_matrix(coords)
    np.testing.assert_array_almost_equal(D, expected_D)

    # Test with empty coordinates
    # Reshape to (0, 2) if your function expects 2D coordinates even for empty
    # Depending on implementation, this might raise an error or return empty array
    # For now, let's assume it should return an empty array of shape (0, 0) or (0, N)
    # build_distance_matrix in the provided snippet will likely have issues with (0, ) shape
    # If coords_empty is np.empty((0, 2)), then:
    coords_empty_2d = np.empty((0, 2))
    D_empty = build_distance_matrix(coords_empty_2d)
    assert D_empty.shape == (0, 0)  # Or (0, 2) depending on strictness, (0, 0) is more typical for dist matrix

    # Test with one coordinate
    coords_one = np.array([[1, 1]])
    D_one = build_distance_matrix(coords_one)
    expected_D_one = np.array([[0.0]])
    np.testing.assert_array_almost_equal(D_one, expected_D_one)


def test_delaunay_neighbors():
    # Test case: num_vertices < 3
    # delaunay_neighbors expects (N, 2), so create an empty array with correct dimensions
    assert delaunay_neighbors(np.empty((0, 2))) == []

    coords1 = np.array([[0, 0]])
    assert delaunay_neighbors(coords1) == [[]]

    coords2 = np.array([[0, 0], [1, 1]])
    # Expected: node 0 neighbors [1], node 1 neighbors [0] (sorted)
    neighbors2 = delaunay_neighbors(coords2)
    assert len(neighbors2) == 2
    assert sorted(neighbors2[0]) == [1]
    assert sorted(neighbors2[1]) == [0]

    # Test case: Triangle (3 vertices)
    coords_triangle = np.array([[0, 0], [1, 0], [0, 1]])
    # Each node should be connected to the other two
    # Expected: [[1, 2], [0, 2], [0, 1]] (inner lists sorted)
    neighbors_triangle = delaunay_neighbors(coords_triangle)
    assert len(neighbors_triangle) == 3
    assert sorted(neighbors_triangle[0]) == [1, 2]
    assert sorted(neighbors_triangle[1]) == [0, 2]
    assert sorted(neighbors_triangle[2]) == [0, 1]

    # Test case: Square (4 vertices) - neighbors depend on triangulation
    # For a square like [[0, 0], [1, 0], [1, 1], [0, 1]],
    # a typical Delaunay triangulation connects each vertex to its adjacent
    # vertices and one of the diagonal vertices.
    coords_square = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # Order: (0, 0), (1, 0), (0, 1), (1, 1)
    # Indices:   0,     1,     2,     3
    # SciPy's Delaunay might produce different triangulations based on floating point details
    # or internal algorithms. A robust test checks properties.
    # For instance, each node must have at least 2 neighbors in a convex hull of 4 points.
    # And degree of nodes in Delaunay is usually small.
    # Let's check a known output for a specific order if possible, or general properties.
    # For [[0, 0], [1, 0], [1, 1], [0, 1]] (node 2 is (0, 1), node 3 is (1, 1))
    # Node 0 (0, 0) connects to 1 (1, 0) and 2 (0, 1). One diagonal, e.g. to 3 (1, 1).
    # So, neighbors_square[0] could be [1, 2, 3]
    # Node 1 (1, 0) connects to 0 (0, 0) and 3 (1, 1). One diagonal, e.g. to 2 (0, 1).
    # So, neighbors_square[1] could be [0, 2, 3]
    neighbors_square = delaunay_neighbors(coords_square)
    assert len(neighbors_square) == 4
    for i in range(4):
        assert len(neighbors_square[i]) >= 2  # Each node should have at least 2 neighbors
        for neighbor in neighbors_square[i]:
            assert i in neighbors_square[neighbor]  # Symmetry: if j is a neighbor of i, i is a neighbor of j

    # Example of a more specific check if triangulation is predictable:
    # coords_stable_square = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    # neighbors_stable_square = delaunay_neighbors(coords_stable_square)
    # expected_neighbors_for_stable_square = [
    #     sorted([1, 2, 3]),  # Neighbors of node 0
    #     sorted([0, 2, 3]),  # Neighbors of node 1
    #     sorted([0, 1, 3]),  # Neighbors of node 2
    #     sorted([0, 1, 2])  # Neighbors of node 3
    # ]
    # This specific expectation might fail due to triangulation variance.
    # The general property check above is more robust.


def test_double_bridge_kick():
    np.random.seed(42)  # for reproducibility

    # Test with N >= 5 (e.g., N=10), should be modified
    # The double_bridge implementation modifies if N > 4.
    original_tour_10 = list(range(10))
    kicked_tour_10 = double_bridge(original_tour_10.copy())
    assert len(kicked_tour_10) == len(original_tour_10), "Length should not change"
    assert sorted(kicked_tour_10) == sorted(original_tour_10), "Nodes should be a permutation of original"
    assert kicked_tour_10 != original_tour_10, "Tour with N=10 should be modified"

    # Test with N = 5, should be modified
    original_tour_5 = list(range(5))
    kicked_tour_5 = double_bridge(original_tour_5.copy())
    assert len(kicked_tour_5) == len(original_tour_5)
    assert sorted(kicked_tour_5) == sorted(original_tour_5)
    assert kicked_tour_5 != original_tour_5, "Tour with N=5 should be modified"

    # Test with N <= 4, should return original
    small_tour_4 = list(range(4))  # N=4
    kicked_small_tour_4 = double_bridge(small_tour_4.copy())
    assert kicked_small_tour_4 == small_tour_4, "Tour with N=4 should be unchanged"

    small_tour_3 = list(range(3))  # N=3
    kicked_small_tour_3 = double_bridge(small_tour_3.copy())
    assert kicked_small_tour_3 == small_tour_3, "Tour with N=3 should be unchanged"

    small_tour_1 = list(range(1))  # N=1
    kicked_small_tour_1 = double_bridge(small_tour_1.copy())
    assert kicked_small_tour_1 == small_tour_1, "Tour with N=1 should be unchanged"

    small_tour_0 = list(range(0))  # N=0
    kicked_small_tour_0 = double_bridge(small_tour_0.copy())
    assert kicked_small_tour_0 == small_tour_0, "Tour with N=0 should be unchanged"


def test_read_opt_tour(tmp_path: Path):
    # Test with a valid .opt.tour file
    content_valid = """
NAME : test_valid.opt.tour
TYPE : TOUR
DIMENSION : 4
TOUR_SECTION
1
2
3
4
-1
EOF
"""
    opt_tour_file_valid = tmp_path / "test_valid.opt.tour"
    opt_tour_file_valid.write_text(content_valid)
    tour = read_opt_tour(str(opt_tour_file_valid))
    assert tour == [0, 1, 2, 3]  # Nodes should be 0-indexed

    # Test with a non-existent file
    tour_non_existent = read_opt_tour(str(tmp_path / "non_existent.opt.tour"))
    assert tour_non_existent is None

    # Test with a file missing TOUR_SECTION
    content_no_tour_section = """
NAME : test_no_section.opt.tour
TYPE : TOUR
DIMENSION : 3
1
2
3
-1
EOF
"""
    opt_tour_file_no_section = tmp_path / "test_no_section.opt.tour"
    opt_tour_file_no_section.write_text(content_no_tour_section)
    tour_no_section = read_opt_tour(str(opt_tour_file_no_section))
    assert tour_no_section is None  # Or handle as error, current impl likely returns None or empty

    # Test with a file with non-integer node in TOUR_SECTION
    content_non_integer = """
NAME : test_non_integer.opt.tour
TYPE : TOUR
DIMENSION : 3
TOUR_SECTION
1
A
3
-1
EOF
"""
    opt_tour_file_non_integer = tmp_path / "test_non_integer.opt.tour"
    opt_tour_file_non_integer.write_text(content_non_integer)
    tour_non_integer = read_opt_tour(str(opt_tour_file_non_integer))
    assert tour_non_integer is None  # Or specific error handling, current impl might error or return partial

    # Test with a file that doesn't end with -1 before EOF
    content_no_neg_one = """
NAME : test_no_neg_one.opt.tour
TYPE : TOUR
DIMENSION : 3
TOUR_SECTION
1
2
3
EOF
"""
    opt_tour_file_no_neg_one = tmp_path / "test_no_neg_one.opt.tour"
    opt_tour_file_no_neg_one.write_text(content_no_neg_one)
    tour_no_neg_one = read_opt_tour(str(opt_tour_file_no_neg_one))
    assert tour_no_neg_one is None  # Or specific error handling


# --- Tour class cost calculation tests ---

def test_tour_init_with_cost():
    order = [0, 1, 2]
    # Coords: (0, 0), (3, 0), (0, 4)
    # Dists: d(0, 1)=3, d(1, 2)=5, d(2, 0)=4
    # Tour: 0-1-2-0. Cost = 3 + 5 + 4 = 12
    coords = np.array([[0, 0], [3, 0], [0, 4]])
    dist_matrix = build_distance_matrix(coords)

    tour = Tour(order, dist_matrix)
    assert tour.get_tour() == order
    assert tour.n == 3
    assert tour.cost == pytest.approx(12.0)

    # Test with an empty tour
    tour_empty_with_d = Tour([], dist_matrix)  # dist_matrix can be non-empty or empty
    assert tour_empty_with_d.cost == 0.0       # Cost is 0.0 as per Tour.__init__ for n=0

    tour_empty_no_d = Tour([])
    # MODIFIED ASSERTION: Expect 0.0 as per Tour.__init__ for n=0
    assert tour_empty_no_d.cost == 0.0

    # Test with a tour of one node
    tour_one_node = Tour([0], dist_matrix)  # D is provided
    assert tour_one_node.cost == 0.0  # init_cost will calculate this as 0

    tour_one_node_no_d = Tour([0])  # D is NOT provided
    assert tour_one_node_no_d.cost is None  # For n > 0 and D is None, cost remains None


def test_tour_flip_and_update_cost():
    # Order: 0-1-2-3-0
    # Coords: (0,0), (1,0), (1,1), (0,1) - a unit square
    # Dists: d(0,1)=1, d(1,2)=1, d(2,3)=1, d(3,0)=1. Diagonals = sqrt(2) -> No, for unit square, diagonals are sqrt(1^2+1^2)=sqrt(2)
    # Corrected Dists for unit square:
    # d(0,1)=1 (node (0,0) to (1,0))
    # d(1,2)=1 (node (1,0) to (1,1))
    # d(2,3)=1 (node (1,1) to (0,1))
    # d(3,0)=1 (node (0,1) to (0,0))
    # d(0,2)=sqrt(2) (node (0,0) to (1,1))
    # d(1,3)=sqrt(2) (node (1,0) to (0,1))

    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # 0:(0,0), 1:(1,0), 2:(1,1), 3:(0,1)
    dist_matrix = build_distance_matrix(coords)
    initial_order_list = [0, 1, 2, 3]
    tour = Tour(initial_order_list, dist_matrix)
    assert tour.cost == pytest.approx(4.0)

    # To get internal order [2,1,0,3] from [0,1,2,3], we flip segment (0,1,2).
    # This means segment_start_node=0, segment_end_node=2.
    # Original tour: 0-1-2-3-0.
    # Flipping segment 0-1-2 (inclusive) means it becomes 2-1-0.
    # The tour becomes 2-1-0-3-2. Internal order: [2,1,0,3].
    # Edges broken: (prev(0),0) = (3,0) and (2,next(2)) = (2,3). Costs: D[3,0]=1, D[2,3]=1. Sum=2.
    # Edges added: (prev(0),2) = (3,2) and (0,next(2)) = (0,3). Costs: D[3,2]=1, D[0,3]=1. Sum=2.
    # Cost change = (D[3,2]+D[0,3]) - (D[3,0]+D[2,3]) = (1+1) - (1+1) = 0.
    # New cost = 4.0 + 0 = 4.0.

    segment_to_flip_start_node, segment_to_flip_end_node = 0, 2
    tour.flip_and_update_cost(segment_to_flip_start_node, segment_to_flip_end_node, dist_matrix)

    expected_raw_order = [2, 1, 0, 3]
    # Cost for tour 2-1-0-3-2:
    # D[2,1]=1, D[1,0]=1, D[0,3]=1, D[3,2]=1. Sum = 4.0
    expected_new_cost = dist_matrix[2, 1] + dist_matrix[1, 0] + dist_matrix[0, 3] + dist_matrix[3, 2]

    assert list(tour.order) == expected_raw_order  # Check raw internal order
    assert tour.cost == pytest.approx(expected_new_cost)
    assert tour.cost == pytest.approx(4.0)

    # To flip back from internal order [2,1,0,3] to [0,1,2,3]:
    # We need to reverse the segment (2,1,0) in [2,1,0,3] to get (0,1,2).
    # So, segment_start_node=2, segment_end_node=0 (using node values).
    # Edges broken: (prev(2),2) = (3,2) and (0,next(0)) = (0,1). Costs: D[3,2]=1, D[0,1]=1. Sum=2.
    # Edges added: (prev(2),0) = (3,0) and (2,next(0)) = (2,1). Costs: D[3,0]=1, D[2,1]=1. Sum=2.
    # Cost change = 0. New cost = 4.0.
    tour.flip_and_update_cost(2, 0, dist_matrix)
    assert list(tour.order) == initial_order_list  # Check raw internal order
    assert tour.cost == pytest.approx(4.0)

    # Also check get_tour() for completeness after the first flip
    # If tour.order is [2,1,0,3], then get_tour() normalizes to start with 0.
    # pos[0] will be 2. Rotated order: order[2:] + order[:2] = [0,3] + [2,1] = [0,3,2,1]
    tour_after_first_flip = Tour(expected_raw_order, dist_matrix)  # Create a new tour to test get_tour
    assert tour_after_first_flip.get_tour() == [0, 3, 2, 1]


def test_read_tsp_file_valid_euc_2d(tmp_path: Path):
    content = """
NAME: test_valid
TYPE: TSP
COMMENT: A valid test file for EUC_2D
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 10.0 20.0
2 30.0 40.0
3 50.0 60.0
EOF
"""
    tsp_file = tmp_path / "test_valid.tsp"
    tsp_file.write_text(content)

    # read_tsp_file in the current lin_kernighan_tsp_solver.py returns only the coordinates np.ndarray
    coords = read_tsp_file(str(tsp_file))
    assert coords is not None, "read_tsp_file should return coordinates for a valid file"

    expected_coords = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    np.testing.assert_array_almost_equal(coords, expected_coords)

    # Assertions for name, comment, edge_type cannot be made from the return of the current read_tsp_file.
    # These would require read_tsp_file to be modified to return metadata.
    # For now, we only test the coordinates it does return.


def test_read_tsp_file_not_found(tmp_path: Path):
    # The current read_tsp_file raises FileNotFoundError, it doesn't return None.
    with pytest.raises(FileNotFoundError):
        read_tsp_file(str(tmp_path / "non_existent.tsp"))


def test_read_tsp_file_error_cases(tmp_path: Path):
    # The current read_tsp_file is very basic and doesn't perform
    # extensive validation like checking DIMENSION, specific sections, or node indexing.
    # It primarily focuses on extracting coordinates after NODE_COORD_SECTION.
    # The error tests below would need read_tsp_file to be more comprehensive.
    # For now, we can test a few basic parsing issues it might encounter.

    # Case: Non-numeric coordinate value (will cause ValueError during float conversion)
    content_non_numeric_coord = """
NAME: test_non_numeric
TYPE: TSP
DIMENSION: 1
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 10.0 ABC
EOF
"""
    tsp_file_non_numeric_coord = tmp_path / "test_non_numeric_coord.tsp"
    tsp_file_non_numeric_coord.write_text(content_non_numeric_coord)
    # The warning is printed, but the function might still proceed or error later.
    # Depending on exact behavior, this might raise ValueError or return partial/empty.
    # The current version prints a warning and skips the line.
    # If it skips '1 10.0 ABC' and there are no other valid nodes, it returns an empty array.
    coords = read_tsp_file(str(tsp_file_non_numeric_coord))
    assert coords.shape == (0, 2) or coords.shape == (0, ), "Expected empty array for non-parseable coords if skipped"

    # Case: Empty NODE_COORD_SECTION or no valid nodes
    content_empty_nodes = """
NAME: test_empty_nodes
TYPE: TSP
DIMENSION: 0
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
EOF
"""
    tsp_file_empty_nodes = tmp_path / "test_empty_nodes.tsp"
    tsp_file_empty_nodes.write_text(content_empty_nodes)
    coords_empty = read_tsp_file(str(tsp_file_empty_nodes))
    assert coords_empty.shape == (0, 2) or coords_empty.shape == (0, ), "Expected empty array for no nodes"

    # The more specific error cases (missing DIMENSION, wrong node index, etc.)
    # are not handled by the current basic read_tsp_file.
    # To test those, read_tsp_file would need to be enhanced significantly.
    # For example, the "DIMENSION not found" test:
    content_no_dim = """
NAME: test_no_dim
TYPE: TSP
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 10.0 20.0
EOF
"""
    tsp_file_no_dim = tmp_path / "test_no_dim.tsp"
    tsp_file_no_dim.write_text(content_no_dim)
    # Current read_tsp_file doesn't use DIMENSION, so it will parse successfully.
    coords_no_dim = read_tsp_file(str(tsp_file_no_dim))
    expected_coords_no_dim = np.array([[10.0, 20.0]])
    np.testing.assert_array_almost_equal(coords_no_dim, expected_coords_no_dim)


# --- LK Algorithm Core Function Tests ---

@pytest.fixture
def simple_tsp_setup():
    """
    Provides a simple 5-node TSP instance for testing LK components.
    Nodes are on a line: 0,0; 1,0; 2,0; 3,0; 4,0.
    Optimal tour: [0, 1, 2, 3, 4], cost = 4 * 1 = 4.0 (if unit steps)
    Let's make it slightly more interesting for 2-opt:
    Nodes: (0,0), (1,0), (3,0), (2,0), (4,0)
    Initial tour: [0,1,2,3,4] (node indices)
    Actual path: (0,0)-(1,0)-(3,0)-(2,0)-(4,0)-(0,0)
    Cost: 1 (0-1) + 2 (1-2 i.e. (1,0)-(3,0)) + 1 (2-3 i.e. (3,0)-(2,0)) + 2 (3-4 i.e. (2,0)-(4,0)) + 4 (4-0 i.e. (4,0)-(0,0))
         = 1 + 2 + 1 + 2 + 4 = 10.0

    Optimal tour for these coords: [0,1,3,2,4] (node indices)
    Path: (0,0)-(1,0)-(2,0)-(3,0)-(4,0)-(0,0)
    Cost: 1 (0-1) + 1 (1-3 i.e. (1,0)-(2,0)) + 1 (3-2 i.e. (2,0)-(3,0)) + 1 (2-4 i.e. (3,0)-(4,0)) + 4 (4-0 i.e. (4,0)-(0,0))
         = 1 + 1 + 1 + 1 + 4 = 8.0
    """
    coords = np.array([
        [0.0, 0.0],  # Node 0
        [1.0, 0.0],  # Node 1
        [3.0, 0.0],  # Node 2
        [2.0, 0.0],  # Node 3
        [4.0, 0.0]   # Node 4
    ])
    dist_matrix = build_distance_matrix(coords)

    # Initial non-optimal tour (node indices): [0, 1, 2, 3, 4]
    # Path: (0,0)-(1,0)-(3,0)-(2,0)-(4,0)-(0,0)
    # Cost: D[0,1]+D[1,2]+D[2,3]+D[3,4]+D[4,0]
    #     = 1 + 2 + 1 + 2 + 4 = 10.0
    initial_tour_order = [0, 1, 2, 3, 4]
    tour_obj = Tour(initial_tour_order, dist_matrix)
    assert tour_obj.cost == pytest.approx(10.0)

    # Neighbors: fully connected for simplicity in small tests
    # or use delaunay_neighbors if preferred and robust for the geometry
    num_nodes = len(coords)
    neighbors = [list(range(num_nodes)) for _ in range(num_nodes)]
    for i in range(num_nodes):
        if i in neighbors[i]:  # remove self-loops
            neighbors[i].remove(i)
        neighbors[i].sort(key=lambda x: dist_matrix[i, x])  # Sort by distance

    return coords, dist_matrix, tour_obj, neighbors


def test_step_finds_simple_2_opt(simple_tsp_setup):
    """
    Test if step() can find a basic 2-opt improvement.
    Initial tour: [0,1,2,3,4] (cost 10.0) for coords (0,0),(1,0),(3,0),(2,0),(4,0)
    Optimal tour: [0,1,3,2,4] (cost 8.0)
    This involves swapping edges (1,2) and (3,4) with (1,3) and (2,4).
    In tour [0,1,2,3,4]:
    Edges are (0,1), (1,2), (2,3), (3,4), (4,0)
    Flip segment (2,3) -> (3,2)
    Tour becomes [0,1,3,2,4]
    Broken: (1,2) and (3,4)
    Added: (1,3) and (2,4)
    """
    _coords, dist_matrix, tour, neighbors = simple_tsp_setup

    original_lk_config = LK_CONFIG.copy()  # Save to restore later
    LK_CONFIG["MAX_LEVEL"] = 2  # A 2-opt is found at level 1 of recursion
    LK_CONFIG["BREADTH"] = [1] * LK_CONFIG["MAX_LEVEL"]  # Minimal breadth

    # We need to choose a `base` node that allows the 2-opt.
    # The 2-opt involves edges (1,2) and (3,4) -> (1,3) and (2,4).
    # Let base = 1. Then s1 = tour.next(1) = 2.
    # We look for t2 (a_candidate_node) such that D[1,2] > D[2,t2].
    # If t2 = 3 (node value), then D[1,2]=dist_matrix[1,2]=2. D[2,3]=dist_matrix[2,3]=1.
    # gain_first_exchange = D[s1,base] - D[s1,t2] = D[2,1] - D[2,3] = 2 - 1 = 1. This is > 0.
    # Now check G_star = gain_first_exchange + D[t2, tour.next(t2)] - D[base, tour.next(t2)]
    # Here, t2_val=3. next_t2_val = tour.next(3) = 4.
    # G_star = 1 + D[3,4] - D[1,4]
    # D[3,4] = dist_matrix[node_3_val, node_4_val] = dist_matrix[2,4] = 1 (dist between (2,0) and (4,0))
    # D[1,4] = dist_matrix[node_1_val, node_4_val] = dist_matrix[1,4] = 3 (dist between (1,0) and (4,0))
    # G_star = 1 + 1 - 3 = -1. This is not > 0. So this 2-opt is not chosen this way.

    # Let's re-evaluate the 2-opt:
    # Tour [0,1,2,3,4]. Edges: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Costs: D[0,1]=1, D[1,2]=2, D[2,3]=1, D[3,4]=2, D[4,0]=4. Sum = 10.
    # Target [0,1,3,2,4]. Edges: (0,1), (1,3), (3,2), (2,4), (4,0)
    # Costs: D[0,1]=1, D[1,3]=1, D[3,2]=1, D[2,4]=1, D[4,0]=4. Sum = 8.
    # Improvement = 2.
    # This is achieved by breaking (1,2) and (3,4), adding (1,3) and (2,4).
    # This corresponds to flipping the segment between next(1) and 3, i.e., segment [2].
    # Or flipping segment between next(3) and 1, i.e., segment [4,0].
    # The flip sequence for this 2-opt is (node_after_s1, node_s2) = (2,3)
    # where s1=1, s2=tour.prev(3)=2. No, this is not right.
    # A 2-opt involves choosing two edges (t1,t2) and (t3,t4) and replacing them with (t1,t3) and (t2,t4).
    # This means reversing the path from t2 to t3.
    # Let t1=1, t2=tour.next(1)=2. Let t3=3, t4=tour.next(3)=4.
    # We break (1,2) and (3,4). We add (1,3) and (2,4).
    # The segment to reverse is from node 2 to node 3 (inclusive).
    # So the flip is (2,3).

    # Try base = 1. s1 = tour.order[tour.pos[1]] = 1.
    # step(level, delta, base_node, tour, D, neigh, start_cost, best_cost, deadline, current_flips)
    # The `tour` object is modified by `step` if it finds an improvement locally,
    # but `step` itself returns the sequence of flips.

    initial_cost = tour.cost

    # We need to iterate through possible base nodes or pick one that works.
    # Let's try all base nodes.
    improved_overall = False
    best_flip_sequence = None

    for base_node_idx in range(tour.n):
        base_node = tour.order[base_node_idx]

        # Reset tour to original state for each base_node test if step modifies it and doesn't revert
        # For this test, we assume step is called on a fresh tour or a tour that step can manage.
        # The step function itself should handle temporary flips and reversions.
        # We are testing if *any* call to step from a base node finds the 2-opt.

        # Create a fresh tour object for each call to step to ensure independence
        # if step modifies the tour and doesn't perfectly revert for non-improving paths.
        current_tour_for_step = Tour(tour.get_tour(), dist_matrix)

        improved, flip_sequence = step(
            level=1,
            delta=0.0,
            base=base_node,
            tour=current_tour_for_step,  # Pass the fresh tour
            D=dist_matrix,
            neigh=neighbors,
            flip_seq=[],  # CORRECTED: Use the parameter name 'flip_seq'
            start_cost=initial_cost,
            best_cost=initial_cost,  # best_cost seen so far for this search branch
            deadline=time.time() + 10  # Ample time
        )
        if improved:
            improved_overall = True
            best_flip_sequence = flip_sequence
            # For a 2-opt, the flip_sequence should be like [(node_x, node_y)]
            # The tour object current_tour_for_step would have been modified by these flips.
            # We can check its cost.
            # cost_after_step_flips = 0
            # final_order_in_step_tour = current_tour_for_step.get_tour()
            # for i in range(current_tour_for_step.n):
            #     n1 = final_order_in_step_tour[i]
            #     n2 = final_order_in_step_tour[(i + 1) % current_tour_for_step.n]
            #     cost_after_step_flips += dist_matrix[n1, n2]
            # assert cost_after_step_flips < initial_cost
            break  # Found an improvement

    assert improved_overall, "Step function should have found an improvement"
    assert best_flip_sequence is not None

    # The expected 2-opt flip is reversing segment (2,3)
    # In the context of step's flip_sequence, it's (t_i, t_{i+1}) where t_i is s_2k-1 and t_{i+1} is s_2k
    # For the 2-opt: (1,2) (3,4) -> (1,3) (2,4).
    # s1=1. s2=2. s3=3. s4=4.
    # Flip is (s2,s3) = (2,3).
    assert len(best_flip_sequence) == 1, "Expected a single flip for a 2-opt"
    # The flip is (node_before_segment_to_reverse, last_node_of_segment_to_reverse)
    # No, step returns (t_2i-1, t_2i) pairs. For a 2-opt, it's (s2, s3).
    # s1=base, s2=next(base). s3=candidate_t2. s4=next(s3).
    # Flip is (s2, s3).
    # If base=1, s1_val=1, s2_val=2. If s3_val=3. Flip is (2,3).
    # This means segment from tour.next(2) to 3 is reversed.
    # tour.next(2) is 3. Segment is just [3]. This is not right.

    # Let's re-check how flips are applied from step's output.
    # If step returns [(x,y)], Tour.apply_flips calls tour.flip(x,y).
    # Tour.flip(x,y) reverses segment x...y.
    # To get [0,1,3,2,4] from [0,1,2,3,4], we flip segment (2,3).
    # So, expected flip is (2,3).

    # We need to ensure the flip_sequence makes sense.
    # The actual nodes involved in the flip (2,3) are node index 2 and node index 3.
    # Their values are coords[2]=(3,0) and coords[3]=(2,0).
    # The flip should be ([node value for index 2], [node value for index 3])
    # No, the flip sequence uses the actual node *values* present in the tour.
    # Initial tour order: [0, 1, 2, 3, 4] (these are the node values/labels)
    # Segment to flip is (node 2, node 3).
    assert best_flip_sequence[0] == (2, 3) or best_flip_sequence[0] == (3, 2)

    # Apply this flip to the original tour and check cost
    original_tour_obj = Tour(simple_tsp_setup[2].get_tour(), dist_matrix)  # Fresh copy
    # Iterate through the flip sequence and apply each flip
    for flip_pair in best_flip_sequence:
        original_tour_obj.flip_and_update_cost(flip_pair[0], flip_pair[1], dist_matrix)

    assert original_tour_obj.cost == pytest.approx(8.0)
    assert original_tour_obj.get_tour() == [0, 1, 3, 2, 4]  # Check final tour order

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)  # Restore config


def test_step_no_improvement_on_optimal_tour(simple_tsp_setup):
    """
    Test if step() correctly finds no improvement if the tour is already optimal
    for simple 2-opt moves.
    Uses the simple_tsp_setup: coords (0,0),(1,0),(3,0),(2,0),(4,0)
    Optimal tour: [0,1,3,2,4] (cost 8.0)
    """
    _coords, dist_matrix, _initial_non_optimal_tour, neighbors = simple_tsp_setup

    # Create the optimal tour for this setup
    optimal_order = [0, 1, 3, 2, 4]
    optimal_tour = Tour(optimal_order, dist_matrix)
    assert optimal_tour.cost == pytest.approx(8.0)

    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 2
    LK_CONFIG["BREADTH"] = [1] * LK_CONFIG["MAX_LEVEL"]

    initial_cost = optimal_tour.cost
    assert initial_cost is not None, "Initial cost should be a float for an initialized tour with a distance matrix."

    for base_node_idx in range(optimal_tour.n):
        base_node = optimal_tour.order[base_node_idx]

        # Create a fresh tour object for each call to step
        current_tour_for_step = Tour(optimal_tour.get_tour(), dist_matrix)

        improved, flip_sequence = step(
            level=1,
            delta=0.0,
            base=base_node,
            tour=current_tour_for_step,
            D=dist_matrix,
            neigh=neighbors,
            flip_seq=[],
            start_cost=initial_cost,  # Now Pylance knows initial_cost is float
            best_cost=initial_cost,  # Now Pylance knows initial_cost is float
            deadline=time.time() + 10
        )
        # For an optimal tour (at least for 2-opts), step should not find improvements
        assert not improved, f"Step should not find improvement from base {base_node} on an optimal tour"
        assert flip_sequence is None, f"Flip sequence should be None if no improvement from base {base_node}"

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_step_pruning_gain_first_exchange():
    """
    Test if step() correctly prunes branches where the first exchange (G1)
    does not offer a strictly positive gain.
    Setup: 3 nodes in a line: 0-1-2. Tour [0,1,2].
    Base=0, s1=1. Candidate for t2 is 2.
    D[base,s1] = D[0,1] = 1. D[s1,t2] = D[1,2] = 1.
    gain_first_exchange = D[0,1] - D[1,2] = 1 - 1 = 0.
    This should be pruned as it's not > FLOAT_COMPARISON_TOLERANCE.
    """
    coords = np.array([
        [0.0, 0.0],  # Node 0
        [1.0, 0.0],  # Node 1
        [2.0, 0.0]   # Node 2
    ])
    dist_matrix = build_distance_matrix(coords)
    # D = [[0,1,2],[1,0,1],[2,1,0]]

    initial_tour_order = [0, 1, 2]  # Cost = D[0,1]+D[1,2]+D[2,0] = 1+1+2 = 4.0
    tour = Tour(initial_tour_order, dist_matrix)
    assert tour.cost == pytest.approx(4.0)

    num_nodes = len(coords)
    neighbors = [list(range(num_nodes)) for _ in range(num_nodes)]
    for i in range(num_nodes):
        if i in neighbors[i]:
            neighbors[i].remove(i)
        neighbors[i].sort(key=lambda x: dist_matrix[i, x])
    # neigh[0] = [1,2] (sorted by dist: D[0,1]=1, D[0,2]=2)
    # neigh[1] = [0,2] (sorted by dist: D[1,0]=1, D[1,2]=1)
    # neigh[2] = [1,0] (sorted by dist: D[2,1]=1, D[2,0]=2)

    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 2
    LK_CONFIG["BREADTH"] = [1] * LK_CONFIG["MAX_LEVEL"]

    # Test with base = 0. s1 = tour.next(0) = 1.
    # Candidates for a_candidate_node (t2) from neigh[s1] (neigh[1]):
    # - node 0: invalid (base)
    # - node 2: gain_first_exchange = D[base,s1] - D[s1,a_candidate_node]
    #                             = D[0,1] - D[1,2] = 1 - 1 = 0.
    # Since 0 is not > FLOAT_COMPARISON_TOLERANCE, this path should be pruned.
    base_node = 0
    initial_cost = tour.cost
    assert initial_cost is not None, "Initial cost should be a float for an initialized tour with a distance matrix."

    current_tour_for_step = Tour(tour.get_tour(), dist_matrix)
    improved, flip_sequence = step(
        level=1,
        delta=0.0,
        base=base_node,
        tour=current_tour_for_step,
        D=dist_matrix,
        neigh=neighbors,
        flip_seq=[],
        start_cost=initial_cost,  # Now Pylance knows initial_cost is float
        best_cost=initial_cost,  # Now Pylance knows initial_cost is float
        deadline=time.time() + 10
    )

    assert not improved, "Step should not find improvement due to pruning (gain_first_exchange <= 0)"
    assert flip_sequence is None, "Flip sequence should be None if pruned"

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_step_respects_max_level(simple_tsp_setup):
    """
    Test if step() correctly respects LK_CONFIG["MAX_LEVEL"].
    - MAX_LEVEL = 0: Should find no improvement (2-opts are level 1).
    - MAX_LEVEL = 1: Should find the available 2-opt.
    """
    _coords, dist_matrix, tour, neighbors = simple_tsp_setup
    # tour is [0,1,2,3,4] with cost 10.0. Optimal is [0,1,3,2,4] with cost 8.0 (a 2-opt).

    original_lk_config = LK_CONFIG.copy()
    initial_cost = tour.cost
    assert initial_cost is not None, "Initial cost should be a float."

    # Scenario 1: MAX_LEVEL = 0 (prevents 2-opts, which are level 1 in recursion)
    LK_CONFIG["MAX_LEVEL"] = 0
    LK_CONFIG["BREADTH"] = [1] * 1  # Dummy breadth for level 0 if it were used

    improved_overall_max_level_0 = False
    for base_node_idx_0 in range(tour.n):
        base_node_0 = tour.order[base_node_idx_0]
        current_tour_for_step_0 = Tour(tour.get_tour(), dist_matrix)
        improved_0, _ = step(
            level=1, delta=0.0, base=base_node_0,
            tour=current_tour_for_step_0, D=dist_matrix, neigh=neighbors,
            flip_seq=[], start_cost=initial_cost, best_cost=initial_cost,
            deadline=time.time() + 10
        )
        if improved_0:
            improved_overall_max_level_0 = True
            break
    assert not improved_overall_max_level_0, \
        "Step should find no improvement if MAX_LEVEL = 0"

    # Scenario 2: MAX_LEVEL = 1 (allows 2-opts)
    LK_CONFIG["MAX_LEVEL"] = 1
    # Ensure BREADTH array is long enough for MAX_LEVEL
    # If MAX_LEVEL is n, BREADTH should have at least n elements.
    # step uses LK_CONFIG["BREADTH"][level - 1]. So for level 1, needs BREADTH[0].
    LK_CONFIG["BREADTH"] = [1] * LK_CONFIG["MAX_LEVEL"]

    improved_overall_max_level_1 = False
    best_flip_sequence_max_level_1 = None
    for base_node_idx_1 in range(tour.n):
        base_node_1 = tour.order[base_node_idx_1]
        current_tour_for_step_1 = Tour(tour.get_tour(), dist_matrix)
        improved_1, flip_sequence_1 = step(
            level=1, delta=0.0, base=base_node_1,
            tour=current_tour_for_step_1, D=dist_matrix, neigh=neighbors,
            flip_seq=[], start_cost=initial_cost, best_cost=initial_cost,
            deadline=time.time() + 10
        )
        if improved_1:
            improved_overall_max_level_1 = True
            best_flip_sequence_max_level_1 = flip_sequence_1
            break

    assert improved_overall_max_level_1, \
        "Step should find an improvement if MAX_LEVEL = 1 and a 2-opt exists"
    assert best_flip_sequence_max_level_1 is not None

    # Verify it's the expected 2-opt
    assert len(best_flip_sequence_max_level_1) == 1
    flip_pair = best_flip_sequence_max_level_1[0]
    assert flip_pair == (2, 3) or flip_pair == (3, 2)

    temp_tour_check = Tour(tour.get_tour(), dist_matrix)
    for fp_pair in best_flip_sequence_max_level_1:
        temp_tour_check.flip_and_update_cost(fp_pair[0], fp_pair[1], dist_matrix)
    assert temp_tour_check.cost == pytest.approx(8.0)
    assert temp_tour_check.get_tour() == [0, 1, 3, 2, 4]

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)


def test_step_respects_deadline(simple_tsp_setup):
    """
    Test if step() correctly respects the deadline.
    """
    _coords, dist_matrix, tour, neighbors = simple_tsp_setup
    initial_cost = tour.cost
    assert initial_cost is not None, "Initial cost should be a float."

    original_lk_config = LK_CONFIG.copy()
    LK_CONFIG["MAX_LEVEL"] = 2  # Allow some search depth
    LK_CONFIG["BREADTH"] = [5] * LK_CONFIG["MAX_LEVEL"]  # Allow some breadth

    # We'll use the first base node for simplicity in this test
    base_node = tour.order[0]
    current_tour_for_step = Tour(tour.get_tour(), dist_matrix)

    # Scenario: Deadline is immediate (or in the past)
    # time.time() will be called inside step. We set deadline to current time.
    # Any operation inside step will make current_time > deadline.
    immediate_deadline = time.time()

    improved, flip_sequence = step(
        level=1, delta=0.0, base=base_node,
        tour=current_tour_for_step, D=dist_matrix, neigh=neighbors,
        flip_seq=[], start_cost=initial_cost, best_cost=initial_cost,
        deadline=immediate_deadline
    )

    assert not improved, "Step should not find improvement if deadline is immediately hit"
    assert flip_sequence is None, "Flip sequence should be None if deadline is hit early"

    # It might be good to also test with a deadline that allows some work but not all.
    # This requires more intricate mocking of time.time() or a very slow operation.
    # For now, an immediate deadline test is a good start.

    LK_CONFIG.clear()
    LK_CONFIG.update(original_lk_config)

# ... More tests for alternate_step, lk_search etc. would follow ...
