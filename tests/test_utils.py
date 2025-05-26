"""
Unit tests for utility functions in the Lin-Kernighan TSP solver.

This module tests functionalities such as distance matrix calculation,
Delaunay neighbor finding, tour perturbation (double bridge), and
parsing of TSPLIB and optimal tour files.
"""
import pytest
import numpy as np
from pathlib import Path
from lin_kernighan_tsp_solver.lin_kernighan_tsp_solver import (
    build_distance_matrix,
    delaunay_neighbors,
    double_bridge,
    read_opt_tour,
    read_tsp_file
)


def test_build_distance_matrix():
    """Tests the calculation of the distance matrix from coordinates."""
    coords = np.array([[0, 0], [3, 0], [0, 4]])
    expected_d_matrix = np.array([
        [0, 3, 4],
        [3, 0, 5],
        [4, 5, 0]
    ])
    d_matrix = build_distance_matrix(coords)
    np.testing.assert_array_almost_equal(d_matrix, expected_d_matrix)

    coords_empty_2d = np.empty((0, 2))
    d_matrix_empty = build_distance_matrix(coords_empty_2d)
    assert d_matrix_empty.shape == (0, 0)

    coords_one = np.array([[1, 1]])
    d_matrix_one = build_distance_matrix(coords_one)
    expected_d_matrix_one = np.array([[0.0]])
    np.testing.assert_array_almost_equal(d_matrix_one, expected_d_matrix_one)


def test_delaunay_neighbors():
    """Tests the Delaunay neighbor finding functionality."""
    assert delaunay_neighbors(np.empty((0, 2))) == []

    coords1 = np.array([[0, 0]])
    assert delaunay_neighbors(coords1) == [[]]

    coords2 = np.array([[0, 0], [1, 1]])
    neighbors2 = delaunay_neighbors(coords2)
    assert len(neighbors2) == 2
    assert sorted(neighbors2[0]) == [1]
    assert sorted(neighbors2[1]) == [0]

    coords_triangle = np.array([[0, 0], [1, 0], [0, 1]])
    neighbors_triangle = delaunay_neighbors(coords_triangle)
    assert len(neighbors_triangle) == 3
    assert sorted(neighbors_triangle[0]) == [1, 2]
    assert sorted(neighbors_triangle[1]) == [0, 2]
    assert sorted(neighbors_triangle[2]) == [0, 1]

    coords_square = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    neighbors_square = delaunay_neighbors(coords_square)
    assert len(neighbors_square) == 4
    for i in range(4):
        assert len(neighbors_square[i]) >= 2
        for neighbor in neighbors_square[i]:
            assert i in neighbors_square[neighbor]


def test_double_bridge_kick():
    """Tests the double bridge tour perturbation."""
    np.random.seed(42)
    original_tour_10 = list(range(10))
    kicked_tour_10 = double_bridge(original_tour_10.copy())
    assert len(kicked_tour_10) == len(original_tour_10)
    assert sorted(kicked_tour_10) == sorted(original_tour_10)
    assert kicked_tour_10 != original_tour_10

    original_tour_5 = list(range(5))
    kicked_tour_5 = double_bridge(original_tour_5.copy())
    assert kicked_tour_5 != original_tour_5

    for n_nodes in [0, 1, 2, 3, 4]:
        small_tour = list(range(n_nodes))
        kicked_small_tour = double_bridge(small_tour.copy())
        assert kicked_small_tour == small_tour


def test_read_opt_tour(tmp_path: Path):
    """Tests reading of .opt.tour files."""
    content_valid = (
        "NAME : test.opt.tour\nTYPE : TOUR\nDIMENSION : 4\n"
        "TOUR_SECTION\n1\n2\n3\n4\n-1\nEOF\n"
    )
    opt_tour_file_valid = tmp_path / "test_valid.opt.tour"
    opt_tour_file_valid.write_text(content_valid)
    tour = read_opt_tour(str(opt_tour_file_valid))
    assert tour == [0, 1, 2, 3]

    assert read_opt_tour(str(tmp_path / "non_existent.opt.tour")) is None

    content_no_tour_section = "NAME : test.opt.tour\nTYPE : TOUR\n1\n2\n-1\nEOF\n"
    opt_tour_file_no_section = tmp_path / "test_no_section.opt.tour"
    opt_tour_file_no_section.write_text(content_no_tour_section)
    assert read_opt_tour(str(opt_tour_file_no_section)) is None

    content_non_integer = "NAME : test.opt.tour\nTOUR_SECTION\n1\nA\n3\n-1\nEOF\n"
    opt_tour_file_non_integer = tmp_path / "test_non_integer.opt.tour"
    opt_tour_file_non_integer.write_text(content_non_integer)
    assert read_opt_tour(str(opt_tour_file_non_integer)) is None

    content_no_neg_one = "NAME : test.opt.tour\nTOUR_SECTION\n1\n2\n3\nEOF\n"
    opt_tour_file_no_neg_one = tmp_path / "test_no_neg_one.opt.tour"
    opt_tour_file_no_neg_one.write_text(content_no_neg_one)
    assert read_opt_tour(str(opt_tour_file_no_neg_one)) is None


def test_read_tsp_file_valid_euc_2d(tmp_path: Path):
    """Tests reading a valid EUC_2D .tsp file."""
    content = """
NAME: test_valid
TYPE: TSP
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
    coords = read_tsp_file(str(tsp_file))
    expected_coords = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    np.testing.assert_array_almost_equal(coords, expected_coords)


def test_read_tsp_file_not_found(tmp_path: Path):
    """Tests reading a non-existent .tsp file."""
    with pytest.raises(FileNotFoundError):
        read_tsp_file(str(tmp_path / "non_existent.tsp"))


def test_read_tsp_file_error_cases(tmp_path: Path):
    """Tests reading .tsp files with various malformed content."""
    content_non_numeric_coord = "NODE_COORD_SECTION\n1 10.0 ABC\nEOF\n"
    tsp_file_non_numeric_coord = tmp_path / "test_non_numeric_coord.tsp"
    tsp_file_non_numeric_coord.write_text(content_non_numeric_coord)
    coords_non_numeric = read_tsp_file(str(tsp_file_non_numeric_coord))
    assert coords_non_numeric.shape == (0, 2) or coords_non_numeric.shape == (0,)

    content_empty_nodes = "NODE_COORD_SECTION\nEOF\n"
    tsp_file_empty_nodes = tmp_path / "test_empty_nodes.tsp"
    tsp_file_empty_nodes.write_text(content_empty_nodes)
    coords_empty = read_tsp_file(str(tsp_file_empty_nodes))
    assert coords_empty.shape == (0, 2) or coords_empty.shape == (0,)

    content_no_dim_but_parses = "NODE_COORD_SECTION\n1 10.0 20.0\nEOF\n"
    tsp_file_no_dim = tmp_path / "test_no_dim.tsp"
    tsp_file_no_dim.write_text(content_no_dim_but_parses)
    coords_no_dim = read_tsp_file(str(tsp_file_no_dim))
    np.testing.assert_array_almost_equal(coords_no_dim, np.array([[10.0, 20.0]]))
