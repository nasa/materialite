import numpy as np
import pytest
from materialite import Orientation, SlipSystem, Vector, get_ipf, get_ipf_colors
from materialite.get_ipf_colors import get_symmetry_operators
from numpy.testing import assert_allclose


@pytest.fixture
def rotation_matrices_cubic():
    slip_systems = SlipSystem.octahedral()
    normals = slip_systems.normal.components
    directions = slip_systems.direction.components
    normals = np.concatenate([normals, -normals], axis=0)
    directions = np.concatenate([directions, -directions], axis=0)
    transverse_111 = np.cross(normals, directions)
    transverse_110 = np.cross(directions, normals)
    # specimen [0,0,1] aligns with crystal [1,1,1]
    R_111 = []
    # specimen [0,0,1] aligns with crystal [1,1,0]
    R_110 = []
    for n, d, t111, t110 in zip(normals, directions, transverse_111, transverse_110):
        R_111.append(np.c_[d, t111, n])
        R_110.append(np.c_[n, t110, d])
    # specimen [0,0,1] aligns with crystal [1,0,0]
    R_100 = get_symmetry_operators("cubic")
    return np.concatenate([R_100, R_110, R_111], axis=0)


@pytest.fixture
def rotation_matrices_hcp():
    slip_systems = SlipSystem.prismatic()
    normals = slip_systems.normal.components
    directions = slip_systems.direction.components
    normals = np.concatenate([normals, -normals], axis=0)
    directions = np.concatenate([directions, -directions], axis=0)
    transverse_nd = np.cross(normals, directions)
    transverse_dn = np.cross(directions, normals)
    R_2110 = []
    R_1010 = []
    R_0001 = []
    for n, d, tnd, tdn in zip(normals, directions, transverse_nd, transverse_dn):
        R_2110.append(np.c_[n, tdn, d])
        R_2110.append(np.c_[tnd, n, d])
        R_1010.append(np.c_[d, tnd, n])
        R_1010.append(np.c_[tdn, d, n])
        R_0001.append(np.c_[n, d, tnd])
        R_0001.append(np.c_[d, n, tdn])
    return np.concatenate([R_2110, R_1010, R_0001], axis=0)


def test_ipf_corners_map_to_rgb_cubic(rotation_matrices_cubic):
    direction = Vector([0, 0, 1])
    orientations = Orientation(rotation_matrices_cubic)
    expected_red = np.repeat([[1, 0, 0]], 24, axis=0)
    expected_green = np.repeat([[0, 1, 0]], 24, axis=0)
    expected_blue = np.repeat([[0, 0, 1]], 24, axis=0)

    colors = get_ipf_colors(direction, orientations, "cubic")
    red = colors[:24, :]
    green = colors[24:48, :]
    blue = colors[48:, :]
    assert_allclose(red, expected_red)
    assert_allclose(green, expected_green)
    assert_allclose(blue, expected_blue)


def test_ipf_corners_map_to_rgb_cubic_no_tensor(rotation_matrices_cubic):
    direction = [0, 0, 1]
    orientations = Orientation(rotation_matrices_cubic)
    expected_red = np.repeat([[1, 0, 0]], 24, axis=0)
    expected_green = np.repeat([[0, 1, 0]], 24, axis=0)
    expected_blue = np.repeat([[0, 0, 1]], 24, axis=0)

    colors = get_ipf_colors(direction, orientations, "cubic")
    red = colors[:24, :]
    green = colors[24:48, :]
    blue = colors[48:, :]
    assert_allclose(red, expected_red)
    assert_allclose(green, expected_green)
    assert_allclose(blue, expected_blue)


def test_ipf_corners_map_to_rgb_hcp(rotation_matrices_hcp):
    direction = Vector(np.array([0, 0, 1]))
    orientations = Orientation(rotation_matrices_hcp)
    expected_red = np.repeat([[1, 0, 0]], 12, axis=0)
    expected_green = np.repeat([[0, 1, 0]], 12, axis=0)
    expected_blue = np.repeat([[0, 0, 1]], 12, axis=0)

    colors = get_ipf_colors(direction, orientations, "hcp")
    green = colors[:12, :]
    blue = colors[12:24, :]
    red = colors[24:, :]
    assert_allclose(red, expected_red, atol=1.0e-7)
    assert_allclose(green, expected_green, atol=1.0e-7)
    assert_allclose(blue, expected_blue, atol=1.0e-7)


def test_ipf_corners_stereographic_projection_cubic(rotation_matrices_cubic):
    direction = Vector(np.array([0, 0, 1]))
    orientations = Orientation(rotation_matrices_cubic)
    xy111 = 1 / (np.sqrt(3) + 1)
    expected_100_corner = np.repeat([[0, 0]], 24, axis=0)
    expected_101_corner = np.repeat([[np.sqrt(2) - 1, 0]], 24, axis=0)
    expected_111_corner = np.repeat([[xy111, xy111]], 24, axis=0)
    expected_corners = np.concatenate(
        [expected_100_corner, expected_101_corner, expected_111_corner], axis=0
    )
    ipf_points, _ = get_ipf(direction, orientations, "cubic")
    assert_allclose(ipf_points, expected_corners)
