import numpy as np
import pytest
from materialite import SlipSystem, Vector
from numpy.testing import assert_allclose


@pytest.fixture
def octahedral_slip_systems():
    normals = np.zeros((12, 3))
    directions = np.zeros((12, 3))
    normals[0, :] = [1, 1, -1]
    normals[1, :] = [1, 1, -1]
    normals[2, :] = [1, 1, -1]
    normals[3, :] = [1, -1, -1]
    normals[4, :] = [1, -1, -1]
    normals[5, :] = [1, -1, -1]
    normals[6, :] = [1, -1, 1]
    normals[7, :] = [1, -1, 1]
    normals[8, :] = [1, -1, 1]
    normals[9, :] = [1, 1, 1]
    normals[10, :] = [1, 1, 1]
    normals[11, :] = [1, 1, 1]

    directions[0, :] = [0, 1, 1]
    directions[1, :] = [1, 0, 1]
    directions[2, :] = [1, -1, 0]
    directions[3, :] = [0, 1, -1]
    directions[4, :] = [1, 0, 1]
    directions[5, :] = [1, 1, 0]
    directions[6, :] = [0, 1, 1]
    directions[7, :] = [1, 0, -1]
    directions[8, :] = [1, 1, 0]
    directions[9, :] = [0, 1, -1]
    directions[10, :] = [1, 0, -1]
    directions[11, :] = [1, -1, 0]
    return normals / np.sqrt(3), directions / np.sqrt(2)


def check_slip_systems(slip_systems, expected_normals, expected_directions):
    assert_allclose(slip_systems.normal.components, expected_normals)
    assert_allclose(slip_systems.direction.components, expected_directions)


def test_initialize_slip_system():
    slip_plane_normal = Vector(np.array([1, 1, 1]))
    expected_normal = slip_plane_normal.components / np.sqrt(3)
    slip_direction = Vector(np.array([1, -1, 0]))
    expected_direction = slip_direction.components / np.sqrt(2)

    slip_system = SlipSystem(slip_plane_normal, slip_direction)

    assert_allclose(slip_system.normal.components, expected_normal)
    assert_allclose(slip_system.direction.components, expected_direction)


def test_octahedral(octahedral_slip_systems):
    expected_normals = octahedral_slip_systems[0]
    expected_directions = octahedral_slip_systems[1]
    slip_systems = SlipSystem.octahedral()
    check_slip_systems(slip_systems, expected_normals, expected_directions)


def test_basal():
    r3b2 = np.sqrt(3) / 2
    expected_normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    expected_directions = np.array([[1, 0, 0], [-0.5, r3b2, 0], [-0.5, -r3b2, 0]])
    slip_systems = SlipSystem.basal()
    check_slip_systems(slip_systems, expected_normals, expected_directions)


def test_prismatic():
    r3b2 = np.sqrt(3) / 2
    expected_normals = np.array([[r3b2, 0.5, 0], [0, -1, 0], [-r3b2, 0.5, 0]])
    expected_directions = np.array([[-0.5, r3b2, 0], [1, 0, 0], [-0.5, -r3b2, 0]])
    slip_systems = SlipSystem.prismatic()
    check_slip_systems(slip_systems, expected_normals, expected_directions)


def test_pyramidal_a():
    # this choice helps with manual calculations
    c_over_a = 1.5
    a_over_c = 2 / 3
    r3 = np.sqrt(3)
    r3b2 = r3 / 2
    expected_normals = (
        np.array(
            [
                [1, 1 / r3, a_over_c],
                [0, -2 / r3, a_over_c],
                [-1, 1 / r3, a_over_c],
                [-1, -1 / r3, a_over_c],
                [0, 2 / r3, a_over_c],
                [1, -1 / r3, a_over_c],
            ]
        )
        * 0.75
    )
    expected_directions = np.array(
        [
            [-0.5, r3b2, 0],
            [1, 0, 0],
            [-0.5, -r3b2, 0],
            [-0.5, r3b2, 0],
            [1, 0, 0],
            [0.5, r3b2, 0],
        ]
    )
    slip_systems = SlipSystem.pyramidal_a(c_over_a=c_over_a)
    check_slip_systems(slip_systems, expected_normals, expected_directions)


def test_pyramidal_ca():
    # this choice helps with manual calculations
    c_over_a = 1.5
    a_over_c = 2 / 3
    r3 = np.sqrt(3)
    r3b2 = r3 / 2
    expected_normals = (
        np.array(
            [
                [1, 1 / r3, a_over_c],
                [1, 1 / r3, a_over_c],
                [0, -2 / r3, a_over_c],
                [0, -2 / r3, a_over_c],
                [-1, 1 / r3, a_over_c],
                [-1, 1 / r3, a_over_c],
                [-1, -1 / r3, a_over_c],
                [-1, -1 / r3, a_over_c],
                [0, 2 / r3, a_over_c],
                [0, 2 / r3, a_over_c],
                [1, -1 / r3, a_over_c],
                [1, -1 / r3, a_over_c],
            ]
        )
        * 0.75
    )
    expected_directions = (
        np.array(
            [
                [-3 / 2, -3 * r3b2, 3 * c_over_a],
                [-3, 0, 3 * c_over_a],
                [3 / 2, 3 * r3b2, 3 * c_over_a],
                [-3 / 2, 3 * r3b2, 3 * c_over_a],
                [3, 0, 3 * c_over_a],
                [3 / 2, -3 * r3b2, 3 * c_over_a],
                [3, 0, 3 * c_over_a],
                [3 / 2, 3 * r3b2, 3 * c_over_a],
                [-3 / 2, -3 * r3b2, 3 * c_over_a],
                [3 / 2, -3 * r3b2, 3 * c_over_a],
                [-3, 0, 3 * c_over_a],
                [-3 / 2, 3 * r3b2, 3 * c_over_a],
            ]
        )
        * 2
        / (3 * np.sqrt(13))
    )
    slip_systems = SlipSystem.pyramidal_ca(c_over_a=c_over_a)
    check_slip_systems(slip_systems, expected_normals, expected_directions)


def test_max_schmid_factor():
    c_over_a = 1.5
    direction_111 = Vector([1.0, 1.0, 1.0])
    directions_111_100 = Vector([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])

    r3 = np.sqrt(3)
    r6 = np.sqrt(6)
    r13 = np.sqrt(13)
    expected_octahedral_max_schmid_factor_111 = 2 / (3 * r6)  # 0.272
    expected_octahedral_max_schmid_factor_100 = 1 / r6  # 0.408
    expected_basal_max_schmid_factor_111 = (1 + r3) / 6  # 0.455
    expected_prismatic_max_schmid_factor_111 = 1 / 3
    expected_pyramidal_a_max_schmid_factor_111 = (1 + r3) / 6  # 0.455
    expected_pyramidal_ca_max_schmid_factor_111 = (7 + 5 * r3) / (12 * r13)  # 0.362

    assert_allclose(
        SlipSystem.octahedral().max_schmid_factor(direction_111).components,
        expected_octahedral_max_schmid_factor_111,
    )
    assert_allclose(
        SlipSystem.basal().max_schmid_factor(direction_111).components,
        expected_basal_max_schmid_factor_111,
    )
    assert_allclose(
        SlipSystem.prismatic().max_schmid_factor(direction_111).components,
        expected_prismatic_max_schmid_factor_111,
    )
    assert_allclose(
        SlipSystem.pyramidal_a(c_over_a=c_over_a)
        .max_schmid_factor(direction_111)
        .components,
        expected_pyramidal_a_max_schmid_factor_111,
    )
    assert_allclose(
        SlipSystem.pyramidal_ca(c_over_a=c_over_a)
        .max_schmid_factor(direction_111)
        .components,
        expected_pyramidal_ca_max_schmid_factor_111,
    )
    assert_allclose(
        SlipSystem.octahedral().max_schmid_factor(directions_111_100).components,
        [
            expected_octahedral_max_schmid_factor_111,
            expected_octahedral_max_schmid_factor_100,
        ],
    )


def test_concatenate():
    r3b2 = np.sqrt(3) / 2
    expected_normals = np.array(
        [[0, 0, 1], [0, 0, 1], [0, 0, 1], [r3b2, 0.5, 0], [0, -1, 0], [-r3b2, 0.5, 0]]
    )
    expected_directions = np.array(
        [
            [1, 0, 0],
            [-0.5, r3b2, 0],
            [-0.5, -r3b2, 0],
            [-0.5, r3b2, 0],
            [1, 0, 0],
            [-0.5, -r3b2, 0],
        ]
    )
    slip_systems = SlipSystem.basal().concatenate(SlipSystem.prismatic())
    check_slip_systems(slip_systems, expected_normals, expected_directions)
