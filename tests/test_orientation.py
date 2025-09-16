import numpy as np
import pytest  # Includes: tmp_path, mocker
from numpy import array, pi
from numpy.testing import assert_allclose, assert_array_equal

from materialite import Orientation, Vector


@pytest.fixture
def seeded_rng():
    return np.random.default_rng(seed=12345)


@pytest.fixture
def cubic_symmetry():
    i = np.array([1, 0, 0])
    j = np.array([0, 1, 0])
    k = np.array([0, 0, 1])
    symmetry_matrices = [0] * 12
    symmetry_matrices[0] = np.eye(3)
    symmetry_matrices[1] = np.array([k, i, j])
    symmetry_matrices[2] = np.array([j, k, i])
    symmetry_matrices[3] = np.array([-j, k, -i])
    symmetry_matrices[4] = np.array([-j, -k, i])
    symmetry_matrices[5] = np.array([j, -k, -i])
    symmetry_matrices[6] = np.array([-k, i, -j])
    symmetry_matrices[7] = np.array([-k, -i, j])
    symmetry_matrices[8] = np.array([k, -i, -j])
    symmetry_matrices[9] = np.array([-i, j, -k])
    symmetry_matrices[10] = np.array([-i, -j, k])
    symmetry_matrices[11] = np.array([i, -j, -k])
    return Orientation(symmetry_matrices)


def test_get_rotation_matrix():
    rotation_matrix = np.eye(3)
    orientation = Orientation.from_rotation_matrix(rotation_matrix)
    assert_array_equal(rotation_matrix, orientation.rotation_matrix)


def test_initialize_with_euler_angles():
    euler_angles = [pi / 2, pi / 2, pi / 2]
    expected_rotation_matrix = array(
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    orientation = Orientation.from_euler_angles(euler_angles)
    assert_allclose(expected_rotation_matrix, orientation.rotation_matrix, atol=1e-14)


def test_euler_angles_R22_equals_one():
    euler_angles = [pi / 6, 0, pi / 4]
    expected_euler_angles = [5 * pi / 12, 0, 0]
    orientation = Orientation.from_euler_angles(euler_angles)
    assert_allclose(orientation.euler_angles, expected_euler_angles)
    assert_allclose(
        orientation.rotation_matrix,
        Orientation.from_euler_angles(expected_euler_angles).rotation_matrix,
        atol=1e-15,
    )


def test_euler_angles_R22_equals_minus_one():
    euler_angles = [pi / 6, pi, pi / 4]
    expected_euler_angles = [-pi / 12, pi, 0]
    orientation = Orientation.from_euler_angles(euler_angles)
    assert_allclose(orientation.euler_angles, expected_euler_angles)
    assert_allclose(
        orientation.rotation_matrix,
        Orientation.from_euler_angles(expected_euler_angles).rotation_matrix,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    "euler_angles", [[pi / 3, pi / 6, pi / 2], [pi / 2, pi / 3, pi / 6], [0, 0, 0]]
)
def test_get_orientation_from_euler_angles(euler_angles):
    orientation = Orientation.from_euler_angles(euler_angles)
    assert_allclose(euler_angles, orientation.euler_angles)


def test_get_orientation_from_euler_angles_in_degrees():
    euler_angles = [90, 60, 30]
    orientation = Orientation.from_euler_angles(euler_angles, in_degrees=True)
    assert_allclose(euler_angles, orientation.euler_angles_in_degrees)


def test_get_orientations_from_euler_angles():
    euler_angles = [[pi / 3, pi / 6, pi / 2], [pi / 2, pi / 3, pi / 6], [0, 0, 0]]
    orientations = Orientation.from_euler_angles(euler_angles)
    assert_allclose(orientations.euler_angles, euler_angles)


def test_initialize_with_orientations():
    euler_angles = [[pi / 3, pi / 6, pi / 2], [pi / 2, pi / 3, pi / 6], [0, 0, 0]]
    orientations = Orientation.from_euler_angles(euler_angles)
    orientations2 = Orientation(orientations)
    assert_allclose(orientations2.euler_angles, euler_angles)
    assert orientations2.dims_str == "p"


def test_get_random_orientation(seeded_rng):
    orientation = Orientation.random(1, rng=seeded_rng)
    expected_euler_angles = np.array([1.42839436, 1.94602287, 5.00999493 - 2 * np.pi])
    assert_allclose(orientation.euler_angles, expected_euler_angles)


def test_get_rotation_matrices():
    rotation_matrices = np.tile(np.eye(3), (2, 1, 1))
    orientations = Orientation.from_rotation_matrix(rotation_matrices)
    assert_array_equal(rotation_matrices, orientations.rotation_matrix)


def test_get_orientations_from_euler_angles_with_s_dimension():
    euler_angles = [
        [[pi / 3, pi / 6, pi / 2], [pi / 2, pi / 3, pi / 6]],
        [[pi / 6, pi / 2, pi / 3], [0, 0, 0]],
    ]
    orientations = Orientation.from_euler_angles(euler_angles)
    assert orientations.dims_str == "ps"
    assert_allclose(orientations.euler_angles, euler_angles)


def test_consistent_rotation_matrices_with_extracted_euler_angles(seeded_rng):
    n = 10**3
    orientations = Orientation.random(n, rng=seeded_rng)
    expected_rotation_matrices = orientations.rotation_matrix
    euler_angles = orientations.euler_angles
    rotation_matrices = Orientation.from_euler_angles(euler_angles).rotation_matrix
    assert_allclose(rotation_matrices, expected_rotation_matrices)


def test_get_random_orientations(seeded_rng):
    orientations = Orientation.random(2, rng=seeded_rng)
    expected_euler_angles = np.array(
        [[1.42839436, 0.93386543, 2.45741378], [1.99025135, 1.2105451, 2.09113158]]
    )
    assert_allclose(orientations.euler_angles, expected_euler_angles)


@pytest.mark.parametrize(
    "euler_angles, plane, direction",
    [
        ([35.264389682754654, 45, 0], [0, 1, 1], [2, -1, 1]),
        (
            [[0, 0, 0], [35.264389682754654, 45, 0]],
            [[0, 0, 1], [0, 1, 1]],
            [[1, 0, 0], [2, -1, 1]],
        ),
    ],
)
def test_miller_indices(euler_angles, plane, direction):
    expected_rotation_matrix = Orientation.from_euler_angles(
        euler_angles, in_degrees=True
    ).rotation_matrix
    rotation_matrix = Orientation.from_miller_indices(plane, direction).rotation_matrix
    assert_allclose(rotation_matrix, expected_rotation_matrix)


def test_identity():
    orientation = Orientation.identity()
    assert_array_equal(orientation.rotation_matrix, np.eye(3))
    assert_array_equal(orientation.euler_angles, [0, 0, 0])


def test_trace():
    orientation = Orientation.identity()
    assert orientation.trace.components == 3
    rng = np.random.default_rng(0)
    orientations = Orientation.random(100, rng=rng)
    assert_allclose(
        orientations.trace.components,
        np.trace(orientations.rotation_matrix, axis1=1, axis2=2),
    )


def test_compose_orientations():
    euler_angles = np.array([30, 0, 0])
    expected_euler_angles = np.array([60, 0, 0])
    orientation = Orientation.from_euler_angles(euler_angles, in_degrees=True)
    assert_allclose(
        (orientation @ orientation).euler_angles_in_degrees,
        expected_euler_angles,
        atol=1.0e-13,
    )


def test_compose_orientations_with_burgers_orientation_relationship(cubic_symmetry):
    euler_angles = [135.0, 90.0, 324.74]
    orientation = Orientation.from_euler_angles(euler_angles, in_degrees=True)
    variants = orientation @ cubic_symmetry
    alpha_plane = Vector([0, 0, 1])
    beta_planes = np.array(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [0, -1, 1],
            [0, -1, -1],
            [0, 1, -1],
            [1, 0, -1],
            [-1, 0, -1],
            [-1, 0, 1],
            [-1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0],
        ]
    )
    alpha_direction = Vector([1, 0, 0])
    beta_directions = np.array(
        [
            [-1, 1, -1],
            [1, -1, -1],
            [-1, -1, 1],
            [1, 1, 1],
            [-1, 1, -1],
            [1, -1, -1],
            [1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, 1, 1],
            [1, -1, -1],
            [-1, -1, 1],
        ]
    )
    calculated_beta_planes = alpha_plane.to_specimen_frame(
        variants
    ).components * np.sqrt(2)
    calculated_beta_directions = alpha_direction.to_specimen_frame(
        variants
    ).components * np.sqrt(3)
    assert_allclose(calculated_beta_planes, beta_planes, atol=1.0e-13)
    assert_allclose(
        calculated_beta_directions, beta_directions, rtol=1.0e-3, atol=1.0e-13
    )


def test_compose_orientations_with_burgers_orientation_relationship_s_dimension(
    cubic_symmetry,
):
    euler_angles = [[135.0, 90.0, 324.74]] * 2
    orientation = Orientation.from_euler_angles(euler_angles, in_degrees=True, dims="s")
    variants = (orientation @ cubic_symmetry[:3]).reorder("ps")
    alpha_plane = Vector([0, 0, 1])
    beta_planes = np.array(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ]
    )
    beta_planes = np.repeat(beta_planes[:, np.newaxis, :], 2, axis=1)
    alpha_direction = Vector([1, 0, 0])
    beta_directions = np.array(
        [
            [-1, 1, -1],
            [1, -1, -1],
            [-1, -1, 1],
        ]
    )
    beta_directions = np.repeat(beta_directions[:, np.newaxis, :], 2, axis=1)
    calculated_beta_planes = alpha_plane.to_specimen_frame(
        variants
    ).components * np.sqrt(2)
    calculated_beta_directions = alpha_direction.to_specimen_frame(
        variants
    ).components * np.sqrt(3)
    assert_allclose(calculated_beta_planes, beta_planes, atol=1.0e-13)
    assert_allclose(
        calculated_beta_directions, beta_directions, rtol=1.0e-3, atol=1.0e-13
    )
