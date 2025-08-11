import numpy as np
import pandas as pd
import pytest  # Includes: tmp_path, mocker
from materialite import Material
from materialite.models import GrainCoarseningModel
from materialite.models.grain_coarsening_model import (
    _get_energy,
    _get_neighbors,
    calculate_potts_energy,
)
from numpy.testing import assert_array_equal


@pytest.fixture
def model():
    max_grain_id = 1
    num_attempts = 300
    seed = 12345
    # sample_points = [5, 2, 3]
    # test_spins = [0, 0, 1]
    return GrainCoarseningModel(
        max_grain_id=max_grain_id, num_flip_attempts=num_attempts, seed=seed
    )


@pytest.fixture
def material():
    material = Material(dimensions=[2, 2, 2], spacing=[3, 3, 3])
    spin_field = np.ones(8)
    spin_field[3] = 0
    material = material.create_fields({"grain": spin_field})
    material = material.create_uniform_field("temperature", 1.0)
    material = material.create_uniform_field("mobility", 1.0)

    return material


def test_assign_neighbors():
    material = Material(dimensions=[2, 2, 3])
    neighbors, num_neighbors = _get_neighbors(material, np.sqrt(3))
    expected_num_neighbors = [7, 11, 7, 7, 11, 7, 7, 11, 7, 7, 11, 7]
    raw_neighbors = [
        [10, 9, 7, 6, 4, 3, 1],
        [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 0],
        [11, 10, 8, 7, 5, 4, 1],
        [10, 9, 7, 6, 4, 1, 0],
        [11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 0],
        [11, 10, 8, 7, 4, 2, 1],
        [10, 9, 7, 4, 3, 1, 0],
        [11, 10, 9, 8, 6, 5, 4, 3, 2, 1, 0],
        [11, 10, 7, 5, 4, 2, 1],
        [10, 7, 6, 4, 3, 1, 0],
        [11, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [10, 8, 7, 5, 4, 2, 1],
    ]
    expected_neighbors = np.array([r + [-1] * (27 - len(r)) for r in raw_neighbors])
    assert_array_equal(num_neighbors, expected_num_neighbors)

    assert_array_equal(np.sort(neighbors[0])[::-1], expected_neighbors[0])
    assert_array_equal(np.sort(neighbors[2])[::-1], expected_neighbors[2])
    assert_array_equal(np.sort(neighbors[5])[::-1], expected_neighbors[5])


def test_run_model(model, material):
    expected_spins = np.ones(8, dtype=int)
    expected_successful_attempts = 300

    new_material = model(material)

    assert_array_equal(new_material.extract("grain"), expected_spins)
    assert (
        new_material.state["successful_flip_attempts"] == expected_successful_attempts
    )


def test_get_energy():
    material = Material(dimensions=[2, 2, 3])
    neighbors, num_neighbors = _get_neighbors(material, np.sqrt(3))

    spin_field = np.ones(12)
    material = material.create_fields({"grain": spin_field})
    num_points = 12

    current_energy = _get_energy(num_points, neighbors, num_neighbors, spin_field)
    expected_energy = 0
    assert_array_equal(current_energy, expected_energy)

    spin_field[3] = 0

    current_energy = _get_energy(num_points, neighbors, num_neighbors, spin_field)
    expected_energy = 14
    assert_array_equal(current_energy, expected_energy)


def test_calculate_potts_energy(model, material):
    current_energy = calculate_potts_energy(material, model)
    expected_energy = 14
    assert_array_equal(current_energy, expected_energy)
