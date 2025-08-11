import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from materialite import Material, Scalar, read_from_json, write_to_json


@pytest.fixture
def material():
    num_regions = 10
    rng = np.random.default_rng(12345)
    test_field = rng.random(8**3)
    array_test_field = rng.random((8**3, 3))
    regions = np.arange(num_regions)
    region_field = pd.DataFrame(
        {
            "grain": regions,
            "grain_data": regions * 0.1,
        }
    )
    state = {
        "1.0": {"stress": 1},
        "2.0": {"strain": [1, 2, 3]},
    }
    material = (
        Material(dimensions=[8, 8, 8], origin=[1, 2, 3], spacing=[3, 2, 1])
        .create_voronoi(num_regions=num_regions, label="grain", rng=rng)
        .assign_random_orientations(region_label="grain")
        .create_fields(
            {
                "test": test_field,
                "scalar_test": Scalar(test_field * 100),
                "array_test": list(array_test_field),
            }
        )
        .create_regional_fields("grain", region_field)
    )
    material.state = state
    return material


def test_write_and_read_yields_same_material(tmp_path, material):
    fname = tmp_path / "test.txt"
    write_to_json(material, fname)
    loaded_material = read_from_json(fname)
    columns = [
        "x",
        "y",
        "z",
        "x_id",
        "y_id",
        "z_id",
        "grain",
        "test",
        "array_test",
    ]
    assert_array_equal(loaded_material.dimensions, material.dimensions)
    assert_array_equal(loaded_material.spacing, material.spacing)
    assert_array_equal(loaded_material.sizes, material.sizes)
    assert_frame_equal(
        material.get_fields()[columns], loaded_material.get_fields()[columns]
    )
    assert_allclose(
        material.extract("orientation").rotation_matrix,
        loaded_material.extract("orientation").rotation_matrix,
    )
    assert_allclose(
        material.extract("scalar_test").components,
        loaded_material.extract("scalar_test").components,
    )
    assert_frame_equal(
        loaded_material.extract_regional_field("grain").drop("orientation", axis=1),
        material.extract_regional_field("grain").drop("orientation", axis=1),
    )
    assert loaded_material.state == material.state


def test_compressed_write_and_read_yields_same_material(tmp_path, material):
    fname = tmp_path / "test.txt"
    write_to_json(material, fname, compress=True)
    loaded_material = read_from_json(fname, decompress=True)
    columns = [
        "x",
        "y",
        "z",
        "x_id",
        "y_id",
        "z_id",
        "grain",
        "test",
        "array_test",
    ]
    assert_array_equal(loaded_material.dimensions, material.dimensions)
    assert_array_equal(loaded_material.spacing, material.spacing)
    assert_array_equal(loaded_material.sizes, material.sizes)
    assert_frame_equal(
        material.get_fields()[columns], loaded_material.get_fields()[columns]
    )
    assert_allclose(
        material.extract("orientation").rotation_matrix,
        loaded_material.extract("orientation").rotation_matrix,
    )
    assert_allclose(
        material.extract("scalar_test").components,
        loaded_material.extract("scalar_test").components,
    )
    assert_frame_equal(
        loaded_material.extract_regional_field("grain").drop("orientation", axis=1),
        material.extract_regional_field("grain").drop("orientation", axis=1),
    )
    assert loaded_material.state == material.state


def test_read_and_write_simple_material(tmp_path):
    fname = tmp_path / "test.txt"
    material = Material()
    write_to_json(material, fname)
    loaded_material = read_from_json(fname)
    assert_array_equal(loaded_material.dimensions, material.dimensions)
    assert_array_equal(loaded_material.spacing, material.spacing)
    assert_array_equal(loaded_material.sizes, material.sizes)
    assert_frame_equal(loaded_material.get_fields(), material.get_fields())
    assert not loaded_material.state
    assert not loaded_material._regional_fields
