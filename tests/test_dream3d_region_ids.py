"""DREAM.3D feature arrays are indexed by the cell feature ID."""

import hashlib

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from materialite import import_dream3d


@pytest.fixture
def dream3d_file(tmp_path):
    def create(feature_ids, number_of_rows, components):
        path = tmp_path / "regions.dream3d"
        ids = np.asarray(feature_ids, dtype=np.int32)
        values = (
            np.arange(number_of_rows * components, dtype=np.float32).reshape(
                number_of_rows, components
            )
            + 0.25
        )
        with h5py.File(path, "w") as data:
            geometry = data.create_group("Volume/_SIMPL_GEOMETRY")
            geometry["DIMENSIONS"] = [2, 2, 2]
            geometry["ORIGIN"] = [1.0, 2.0, 3.0]
            geometry["SPACING"] = [0.5, 1.0, 2.0]
            data["Volume/CellData/FeatureIds"] = ids.reshape(2, 2, 2, 1)
            data["Volume/CellData/PointValue"] = np.arange(8).reshape(2, 2, 2, 1)
            data["Volume/FeatureData/Property"] = values
            data["Volume/FeatureData/Count"] = np.arange(
                number_of_rows, dtype=np.int32
            ).reshape(-1, 1)
        return path, ids, values

    return create


@pytest.mark.parametrize("components", [1, 3])
@pytest.mark.parametrize(
    "feature_ids,number_of_rows",
    [
        ([1, 2, 3, 1, 2, 3, 1, 2], 4),
        ([0, 1, 2, 3, 0, 1, 2, 3], 4),
        ([1, 4, 1, 4, 4, 1, 4, 1], 5),
        ([0, 4, 0, 4, 4, 0, 4, 0], 5),
        ([3, 3, 3, 3, 3, 3, 3, 3], 4),
        ([0, 0, 0, 0, 0, 0, 0, 0], 4),
        ([1, 2, 1, 2, 2, 1, 2, 1], 7),
    ],
    ids=[
        "contiguous",
        "contiguous-zero",
        "sparse",
        "sparse-zero",
        "one-feature",
        "background-only",
        "unused-trailing-rows",
    ],
)
def test_feature_rows_follow_present_ids(
    dream3d_file, feature_ids, number_of_rows, components
):
    path, ids, values = dream3d_file(feature_ids, number_of_rows, components)
    original_hash = hashlib.sha256(path.read_bytes()).digest()
    material = import_dream3d(
        path,
        "Volume/_SIMPL_GEOMETRY",
        field_paths=["Volume/CellData/PointValue"],
        region_id_path="Volume/CellData/FeatureIds",
        region_field_paths=["Volume/FeatureData/Property", "Volume/FeatureData/Count"],
    )

    expected_ids = np.unique(ids)
    regional = material.extract_regional_field("feature_ids")
    assert_array_equal(regional["feature_ids"], expected_ids)
    assert_array_equal(regional["count"], expected_ids)
    points = material.get_fields().sort_values("point_value")
    assert_array_equal(points["feature_ids"], ids)
    assert_array_equal(points["count"], ids)
    for index in range(components):
        label = "property" if components == 1 else f"property_{index + 1}"
        assert_array_equal(regional[label], values[expected_ids, index])
        assert_array_equal(points[label], values[ids, index])
        assert regional[label].dtype == np.float32
    assert_array_equal(material.dimensions, [2, 2, 2])
    assert_array_equal(material.origin, [1.0, 2.0, 3.0])
    assert_array_equal(material.spacing, [0.5, 1.0, 2.0])
    assert hashlib.sha256(path.read_bytes()).digest() == original_hash


@pytest.mark.parametrize("region_field_paths", [None, []])
def test_sparse_ids_without_feature_properties(dream3d_file, region_field_paths):
    path, ids, _ = dream3d_file([1, 4, 1, 4, 4, 1, 4, 1], 5, 1)
    material = import_dream3d(
        path,
        "Volume/_SIMPL_GEOMETRY",
        field_paths=["Volume/CellData/PointValue"],
        region_id_path="Volume/CellData/FeatureIds",
        region_field_paths=region_field_paths,
    )
    points = material.get_fields().sort_values("point_value")
    assert_array_equal(points["feature_ids"], ids)
    assert "property" not in points
