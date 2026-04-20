import numpy as np
import pytest
from materialite import Crop, Material
from numpy.testing import assert_array_equal


@pytest.fixture
def material():
    return Material(dimensions=[8, 8, 8])


# --- from_slice_by_id ---


@pytest.mark.parametrize(
    "axis, index, expected_dims",
    [
        ("z", 3, [8, 8, 1]),
        ("y", 0, [8, 1, 8]),
        (0, 5, [1, 8, 8]),
    ],
)
def test_from_slice_by_id(material, axis, index, expected_dims):
    result = Crop.from_slice_by_id(axis=axis, index=index).run(material)
    assert_array_equal(result.dimensions, expected_dims)


# --- from_box_by_id ---


def test_from_box_by_id(material):
    # center_id defaults to [3, 3, 3] for an 8x8x8 material
    result = Crop.from_box_by_id(dimensions=[4, 2, 6]).run(material)
    assert_array_equal(result.dimensions, [4, 2, 6])


def test_from_box_by_id_explicit_center(material):
    result = Crop.from_box_by_id(dimensions=[4, 2, 6], center_id=[4, 4, 4]).run(
        material
    )
    assert_array_equal(result.dimensions, [4, 2, 6])


def test_from_box_by_id_partial_axes(material):
    # None elements in dimensions span the full range for that axis
    result = Crop.from_box_by_id(dimensions=[None, None, 2]).run(material)
    assert_array_equal(result.dimensions, [8, 8, 2])


def test_from_box_by_id_no_args_unchanged(material):
    result = Crop.from_box_by_id().run(material)
    assert_array_equal(result.dimensions, material.dimensions)


# --- from_crop ---


@pytest.mark.parametrize(
    "x, y, z, expected_dims",
    [
        ((2, 5), None, None, [4, 8, 8]),
        ((2, 5), (1, 4), (0, 3), [4, 4, 4]),
        (None, None, None, [8, 8, 8]),
    ],
)
def test_from_crop(material, x, y, z, expected_dims):
    result = Crop.from_crop(x=x, y=y, z=z).run(material)
    assert_array_equal(result.dimensions, expected_dims)


# --- from_crop_by_id ---


@pytest.mark.parametrize(
    "x, y, z, expected_dims",
    [
        ((1, 4), None, None, [4, 8, 8]),
        ((0, 3), (2, 5), (1, 6), [4, 4, 6]),
        (None, None, None, [8, 8, 8]),
    ],
)
def test_from_crop_by_id(material, x, y, z, expected_dims):
    result = Crop.from_crop_by_id(x=x, y=y, z=z).run(material)
    assert_array_equal(result.dimensions, expected_dims)


# --- from_fraction ---
# Material(dimensions=[8,8,8]): int(8*0.5+0.5) = 4, starts=(8-4)//2=2 when centered.


@pytest.mark.parametrize(
    "fractions, centered, expected_dims",
    [
        ([0.5, 1.0, 1.0], True, [4, 8, 8]),
        ([0.5, 0.5, 0.5], True, [4, 4, 4]),
        ([0.5, 1.0, 1.0], False, [4, 8, 8]),
    ],
)
def test_from_fraction_dimensions(material, fractions, centered, expected_dims):
    result = Crop.from_fraction(fractions=fractions, centered=centered).run(material)
    assert_array_equal(result.dimensions, expected_dims)


def test_from_fraction_centered_starts_at_midpoint(material):
    result = Crop.from_fraction(fractions=[0.5, 1.0, 1.0], centered=True).run(material)
    assert_array_equal(result.origin, [2, 0, 0])


def test_from_fraction_uncentered_starts_at_origin(material):
    result = Crop.from_fraction(fractions=[0.5, 1.0, 1.0], centered=False).run(material)
    assert_array_equal(result.origin, [0, 0, 0])


# --- from_clip ---


@pytest.mark.parametrize(
    "normal, value, keep, expected_dims",
    [
        ("z", 4.0, "above", [8, 8, 4]),
        ("z", 4.0, "below", [8, 8, 5]),
        ("x", 3.0, "positive", [5, 8, 8]),
        ("y", 2.0, "low", [8, 3, 8]),
    ],
)
def test_from_clip(material, normal, value, keep, expected_dims):
    result = Crop.from_clip(normal=normal, value=value, keep=keep).run(material)
    assert_array_equal(result.dimensions, expected_dims)


def test_from_clip_invalid_keep():
    with pytest.raises(ValueError):
        Crop.from_clip(keep="sideways")


# --- from_clip_by_id ---


@pytest.mark.parametrize(
    "normal, index, keep, expected_dims",
    [
        ("z", 4, "above", [8, 8, 4]),
        ("z", 4, "below", [8, 8, 5]),
        ("x", 3, "high", [5, 8, 8]),
        ("y", 2, "low", [8, 3, 8]),
    ],
)
def test_from_clip_by_id(material, normal, index, keep, expected_dims):
    result = Crop.from_clip_by_id(normal=normal, index=index, keep=keep).run(material)
    assert_array_equal(result.dimensions, expected_dims)


def test_from_clip_by_id_invalid_keep():
    with pytest.raises(ValueError):
        Crop.from_clip_by_id(keep="sideways")
