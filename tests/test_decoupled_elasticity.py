from functools import partial

import numpy as np
import pytest  # Includes: tmp_path, mocker
from materialite.models import DecoupledCrystalElasticity
from numpy.testing import assert_allclose

from materialite import (
    Material,
    Order2SymmetricTensor,
    Order4SymmetricTensor,
    Orientation,
    Scalar,
    Vector,
)

A_TOL = 1e-14
assert_allclose_with_atol = partial(assert_allclose, atol=A_TOL)
MODEL_PATH = "materialite.models.decoupled_crystal_elasticity"


@pytest.fixture
def model():
    strain = Order2SymmetricTensor.from_strain_voigt(
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    )
    return DecoupledCrystalElasticity(applied_strain=strain)


@pytest.fixture
def stiffness_tensor():
    return Order4SymmetricTensor.from_transverse_isotropic_constants(
        252, 152, 152, 202, 90
    )


@pytest.fixture
def material(stiffness_tensor):
    # 90, 45, 90 degree rotations about x axis
    angles = Scalar([90, 45, 90]) * np.pi / 180
    orientations = Orientation.from_axis_angle(Vector.X, angles)
    num_points = 3
    fields = {
        "stiffness": stiffness_tensor.repeat(num_points),
        "orientation": orientations,
    }
    return Material(dimensions=[3, 1, 1]).create_fields(fields)


def test_error_if_material_has_missing_fields(model, material):
    with pytest.raises(AttributeError):
        _ = model(material.remove_field("orientation"))

    with pytest.raises(AttributeError):
        _ = model(material.remove_field("stiffness"))


def test_run_model(model, material):
    expected_stress = np.array(
        [
            [152.0, 152.0, 252.0, 0, 0, 0],
            [152.0, 99.5, 279.5, 12.5, 0, 0],
            [152.0, 152.0, 252.0, 0, 0, 0],
        ]
    )
    stress = model(material).extract("stress").stress_voigt
    assert_allclose_with_atol(stress, expected_stress)
