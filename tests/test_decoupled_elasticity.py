import numpy as np
import pytest  # Includes: tmp_path, mocker
from materialite import (
    Material,
    Order2SymmetricTensor,
    Order4SymmetricTensor,
    Orientation,
)
from materialite.models import DecoupledCrystalElasticity
from numpy.testing import assert_allclose

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
def material(stiffness_tensor, mocker):
    r2 = 1 / np.sqrt(2)
    orientations = Orientation.from_rotation_matrix(
        # 90 degrees about specimen x axis
        np.array(
            [
                [[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]],
                # 45 degrees about specimen x axis
                [[1.0, 0, 0], [0, r2, r2], [0, -r2, r2]],
                [[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]],
            ]
        )
    )
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
    assert_allclose(stress, expected_stress)
