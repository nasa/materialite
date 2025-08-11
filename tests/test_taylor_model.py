import numpy as np
import pandas as pd
import pytest
from materialite import Material, Order2SymmetricTensor, Orientation
from materialite.models import TaylorModel
from numpy.testing import assert_allclose


@pytest.fixture
def applied_strain():
    return Order2SymmetricTensor.from_strain_voigt(np.array([-0.5, -0.5, 1, 0, 0, 0]))


@pytest.fixture
def crss():
    return 2.0


@pytest.fixture
def model(applied_strain, crss):
    return TaylorModel(applied_strain, crss)


@pytest.fixture
def model_with_mask(applied_strain, crss):
    return TaylorModel(applied_strain, crss, "mask")


@pytest.fixture
def expected_stresses(crss):
    return (
        np.array([[-1 / 3, -1 / 3, 2 / 3, 0, 0, 0], [0, -1.0, 1.0, 0, 0, 0]])
        * crss
        * np.sqrt(6)
    )


@pytest.fixture
def expected_taylor_factors():
    return np.array([1.0, 1.5]) * np.sqrt(6)


@pytest.fixture
def material():
    r2 = 1 / np.sqrt(2)
    orientations = Orientation.from_rotation_matrix(
        # 90 degrees about specimen x axis
        np.array(
            [
                [[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]],
                # 45 degrees about specimen x axis
                [[1.0, 0, 0], [0, r2, r2], [0, -r2, r2]],
            ]
        )
    )
    mask = [0, 1]
    fields = {"orientation": orientations, "mask": mask}
    return Material(dimensions=[2, 1, 1]).create_fields(fields)


def test_error_if_material_has_missing_fields(applied_strain, crss, material):
    model = TaylorModel(applied_strain, crss)
    with pytest.raises(AttributeError):
        _ = model(material.remove_field("orientation"))


@pytest.mark.parametrize(
    "applied_strain", [[1, 2, 3], np.arange(9).reshape((3, 3)), np.ones((3, 3))]
)
def test_error_if_invalid_applied_strain(applied_strain):
    with pytest.raises(ValueError):
        _ = TaylorModel(applied_strain=applied_strain)


def test_run(model, material, expected_stresses, expected_taylor_factors):
    new_material = model(material)

    assert new_material.state == pytest.approx(
        {"average_taylor_factor": 1.25 * np.sqrt(6)}
    )
    assert_allclose(new_material.extract("taylor_factor"), expected_taylor_factors)
    assert_allclose(
        new_material.extract("stress").stress_voigt, expected_stresses, atol=1.0e-10
    )


def test_run_with_mask(
    material, model_with_mask, expected_stresses, expected_taylor_factors
):
    expected_stresses[0, :] = 0.0
    expected_taylor_factors[0] = 0.0

    new_material = model_with_mask(material)

    assert new_material.state == pytest.approx(
        {"average_taylor_factor": 0.75 * np.sqrt(6)}
    )
    assert_allclose(new_material.extract("taylor_factor"), expected_taylor_factors)
    assert_allclose(
        new_material.extract("stress").stress_voigt, expected_stresses, atol=1.0e-10
    )
