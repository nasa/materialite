from functools import partial

import numpy as np
import pandas as pd
import pytest  # Includes: tmp_path, mocker
from materialite.models.small_strain_fft import Elastic, LoadSchedule, SmallStrainFFT
from materialite.models.small_strain_fft.temperature_history import TemperatureHistory
from materialite.util import repeat_data
from numpy.testing import assert_allclose

from materialite import (
    Box,
    Material,
    Scalar,
    Order2SymmetricTensor,
    Order4SymmetricTensor,
    Orientation,
)


@pytest.fixture
def modulus():
    return 200


@pytest.fixture
def poisson():
    return 0.3


@pytest.fixture
def alpha():
    return 1.0e-5


@pytest.fixture
def delta_T():
    return 100.0


@pytest.fixture
def elastic_model(modulus, poisson, alpha):
    stiffness = Order4SymmetricTensor.from_isotropic_constants(
        modulus=modulus, shear_modulus=modulus / (2 * (1 + poisson))
    )
    thermal_expansion_coefficients = Order2SymmetricTensor(
        np.array([alpha, alpha, alpha, 0, 0, 0])
    )
    return Elastic(stiffness, thermal_expansion_coefficients)


@pytest.fixture
def material(delta_T):
    temperature_1 = Scalar([0, delta_T], dims="t")
    temperature_2 = Scalar([0, 0], dims="t")
    df = pd.DataFrame(
        {"phase": [1, 2], "temperature_history": [temperature_1, temperature_2]}
    )
    material = Material(dimensions=[4, 4, 4])
    material = (
        material.create_uniform_field(
            "orientation", Orientation.from_euler_angles([0, 0, 0])
        )
        .create_uniform_field("phase", 1)
        .insert_feature(
            Box(max_corner=[None, None, material.sizes[2] / 2]),
            fields={"phase": 2},
        )
        .create_regional_fields("phase", df)
    )
    return material


@pytest.fixture
def expected_data(modulus, poisson, alpha, delta_T):
    sigma_22_2 = modulus * alpha * delta_T / (2 * (1 - poisson))
    sigma_22_1 = -sigma_22_2
    sigma_11_1 = -modulus * alpha * delta_T - poisson * sigma_22_2
    sigma_11_2 = poisson * sigma_22_2
    sigma_1 = np.array([sigma_11_1, sigma_22_1, 0, 0, 0, 0])
    sigma_2 = np.array([sigma_11_2, sigma_22_2, 0, 0, 0, 0])
    epsilon_22 = 1 / modulus * (sigma_22_2 - poisson * sigma_11_2)
    epsilon_33_1 = -poisson / modulus * (sigma_11_1 + sigma_22_1) + alpha * delta_T
    epsilon_33_2 = -poisson / modulus * (sigma_11_2 + sigma_22_2)
    epsilon_1 = np.array([0, epsilon_22, epsilon_33_1, 0, 0, 0])
    epsilon_2 = np.array([0, epsilon_22, epsilon_33_2, 0, 0, 0])
    return {"stress": {1: sigma_1, 2: sigma_2}, "strain": {1: epsilon_1, 2: epsilon_2}}


def test_elastic_thermal_strain(material, elastic_model, expected_data):
    temperatures = material.extract("temperature_history")
    times = [0, 1]
    temperature_history = TemperatureHistory(temperatures, times)
    applied_strain_rate = Order2SymmetricTensor(np.array([0, 0, 0, 0, 0, 0]))
    stress = Order2SymmetricTensor.zero()
    stress_mask = np.array([0, 1, 1, 1, 1, 1])
    load_schedule = LoadSchedule.from_constant_rates(
        applied_strain_rate, stress, stress_mask
    )
    model = SmallStrainFFT(
        load_schedule=load_schedule, end_time=1.0, constitutive_model=elastic_model
    )
    material = model(
        material,
        linear_solver_tolerance=1.0e-7,
        temperature_history=temperature_history,
    )
    indices = material.get_region_indices("phase")
    stress_1 = material.extract("stress")[indices[1]].mean().components
    stress_2 = material.extract("stress")[indices[2]].mean().components
    strain_1 = material.extract("strain")[indices[1]].mean().components
    strain_2 = material.extract("strain")[indices[2]].mean().components
    thermal_strain_1 = material.extract("thermal_strain")[indices[1]].mean().components
    thermal_strain_2 = material.extract("thermal_strain")[indices[2]].mean().components
    assert_allclose(stress_1, expected_data["stress"][1], atol=1.0e-13)
    assert_allclose(stress_2, expected_data["stress"][2], atol=1.0e-13)
    assert_allclose(strain_1, expected_data["strain"][1], atol=1.0e-13)
    assert_allclose(strain_2, expected_data["strain"][2], atol=1.0e-13)
    assert_allclose(
        thermal_strain_1, np.array([0.001, 0.001, 0.001, 0, 0, 0]), atol=1.0e-13
    )
    assert_allclose(thermal_strain_2, np.zeros(6), atol=1.0e-13)
