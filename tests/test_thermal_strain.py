from functools import partial

import numpy as np
import pandas as pd
import pytest  # Includes: tmp_path, mocker
from materialite.models.small_strain_fft import (
    Elastic,
    ElasticViscoplastic,
    IsotropicElasticPlastic,
    LoadSchedule,
    SmallStrainFFT,
    linear,
    perfect_plasticity,
)
from materialite.models.small_strain_fft.temperature_history import TemperatureHistory
from numpy.testing import assert_allclose

from materialite import (
    Box,
    Material,
    Order2SymmetricTensor,
    Order4SymmetricTensor,
    Orientation2,
    Scalar,
    SlipSystem,
    Sphere,
)


@pytest.fixture
def modulus():
    return 200


@pytest.fixture
def poisson():
    return 0.3


@pytest.fixture
def shear_modulus(modulus, poisson):
    return modulus / (2 * (1 + poisson))


@pytest.fixture
def alpha():
    return 1.0e-5


@pytest.fixture
def thermal_expansion_coefficients(alpha):
    return Order2SymmetricTensor([alpha, alpha, alpha, 0, 0, 0])


@pytest.fixture
def delta_T():
    return 100.0


@pytest.fixture
def elastic_model(modulus, thermal_expansion_coefficients, shear_modulus):
    stiffness = Order4SymmetricTensor.from_isotropic_constants(
        modulus=modulus, shear_modulus=shear_modulus
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
            "orientation", Orientation2.from_euler_angles([0, 0, 0])
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
def expected_elastic_data(modulus, poisson, alpha, delta_T):
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


def test_elastic(
    material,
    elastic_model,
    expected_elastic_data,
    thermal_expansion_coefficients,
    delta_T,
):
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
        load_schedule=load_schedule,
        end_time=1.0,
        constitutive_model=elastic_model,
        temperature_history=temperature_history,
    )
    material = model(material, linear_solver_tolerance=1.0e-7)
    indices = material.get_region_indices("phase")
    stress_1 = material.extract("stress")[indices[1]].mean().components
    stress_2 = material.extract("stress")[indices[2]].mean().components
    strain_1 = material.extract("strain")[indices[1]].mean().components
    strain_2 = material.extract("strain")[indices[2]].mean().components
    thermal_strain_1 = material.extract("thermal_strain")[indices[1]].mean().components
    thermal_strain_2 = material.extract("thermal_strain")[indices[2]].mean().components
    assert_allclose(stress_1, expected_elastic_data["stress"][1], atol=1.0e-13)
    assert_allclose(stress_2, expected_elastic_data["stress"][2], atol=1.0e-13)
    assert_allclose(strain_1, expected_elastic_data["strain"][1], atol=1.0e-13)
    assert_allclose(strain_2, expected_elastic_data["strain"][2], atol=1.0e-13)
    assert_allclose(
        thermal_strain_1,
        thermal_expansion_coefficients.components * delta_T,
        atol=1.0e-13,
    )
    assert_allclose(thermal_strain_2, np.zeros(6), atol=1.0e-13)


def test_isotropic_plasticity(
    modulus, shear_modulus, delta_T, thermal_expansion_coefficients, poisson
):
    material = Material(dimensions=[8, 8, 8]).create_uniform_field(
        "orientation", Orientation2.identity()
    )
    yield_ratio = 0.001
    yield_stress = modulus * yield_ratio
    expected_stress = yield_stress
    constitutive_model = IsotropicElasticPlastic(
        modulus,
        shear_modulus,
        yield_stress,
        perfect_plasticity,
        None,
        thermal_expansion_coefficients=thermal_expansion_coefficients,
    )
    strain_rate = 0.003
    load_schedule = LoadSchedule.from_constant_uniaxial_strain_rate(
        direction="z", magnitude=strain_rate
    )

    time_increment = 0.1
    end_time = 1.0
    num_time_steps = 10
    num_points = material.num_points
    temperatures = np.zeros((num_points, 2))
    temperatures[:, 1] = delta_T
    temperatures = Scalar(temperatures, dims="pt")
    times = [0, end_time]
    temperature_history = TemperatureHistory(temperatures, times)
    model = SmallStrainFFT(
        load_schedule=load_schedule,
        end_time=end_time,
        initial_time_increment=time_increment,
        constitutive_model=constitutive_model,
        temperature_history=temperature_history,
    )
    output_times = (np.arange(num_time_steps) + 1) * time_increment
    material = model(
        material,
        output_times=output_times,
    )
    stress = material.extract("stress")
    mean_stress_norm = stress[:, -1].mean().norm.components
    axial_stresses = stress.mean("p").components[:, 2]
    strain = (material.extract("strain") - material.extract("thermal_strain")).mean("p")
    expected_axial_strain = np.arange(0.0002, 0.0021, 0.0002)
    expected_transverse_strain = np.zeros(num_time_steps)
    expected_transverse_strain[:5] = -poisson * expected_axial_strain[:5]
    expected_transverse_strain[5:] = expected_transverse_strain[4] - 0.5 * (
        expected_axial_strain[5:] - expected_axial_strain[4]
    )

    assert_allclose(mean_stress_norm, expected_stress)
    assert_allclose(axial_stresses[4:], expected_stress)
    assert_allclose(strain.components[:, 2], expected_axial_strain)
    assert_allclose(strain.components[:, 0], expected_transverse_strain)
    assert_allclose(strain.components[:, 1], expected_transverse_strain)


def test_elastic_viscoplastic(thermal_expansion_coefficients, delta_T):
    material = Material(dimensions=[4, 4, 4])
    stiffness_tensor = Order4SymmetricTensor.from_cubic_constants(
        C11=243300, C12=156700, C44=117800
    )
    df = pd.DataFrame(
        {
            "grain": [1, 2],
            "orientation": Orientation2.from_euler_angles(
                [[0, 0, 0], [np.pi / 4, np.pi / 4, 0]]
            ),
        }
    )
    material = (
        material.create_uniform_field("grain", 1)
        .insert_feature(
            Box(max_corner=[None, None, material.sizes[2] / 2]),
            fields={"grain": 2},
        )
        .create_regional_fields("grain", df)
    )
    load_schedule = LoadSchedule.from_constant_uniaxial_strain_rate(
        direction="x", magnitude=0.005
    )

    evp_linear = ElasticViscoplastic(
        stiffness=stiffness_tensor,
        slip_systems=SlipSystem.octahedral(),
        reference_slip_rate=1.0,
        rate_exponent=10.0,
        slip_resistance=300.0,
        hardening_function=linear,
        hardening_properties={"hardening_rate": 10},
        thermal_expansion_coefficients=thermal_expansion_coefficients,
    )
    end_time = 1.0
    initial_time_increment = 0.1
    model = SmallStrainFFT(
        load_schedule=load_schedule,
        end_time=end_time,
        initial_time_increment=initial_time_increment,
        constitutive_model=evp_linear,
    )
    output_times = np.arange(0.1, 1.01, 0.1)
    material_basic = model(material, output_times=output_times)

    temperatures = Scalar(np.zeros((material.num_points, 2)))
    temperatures[:, 1] = delta_T
    times = [0, 1]
    temperature_history = TemperatureHistory(times=times, temperatures=temperatures)
    load_schedule_thermal = LoadSchedule.from_constant_uniaxial_strain_rate(
        direction="x", magnitude=0.006
    )
    model_thermal = SmallStrainFFT(
        load_schedule=load_schedule_thermal,
        end_time=end_time,
        initial_time_increment=initial_time_increment,
        constitutive_model=evp_linear,
        temperature_history=temperature_history,
    )
    material_thermal = model_thermal(material, output_times=output_times)

    stress_basic = material_basic.extract("stress").mean("p").components[:, 0]
    strain_basic = material_basic.extract("strain").mean("p").components[:, 0]
    stress_thermal = material_thermal.extract("stress").mean("p").components[:, 0]
    strain_thermal = (
        material_thermal.extract("strain").mean("p")
        - material_thermal.extract("thermal_strain").mean("p")
    ).components[:, 0]

    assert_allclose(stress_thermal, stress_basic, atol=1.0e-13)
    assert_allclose(strain_thermal, strain_basic, atol=1.0e-13)


def test_eshelby_problem():
    expected_stress = -280.3136365
    material = (
        Material([256, 256, 1], sizes=[1023, 1023, 0])
        .create_uniform_field("phase", 1)
        .create_uniform_field("orientation", Orientation2.identity())
    )
    times = np.array([0, 1])
    temperature1 = Scalar([0, 1000.0], dims="t")
    temperature2 = Scalar([0, 1500.0], dims="t")
    sphere = Sphere(radius=43, centroid=material.sizes / 2)
    regional_fields = {
        "phase": [1, 2],
        "temperature_history": [temperature1, temperature2],
    }
    material = material.insert_feature(
        sphere, fields={"phase": 2}
    ).create_regional_fields("phase", regional_fields)
    pointwise_temperature = material.extract("temperature_history")
    temperature_history = TemperatureHistory(
        temperatures=pointwise_temperature, times=times
    )

    alpha = 10.0e-6
    thermal_expansion_coefficients = Order2SymmetricTensor(
        [alpha, alpha, alpha, 0, 0, 0]
    )
    E = 65400.0
    nu = 0.42
    stiffness = Order4SymmetricTensor.from_isotropic_constants(
        modulus=65400.0, shear_modulus=E / (2 * (1 + nu))
    )
    elastic_model = Elastic(stiffness, thermal_expansion_coefficients)
    strain_rate = Order2SymmetricTensor.zero()
    stress_rate = Order2SymmetricTensor.zero()
    stress_mask = np.array([1, 1, 1, 1, 1, 1])
    load_schedule = LoadSchedule.from_constant_rates(
        strain_rate=strain_rate, stress_rate=stress_rate, stress_mask=stress_mask
    )
    model = SmallStrainFFT(
        load_schedule=load_schedule,
        end_time=1.0,
        initial_time_increment=1.0,
        constitutive_model=elastic_model,
        temperature_history=temperature_history,
    )
    material = model(
        material,
        linear_solver_tolerance=1.0e-14,
        global_tolerance=1.0,
        strain_correction_tolerance=1.0,
    )
    inclusion_idx = material.get_region_indices("phase")[2]
    mean_stress_inclusion = (
        material.extract("stress")[inclusion_idx].mean().components[0]
    )
    assert_allclose(mean_stress_inclusion, expected_stress, atol=1)
