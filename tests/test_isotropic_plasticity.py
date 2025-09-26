import numpy as np
import pytest  # Includes: tmp_path, mocker
from materialite import Material, Orientation
from materialite.models.small_strain_fft import (
    IsotropicElasticPlastic,
    LoadSchedule,
    SmallStrainFFT,
    linear,
    perfect_plasticity,
    voce,
)
from numpy.testing import assert_allclose


@pytest.fixture
def material():
    sizes = [3, 3, 3]
    return (
        Material(dimensions=[8, 8, 8], sizes=sizes)
        .create_uniform_field("orientation", Orientation(np.eye(3)))
        .create_uniform_field("phase", 1)
    )


@pytest.fixture
def load_schedule():
    return LoadSchedule.from_constant_uniaxial_strain_rate(direction="z")


def test_with_perfect_plasticity(material, load_schedule):
    expected_mean_stress_norm = 150.0
    modulus = 150000.0
    shear_modulus = 60000.0
    yield_stress = 150.0
    constitutive_model = IsotropicElasticPlastic(
        modulus, shear_modulus, yield_stress, perfect_plasticity, None
    )
    time_increment = 1.0e-4
    end_time = 15.0e-4
    num_time_steps = 15
    model = SmallStrainFFT(
        load_schedule=load_schedule,
        end_time=end_time,
        initial_time_increment=time_increment,
        constitutive_model=constitutive_model,
    )
    output_times = (np.arange(num_time_steps) + 1) * time_increment
    material = model(
        material,
        output_times=output_times,
    )
    stress = material.extract("stress")
    mean_stress_norm = stress[:, -1].mean().norm.components
    axial_stresses = stress.mean("p").components[:, 2]
    assert_allclose(mean_stress_norm, expected_mean_stress_norm)
    assert_allclose(axial_stresses[9:], expected_mean_stress_norm)


def test_with_linear_hardening(material, load_schedule):
    expected_mean_stress_norm = 150.3980086
    modulus = 150000.0
    shear_modulus = 60000.0
    yield_stress = 150.0
    hardening_rate = 1000.0
    constitutive_model = IsotropicElasticPlastic(
        modulus, shear_modulus, yield_stress, linear, {"hardening_rate": hardening_rate}
    )
    time_increment = 1.0e-4
    end_time = 15.0e-4
    num_time_steps = 15
    model = SmallStrainFFT(
        load_schedule=load_schedule,
        end_time=end_time,
        initial_time_increment=time_increment,
        constitutive_model=constitutive_model,
    )
    output_times = np.round((np.arange(num_time_steps) + 1) * time_increment, 10)
    material = model(
        material,
        output_times=output_times,
        output_variables=["eq_plastic_strains"],
    )
    stress = material.extract("stress")
    mean_stress_norm = stress[:, -1].mean().norm.components
    axial_stresses = stress.mean("p").components[:, 2]
    plastic_strains = material.extract("eq_plastic_strains").mean("p").components
    assert_allclose(mean_stress_norm, expected_mean_stress_norm)
    assert_allclose(stress[:, -1].components[:, 2], axial_stresses[-1])
    assert_allclose(
        axial_stresses[12:], hardening_rate * plastic_strains[11:-1] + yield_stress
    )


def test_with_voce_hardening(material, load_schedule):
    expected_mean_stress_norm = 171.694205
    modulus = 150000.0
    shear_modulus = 60000.0
    yield_stress = 150.0
    tau1 = 30.0
    theta0 = 10000.0
    theta1 = 100.0
    hardening_parameters = {"tau_1": tau1, "theta_0": theta0, "theta_1": theta1}
    constitutive_model = IsotropicElasticPlastic(
        modulus, shear_modulus, yield_stress, voce, hardening_parameters
    )
    time_increment = 1.0e-4
    end_time = 50.0e-4
    num_time_steps = 50
    model = SmallStrainFFT(
        load_schedule=load_schedule,
        end_time=end_time,
        initial_time_increment=time_increment,
        constitutive_model=constitutive_model,
    )
    output_times = np.round((np.arange(num_time_steps) + 1) * time_increment, 10)
    material = model(
        material,
        output_times=output_times,
        output_variables=["eq_plastic_strains"],
    )
    stress = material.extract("stress")
    mean_stress_norm = stress[:, -1].mean().norm.components
    axial_stresses = stress.mean("p").components[:, 2]
    plastic_strains = material.extract("eq_plastic_strains").mean("p").components
    integrated_voce = lambda x: yield_stress + (tau1 + theta1 * x) * (
        1 - np.exp(-x * np.abs(theta0 / tau1))
    )
    assert_allclose(mean_stress_norm, expected_mean_stress_norm)
    assert_allclose(stress[:, -1].components[:, 2], axial_stresses[-1])
    assert_allclose(axial_stresses[11:], integrated_voce(plastic_strains[10:-1]))
