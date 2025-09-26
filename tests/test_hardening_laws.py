from collections import defaultdict

import numpy as np
import pytest
from numpy.testing import assert_allclose

from materialite import Material, Order4SymmetricTensor, Orientation, SlipSystem, Vector
from materialite.models.small_strain_fft import (
    ElasticViscoplastic,
    LoadSchedule,
    SmallStrainFFT,
    armstrong_frederick,
    linear,
    perfect_plasticity,
    voce,
)


def extract_outputs(material):
    slip_resistances = material.extract("slip_resistances").mean("p").mean("s").components
    slip_system_shear_strains = material.extract("slip_system_shear_strains").mean("p").mean("s").components
    plastic_strains = material.extract("plastic_strains").mean("p").components
    return slip_resistances, slip_system_shear_strains, plastic_strains


@pytest.fixture
def material():
    return Material(dimensions=[4, 4, 4]).create_uniform_field(
        "orientation", Orientation(np.eye(3))
    )


@pytest.fixture
def model_setup():
    load_schedule = LoadSchedule.from_constant_uniaxial_strain_rate(direction="x")
    time_increment = 1.0e-4
    end_time = 50 * time_increment
    return {
        "load_schedule": load_schedule,
        "initial_time_increment": time_increment,
        "end_time": end_time,
    }


@pytest.fixture
def parameters():
    modulus = 150000.0
    shear_modulus = 60000.0
    stiffness_tensor = Order4SymmetricTensor.from_isotropic_constants(
        modulus=modulus, shear_modulus=shear_modulus
    )
    slip_plane_normal = Vector([[1.0, 1.0, 0.0]], "s")
    slip_direction = Vector([[1.0, -1.0, 0.0]], "s")
    slip_system = SlipSystem(slip_plane_normal, slip_direction)
    return {
        "stiffness": stiffness_tensor,
        "reference_slip_rate": 1.0,
        "rate_exponent": 50.0,
        "slip_resistance": 150.0,
        "slip_systems": slip_system,
    }


@pytest.fixture
def output_times():
    return np.arange(1.0e-4, 50.1e-4, 1.0e-4)


@pytest.fixture
def output_variables():
    return ["slip_system_shear_strains", "plastic_strains", "slip_resistances"]


def test_perfect_plasticity(
    material, model_setup, parameters, output_times, output_variables
):
    constitutive_model = ElasticViscoplastic(
        **parameters, hardening_function=perfect_plasticity, hardening_properties=None
    )
    model = SmallStrainFFT(**model_setup, constitutive_model=constitutive_model)
    material = model(
        material,
        output_times=output_times,
        output_variables=output_variables,
    )
    slip_resistances, _, plastic_strains = extract_outputs(material)
    assert_allclose(slip_resistances, parameters["slip_resistance"])
    assert_allclose(np.sum(plastic_strains[:, :3], axis=1), 0)


def test_linear_hardening(
    material, model_setup, parameters, output_times, output_variables
):
    hardening_rate = 1000.0
    constitutive_model = ElasticViscoplastic(
        **parameters,
        hardening_function=linear,
        hardening_properties={"hardening_rate": hardening_rate},
    )
    model = SmallStrainFFT(**model_setup, constitutive_model=constitutive_model)
    material = model(
        material,
        output_times=output_times,
        output_variables=output_variables,
    )
    slip_resistances, slip_system_shear_strains, plastic_strains = extract_outputs(material)
    output_hardening_rate = (
        slip_resistances[-1] - slip_resistances[0]
    ) / (slip_system_shear_strains[-1])
    assert_allclose(output_hardening_rate, hardening_rate)
    assert_allclose(np.sum(plastic_strains[:, :3], axis=1), 0)


def test_voce_hardening(
    material, model_setup, parameters, output_times, output_variables
):
    tau1 = 30.0
    theta0 = 10000.0
    theta1 = 100.0
    constitutive_model = ElasticViscoplastic(
        **parameters,
        hardening_function=voce,
        hardening_properties={"tau_1": tau1, "theta_0": theta0, "theta_1": theta1},
    )
    model = SmallStrainFFT(**model_setup, constitutive_model=constitutive_model)
    material = model(
        material,
        output_times=output_times,
        output_variables=output_variables,
    )
    slip_resistances, slip_system_shear_strains, plastic_strains = extract_outputs(material)
    integrated_voce = lambda x: parameters["slip_resistance"] + (tau1 + theta1 * x) * (
        1 - np.exp(-x * np.abs(theta0 / tau1))
    )
    idx = slip_system_shear_strains > 1.0e-10
    assert_allclose(
        integrated_voce(slip_system_shear_strains[idx]),
        slip_resistances[idx],
    )
    assert_allclose(np.sum(plastic_strains[:, :3], axis=1), 0)


def test_af_hardening(
    material, model_setup, parameters, output_times, output_variables
):
    hardening = 10000.0
    recovery = 20.0
    constitutive_model = ElasticViscoplastic(
        **parameters,
        hardening_function=armstrong_frederick,
        hardening_properties={
            "direct_hardening": hardening,
            "dynamic_recovery": recovery,
        },
    )
    model = SmallStrainFFT(**model_setup, constitutive_model=constitutive_model)
    material = model(
        material,
        output_times=output_times,
        output_variables=output_variables,
    )
    slip_resistances, slip_system_shear_strains, plastic_strains = extract_outputs(material)
    idx = slip_system_shear_strains > 1.0e-10
    nonzero_slips = slip_system_shear_strains[idx]
    nonzero_crss = slip_resistances[idx]
    expected_slip_increments = nonzero_slips[1:] - nonzero_slips[:-1]
    crss_increments = nonzero_crss[1:] - nonzero_crss[:-1]
    slip_increments = crss_increments / (hardening - recovery * nonzero_crss[1:])
    assert_allclose(slip_increments, expected_slip_increments)
    assert_allclose(np.sum(plastic_strains[:, :3], axis=1), 0)
