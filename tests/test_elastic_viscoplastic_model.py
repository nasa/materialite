import numpy as np
import pandas as pd
import pytest  # Includes: tmp_path, mocker
from numpy.testing import assert_allclose

from materialite import (
    Material,
    Order2SymmetricTensor,
    Order4SymmetricTensor,
    SlipSystem,
    Sphere,
)
from materialite.models.small_strain_fft import (
    Elastic,
    ElasticViscoplastic,
    LoadSchedule,
    SmallStrainFFT,
    armstrong_frederick,
    linear,
    perfect_plasticity,
    voce,
)


@pytest.fixture
def stiffness_tensor():
    return Order4SymmetricTensor.from_voigt(
        np.array(
            [
                [160e3, 90e3, 66e3, 0, 0, 0],
                [90e3, 160e3, 66e3, 0, 0, 0],
                [66e3, 66e3, 181.7e3, 0, 0, 0],
                [0, 0, 0, 46.5e3, 0, 0],
                [0, 0, 0, 0, 46.5e3, 0],
                [0, 0, 0, 0, 0, 35e3],
            ]
        )
    )


@pytest.fixture
def material_no_defect():
    rng = np.random.default_rng(0)
    sizes = [3, 3, 3]
    return (
        Material(dimensions=[8, 8, 8], sizes=sizes)
        .create_voronoi(num_regions=10, rng=rng)
        .assign_random_orientations(rng=rng)
        .create_uniform_field("phase", 1)
    )


@pytest.fixture
def material_with_defect(material_no_defect):
    sizes = material_no_defect.sizes
    midpoint = sizes[2] / 2
    return material_no_defect.insert_feature(
        Sphere(radius=1, centroid=[midpoint, midpoint, midpoint]),
        fields={"phase": 2},
    )


@pytest.fixture
def evp_linear(stiffness_tensor):
    return ElasticViscoplastic(
        stiffness=stiffness_tensor,
        slip_systems=SlipSystem.octahedral(),
        reference_slip_rate=1.0,
        rate_exponent=10.0,
        slip_resistance=300.0,
        hardening_function=linear,
        hardening_properties={"hardening_rate": 10},
    )


@pytest.fixture
def evp_voce(stiffness_tensor):
    return ElasticViscoplastic(
        stiffness=stiffness_tensor,
        slip_systems=SlipSystem.octahedral(),
        reference_slip_rate=1.0,
        rate_exponent=10.0,
        slip_resistance=300.0,
        hardening_function=voce,
        hardening_properties={"tau_1": 100.0, "theta_0": 20000.0, "theta_1": 10.0},
    )


@pytest.fixture
def evp_af(stiffness_tensor):
    return ElasticViscoplastic(
        stiffness=stiffness_tensor,
        slip_systems=SlipSystem.octahedral(),
        reference_slip_rate=1.0,
        rate_exponent=10.0,
        slip_resistance=300.0,
        hardening_function=armstrong_frederick,
        hardening_properties={"direct_hardening": 10000.0, "dynamic_recovery": 15.0},
    )


@pytest.fixture
def evp_pp(stiffness_tensor):
    return ElasticViscoplastic(
        stiffness=stiffness_tensor,
        slip_systems=SlipSystem.octahedral(),
        reference_slip_rate=1.0,
        rate_exponent=10.0,
        slip_resistance=300.0,
        hardening_function=perfect_plasticity,
        hardening_properties=None,
    )


@pytest.fixture
def load_schedule():
    return LoadSchedule.from_constant_uniaxial_strain_rate(direction="z")


def assign_constitutive_models(material, evp_model):
    phases = [1, 2]
    pore_model = Elastic(evp_model.stiffness * 0.0)
    models = [evp_model, pore_model]
    regional_fields = pd.DataFrame({"phase": phases, "constitutive_model": models})
    return material.create_regional_fields("phase", regional_fields)


def test_evp_with_linear_hardening_defect(
    material_with_defect, evp_linear, load_schedule
):
    # evpfft mean stress norm: 354.8037
    expected_mean_stress_norm = 361.276725
    material_linear = assign_constitutive_models(material_with_defect, evp_linear)

    model = SmallStrainFFT(
        load_schedule=load_schedule,
        end_time=4.0e-3,
        initial_time_increment=1.0e-3,
    )
    material = model(material_linear, phase_label="phase", global_tolerance=1.0e-7)
    mean_stress_norm = material.extract("stress").mean().norm.components
    assert_allclose(mean_stress_norm, expected_mean_stress_norm)
    mean_strain = material.extract("strain").mean().components
    assert_allclose(mean_strain[2], 0.004)


def test_evp_with_voce_hardening(material_no_defect, evp_voce, load_schedule):
    # evpfft mean stress norm: 540.14502
    expected_mean_stress_norm = 541.043871

    model = SmallStrainFFT(
        load_schedule=load_schedule,
        end_time=5.0e-3,
        initial_time_increment=1.0e-3,
        constitutive_model=evp_voce,
    )
    material = model(material_no_defect, global_tolerance=1.0e-7)
    mean_stress_norm = material.extract("stress").mean().norm.components
    assert_allclose(mean_stress_norm, expected_mean_stress_norm)
    mean_strain = material.extract("strain").mean().components
    assert_allclose(mean_strain[2], 0.005)


def test_evp_with_af_hardening(material_no_defect, evp_af, load_schedule):
    expected_mean_stress_norm = 539.345343

    model = SmallStrainFFT(
        load_schedule=load_schedule,
        end_time=5.0e-3,
        initial_time_increment=1.0e-3,
        constitutive_model=evp_af,
    )
    material = model(material_no_defect, global_tolerance=1.0e-7)
    mean_stress_norm = material.extract("stress").mean().norm.components
    assert_allclose(mean_stress_norm, expected_mean_stress_norm)
    mean_strain = material.extract("strain").mean().components
    assert_allclose(mean_strain[2], 0.005)


def test_evp_with_no_hardening(material_no_defect, evp_pp, load_schedule):
    # evpfft mean stress norm: 441.6827
    expected_mean_stress_norm = 442.608112

    model = SmallStrainFFT(
        load_schedule=load_schedule,
        end_time=4.0e-3,
        initial_time_increment=1.0e-3,
        constitutive_model=evp_pp,
    )
    material = model(material_no_defect, global_tolerance=1.0e-7)
    mean_stress_norm = material.extract("stress").mean().norm.components
    assert_allclose(mean_stress_norm, expected_mean_stress_norm)
    mean_strain = material.extract("strain").mean().components
    assert_allclose(mean_strain[2], 0.004)


def test_strain_bcs(material_no_defect, evp_pp):
    # evpfft S33: 426.2354431
    expected_S33 = 426.923816
    expected_mean_stress_norm = 428.328875
    velocity_gradient = Order2SymmetricTensor.from_strain_voigt(
        np.array([-0.35, -0.35, 1.0, 0, 0, 0])
    )
    stress_mask = np.zeros(6)

    load_schedule = LoadSchedule.from_constant_rates(
        strain_rate=velocity_gradient,
        stress_rate=Order2SymmetricTensor.zero(),
        stress_mask=stress_mask,
    )

    model = SmallStrainFFT(
        load_schedule=load_schedule,
        end_time=4.0e-3,
        initial_time_increment=1.0e-3,
        constitutive_model=evp_pp,
    )
    material = model(material_no_defect, global_tolerance=1.0e-7)
    mean_stress_norm = material.extract("stress").mean().norm.components
    assert_allclose(mean_stress_norm, expected_mean_stress_norm)
    S33 = material.extract("stress").mean().components[2]
    assert_allclose(S33, expected_S33)
    mean_strain = material.extract("strain").mean().components
    assert_allclose(mean_strain, velocity_gradient.components * 0.004, atol=1e-14)
