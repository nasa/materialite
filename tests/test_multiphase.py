import numpy as np
import pytest  # Includes: tmp_path, mocker
from materialite import Box, Material, Orientation, Scalar
from materialite.models.small_strain_fft import (
    IsotropicElasticPlastic,
    LoadSchedule,
    SmallStrainFFT,
    linear,
)
from numpy.testing import assert_allclose


@pytest.fixture
def material():
    sizes = [3, 3, 3]
    box = Box(max_corner=[2, 2, 2])
    modulus = 150000.0
    shear_modulus = 60000.0
    yield_stress = 150.0
    hardening_rate = 1000.0
    constitutive_model = IsotropicElasticPlastic(
        modulus, shear_modulus, yield_stress, linear, {"hardening_rate": hardening_rate}
    )
    constitutive_model2 = IsotropicElasticPlastic(
        modulus, shear_modulus, yield_stress, linear, {"hardening_rate": hardening_rate}
    )
    return (
        Material(dimensions=[8, 8, 8], sizes=sizes)
        .create_uniform_field("orientation", Orientation(np.eye(3)))
        .create_uniform_field("phase", 0)
        .insert_feature(box, fields={"phase": 1})
        .create_regional_fields(
            region_label="phase",
            regional_fields={
                "phase": [0, 1],
                "constitutive_model": [constitutive_model, constitutive_model2],
            },
        )
    )


def test_with_linear_hardening(material):
    expected_mean_stress_norm = 150.3980086
    yield_stress = 150.0
    hardening_rate = 1000.0
    load_schedule = LoadSchedule.from_constant_uniaxial_strain_rate(direction="z")
    time_increment = 1.0e-4
    end_time = 15.0e-4
    num_time_steps = 15
    model = SmallStrainFFT(
        load_schedule=load_schedule,
        end_time=end_time,
        initial_time_increment=time_increment,
    )
    output_times = np.round((np.arange(num_time_steps) + 1) * time_increment, 10)
    material = model(
        material,
        output_times=output_times,
        output_variables=["eq_plastic_strains"],
        phase_label="phase",
    )
    stress = material.extract("stress")
    mean_stress_norm = stress[:, -1].mean().norm.components
    axial_stresses = stress.mean("p").components[:, 2]
    eq_plastic_strains = material.extract("eq_plastic_strains")
    mean_eq_plastic_strains = eq_plastic_strains.mean("p").components
    assert_allclose(mean_stress_norm, expected_mean_stress_norm)
    assert_allclose(stress[:, -1].components[:, 2], axial_stresses[-1])
    assert_allclose(
        axial_stresses[12:], hardening_rate * mean_eq_plastic_strains[11:-1] + yield_stress
    )
    mean_eq_plastic_strains = np.tile(mean_eq_plastic_strains, (len(eq_plastic_strains), 1))
    assert_allclose(eq_plastic_strains.components, mean_eq_plastic_strains)
