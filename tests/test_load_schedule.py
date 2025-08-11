import numpy as np
import pytest
from materialite import Order2SymmetricTensor
from materialite.models.small_strain_fft import LoadSchedule
from numpy.testing import assert_allclose


def check_outputs(
    load_schedule, time, time_step, expected_dstrain, expected_dstress, expected_stress
):
    dstrain = load_schedule.strain_increment(time, time_step)
    dstress = load_schedule.stress_increment(time, time_step)
    stress = load_schedule.stress(time, time_step)
    assert_allclose(dstrain.components, expected_dstrain, atol=1.0e-14)
    assert_allclose(dstress.components, expected_dstress, atol=1.0e-14)
    assert_allclose(stress.components, expected_stress, atol=1.0e-14)


def test_from_constant_rates():
    strain_rate = Order2SymmetricTensor([1.0, 0, 2.0, 0, 0, 0])
    stress_rate = Order2SymmetricTensor([0, 3.0, 1.0, 0, 0, 0])
    stress_mask = np.array([1, 1, 0, 1, 1, 1])
    dt = 2.0
    t = 3.0
    load_schedule = LoadSchedule.from_constant_rates(
        strain_rate, stress_rate, stress_mask
    )
    expected_dstrain = [0, 0, 4.0, 0, 0, 0]
    expected_dstress = [0, 6.0, 0, 0, 0, 0]
    expected_stress = [0, 15.0, 0, 0, 0, 0]
    check_outputs(
        load_schedule, t, dt, expected_dstrain, expected_dstress, expected_stress
    )


def test_from_constant_uniaxial_strain_rate():
    expected_dstrain_x = [2.0, 0, 0, 0, 0, 0]
    expected_dstrain_y = [0, 2.0, 0, 0, 0, 0]
    expected_dstrain_z = [0, 0, 2.0, 0, 0, 0]
    expected_dstress = np.zeros(6)
    expected_stress = np.zeros(6)
    load_schedule_x = LoadSchedule.from_constant_uniaxial_strain_rate(direction="x")
    load_schedule_y = LoadSchedule.from_constant_uniaxial_strain_rate(direction="y")
    load_schedule_z = LoadSchedule.from_constant_uniaxial_strain_rate(direction="z")
    t = 1.0
    dt = 2.0
    check_outputs(
        load_schedule_x, t, dt, expected_dstrain_x, expected_dstress, expected_stress
    )
    check_outputs(
        load_schedule_y, t, dt, expected_dstrain_y, expected_dstress, expected_stress
    )
    check_outputs(
        load_schedule_z, t, dt, expected_dstrain_z, expected_dstress, expected_stress
    )


def test_from_ramp():
    strains = Order2SymmetricTensor([[1.0, 0, 1.0, 0, 0, 0], [1.0, 0, 3.0, 0, 0, 0]])
    stresses = Order2SymmetricTensor([[0, 1.0, 1.0, 0, 0, 0], [0, 4.0, 1.0, 0, 0, 0]])
    stress_mask = np.array([1, 1, 0, 1, 1, 1])
    times = [1.0, 5.0]
    dt = 2.0
    t = 3.0
    load_schedule = LoadSchedule.from_ramp(times, strains, stresses, stress_mask)
    expected_dstrain = [0, 0, 1.0, 0, 0, 0]
    expected_dstress = [0, 1.5, 0, 0, 0, 0]
    expected_stress = [0, 4.0, 0, 0, 0, 0]
    check_outputs(
        load_schedule, t, dt, expected_dstrain, expected_dstress, expected_stress
    )


def test_from_stress_amplitude():
    stress_amplitude = Order2SymmetricTensor([0, 0, 2.0, 0, 0, 0])
    frequency = 1.0
    load_schedule = LoadSchedule.from_cyclic_stress(stress_amplitude, frequency)
    t = 0.0
    dt = 0.25
    expected_dstrain = np.zeros(6)
    expected_dstress = [0, 0, 2.0, 0, 0, 0]
    expected_stress = [0, 0, 2.0, 0, 0, 0]
    check_outputs(
        load_schedule, t, dt, expected_dstrain, expected_dstress, expected_stress
    )

    t = 0.25
    dt = 0.25
    expected_dstrain = np.zeros(6)
    expected_dstress = [0, 0, -2.0, 0, 0, 0]
    expected_stress = [0, 0, 0, 0, 0, 0]
    check_outputs(
        load_schedule, t, dt, expected_dstrain, expected_dstress, expected_stress
    )


def test_from_strain_amplitude():
    strain_amplitude = Order2SymmetricTensor([0, 0, 2.0, 0, 0, 0])
    stress_mask = np.array([1, 1, 0, 1, 1, 1])
    frequency = 1.0
    load_schedule = LoadSchedule.from_cyclic_strain(
        strain_amplitude, frequency, stress_mask
    )
    t = 0.0
    dt = 0.25
    expected_dstrain = [0, 0, 2.0, 0, 0, 0]
    expected_dstress = np.zeros(6)
    expected_stress = np.zeros(6)
    check_outputs(
        load_schedule, t, dt, expected_dstrain, expected_dstress, expected_stress
    )

    t = 0.25
    dt = 0.25
    expected_dstrain = [0, 0, -2.0, 0, 0, 0]
    expected_dstress = np.zeros(6)
    expected_stress = np.zeros(6)
    check_outputs(
        load_schedule, t, dt, expected_dstrain, expected_dstress, expected_stress
    )
