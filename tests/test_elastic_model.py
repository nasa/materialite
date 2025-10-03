from functools import partial

import numpy as np
import pandas as pd
import pytest  # Includes: tmp_path, mocker
from materialite.models.small_strain_fft import Elastic, LoadSchedule, SmallStrainFFT
from materialite.util import repeat_data
from numpy.testing import assert_allclose

from materialite import (
    Box,
    Material,
    Order2SymmetricTensor,
    Order4SymmetricTensor,
    Orientation,
    Sphere,
)

A_TOL = 1e-14
assert_allclose_with_atol = partial(assert_allclose, atol=A_TOL)


def create_grains(material, grains_per_side):
    n = material.sizes[0]
    points = np.linspace(0, n, grains_per_side + 1)
    points_min = points[:-1]
    points_max = points[1:]
    x, y, z = np.meshgrid(points_min, points_min, points_min, indexing="ij")
    x_min = x.ravel()
    y_min = y.ravel()
    z_min = z.ravel()
    x, y, z = np.meshgrid(points_max, points_max, points_max, indexing="ij")
    x_max = x.ravel()
    y_max = y.ravel()
    z_max = z.ravel()
    for i, (x0, y0, z0, x1, y1, z1) in enumerate(
        zip(x_min, y_min, z_min, x_max, y_max, z_max)
    ):
        grain = Box(min_corner=[x0, y0, z0], max_corner=[x1, y1, z1])
        material = material.insert_feature(grain, fields={"grain": i + 1})
    return material


def shift_origin(material, xyz_shift):
    domain_max = material.dimensions[0] + xyz_shift
    coords = material.extract(["x", "y", "z"])
    periodic_grains = np.tile(material.extract("grain"), 27)
    periodic_coords = repeat_data(coords, *(material.sizes + material.spacing))
    all_cond = True
    for i in range(3):
        data = periodic_coords[:, i]
        cond = np.logical_and(data >= xyz_shift, data < domain_max)
        all_cond = np.logical_and(all_cond, cond)
    df = (
        pd.DataFrame(
            data=np.c_[periodic_coords[all_cond, :], periodic_grains[all_cond]],
            columns=["x", "y", "z", "grain"],
        )
        .sort_values(by=["x", "y", "z"])
        .reset_index(drop=True)
    )
    return Material(
        dimensions=material.dimensions, origin=material.origin + xyz_shift, fields=df
    )


@pytest.fixture
def elastic_model():
    stiffness = Order4SymmetricTensor.from_voigt(
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
    return Elastic(stiffness)


@pytest.fixture
def material():
    material = Material(dimensions=[4, 4, 4])
    material = material.create_uniform_field(
        "orientation", Orientation.from_euler_angles([0, np.pi / 4, 0])
    ).insert_feature(
        Box(max_corner=[None, None, material.sizes[2] / 2]),
        fields={"orientation": Orientation.identity()},
    )
    return material


@pytest.fixture
def simple_material():
    return Material(dimensions=[3, 4, 5]).create_uniform_field(
        "orientation", Orientation.identity()
    )


@pytest.fixture
def material_with_defect(elastic_model):
    rng = np.random.default_rng(0)
    sizes = [3, 3, 3]
    midpoint = sizes[2] / 2
    phases = [1, 2]
    elastic_models = [elastic_model, Elastic(elastic_model.stiffness * 0.0)]

    regional_fields = pd.DataFrame(
        columns=["phase", "constitutive_model"],
        data=np.c_[phases, elastic_models],
    )

    material = (
        Material(dimensions=[8, 8, 8], sizes=sizes)
        .create_voronoi(num_regions=10, rng=rng)
        .assign_random_orientations(rng=rng)
        .create_uniform_field("phase", 1)
        .insert_feature(
            Sphere(radius=1, centroid=[midpoint, midpoint, midpoint]),
            fields={"phase": 2},
        )
        .create_regional_fields("phase", regional_fields)
    )
    return material


def test_compare_elasticity_with_evpfft(material, elastic_model):
    # evpfft mean stress norm (Voigt) = 6.583298119199364
    # mean stress norm (Voigt) = 6.5862008
    expected_mean_stress_norm = 6.5890252
    applied_strain_rate = Order2SymmetricTensor([0, 0, 1.0, 0, 0, 0])
    stress = Order2SymmetricTensor.zero()
    stress_mask = np.array([1, 1, 0, 0, 0, 0])
    load_schedule = LoadSchedule.from_constant_rates(
        applied_strain_rate, stress, stress_mask
    )
    model = SmallStrainFFT(
        load_schedule=load_schedule, end_time=5.0e-5, constitutive_model=elastic_model
    )
    material = model(material, linear_solver_tolerance=1.0e-7)
    mean_stress_norm = material.extract("stress").mean().norm.components
    assert_allclose(mean_stress_norm, expected_mean_stress_norm)


def test_stress_bc(material_with_defect):
    end_time = 1.0e-4
    expected_mean_stress_norm = 6.0
    applied_strain_rate = Order2SymmetricTensor.zero()
    stress_rate = Order2SymmetricTensor(np.array([0, 0, 6, 0, 0, 0])) / end_time
    stress_mask = np.ones(6)
    load_schedule = LoadSchedule.from_constant_rates(
        applied_strain_rate, stress_rate, stress_mask
    )
    model = SmallStrainFFT(
        load_schedule=load_schedule,
        end_time=end_time,
    )
    material = model(
        material_with_defect, phase_label="phase", linear_solver_tolerance=1.0e-7
    )
    mean_stress_norm = material.extract("stress").mean().norm.components
    assert_allclose(mean_stress_norm, expected_mean_stress_norm)


def test_elasticity_with_defect(material_with_defect):
    # evpfft mean stress norm: 4.472936630
    expected_mean_stress_norm = 4.547568053
    load_schedule = LoadSchedule.from_constant_uniaxial_strain_rate(direction="z")
    model = SmallStrainFFT(
        load_schedule=load_schedule, end_time=5.0e-5, initial_time_increment=5.0e-5
    )
    material = model(
        material_with_defect, phase_label="phase", linear_solver_tolerance=1.0e-7
    )
    mean_stress_norm = material.extract("stress").mean().norm.components
    print(mean_stress_norm)
    assert_allclose(mean_stress_norm, expected_mean_stress_norm)


def test_single_crystal_tension(simple_material, elastic_model):
    end_time = 0.001
    applied_strain_rate = Order2SymmetricTensor([-0.35, -0.35, 1.0, 0, 0, 0])
    stress = Order2SymmetricTensor.zero()
    stress_mask = np.zeros(6)
    stress_isostrain = (
        elastic_model.stiffness @ applied_strain_rate * end_time
    ).components
    load_schedule = LoadSchedule.from_constant_rates(
        applied_strain_rate, stress, stress_mask
    )
    model = SmallStrainFFT(
        load_schedule=load_schedule, end_time=end_time, constitutive_model=elastic_model
    )
    material = model(simple_material, linear_solver_tolerance=1.0e-7)
    stress_elasticity = material.extract("stress").components
    for s in stress_elasticity:
        assert_allclose_with_atol(s, stress_isostrain)


def test_single_crystal_shear(simple_material, elastic_model):
    end_time = 0.001
    applied_strain_rate = Order2SymmetricTensor.from_cartesian(
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    )
    stress = Order2SymmetricTensor.zero()
    stress_mask = np.zeros(6)
    stress_isostrain = (
        elastic_model.stiffness @ applied_strain_rate * end_time
    ).components
    load_schedule = LoadSchedule.from_constant_rates(
        applied_strain_rate, stress, stress_mask
    )
    model = SmallStrainFFT(
        load_schedule=load_schedule, end_time=end_time, constitutive_model=elastic_model
    )
    material = model(simple_material, linear_solver_tolerance=1.0e-7)
    stress_elasticity = material.extract("stress").components
    for s in stress_elasticity:
        assert_allclose_with_atol(s, stress_isostrain)


def test_periodicity(elastic_model):
    grains_per_side = 4
    num_grains = grains_per_side**3
    grains_list = np.arange(num_grains) + 1
    rng = np.random.default_rng(12345)
    orientations = Orientation.random(num_grains, rng=rng)
    regional_field = {"grain": grains_list, "orientation": orientations}

    end_time = 1.0e-4
    load_schedule = LoadSchedule.from_constant_uniaxial_strain_rate(direction="z")
    model = SmallStrainFFT(
        load_schedule=load_schedule, end_time=end_time, constitutive_model=elastic_model
    )

    material = (
        Material(dimensions=[16, 16, 16])
        .create_uniform_field("grain", 0)
        .run(create_grains, grains_per_side=grains_per_side)
        .create_regional_fields("grain", regional_field)
        .run(model, linear_solver_tolerance=1.0e-7)
    )
    grain_size = material.sizes[0] / grains_per_side
    xyz_shift_half = grain_size / 2.0
    xyz_shift_double = grain_size * 2.0
    material_half = (
        material.run(shift_origin, xyz_shift=xyz_shift_half)
        .create_regional_fields("grain", regional_field)
        .run(model, linear_solver_tolerance=1.0e-7)
    )
    material_double = (
        material.run(shift_origin, xyz_shift=xyz_shift_double)
        .create_regional_fields("grain", regional_field)
        .run(model, linear_solver_tolerance=1.0e-7)
    )
    stress = material.extract("stress").norm.components
    stress_half = material_half.extract("stress").norm.components
    stress_double = material_double.extract("stress").norm.components
    indices = material.get_region_indices("grain")
    indices_half = material_half.get_region_indices("grain")
    indices_double = material_double.get_region_indices("grain")
    for i in grains_list:
        data = np.sort(stress[indices[i]])
        half_data = np.sort(stress_half[indices_half[i]])
        double_data = np.sort(stress_double[indices_double[i]])
        assert_allclose(half_data, data)
        assert_allclose(double_data, data)
