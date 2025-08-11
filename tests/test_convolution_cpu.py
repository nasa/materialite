import numpy as np
import pytest  # Includes: tmp_path, mocker
from materialite import Material
from materialite.models.convolution_model import ConvolutionModel, LaserBeam


@pytest.fixture
def model():
    thermophysical_props = {"thermal_diffusivity": 12e-6}
    thermophysical_props.update({"volumetric_specific_heat": 678.6 * 4252})
    thermophysical_props.update({"power_absorptivity": 0.5})
    thermophysical_props.update({"melt_temperature": 1923.0})

    enth_adjustment = 1.0
    rho = 4000.0
    enthalpy_props = {"vapor_temperature": 3000.0}
    enthalpy_props.update({"heat_of_fusion": 370000.0 * rho * enth_adjustment})
    enthalpy_props.update({"heat_of_vapor": 10000000 * rho * enth_adjustment})
    enthalpy_props.update({"melt_range": 25.0})
    enthalpy_props.update({"vapor_range": 10.0})

    return ConvolutionModel(
        save_frequency=50,
        time_steps=50,
        thermophysical_props=thermophysical_props,
        enthalpy_props=enthalpy_props,
        adjust_xy_radii_multiple=3,
        adjust_xy_extra_voxels=0,
        z_range=int(35),
    )


@pytest.fixture
def material():
    material = Material(dimensions=[50, 50, 30], spacing=[5e-6, 5e-6, 5e-6])
    material = material.create_uniform_field("temperature", 300.0)

    return material


@pytest.fixture
def laser(material):

    laser = LaserBeam(
        material,
        laser_power=100,
        hatch_spacing=20,
        beam_x_radius=50.0e-6,
        beam_y_radius=50.0e-6,
        beam_z_radius=30.0e-6,
        num_layers=1,
        scan_offset_start=0,
        hatch_offset_start=0,
        hatch_offset_end=0,
        z_start=30,
        layer_thickness=8,
        time_between_scans=0.000,
        time_between_layers=0.0,
        rotation_angle_increment=np.deg2rad(67),
    )

    domain_epsilon = 5e-6
    laser.create_build_path(material, adjust_domain=True, domain_epsilon=domain_epsilon)

    return laser


def test_laser_create_build_path(material, laser):
    laser.create_build_path(material, adjust_domain=True, domain_epsilon=5e-6)

    assert laser.x_pos.size > 0
    assert laser.y_pos.size > 0
    assert laser.z_pos.size > 0
    assert laser.time.size > 0
    assert laser.beam_power.size > 0
    assert laser.scan_number.size > 0
    assert laser.layer_number.size > 0
    assert laser.x_pos.size == laser.y_pos.size == laser.z_pos.size
    assert laser.x_pos.size == laser.beam_power.size == laser.scan_number.size
    assert laser.x_pos.size == laser.scan_number.size == laser.layer_number.size

    assert np.isnan(laser.x_pos).any() == np.False_
    assert np.isnan(laser.y_pos).any() == np.False_
    assert np.isnan(laser.z_pos).any() == np.False_
    assert np.isnan(laser.time).any() == np.False_
    assert np.isnan(laser.beam_power).any() == np.False_
    assert np.isnan(laser.scan_number).any() == np.False_
    assert np.isnan(laser.layer_number).any() == np.False_


def test_laser_adjust_build_path(material):
    laser = LaserBeam(
        material,
        laser_power=100,
        hatch_spacing=20,
        beam_x_radius=50.0e-6,
        beam_y_radius=50.0e-6,
        beam_z_radius=30.0e-6,
        num_layers=10,
        scan_offset_start=0,
        hatch_offset_start=0,
        hatch_offset_end=0,
        z_start=30,
        layer_thickness=8,
        time_between_scans=0.000,
        time_between_layers=0.0,
        rotation_angle_increment=np.deg2rad(67),
    )

    laser.create_build_path(material, adjust_domain=False, domain_epsilon=5e-6)
    assert np.min(laser.x_pos) < 0
    assert np.min(laser.y_pos) < 0

    domain_epsilon = 5e-6
    laser.create_build_path(material, adjust_domain=True, domain_epsilon=domain_epsilon)
    assert np.min(laser.x_pos) > -domain_epsilon
    assert np.min(laser.y_pos) > -domain_epsilon


def test_run_model(model, material, laser):
    new_material = model(material, laser=laser, temperature_label="temperature")

    start_minimum_temperature = np.min(new_material.state["temperature_history"][0])
    expected_start_minimum_temperature = 300.0
    assert start_minimum_temperature == expected_start_minimum_temperature

    end_max_temperature = np.max(new_material.state["temperature_history"][50])
    assert end_max_temperature > start_minimum_temperature

    assert np.isnan(new_material.state["temperature_history"][0]).any() == np.False_
    assert np.isnan(new_material.state["temperature_history"][50]).any() == np.False_
