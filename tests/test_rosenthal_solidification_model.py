import numpy as np
import pytest
from materialite import Material
from materialite.models import LaserPath, RosenthalSolidificationModel
from materialite.tensor import Scalar
from numpy.testing import assert_allclose, assert_array_equal


@pytest.fixture
def small_material():
    rng = np.random.default_rng(42)
    material = Material(
        dimensions=[10, 10, 10], spacing=[0.001, 0.001, 0.001]
    ).create_random_integer_field("grain", low=0, high=10**3, rng=rng)
    return material


@pytest.fixture
def laser_path():
    return LaserPath.single_line_scan(
        start=[0, 0, 0.009],
        end=[0.02, 0, 0.009],
        velocity=1.0,
        power=50,
    )


@pytest.fixture
def model(laser_path):
    """Create a basic RosenthalSolidificationModel."""
    return RosenthalSolidificationModel(
        laser_path=laser_path,
        initial_temperature=300,
        thermophysical_properties={
            "diffusivity": 5.0e-6,
            "conductivity": 10.0,
            "absorptivity": 0.5,
            "melt_temperature": 1500.0,
        },
        time_steps=10,
        time_step_duration=0.001,
    )


def test_model_initialization(laser_path):
    """Test model initialization with default parameters."""
    model = RosenthalSolidificationModel(laser_path)
    assert model.initial_temperature == 300
    assert model.diffusivity == 5.0e-6
    assert model.conductivity == 10.0
    assert model.absorptivity == 0.5
    assert model.melt_temperature == 1500.0
    assert model.time_step_duration == 0.001
    assert model.output_times is None
    assert model.time_steps == int(np.ceil(laser_path.total_time / 0.001)) + 1


def test_model_initialization_custom(laser_path):
    """Test model initialization with custom parameters."""
    model = RosenthalSolidificationModel(
        laser_path=laser_path,
        initial_temperature=400,
        time_steps=50,
        output_times=[0.01, 0.015],
    )
    assert model.initial_temperature == 400
    assert model.time_steps == 50
    assert_array_equal(model.output_times, [0.01, 0.015, 0.02])


def test_rosenthal_temperature():
    """Test Rosenthal temperature calculation."""
    laser_path = LaserPath.single_line_scan(
        start=[0, 0, 0],
        end=[0.01, 0, 0],
        velocity=0.5,
        power=50,
    )
    model = RosenthalSolidificationModel(laser_path=laser_path)

    x_prime = Scalar(np.array([0.0, 0.001, 0.002]), dims="p")
    R = Scalar(np.array([0.001, 0.001, 0.001]), dims="p")

    temperature = model.rosenthal_temperature(
        x_prime=x_prime,
        R=R,
        Qp=50,
        v=0.5,
    )

    assert isinstance(temperature, Scalar)
    assert temperature.dims_str == "p"

    # Check that temperatures are reasonable
    assert np.all(temperature.components >= 300)
    assert np.all(temperature.components <= 1500)


def test_rosenthal_temperature_cap():
    """Test that temperature is capped at melt temperature."""
    laser_path = LaserPath.single_line_scan(
        start=[0, 0, 0],
        end=[0.01, 0, 0],
        velocity=0.1,
        power=1000,
    )
    model = RosenthalSolidificationModel(laser_path=laser_path)

    # Create conditions that would exceed melt temp
    x_prime = Scalar(np.array([0.0]), dims="p")
    R = Scalar(np.array([0.0001]), dims="p")

    temperature = model.rosenthal_temperature(
        x_prime=x_prime,
        R=R,
        Qp=1000,
        v=0.1,
    )

    assert np.all(temperature.components <= 1500)


def test_run_basic(small_material, model):
    new_material = model.run(small_material)

    assert "temperature" in list(new_material.fields)

    temperature = new_material.extract("temperature")

    assert isinstance(temperature, Scalar)
    assert temperature.dims_str == "p"

    assert np.all(temperature.components >= model.initial_temperature)
    assert np.all(temperature.components <= model.melt_temperature)


def test_run_with_history(small_material, laser_path):
    """Test run with history saving enabled."""
    output_times = [0.005, 0.01, 0.015]
    model = RosenthalSolidificationModel(
        laser_path=laser_path,
        time_step_duration=0.001,
        output_times=output_times,
    )

    new_material = model.run(small_material)

    assert "temperature" in list(new_material.fields)
    assert "time_history" in new_material.state

    temp_history = new_material.extract("temperature")
    time_history = new_material.state["time_history"]

    assert isinstance(temp_history, Scalar)
    assert temp_history.dims_str == "pt"

    # total_time is automatically appended to output_times
    assert temp_history.components.shape[0] == small_material.num_points
    assert temp_history.components.shape[1] == len(model.output_times)
    assert_array_equal(time_history, output_times + [0.02])


def test_output_times_none_vs_specified(small_material, laser_path):
    """Test difference between output_times=None and output_times specified."""
    model_no_history = RosenthalSolidificationModel(
        laser_path=laser_path,
        time_step_duration=0.001,
        output_times=None,
    )
    material_no_history = model_no_history.run(small_material)
    temp_no_history = material_no_history.extract("temperature")

    assert isinstance(temp_no_history, Scalar)
    assert temp_no_history.dims_str == "p"
    assert temp_no_history.components.shape == (small_material.num_points,)
    assert "time_history" not in material_no_history.state

    output_times = [0.005, 0.01, 0.015]
    model_with_history = RosenthalSolidificationModel(
        laser_path=laser_path,
        time_step_duration=0.001,
        output_times=output_times,
    )
    material_with_history = model_with_history.run(small_material)
    temp_with_history = material_with_history.extract("temperature")

    assert isinstance(temp_with_history, Scalar)
    assert temp_with_history.dims_str == "pt"
    assert temp_with_history.components.shape[0] == small_material.num_points
    assert "time_history" in material_with_history.state


def test_temperature_field_shape(small_material, model):
    """Test that output temperature field has correct shape."""
    new_material = model.run(small_material)
    temperature = new_material.extract("temperature")

    assert temperature.components.shape == (small_material.num_points,)


def test_custom_grain_id_label(small_material, laser_path):
    """Test using a custom grain ID field label."""
    rng = np.random.default_rng(42)
    material_custom = Material(dimensions=[10, 10, 10], spacing=[0.001, 0.001, 0.001])
    material_custom = material_custom.create_voronoi(
        num_regions=5, label="grain_custom", rng=rng
    )

    model = RosenthalSolidificationModel(
        laser_path=laser_path,
        time_steps=10,
        time_step_duration=0.001,
    )

    new_material = model.run(material_custom, grain_id_label="grain_custom")

    assert "grain_custom" in list(new_material.fields)
    assert "temperature" in list(
        new_material.fields
    )


def test_laser_movement(small_material):
    """Test that laser position changes over time when history is saved."""
    # small_material is 0.01 x 0.01 x 0.01 meters
    laser_path = LaserPath.single_line_scan(
        start=[0.0, 0.005, 0.009],
        end=[0.01, 0.005, 0.009],
        velocity=0.5,  # 0.01 m distance / 0.5 m/s = 0.02 seconds duration
        power=100,
    )

    output_times = [0.005, 0.01, 0.015]
    model = RosenthalSolidificationModel(
        time_step_duration=0.001,
        output_times=output_times,
        laser_path=laser_path,
    )

    new_material = model.run(small_material)

    temp_history = new_material.extract("temperature")

    assert isinstance(temp_history, Scalar)
    assert temp_history.dims_str == "pt"

    # Temperature distributions should be different at different times
    temp_0 = temp_history.components[:, 0]
    temp_1 = temp_history.components[:, 1]
    temp_2 = temp_history.components[:, 2]

    assert not np.allclose(temp_0, temp_1)
    assert not np.allclose(temp_1, temp_2)


def test_model_with_raster_scan(small_material):
    """Test model with a raster scan pattern."""
    laser_path = LaserPath.raster_scan(
        x_start=0,
        x_end=0.01,
        y_start=0,
        y_end=0.01,
        num_passes=3,
        velocity=0.5,
        power=100,
        z_height=0.009,
    )

    output_times = [0.01, 0.02, 0.03]
    model = RosenthalSolidificationModel(
        laser_path=laser_path,
        time_step_duration=0.001,
        output_times=output_times,
    )

    new_material = model.run(small_material)

    temperature = new_material.extract("temperature")
    assert isinstance(temperature, Scalar)
    assert temperature.dims_str == "pt"
    assert_array_equal(output_times + [laser_path.total_time], new_material.state["time_history"])


def test_model_with_multi_segment_path(small_material):
    """Test model with multiple scan segments."""
    segments = [
        {"start": [0, 0, 0.009], "end": [0.01, 0, 0.009], "velocity": 1.0, "power": 75},
        {
            "start": [0.01, 0.001, 0.009],
            "end": [0, 0.001, 0.009],
            "velocity": 1.0,
            "power": 75,
        },
        {
            "start": [0, 0.002, 0.009],
            "end": [0.01, 0.002, 0.009],
            "velocity": 1.0,
            "power": 75,
        },
    ]

    laser_path = LaserPath.from_segments(segments)

    model = RosenthalSolidificationModel(
        laser_path=laser_path, time_step_duration=0.0005
    )

    new_material = model.run(small_material)
    temperature = new_material.extract("temperature")

    assert isinstance(temperature, Scalar)
    assert np.all(temperature.components >= model.initial_temperature)


def test_laser_off_time_handling(small_material):
    """Test that model handles times when laser is not active."""
    path1 = LaserPath.single_line_scan(
        start=[0, 0.005, 0.009],
        end=[0.005, 0.005, 0.009],
        velocity=0.5,
        start_time=0.0,
        power=50,
    )

    # Use this path but simulate more time than it covers
    model = RosenthalSolidificationModel(
        laser_path=path1, time_steps=20, time_step_duration=0.001
    )

    new_material = model.run(small_material)
    temperature = new_material.extract("temperature")
    assert isinstance(temperature, Scalar)
    assert_allclose(temperature.components, model.initial_temperature)
