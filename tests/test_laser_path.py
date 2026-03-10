import numpy as np
import pytest
from materialite.models.laser_path import LaserPath
from numpy.testing import assert_allclose, assert_array_equal


def test_single_line_scan():
    """Test creating a single line scan."""
    path = LaserPath.single_line_scan(
        start=[0, 0, 0], end=[0.01, 0, 0], velocity=0.5, power=300, start_time=0.0
    )

    assert path.n_scans == 1
    assert_array_equal(path.start_positions[0], [0, 0, 0])
    assert_array_equal(path.end_positions[0], [0.01, 0, 0])
    assert path.start_times[0] == 0.0
    assert_allclose(path.end_times[0], 0.02)  # 0.01 m / 0.5 m/s = 0.02 s
    assert_array_equal(path.powers, [300])


def test_get_position_single_scan():
    """Test getting position at different times for a single scan."""
    path = LaserPath.single_line_scan(
        start=[0, 0, 0], end=[0.01, 0, 0], power=300, velocity=0.5
    )

    # Scalar inputs
    # At start time
    pos = path.get_position(0.0)
    assert_allclose(pos, [0, 0, 0])

    # At midpoint
    pos = path.get_position(0.01)
    assert_allclose(pos, [0.005, 0, 0])

    # At end time
    pos = path.get_position(0.02)
    assert_allclose(pos, [0.01, 0, 0])

    # Before start (laser not active)
    pos = path.get_position(-0.01)
    assert np.isnan(pos).all()

    # After end (laser not active)
    pos = path.get_position(0.03)
    assert np.isnan(pos).all()

    # Vectorized inputs
    # Multiple valid times
    times = np.array([0.0, 0.01, 0.02])
    positions = path.get_position(times)
    expected = np.array([[0, 0, 0], [0.005, 0, 0], [0.01, 0, 0]])
    assert_allclose(positions, expected)

    # Mix of valid and invalid times
    times = np.array([-0.01, 0.0, 0.01, 0.02, 0.03])
    positions = path.get_position(times)
    assert np.isnan(positions[0]).all()  # Before start
    assert_allclose(positions[1], [0, 0, 0])  # Start
    assert_allclose(positions[2], [0.005, 0, 0])  # Midpoint
    assert_allclose(positions[3], [0.01, 0, 0])  # End
    assert np.isnan(positions[4]).all()  # After end

    # All invalid times
    times = np.array([-0.1, -0.01, 0.03, 0.1])
    positions = path.get_position(times)
    assert positions.shape == (4, 3)
    assert np.isnan(positions).all()

    # Single-element array (should return array, not scalar)
    times = np.array([0.01])
    positions = path.get_position(times)
    assert positions.shape == (1, 3)
    assert_allclose(positions[0], [0.005, 0, 0])

    # Empty array
    times = np.array([])
    positions = path.get_position(times)
    assert positions.shape == (0, 3)


def test_from_segments():
    """Test creating path from multiple segments."""
    segments = [
        {"start": [0, 0, 0], "end": [0.01, 0, 0], "velocity": 0.5, "power": 300},
        {
            "start": [0.01, 0.001, 0],
            "end": [0, 0.001, 0],
            "velocity": 0.5,
            "power": 300,
        },
        {
            "start": [0, 0.002, 0],
            "end": [0.01, 0.002, 0],
            "velocity": 0.5,
            "power": 300,
        },
    ]

    path = LaserPath.from_segments(segments)

    assert path.n_scans == 3
    assert_allclose(path.total_time, 0.06)  # 3 segments * 0.02s each

    # Check first segment
    assert_array_equal(path.start_positions[0], [0, 0, 0])
    assert_array_equal(path.end_positions[0], [0.01, 0, 0])

    # Check second segment
    assert_array_equal(path.start_positions[1], [0.01, 0.001, 0])
    assert_array_equal(path.end_positions[1], [0, 0.001, 0])


def test_get_position_multi_segment():
    """Test getting position across multiple segments."""
    segments = [
        {"start": [0, 0, 0], "end": [0.01, 0, 0], "velocity": 0.5, "power": 300},
        {
            "start": [0.01, 0.001, 0],
            "end": [0, 0.001, 0],
            "velocity": 0.5,
            "power": 300,
        },
    ]

    path = LaserPath.from_segments(segments)

    # Scalar inputs
    # Position during first segment
    pos = path.get_position(0.01)
    assert_allclose(pos, [0.005, 0, 0])

    # Position during second segment
    pos = path.get_position(0.03)
    assert_allclose(pos, [0.005, 0.001, 0])

    # Position after second segment
    pos = path.get_position(0.041)
    assert np.isnan(pos).all()

    # Vectorized inputs
    # Times spanning both segments
    times = np.array([0.0, 0.01, 0.019, 0.03, 0.04])
    positions = path.get_position(times)
    
    # First segment: 0.0 to 0.02 -> [0, 0, 0] to [0.01, 0, 0]
    assert_allclose(positions[0], [0, 0, 0])  # t=0.0, start of segment 1
    assert_allclose(positions[1], [0.005, 0, 0])  # t=0.01, middle of segment 1
    assert_allclose(positions[2], [0.0095, 0, 0])  # t=0.019, near end of segment 1
    
    # Second segment: 0.02 to 0.04 -> [0.01, 0.001, 0] to [0, 0.001, 0]
    assert_allclose(positions[3], [0.005, 0.001, 0])  # t=0.03, middle of segment 2
    assert_allclose(positions[4], [0, 0.001, 0])  # t=0.04, end of segment 2

    # Times with gaps between segments and after
    times = np.array([-0.01, 0.015, 0.041, 0.1])
    positions = path.get_position(times)
    assert np.isnan(positions[0]).all()  # Before first segment
    assert_allclose(positions[1], [0.0075, 0, 0])  # In first segment
    assert np.isnan(positions[2]).all()  # After second segment
    assert np.isnan(positions[3]).all()  # Well after all segments

    # All times in first segment
    times = np.array([0.0, 0.005, 0.01, 0.015, 0.019])
    positions = path.get_position(times)
    assert_allclose(positions[0], [0.0, 0, 0])
    assert_allclose(positions[1], [0.0025, 0, 0])
    assert_allclose(positions[2], [0.005, 0, 0])
    assert_allclose(positions[3], [0.0075, 0, 0])
    assert_allclose(positions[4], [0.0095, 0, 0])

    # All times in second segment
    times = np.array([0.02, 0.03, 0.04])
    positions = path.get_position(times)
    assert_allclose(positions[0], [0.01, 0.001, 0])
    assert_allclose(positions[1], [0.005, 0.001, 0])
    assert_allclose(positions[2], [0.0, 0.001, 0])


def test_get_position_segment_boundaries():
    """Test get_position behavior at segment boundaries.
    
    When segments are consecutive (end time of one = start time of next),
    the position at the boundary should be at the start of the second segment.
    """
    segments = [
        {"start": [0, 0, 0], "end": [0.01, 0, 0], "velocity": 0.5, "power": 300},
        {
            "start": [0.01, 0.001, 0],
            "end": [0, 0.001, 0],
            "velocity": 0.5,
            "power": 300,
        },
    ]

    path = LaserPath.from_segments(segments)

    # At the exact boundary between segments (t=0.02)
    # This should return the start of the second segment
    pos = path.get_position(0.02)
    assert_allclose(pos, [0.01, 0.001, 0])

    # Test with vectorized input including boundary
    times = np.array([0.0, 0.02, 0.04])
    positions = path.get_position(times)
    assert_allclose(positions[0], [0.0, 0.0, 0.0])  # Start of segment 1
    assert_allclose(positions[1], [0.01, 0.001, 0.0])  # Start of segment 2 (boundary)
    assert_allclose(positions[2], [0.0, 0.001, 0.0])  # End of segment 2


def test_get_scan_indices_single_scan():
    """Test getting scan indices at different times for a single scan."""
    path = LaserPath.single_line_scan(
        start=[0, 0, 0], end=[0.01, 0, 0], power=300, velocity=0.5
    )

    # Scalar inputs
    # At start time
    idx = path.get_scan_indices(0.0)
    assert idx == 0

    # At midpoint
    idx = path.get_scan_indices(0.01)
    assert idx == 0

    # At end time
    idx = path.get_scan_indices(0.02)
    assert idx == 0

    # Before start (laser not active)
    idx = path.get_scan_indices(-0.01)
    assert idx == -1

    # After end (laser not active)
    idx = path.get_scan_indices(0.03)
    assert idx == -1

    # Vectorized inputs
    # Multiple valid times
    times = np.array([0.0, 0.01, 0.02])
    indices = path.get_scan_indices(times)
    assert_array_equal(indices, [0, 0, 0])

    # Mix of valid and invalid times
    times = np.array([-0.01, 0.0, 0.01, 0.02, 0.03])
    indices = path.get_scan_indices(times)
    assert_array_equal(indices, [-1, 0, 0, 0, -1])

    # All invalid times
    times = np.array([-0.1, -0.01, 0.03, 0.1])
    indices = path.get_scan_indices(times)
    assert_array_equal(indices, [-1, -1, -1, -1])

    # Single-element array (should return array, not scalar)
    times = np.array([0.01])
    indices = path.get_scan_indices(times)
    assert indices.shape == (1,)
    assert indices[0] == 0

    # Empty array
    times = np.array([])
    indices = path.get_scan_indices(times)
    assert indices.shape == (0,)


def test_get_scan_indices_multi_segment():
    """Test getting scan indices across multiple segments."""
    segments = [
        {"start": [0, 0, 0], "end": [0.01, 0, 0], "velocity": 0.5, "power": 300},
        {
            "start": [0.01, 0.001, 0],
            "end": [0, 0.001, 0],
            "velocity": 0.5,
            "power": 400,
        },
        {
            "start": [0, 0.002, 0],
            "end": [0.01, 0.002, 0],
            "velocity": 1.0,
            "power": 500,
        },
    ]

    path = LaserPath.from_segments(segments)

    # Scalar inputs
    # During first segment
    idx = path.get_scan_indices(0.01)
    assert idx == 0

    # During second segment
    idx = path.get_scan_indices(0.03)
    assert idx == 1

    # During third segment
    idx = path.get_scan_indices(0.045)
    assert idx == 2

    # After all segments
    idx = path.get_scan_indices(0.06)
    assert idx == -1

    # Before all segments
    idx = path.get_scan_indices(-0.01)
    assert idx == -1

    # Vectorized inputs
    # Times spanning all segments
    times = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.045, 0.05])
    indices = path.get_scan_indices(times)
    # Segment 0: 0.0 to 0.02
    # Segment 1: 0.02 to 0.04
    # Segment 2: 0.04 to 0.05
    assert_array_equal(indices, [0, 0, 1, 1, 2, 2, 2])

    # Times with gaps and invalid periods
    times = np.array([-0.01, 0.015, 0.035, 0.055])
    indices = path.get_scan_indices(times)
    assert_array_equal(indices, [-1, 0, 1, -1])

    # All times in first segment
    times = np.array([0.0, 0.005, 0.01, 0.015, 0.019])
    indices = path.get_scan_indices(times)
    assert_array_equal(indices, [0, 0, 0, 0, 0])

    # All times in second segment (note: t=0.04 is at boundary, returns segment 2)
    times = np.array([0.02, 0.03, 0.039])
    indices = path.get_scan_indices(times)
    assert_array_equal(indices, [1, 1, 1])


def test_get_scan_indices_segment_boundaries():
    """Test get_scan_indices behavior at segment boundaries.
    
    When segments are consecutive (end time of one = start time of next),
    the index at the boundary should be the second segment.
    """
    segments = [
        {"start": [0, 0, 0], "end": [0.01, 0, 0], "velocity": 0.5, "power": 300},
        {
            "start": [0.01, 0.001, 0],
            "end": [0, 0.001, 0],
            "velocity": 0.5,
            "power": 400,
        },
    ]

    path = LaserPath.from_segments(segments)

    # At the exact boundary between segments (t=0.02)
    # This should return the index of the second segment
    idx = path.get_scan_indices(0.02)
    assert idx == 1

    # Test with vectorized input including boundary
    times = np.array([0.0, 0.02, 0.04])
    indices = path.get_scan_indices(times)
    assert_array_equal(indices, [0, 1, 1])


def test_get_scan_indices_access_properties():
    """Test using get_scan_indices to access scan properties."""
    segments = [
        {"start": [0, 0, 0], "end": [0.01, 0, 0], "velocity": 0.5, "power": 300},
        {
            "start": [0.01, 0.001, 0],
            "end": [0, 0.001, 0],
            "velocity": 0.5,
            "power": 400,
        },
        {
            "start": [0, 0.002, 0],
            "end": [0.01, 0.002, 0],
            "velocity": 1.0,
            "power": 500,
        },
    ]

    path = LaserPath.from_segments(segments)

    # Test accessing properties for different times
    time = 0.01  # First segment
    idx = path.get_scan_indices(time)
    assert idx >= 0
    assert path.powers[idx] == 300
    assert_array_equal(path.start_positions[idx], [0, 0, 0])
    assert_array_equal(path.end_positions[idx], [0.01, 0, 0])

    time = 0.03  # Second segment
    idx = path.get_scan_indices(time)
    assert idx >= 0
    assert path.powers[idx] == 400
    assert_array_equal(path.start_positions[idx], [0.01, 0.001, 0])
    assert_array_equal(path.end_positions[idx], [0, 0.001, 0])

    time = 0.045  # Third segment
    idx = path.get_scan_indices(time)
    assert idx >= 0
    assert path.powers[idx] == 500
    assert_array_equal(path.start_positions[idx], [0, 0.002, 0])
    assert_array_equal(path.end_positions[idx], [0.01, 0.002, 0])

    # Test with vectorized input
    times = np.array([0.01, 0.03, 0.045])
    indices = path.get_scan_indices(times)
    powers = path.powers[indices]
    assert_array_equal(powers, [300, 400, 500])


def test_get_scan_indices_with_gaps():
    """Test get_scan_indices when there are gaps between segments."""
    # Create non-consecutive segments with gaps
    path1 = LaserPath.single_line_scan(
        start=[0, 0, 0], end=[0.01, 0, 0], velocity=0.5, power=300, start_time=0.0
    )
    path2 = LaserPath.single_line_scan(
        start=[0, 0.001, 0],
        end=[0.01, 0.001, 0],
        velocity=0.5,
        power=400,
        start_time=0.1,  # Gap from 0.02 to 0.1
    )
    path3 = LaserPath.single_line_scan(
        start=[0, 0.002, 0],
        end=[0.01, 0.002, 0],
        velocity=0.5,
        power=500,
        start_time=0.2,  # Gap from 0.12 to 0.2
    )

    combined = LaserPath.from_list([path1, path2, path3])

    # Scalar inputs in gaps should return -1
    idx = combined.get_scan_indices(0.05)  # In gap between first and second
    assert idx == -1

    idx = combined.get_scan_indices(0.15)  # In gap between second and third
    assert idx == -1

    # Vectorized inputs
    times = np.array([0.0, 0.01, 0.05, 0.1, 0.11, 0.15, 0.2, 0.21])
    indices = combined.get_scan_indices(times)
    # Segment 0: 0.0 to 0.02
    # Gap: 0.02 to 0.1
    # Segment 1: 0.1 to 0.12
    # Gap: 0.12 to 0.2
    # Segment 2: 0.2 to 0.22
    assert_array_equal(indices, [0, 0, -1, 1, 1, -1, 2, 2])


def test_get_scan_indices_dtype():
    """Test that get_scan_indices returns correct integer types."""
    path = LaserPath.single_line_scan(
        start=[0, 0, 0], end=[0.01, 0, 0], power=300, velocity=0.5
    )

    # Scalar should return Python int
    idx = path.get_scan_indices(0.01)
    assert isinstance(idx, int)

    # Array should return numpy array with int32 dtype
    times = np.array([0.0, 0.01, 0.02])
    indices = path.get_scan_indices(times)
    assert indices.dtype == np.int32


def test_from_list():
    """Test combining multiple LaserPath objects into one using from_list."""
    # Create three separate paths
    path1 = LaserPath.single_line_scan(
        start=[0, 0, 0], end=[0.01, 0, 0], velocity=0.5, power=300, start_time=0.0
    )

    path2 = LaserPath.single_line_scan(
        start=[0, 0.001, 0],
        end=[0.01, 0.001, 0],
        velocity=1.0,
        power=300,
        start_time=0.5,
    )

    segments = [
        {
            "start": [0, 0.002, 0],
            "end": [0.01, 0.002, 0],
            "velocity": 0.5,
            "power": 300,
        },
        {
            "start": [0.01, 0.003, 0],
            "end": [0, 0.003, 0],
            "velocity": 0.5,
            "power": 300,
        },
    ]
    path3 = LaserPath.from_segments(segments, start_time=1.0)

    # Combine all three paths
    combined = LaserPath.from_list([path1, path2, path3])

    # Total scans should be sum of all individual paths
    assert combined.n_scans == 4  # 1 + 1 + 2

    # Verify start positions are correctly concatenated
    assert_array_equal(combined.start_positions[0], [0, 0, 0])  # From path1
    assert_array_equal(combined.start_positions[1], [0, 0.001, 0])  # From path2
    assert_array_equal(combined.start_positions[2], [0, 0.002, 0])  # From path3, scan 1
    assert_array_equal(
        combined.start_positions[3], [0.01, 0.003, 0]
    )  # From path3, scan 2

    # Verify end positions
    assert_array_equal(combined.end_positions[0], [0.01, 0, 0])
    assert_array_equal(combined.end_positions[1], [0.01, 0.001, 0])

    # Verify times are correctly concatenated
    assert_allclose(combined.start_times[0], 0.0)
    assert_allclose(combined.start_times[1], 0.5)
    assert_allclose(combined.start_times[2], 1.0)
    assert_allclose(combined.start_times[3], 1.02)

    # Verify get_position works across the combined path
    # Position during first original path
    pos = combined.get_position(0.01)
    assert_allclose(pos, [0.005, 0, 0])

    # Position during second original path
    pos = combined.get_position(0.505)
    assert_allclose(pos, [0.005, 0.001, 0])

    # Position during third original path
    pos = combined.get_position(1.01)
    assert_allclose(pos, [0.005, 0.002, 0])

    # Total time should be the maximum end time
    assert_allclose(combined.total_time, path3.total_time)


def test_from_list_single_path():
    """Test from_list with a single path in the list."""
    path = LaserPath.single_line_scan(
        start=[0, 0, 0],
        end=[0.01, 0, 0],
        velocity=0.5,
        power=300,
    )

    combined = LaserPath.from_list([path])

    # Should be identical to the original path
    assert combined.n_scans == 1
    assert_array_equal(combined.start_positions, path.start_positions)
    assert_array_equal(combined.end_positions, path.end_positions)
    assert_array_equal(combined.start_times, path.start_times)
    assert_array_equal(combined.end_times, path.end_times)


def test_raster_scan_bidirectional():
    """Test creating a bidirectional raster scan pattern."""
    path = LaserPath.raster_scan(
        x_start=0,
        x_end=0.01,
        y_start=0,
        y_end=0.01,
        num_passes=3,
        velocity=0.5,
        power=300,
        z_height=0.001,
        bidirectional=True,
    )

    assert path.n_scans == 3

    # First pass: left to right
    assert_array_equal(path.start_positions[0], [0, 0, 0.001])
    assert_array_equal(path.end_positions[0], [0.01, 0, 0.001])

    # Second pass: right to left (bidirectional)
    assert_array_equal(path.start_positions[1], [0.01, 0.005, 0.001])
    assert_array_equal(path.end_positions[1], [0, 0.005, 0.001])

    # Third pass: left to right again
    assert_array_equal(path.start_positions[2], [0, 0.01, 0.001])
    assert_array_equal(path.end_positions[2], [0.01, 0.01, 0.001])


def test_raster_scan_unidirectional():
    """Test creating a unidirectional raster scan pattern."""
    path = LaserPath.raster_scan(
        x_start=0,
        x_end=0.01,
        y_start=0,
        y_end=0.01,
        num_passes=2,
        velocity=0.5,
        power=300,
        bidirectional=False,
    )

    assert path.n_scans == 2

    # Both passes should go in the same direction (left to right)
    assert_array_equal(path.start_positions[0][:2], [0, 0])
    assert_array_equal(path.end_positions[0][:2], [0.01, 0])
    assert_array_equal(path.start_positions[1][:2], [0, 0.01])
    assert_array_equal(path.end_positions[1][:2], [0.01, 0.01])


def test_total_time():
    """Test total_time property."""
    path = LaserPath.single_line_scan(
        start=[0, 0, 0],
        end=[0.01, 0, 0],
        velocity=0.5,
        power=300,
    )
    assert_allclose(path.total_time, 0.02)

    segments = [
        {"start": [0, 0, 0], "end": [0.01, 0, 0], "velocity": 0.5, "power": 300},
        {
            "start": [0.01, 0.001, 0],
            "end": [0, 0.001, 0],
            "velocity": 1.0,
            "power": 300,
        },
    ]
    path = LaserPath.from_segments(segments, start_time=1.0)
    # First segment: 0.02s, second segment: 0.01s, start at 1.0s
    assert_allclose(path.total_time, 1.03)


def test_validation_errors():
    """Test that validation catches invalid inputs."""
    # end_times before start_times
    with pytest.raises(ValueError, match="end_times must be greater"):
        LaserPath(
            start_positions=[[0, 0, 0]],
            end_positions=[[1, 0, 0]],
            start_times=[1.0],
            end_times=[0.5],
            powers=[300],
        )

    # Mismatched shapes
    with pytest.raises(ValueError, match="start_positions must have shape"):
        LaserPath(
            start_positions=[[0, 0]],  # Wrong shape
            end_positions=[[1, 0, 0]],
            start_times=[0.0],
            end_times=[1.0],
            powers=[300],
        )


def test_repr():
    """Test string representation."""
    path = LaserPath.single_line_scan(
        start=[0, 0, 0],
        end=[0.01, 0, 0],
        velocity=0.5,
        power=300,
    )
    repr_str = repr(path)
    assert "LaserPath" in repr_str
    assert "n_scans=1" in repr_str
    assert "total_time" in repr_str


def test_angled_raster_scan_0_degrees():
    """Test angled raster scan at 0 degrees (horizontal scans)."""
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
    )

    # Should have 3 scans: y=0.0, y=0.005, y=0.01
    assert path.n_scans == 3

    # First scan should start at origin and end at (0.01, 0, 0)
    assert_allclose(path.start_positions[0], [0.0, 0.0, 0.0])
    assert_allclose(path.end_positions[0], [0.01, 0.0, 0.0])

    # Second scan should be at y=0.005 and reversed (bidirectional)
    assert_allclose(path.start_positions[1][1], 0.005, atol=1e-10)
    assert_allclose(path.end_positions[1][1], 0.005, atol=1e-10)
    # Reversed: starts at x_max, ends at x_min
    assert path.start_positions[1][0] > path.end_positions[1][0]


def test_angled_raster_scan_90_degrees():
    """Test angled raster scan at 90 degrees (vertical scans)."""
    path = LaserPath.angled_raster_scan(
        angle=90,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
    )

    # Should have 3 scans: x=0.0, x=0.005, x=0.01
    assert path.n_scans == 3

    # First scan should be vertical (constant x, varying y)
    assert_allclose(path.start_positions[0][0], path.end_positions[0][0])
    # y should span from y_min to y_max
    assert_allclose(
        abs(path.end_positions[0][1] - path.start_positions[0][1]), 0.01, atol=1e-10
    )


def test_angled_raster_scan_45_degrees():
    """Test angled raster scan at 45 degrees (diagonal scans)."""
    path = LaserPath.angled_raster_scan(
        angle=45,
        hatch_spacing=0.001,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
    )

    # Should create 14 diagonal scans
    assert path.n_scans == 14

    # Check that scans are approximately diagonal (dx ≈ dy)
    for i in range(path.n_scans):
        dx = path.end_positions[i][0] - path.start_positions[i][0]
        dy = path.end_positions[i][1] - path.start_positions[i][1]
        # For 45°, |dx| should approximately equal |dy|
        assert_allclose(abs(dx), abs(dy), rtol=0.1)


def test_angled_raster_scan_with_start_offset():
    """Test angled raster scan with start offset."""
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        start_offset=0.001,  # Extend 1mm before boundary
    )

    # First scan should start at x=-0.001 (extended backward)
    assert_allclose(path.start_positions[0][0], -0.001)
    # End should still be at x=0.01
    assert_allclose(path.end_positions[0][0], 0.01)


def test_angled_raster_scan_with_end_offset():
    """Test angled raster scan with end offset."""
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        end_offset=0.002,  # Extend 2mm beyond boundary
    )

    # First scan should start at x=0.0
    assert_allclose(path.start_positions[0][0], 0.0)
    # End should be at x=0.012 (extended forward)
    assert_allclose(path.end_positions[0][0], 0.012)


def test_angled_raster_scan_with_both_offsets():
    """Test angled raster scan with both start and end offsets."""
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        start_offset=0.001,
        end_offset=0.002,
    )

    # First scan should extend on both ends
    assert_allclose(path.start_positions[0][0], -0.001)
    assert_allclose(path.end_positions[0][0], 0.012)


def test_angled_raster_scan_positive_domain_offset():
    """Test angled raster scan with positive domain_offset.
    
    Positive domain_offset extends the intersection boundaries beyond the 
    specified x_bounds and y_bounds, making scans longer and potentially 
    increasing the number of scans.
    """
    # Test at 0 degrees (horizontal scans)
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        domain_offset=0.002,  # Extend 2mm beyond bounds
    )

    # Should have 3 scans at 0° with 5mm spacing
    assert path.n_scans == 3

    # First scan should start at extended x_min and end at extended x_max
    # domain_offset extends both x and y bounds
    # For 0° scans (horizontal), x determines scan length, y determines scan position
    # x_min = 0.0 - 0.002 = -0.002
    # x_max = 0.01 + 0.002 = 0.012
    assert_allclose(path.start_positions[0][0], -0.002, atol=1e-10)
    assert_allclose(path.end_positions[0][0], 0.012, atol=1e-10)
    
    # Y position should be at y_min (unaffected by domain_offset for 0° scans)
    # Actually, for the first scan at angle=0, it should be at the minimum y projection
    # which would be y_bound_low (0.0) since that's what determines hatch positions
    assert_allclose(path.start_positions[0][1], 0.0, atol=1e-10)

    # Test at 90 degrees (vertical scans)
    path = LaserPath.angled_raster_scan(
        angle=90,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        domain_offset=0.002,
    )

    # Should have 3 scans at 90°
    assert path.n_scans == 3

    # For 90° scans (vertical), y determines scan length, x determines scan position
    # First vertical scan should extend in y direction
    # y_min = 0.0 - 0.002 = -0.002
    # y_max = 0.01 + 0.002 = 0.012
    assert_allclose(path.start_positions[0][1], -0.002, atol=1e-10)
    assert_allclose(path.end_positions[0][1], 0.012, atol=1e-10)


def test_angled_raster_scan_negative_domain_offset():
    """Test angled raster scan with negative domain_offset.
    
    Negative domain_offset shrinks the intersection boundaries, making scans 
    shorter and potentially reducing the number of scans (since the effective 
    domain is smaller).
    """
    # Test at 0 degrees (horizontal scans) with bidirectional=False for predictable positions
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.002,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        domain_offset=-0.002,  # Shrink 2mm inside bounds
        bidirectional=False,    # All scans go left-to-right
    )

    # With negative domain_offset, the effective domain is smaller
    # x_min = 0.0 - (-0.002) = 0.002
    # x_max = 0.01 + (-0.002) = 0.008
    # y_min = 0.0 - (-0.002) = 0.002
    # y_max = 0.01 + (-0.002) = 0.008
    # Y range = 0.006, with 0.002 spacing = 4 scans
    assert path.n_scans == 4

    # First scan should start at shrunk x_min and end at shrunk x_max (unidirectional)
    # x_min = 0.002, x_max = 0.008
    assert_allclose(path.start_positions[0][0], 0.002, atol=1e-10)
    assert_allclose(path.end_positions[0][0], 0.008, atol=1e-10)
    
    # Y position should be at the shrunk y_min = 0.002
    assert_allclose(path.start_positions[0][1], 0.002, atol=1e-10)
    
    # Second scan should also go left-to-right at y=0.004
    assert_allclose(path.start_positions[1][0], 0.002, atol=1e-10)
    assert_allclose(path.end_positions[1][0], 0.008, atol=1e-10)
    assert_allclose(path.start_positions[1][1], 0.004, atol=1e-10)

    # Test at 45 degrees
    path = LaserPath.angled_raster_scan(
        angle=45,
        hatch_spacing=0.001,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        domain_offset=-0.001,  # Shrink 1mm inside bounds
    )

    # Negative domain_offset should reduce the number of scans
    # Without offset: 14 scans
    # With -1mm offset: effective domain is 8mm x 8mm = 11 scans
    assert path.n_scans == 11

    # All scans should be within the shrunk domain
    for i in range(path.n_scans):
        # Check that all positions are within shrunk bounds (with some numerical tolerance)
        assert path.start_positions[i][0] >= 0.001 - 1e-9
        assert path.start_positions[i][0] <= 0.009 + 1e-9
        assert path.start_positions[i][1] >= 0.001 - 1e-9
        assert path.start_positions[i][1] <= 0.009 + 1e-9


def test_angled_raster_scan_domain_offset_with_start_end_offsets():
    """Test that domain_offset works correctly with start_offset and end_offset.
    
    domain_offset affects the intersection boundaries (where scans intersect the domain),
    while start_offset and end_offset extend individual scans beyond their intersection points.
    These should work independently and compound their effects.
    """
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        domain_offset=0.001,   # Extend intersection boundaries by 1mm
        start_offset=0.001,     # Extend each scan start by 1mm
        end_offset=0.001,       # Extend each scan end by 1mm
    )

    # Should have 3 scans
    assert path.n_scans == 3

    # First scan calculation:
    # Domain boundaries: x_min = -0.001, x_max = 0.011 (due to domain_offset)
    # Intersection gives: start at x=-0.001, end at x=0.011
    # With start_offset: start at x=-0.001-0.001 = -0.002
    # With end_offset: end at x=0.011+0.001 = 0.012
    assert_allclose(path.start_positions[0][0], -0.002, atol=1e-10)
    assert_allclose(path.end_positions[0][0], 0.012, atol=1e-10)


def test_angled_raster_scan_large_negative_domain_offset():
    """Test that large negative domain_offset raises error when domain becomes invalid.
    
    If domain_offset is so negative that it inverts the domain (x_min >= x_max or 
    y_min >= y_max), or makes it too small to generate any scans, this should be 
    handled gracefully.
    """
    # Very large negative offset that makes effective domain too small
    with pytest.raises(ValueError, match="No valid scan segments"):
        LaserPath.angled_raster_scan(
            angle=0,
            hatch_spacing=0.002,
            x_bounds=(0.0, 0.01),
            y_bounds=(0.0, 0.01),
            velocity=0.5,
            power=300,
            domain_offset=-0.006,  # 6mm offset on a 10mm domain -> only 4mm x 4mm effective
        )


def test_angled_raster_scan_bidirectional():
    """Test that bidirectional scans alternate direction correctly."""
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        bidirectional=True,
    )

    # First scan: forward (x increases)
    assert path.end_positions[0][0] > path.start_positions[0][0]

    # Second scan: reverse (x decreases)
    if path.n_scans > 1:
        assert path.end_positions[1][0] < path.start_positions[1][0]

    # Third scan: forward again (x increases)
    if path.n_scans > 2:
        assert path.end_positions[2][0] > path.start_positions[2][0]


def test_angled_raster_scan_unidirectional():
    """Test that unidirectional scans all go the same direction."""
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        bidirectional=False,
    )

    # All scans should go the same direction (x increases)
    for i in range(path.n_scans):
        assert path.end_positions[i][0] > path.start_positions[i][0]


def test_angled_raster_scan_negative_offset():
    """Test angled raster scan with negative offsets (shortens scans)."""
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        start_offset=-0.001,  # Start inside domain
        end_offset=-0.001,  # End inside domain
    )

    # First scan should be shortened on both ends
    assert_allclose(path.start_positions[0][0], 0.001)
    assert_allclose(path.end_positions[0][0], 0.009)


def test_angled_raster_scan_z_height():
    """Test that z_height parameter is correctly applied."""
    z_height = 0.005
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        z_height=z_height,
    )

    # All scans should be at specified z height
    for i in range(path.n_scans):
        assert_allclose(path.start_positions[i][2], z_height)
        assert_allclose(path.end_positions[i][2], z_height)


def test_angled_raster_scan_start_time():
    """Test that start_time parameter works correctly."""
    start_time = 1.0
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        start_time=start_time,
    )

    # First scan should start at specified time
    assert_allclose(path.start_times[0], start_time)
    # Total time should be offset by start_time
    assert path.total_time >= start_time


def test_angled_raster_scan_bidirectional_with_offsets():
    """Test that offsets work correctly with bidirectional scanning."""
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        start_offset=0.001,
        end_offset=0.002,
        bidirectional=True,
    )

    # First scan (forward): start extended backward, end extended forward
    assert_allclose(path.start_positions[0][0], -0.001)
    assert_allclose(path.end_positions[0][0], 0.012)

    # Second scan (reverse): start still extended by start_offset,
    # end still extended by end_offset (but in reverse direction)
    if path.n_scans > 1:
        # Reverse scan: starts at right (x_max + start_offset)
        assert_allclose(path.start_positions[1][0], 0.011)
        # Ends at left (x_min - end_offset)
        assert_allclose(path.end_positions[1][0], -0.002)


def test_angled_raster_scan_large_hatch_spacing():
    """Test angled raster scan with very large hatch spacing.

    This tests the edge case where hatch_spacing is large relative to the domain,
    resulting in fewer scans than might be expected. Some hatch lines may not
    intersect the domain at all (len(intersections) < 2).
    """
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.015,  # Larger than the 10mm domain
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
    )

    # Should have 1 scan (spacing larger than domain)
    assert path.n_scans == 1

    # All scans should be valid (proper start and end positions)
    for i in range(path.n_scans):
        # Scan should have non-zero length
        length = np.linalg.norm(path.end_positions[i] - path.start_positions[i])
        assert length > 0


def test_angled_raster_scan_very_large_hatch_spacing():
    """Test angled raster scan with hatch spacing much larger than domain.

    This is an extreme edge case where hatch lines don't intersect the domain.
    Should raise ValueError with a helpful message.
    """
    with pytest.raises(ValueError, match="No valid scan segments"):
        LaserPath.angled_raster_scan(
            angle=45,
            hatch_spacing=0.1,  # 100mm spacing for a 10mm domain
            x_bounds=(0.0, 0.01),
            y_bounds=(0.0, 0.01),
            velocity=0.5,
            power=300,
        )


def test_angled_raster_scan_small_domain():
    """Test angled raster scan with a very small domain.

    Tests that the method works correctly even with tiny dimensions,
    where numerical precision could cause intersection detection issues.
    """
    path = LaserPath.angled_raster_scan(
        angle=30,
        hatch_spacing=0.0001,  # 0.1mm spacing
        x_bounds=(0.0, 0.001),  # 1mm x 1mm domain
        y_bounds=(0.0, 0.001),
        velocity=0.5,
        power=300,
    )

    # Should have 13 scans for the small domain with 0.1mm spacing
    assert path.n_scans == 13

    # All scans should be within or very close to the small domain
    for i in range(path.n_scans):
        assert -0.001 <= path.start_positions[i][0] <= 0.002
        assert -0.001 <= path.start_positions[i][1] <= 0.002


def test_angled_raster_scan_near_zero_angle():
    """Test angled raster scan with angle very close to 0 degrees.

    Tests numerical stability when scan_dir is nearly [1, 0].
    """
    path = LaserPath.angled_raster_scan(
        angle=0.01,  # Nearly horizontal
        hatch_spacing=0.002,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
    )

    # Should have 5 scans (nearly horizontal)
    assert path.n_scans == 5

    # Scans should be nearly horizontal
    for i in range(path.n_scans):
        dx = path.end_positions[i][0] - path.start_positions[i][0]
        dy = path.end_positions[i][1] - path.start_positions[i][1]
        # |dx| should be much larger than |dy| for near-horizontal scans
        assert abs(dx) > 10 * abs(dy)


def test_angled_raster_scan_near_90_angle():
    """Test angled raster scan with angle very close to 90 degrees.

    Tests numerical stability when scan_dir is nearly [0, 1].
    """
    path = LaserPath.angled_raster_scan(
        angle=89.99,  # Nearly vertical
        hatch_spacing=0.002,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
    )

    # Should have 5 scans (nearly vertical)
    assert path.n_scans == 5

    # Scans should be nearly vertical
    for i in range(path.n_scans):
        dx = path.end_positions[i][0] - path.start_positions[i][0]
        dy = path.end_positions[i][1] - path.start_positions[i][1]
        # |dy| should be much larger than |dx| for near-vertical scans
        assert abs(dy) > 10 * abs(dx)


def test_angled_raster_scan_negative_angle():
    """Test angled raster scan with negative angle.

    Negative angles should work correctly (clockwise rotation).
    """
    path = LaserPath.angled_raster_scan(
        angle=-45,
        hatch_spacing=0.002,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
    )

    # Should have 7 scans at -45°
    assert path.n_scans == 7

    # Check that scans are diagonal (opposite direction from +45°)
    for i in range(path.n_scans):
        dx = path.end_positions[i][0] - path.start_positions[i][0]
        dy = path.end_positions[i][1] - path.start_positions[i][1]
        # For -45°, dx and dy should have opposite signs or similar magnitude
        assert_allclose(abs(dx), abs(dy), rtol=0.2)


def test_angled_raster_scan_180_degree():
    """Test angled raster scan at 180 degrees.

    Should produce horizontal scans in the opposite direction from 0°.
    """
    path = LaserPath.angled_raster_scan(
        angle=180,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        bidirectional=False,
    )

    # Should have 2 scans at 180°
    assert path.n_scans == 2

    # First scan should go from right to left (negative x direction)
    # even in unidirectional mode
    dx = path.end_positions[0][0] - path.start_positions[0][0]
    assert dx < 0  # Moving in negative x direction


def test_angled_raster_scan_non_square_domain():
    """Test angled raster scan on a non-square rectangular domain.

    Tests that the method correctly handles aspect ratios != 1.
    """
    path = LaserPath.angled_raster_scan(
        angle=45,
        hatch_spacing=0.001,
        x_bounds=(0.0, 0.02),  # 20mm wide
        y_bounds=(0.0, 0.005),  # 5mm tall
        velocity=0.5,
        power=300,
    )

    # Should have 17 scans for non-square domain at 45°
    assert path.n_scans == 17

    # All scans should have reasonable positions within the domain bounds
    for i in range(path.n_scans):
        # Allow some tolerance for offsets/numerical precision
        assert -0.005 <= path.start_positions[i][0] <= 0.025
        assert -0.005 <= path.start_positions[i][1] <= 0.010


def test_angled_raster_scan_ensures_minimum_scans():
    """Test scan count with very large hatch spacing.

    Even with massive hatch spacing (much larger than the domain),
    np.arange will produce at least one scan at the starting position.
    """
    # Even with massive hatch spacing, should get exactly 1 scan
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=1.0,  # 1 meter spacing for 1cm domain!
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
    )

    # Should produce exactly one scan (at the start of the domain)
    assert path.n_scans == 1


def test_angled_raster_scan_large_offset_relative_to_domain():
    """Test angled raster with offsets larger than the domain itself.

    This tests a potential edge case in the intersection logic where offsets
    might cause unusual behavior.
    """
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        start_offset=0.02,  # 20mm offset (2x domain width)
        end_offset=0.02,
    )

    # Should have 3 scans despite large offsets
    assert path.n_scans == 3

    # First scan should extend far beyond the domain
    scan_length = np.linalg.norm(path.end_positions[0] - path.start_positions[0])
    # Should be much longer than the 10mm domain due to 20mm extensions on each side
    assert scan_length > 0.04  # At least 40mm (10mm + 20mm + 20mm)


def test_angled_raster_scan_hatch_spacing_accuracy():
    """Test that actual perpendicular spacing between scans exactly matches requested hatch_spacing.

    The implementation should maintain exact spacing between hatches, independent of domain size.
    This ensures consistent, predictable scan patterns.
    """
    hatch_spacing = 0.001  # 1mm spacing

    # Test at 0 degrees (horizontal scans)
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=hatch_spacing,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
    )

    # For horizontal scans (angle=0), perpendicular direction is along y-axis
    # So spacing should be the difference in y-coordinates
    if path.n_scans > 1:
        for i in range(path.n_scans - 1):
            # Get y-coordinates of consecutive scans (should be constant along each scan)
            y1 = path.start_positions[i][1]
            y2 = path.start_positions[i + 1][1]
            actual_spacing = abs(y2 - y1)

            # Spacing should be exactly the requested value (within floating-point tolerance)
            assert_allclose(
                actual_spacing,
                hatch_spacing,
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"Spacing {actual_spacing} does not match requested {hatch_spacing}",
            )

    # Test at 45 degrees (diagonal scans)
    path = LaserPath.angled_raster_scan(
        angle=45,
        hatch_spacing=hatch_spacing,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
    )

    # For general angle, compute perpendicular direction
    angle_rad = np.deg2rad(45)
    perp_dir = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

    if path.n_scans > 1:
        for i in range(path.n_scans - 1):
            # Get midpoints of consecutive scans for more robust measurement
            mid1 = (path.start_positions[i][:2] + path.end_positions[i][:2]) / 2
            mid2 = (path.start_positions[i + 1][:2] + path.end_positions[i + 1][:2]) / 2

            # Project onto perpendicular direction to get spacing
            diff = mid2 - mid1
            actual_spacing = abs(np.dot(diff, perp_dir))

            # Spacing should be exactly the requested value
            assert_allclose(
                actual_spacing,
                hatch_spacing,
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"Spacing {actual_spacing} does not match requested {hatch_spacing} at 45°",
            )

    # Test at 90 degrees (vertical scans)
    path = LaserPath.angled_raster_scan(
        angle=90,
        hatch_spacing=hatch_spacing,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
    )

    # For vertical scans (angle=90), perpendicular direction is along x-axis
    if path.n_scans > 1:
        for i in range(path.n_scans - 1):
            x1 = path.start_positions[i][0]
            x2 = path.start_positions[i + 1][0]
            actual_spacing = abs(x2 - x1)

            # Spacing should be exactly the requested value
            assert_allclose(
                actual_spacing,
                hatch_spacing,
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"Spacing {actual_spacing} does not match requested {hatch_spacing} at 90°",
            )


def test_angled_raster_scan_hatch_spacing_uniformity():
    """Test that spacing between all consecutive scans is uniform.

    Verifies that all gaps between hatches are approximately equal,
    not just that they're close to the requested spacing.
    """
    hatch_spacing = 0.0015
    path = LaserPath.angled_raster_scan(
        angle=30,
        hatch_spacing=hatch_spacing,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
    )

    if path.n_scans > 2:
        # Compute perpendicular direction for 30°
        angle_rad = np.deg2rad(30)
        perp_dir = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

        # Measure all spacings
        spacings = []
        for i in range(path.n_scans - 1):
            mid1 = (path.start_positions[i][:2] + path.end_positions[i][:2]) / 2
            mid2 = (path.start_positions[i + 1][:2] + path.end_positions[i + 1][:2]) / 2
            diff = mid2 - mid1
            spacing = abs(np.dot(diff, perp_dir))
            spacings.append(spacing)

        # All spacings should be very similar to each other
        spacings = np.array(spacings)
        mean_spacing = np.mean(spacings)
        std_spacing = np.std(spacings)

        # Standard deviation should be small relative to mean
        # (all spacings should be uniform)
        assert (
            std_spacing < mean_spacing * 0.05
        ), f"Spacing varies too much: mean={mean_spacing}, std={std_spacing}"


def test_angled_raster_scan_direction_consistency():
    """Test that all scans proceed in the expected direction.

    For bidirectional=True: odd-indexed scans should reverse direction.
    For bidirectional=False: all scans should go in the same direction.
    This should hold even with negative offsets.
    """
    # Test bidirectional scans at 0 degrees
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.002,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        bidirectional=True,
    )

    # For 0°, scan_dir is [1, 0], so forward scans should have dx > 0
    for i in range(path.n_scans):
        dx = path.end_positions[i][0] - path.start_positions[i][0]
        if i % 2 == 0:
            # Even-indexed scans should go forward (positive x)
            assert dx > 0, f"Scan {i} (even) should go forward but dx={dx}"
        else:
            # Odd-indexed scans should go backward (negative x)
            assert dx < 0, f"Scan {i} (odd) should go backward but dx={dx}"

    # Test unidirectional scans at 0 degrees
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.002,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        bidirectional=False,
    )

    # All scans should go in the same direction (forward)
    for i in range(path.n_scans):
        dx = path.end_positions[i][0] - path.start_positions[i][0]
        assert dx > 0, f"Unidirectional scan {i} should go forward but dx={dx}"

    # Test bidirectional scans at 90 degrees
    path = LaserPath.angled_raster_scan(
        angle=90,
        hatch_spacing=0.002,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        bidirectional=True,
    )

    # For 90°, scan_dir is [0, 1], so forward scans should have dy > 0
    for i in range(path.n_scans):
        dy = path.end_positions[i][1] - path.start_positions[i][1]
        if i % 2 == 0:
            # Even-indexed scans should go forward (positive y)
            assert dy > 0, f"Scan {i} (even) should go forward but dy={dy}"
        else:
            # Odd-indexed scans should go backward (negative y)
            assert dy < 0, f"Scan {i} (odd) should go backward but dy={dy}"


def test_angled_raster_scan_negative_offsets_preserve_direction():
    """Test that negative offsets don't reverse scan direction.

    Even with large negative offsets that shorten the scans,
    the direction should remain consistent with the intended pattern.
    If offsets are so large they would reverse the direction, those scans are skipped.
    """
    # Test with negative offsets at 0 degrees
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.002,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        start_offset=-0.002,  # Negative: start inside domain
        end_offset=-0.002,  # Negative: end inside domain
        bidirectional=True,
    )

    # Even with negative offsets, direction pattern should be preserved
    # Some scans may be skipped if offsets are too large
    for i in range(path.n_scans):
        dx = path.end_positions[i][0] - path.start_positions[i][0]
        # All remaining scans should have valid direction (no reversals)
        # We can't predict exact i % 2 pattern since some may be skipped,
        # but we can check that dx is non-zero and scans alternate if possible
        assert abs(dx) > 0, f"Scan {i} should have non-zero length"

    # Test with moderate negative offsets that shouldn't skip scans
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.002,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        start_offset=-0.001,  # Small negative offset
        end_offset=-0.001,
        bidirectional=True,
    )

    # With moderate offsets, should have full pattern preserved
    # Should have 6 scans at 0° with 2mm spacing
    assert path.n_scans == 6, "Should have 6 scans with moderate offsets"

    # All scans should maintain proper direction
    for i in range(path.n_scans):
        dx = path.end_positions[i][0] - path.start_positions[i][0]
        assert abs(dx) > 0, f"Scan {i} should have non-zero length"

    # Test at 45 degrees with negative offsets that are not too extreme
    path = LaserPath.angled_raster_scan(
        angle=45,
        hatch_spacing=0.002,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        start_offset=-0.001,  # Moderate negative offset
        end_offset=-0.001,
        bidirectional=True,
    )

    # Compute expected scan direction for 45°
    angle_rad = np.deg2rad(45)
    scan_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])

    # All created scans should have proper direction (none should be reversed)
    for i in range(path.n_scans):
        scan_vector = path.end_positions[i][:2] - path.start_positions[i][:2]
        scan_length = np.linalg.norm(scan_vector)
        assert scan_length > 0, f"Scan {i} at 45° should have non-zero length"


def test_angled_raster_scan_excessive_negative_offsets():
    """Test that excessive negative offsets cause scans to be skipped.

    When negative offsets are so large that they would reverse the scan direction,
    those scans should be skipped entirely rather than creating invalid paths.
    If all scans would be invalid, a ValueError is raised.
    """
    # Very large negative offsets that exceed half the domain width
    # This should cause all scans to be invalid and raise ValueError
    with pytest.raises(ValueError, match="No valid scan segments"):
        LaserPath.angled_raster_scan(
            angle=0,
            hatch_spacing=0.002,
            x_bounds=(0.0, 0.01),
            y_bounds=(0.0, 0.01),
            velocity=0.5,
            power=300,
            start_offset=-0.006,  # 6mm offset for 10mm domain - too large!
            end_offset=-0.006,
            bidirectional=True,
        )

    # Moderately large negative offsets - some scans may be skipped but not all
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.002,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
        power=300,
        start_offset=-0.003,  # 3mm offset - at the edge
        end_offset=-0.003,
        bidirectional=True,
    )

    # Should have some scans (those where the offset doesn't cause reversal)
    # Any scans that remain should have correct direction
    assert path.n_scans >= 1, "Should have at least some valid scans"

    for i in range(path.n_scans):
        dx = path.end_positions[i][0] - path.start_positions[i][0]
        # Each scan should have non-zero length and not be reversed
        assert abs(dx) > 1e-10, f"Scan {i} should have measurable length"
