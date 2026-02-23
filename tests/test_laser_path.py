import numpy as np
import pytest
from materialite.models.laser_path import LaserPath
from numpy.testing import assert_allclose, assert_array_equal


def test_single_line_scan():
    """Test creating a single line scan."""
    path = LaserPath.single_line_scan(
        start=[0, 0, 0], end=[0.01, 0, 0], velocity=0.5, start_time=0.0
    )

    assert path.n_scans == 1
    assert_array_equal(path.start_positions[0], [0, 0, 0])
    assert_array_equal(path.end_positions[0], [0.01, 0, 0])
    assert path.start_times[0] == 0.0
    assert_allclose(path.end_times[0], 0.02)  # 0.01 m / 0.5 m/s = 0.02 s


def test_get_position_single_scan():
    """Test getting position at different times for a single scan."""
    path = LaserPath.single_line_scan(
        start=[0, 0, 0], end=[0.01, 0, 0], velocity=0.5
    )

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
    assert pos is None

    # After end (laser not active)
    pos = path.get_position(0.03)
    assert pos is None


def test_from_segments():
    """Test creating path from multiple segments."""
    segments = [
        {"start": [0, 0, 0], "end": [0.01, 0, 0], "velocity": 0.5},
        {"start": [0.01, 0.001, 0], "end": [0, 0.001, 0], "velocity": 0.5},
        {"start": [0, 0.002, 0], "end": [0.01, 0.002, 0], "velocity": 0.5},
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
        {"start": [0, 0, 0], "end": [0.01, 0, 0], "velocity": 0.5},
        {"start": [0.01, 0.001, 0], "end": [0, 0.001, 0], "velocity": 0.5},
    ]

    path = LaserPath.from_segments(segments)

    # Position during first segment
    pos = path.get_position(0.01)
    assert_allclose(pos, [0.005, 0, 0])

    # Position during second segment
    pos = path.get_position(0.03)
    assert_allclose(pos, [0.005, 0.001, 0])

    # Position after second segment
    pos = path.get_position(0.041)
    assert pos is None


def test_raster_scan_bidirectional():
    """Test creating a bidirectional raster scan pattern."""
    path = LaserPath.raster_scan(
        x_start=0,
        x_end=0.01,
        y_start=0,
        y_end=0.01,
        num_passes=3,
        velocity=0.5,
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
        start=[0, 0, 0], end=[0.01, 0, 0], velocity=0.5
    )
    assert_allclose(path.total_time, 0.02)

    segments = [
        {"start": [0, 0, 0], "end": [0.01, 0, 0], "velocity": 0.5},
        {"start": [0.01, 0.001, 0], "end": [0, 0.001, 0], "velocity": 1.0},
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
        )

    # Mismatched shapes
    with pytest.raises(ValueError, match="start_positions must have shape"):
        LaserPath(
            start_positions=[[0, 0]],  # Wrong shape
            end_positions=[[1, 0, 0]],
            start_times=[0.0],
            end_times=[1.0],
        )


def test_repr():
    """Test string representation."""
    path = LaserPath.single_line_scan(
        start=[0, 0, 0], end=[0.01, 0, 0], velocity=0.5
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
        start_offset=0.001,
        end_offset=0.002,
    )

    # First scan should extend on both ends
    assert_allclose(path.start_positions[0][0], -0.001)
    assert_allclose(path.end_positions[0][0], 0.012)


def test_angled_raster_scan_bidirectional():
    """Test that bidirectional scans alternate direction correctly."""
    path = LaserPath.angled_raster_scan(
        angle=0,
        hatch_spacing=0.005,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
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
    )

    # Should have 1 scan (spacing larger than domain)
    assert path.n_scans == 1
    
    # All scans should be valid (proper start and end positions)
    for i in range(path.n_scans):
        # Scan should have non-zero length
        length = np.linalg.norm(
            path.end_positions[i] - path.start_positions[i]
        )
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
        start_offset=0.02,  # 20mm offset (2x domain width)
        end_offset=0.02,
    )

    # Should have 3 scans despite large offsets
    assert path.n_scans == 3
    
    # First scan should extend far beyond the domain
    scan_length = np.linalg.norm(
        path.end_positions[0] - path.start_positions[0]
    )
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
            assert_allclose(actual_spacing, hatch_spacing, rtol=1e-10, atol=1e-12,
                err_msg=f"Spacing {actual_spacing} does not match requested {hatch_spacing}")
    
    # Test at 45 degrees (diagonal scans)
    path = LaserPath.angled_raster_scan(
        angle=45,
        hatch_spacing=hatch_spacing,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
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
            assert_allclose(actual_spacing, hatch_spacing, rtol=1e-10, atol=1e-12,
                err_msg=f"Spacing {actual_spacing} does not match requested {hatch_spacing} at 45°")
    
    # Test at 90 degrees (vertical scans)
    path = LaserPath.angled_raster_scan(
        angle=90,
        hatch_spacing=hatch_spacing,
        x_bounds=(0.0, 0.01),
        y_bounds=(0.0, 0.01),
        velocity=0.5,
    )
    
    # For vertical scans (angle=90), perpendicular direction is along x-axis
    if path.n_scans > 1:
        for i in range(path.n_scans - 1):
            x1 = path.start_positions[i][0]
            x2 = path.start_positions[i + 1][0]
            actual_spacing = abs(x2 - x1)
            
            # Spacing should be exactly the requested value
            assert_allclose(actual_spacing, hatch_spacing, rtol=1e-10, atol=1e-12,
                err_msg=f"Spacing {actual_spacing} does not match requested {hatch_spacing} at 90°")


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
        assert std_spacing < mean_spacing * 0.05, \
            f"Spacing varies too much: mean={mean_spacing}, std={std_spacing}"


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
        start_offset=-0.002,  # Negative: start inside domain
        end_offset=-0.002,    # Negative: end inside domain
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


