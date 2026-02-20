# Copyright 2025 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.
#
# The Materialite platform is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np


class LaserPath:
    """
    Define laser scanning paths for welding/additive manufacturing simulations.

    This class manages one or more laser scan segments, each defined by start and
    end positions in 3D space and the times at which the scan begins and ends. It
    provides methods to compute laser position at any given time during the simulation.

    Parameters
    ----------
    start_positions : array_like
        Array of shape (n_scans, 3) containing [x, y, z] coordinates for the
        start of each scan segment.
    end_positions : array_like
        Array of shape (n_scans, 3) containing [x, y, z] coordinates for the
        end of each scan segment.
    start_times : array_like
        Array of shape (n_scans,) containing the start time for each scan segment.
    end_times : array_like
        Array of shape (n_scans,) containing the end time for each scan segment.

    Attributes
    ----------
    n_scans : int
        Number of scan segments.
    start_positions : ndarray
        Start positions of each scan segment, shape (n_scans, 3).
    end_positions : ndarray
        End positions of each scan segment, shape (n_scans, 3).
    start_times : ndarray
        Start times of each scan segment, shape (n_scans,).
    end_times : ndarray
        End times of each scan segment, shape (n_scans,).

    Examples
    --------
    Single linear scan in the x-direction:

    >>> path = LaserPath.single_line_scan(
    ...     start=[0, 0.005, 0.005],
    ...     end=[0.01, 0.005, 0.005],
    ...     velocity=0.5,
    ...     start_time=0.0
    ... )

    Multiple scans in different directions:

    >>> path = LaserPath.from_segments([
    ...     {"start": [0, 0, 0], "end": [0.01, 0, 0], "velocity": 0.5},
    ...     {"start": [0.01, 0.001, 0], "end": [0, 0.001, 0], "velocity": 0.5},
    ...     {"start": [0, 0.002, 0], "end": [0.01, 0.002, 0], "velocity": 0.5},
    ... ])

    Raster scan pattern:

    >>> path = LaserPath.raster_scan(
    ...     x_start=0, x_end=0.01, y_start=0, y_end=0.01,
    ...     num_passes=10, velocity=0.5, z_height=0.0
    ... )
    """

    def __init__(self, start_positions, end_positions, start_times, end_times):
        self.start_positions = np.atleast_2d(np.array(start_positions))
        self.end_positions = np.atleast_2d(np.array(end_positions))
        self.start_times = np.atleast_1d(np.array(start_times))
        self.end_times = np.atleast_1d(np.array(end_times))

        self.n_scans = len(self.start_times)

        # Validation
        if self.start_positions.shape != (self.n_scans, 3):
            raise ValueError(
                f"start_positions must have shape ({self.n_scans}, 3), "
                f"got {self.start_positions.shape}"
            )
        if self.end_positions.shape != (self.n_scans, 3):
            raise ValueError(
                f"end_positions must have shape ({self.n_scans}, 3), "
                f"got {self.end_positions.shape}"
            )
        if len(self.end_times) != self.n_scans:
            raise ValueError(
                f"end_times must have length {self.n_scans}, got {len(self.end_times)}"
            )
        if np.any(self.end_times <= self.start_times):
            raise ValueError("end_times must be greater than start_times for all scans")

    def get_position(self, time):
        """
        Get laser position at a given time.

        Parameters
        ----------
        time : float
            The time at which to compute the laser position.

        Returns
        -------
        ndarray
            The [x, y, z] position of the laser at the given time. Returns None
            if the laser is not active at the given time.
        """
        active_scan = None
        idx = np.searchsorted(self.start_times, time, side="right") - 1
        if 0 <= idx < self.n_scans and self.start_times[idx] <= time <= self.end_times[idx]:
            active_scan = idx
        if active_scan is None:
            return None

        # Interpolate position within the active scan
        t0 = self.start_times[active_scan]
        t1 = self.end_times[active_scan]
        p0 = self.start_positions[active_scan]
        p1 = self.end_positions[active_scan]

        fraction = (time - t0) / (t1 - t0)
        position = p0 + fraction * (p1 - p0)

        return position

    @property
    def total_time(self):
        """Total time spanned by all scan segments."""
        return np.max(self.end_times)

    @classmethod
    def single_line_scan(cls, start, end, velocity, start_time=0.0):
        """
        Create a single linear scan segment.

        Parameters
        ----------
        start : array_like
            Starting position [x, y, z] in meters.
        end : array_like
            Ending position [x, y, z] in meters.
        velocity : float
            Laser velocity in m/s.
        start_time : float, default 0.0
            Time at which the scan begins in seconds.

        Returns
        -------
        LaserPath
            A LaserPath object with a single scan segment.
        """
        start = np.array(start)
        end = np.array(end)
        distance = np.linalg.norm(end - start)
        duration = distance / velocity
        end_time = start_time + duration

        return cls(
            start_positions=[start],
            end_positions=[end],
            start_times=[start_time],
            end_times=[end_time],
        )

    @classmethod
    def from_segments(cls, segments, start_time=0.0):
        """
        Create a laser path from a list of scan segment specifications.

        Each segment should be consecutive, with the end of one being close to
        the start of the next (or with a gap representing laser off time).

        Parameters
        ----------
        segments : list of dict
            List of segment specifications. Each dict should contain:
            - "start": [x, y, z] start position
            - "end": [x, y, z] end position
            - "velocity": scanning velocity in m/s
        start_time : float, default 0.0
            Time at which the first scan begins.

        Returns
        -------
        LaserPath
            A LaserPath object with multiple scan segments.

        Examples
        --------
        >>> segments = [
        ...     {"start": [0, 0, 0], "end": [0.01, 0, 0], "velocity": 0.5},
        ...     {"start": [0.01, 0.001, 0], "end": [0, 0.001, 0], "velocity": 0.5},
        ... ]
        >>> path = LaserPath.from_segments(segments)
        """
        start_positions = []
        end_positions = []
        start_times = []
        end_times = []

        current_time = start_time

        for seg in segments:
            start = np.array(seg["start"])
            end = np.array(seg["end"])
            velocity = seg["velocity"]

            distance = np.linalg.norm(end - start)
            duration = distance / velocity

            start_positions.append(start)
            end_positions.append(end)
            start_times.append(current_time)
            end_times.append(current_time + duration)

            current_time += duration

        return cls(start_positions, end_positions, start_times, end_times)

    @classmethod
    def raster_scan(
        cls,
        x_start,
        x_end,
        y_start,
        y_end,
        num_passes,
        velocity,
        z_height=0.0,
        start_time=0.0,
        bidirectional=True,
    ):
        """
        Create a raster (back-and-forth) scan pattern in the x-y plane.

        Parameters
        ----------
        x_start : float
            Starting x coordinate.
        x_end : float
            Ending x coordinate.
        y_start : float
            Starting y coordinate.
        y_end : float
            Ending y coordinate.
        num_passes : int
            Number of scan passes (lines) in the y direction.
        velocity : float
            Laser velocity in m/s.
        z_height : float, default 0.0
            Height (z coordinate) at which scanning occurs.
        start_time : float, default 0.0
            Time at which scanning begins.
        bidirectional : bool, default True
            If True, alternates scan direction (typical raster pattern).
            If False, always scans in the same x direction.

        Returns
        -------
        LaserPath
            A LaserPath object with raster scan pattern.

        Examples
        --------
        >>> path = LaserPath.raster_scan(
        ...     x_start=0, x_end=0.01, y_start=0, y_end=0.01,
        ...     num_passes=10, velocity=0.5
        ... )
        """
        segments = []
        y_positions = np.linspace(y_start, y_end, num_passes)

        for i, y in enumerate(y_positions):
            if bidirectional and i % 2 == 1:
                # Scan in reverse direction
                start = [x_end, y, z_height]
                end = [x_start, y, z_height]
            else:
                # Scan in forward direction
                start = [x_start, y, z_height]
                end = [x_end, y, z_height]

            segments.append({"start": start, "end": end, "velocity": velocity})

        return cls.from_segments(segments, start_time=start_time)

    @classmethod
    def angled_raster_scan(
        cls,
        angle,
        hatch_spacing,
        x_bounds,
        y_bounds,
        velocity,
        z_height=0.0,
        start_time=0.0,
        bidirectional=True,
        start_offset=0.0,
        end_offset=0.0,
    ):
        """
        Create a raster scan pattern at an arbitrary angle in the x-y plane.

        This method generates parallel scan lines (hatches) rotated by a specified
        angle, with each scan starting and ending where it intersects the
        rectangular domain defined by x_bounds and y_bounds. Optional offsets
        extend scans beyond (or shorten them within) the domain boundaries.

        Parameters
        ----------
        angle : float
            Angle of scan direction in degrees. 0° scans along the x-axis,
            90° scans along the y-axis. Positive angles rotate counterclockwise.
        hatch_spacing : float
            Perpendicular distance between parallel scan lines in meters.
        x_bounds : tuple of float
            (x_min, x_max) bounds of the rectangular domain in meters.
        y_bounds : tuple of float
            (y_min, y_max) bounds of the rectangular domain in meters.
        velocity : float
            Laser scanning velocity in m/s.
        z_height : float, default 0.0
            Height (z-coordinate) at which all scans occur in meters.
        start_time : float, default 0.0
            Time at which the first scan begins in seconds.
        bidirectional : bool, default True
            If True, alternates scan direction for consecutive hatches (typical
            raster pattern). If False, all scans proceed in the same direction.
        start_offset : float, default 0.0
            Distance to extend the start of each scan beyond the domain boundary
            in meters. Positive values extend the scan backward (before the
            intersection), negative values shorten it (start inside domain).
        end_offset : float, default 0.0
            Distance to extend the end of each scan beyond the domain boundary
            in meters. Positive values extend the scan forward (beyond the
            intersection), negative values shorten it (end inside domain).

        Returns
        -------
        LaserPath
            A LaserPath object containing the angled raster scan pattern.

        Examples
        --------
        >>> # Horizontal raster scan (0°) across a 10mm x 10mm domain
        >>> path = LaserPath.angled_raster_scan(
        ...     angle=0,
        ...     hatch_spacing=0.001,
        ...     x_bounds=(0.0, 0.01),
        ...     y_bounds=(0.0, 0.01),
        ...     velocity=0.5,
        ... )

        >>> # 45° diagonal scan with offsets extending beyond domain
        >>> path = LaserPath.angled_raster_scan(
        ...     angle=45,
        ...     hatch_spacing=0.0005,
        ...     x_bounds=(0.0, 0.01),
        ...     y_bounds=(0.0, 0.01),
        ...     velocity=0.5,
        ...     start_offset=0.001,  # Start 1mm before boundary
        ...     end_offset=0.001,    # End 1mm after boundary
        ... )

        >>> # Unidirectional vertical scan (90°)
        >>> path = LaserPath.angled_raster_scan(
        ...     angle=90,
        ...     hatch_spacing=0.001,
        ...     x_bounds=(0.0, 0.01),
        ...     y_bounds=(0.0, 0.01),
        ...     velocity=0.5,
        ...     bidirectional=False,
        ... )
        """
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        angle_rad = np.deg2rad(angle)
        scan_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        perp_dir = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

        corners = np.array(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ]
        )
        projections = corners @ perp_dir
        perp_min = np.min(projections)
        perp_max = np.max(projections)
        hatch_offsets = np.arange(perp_min, perp_max + hatch_spacing / 2, hatch_spacing)

        segments = []
        for i, offset in enumerate(hatch_offsets):
            reference_point = offset * perp_dir

            intersections = []

            # Edge 1: y = y_min (bottom edge)
            if abs(scan_dir[1]) > 1e-10:
                t = (y_min - reference_point[1]) / scan_dir[1]
                x = reference_point[0] + t * scan_dir[0]
                if x_min <= x <= x_max:
                    intersections.append((x, y_min, t))

            # Edge 2: y = y_max (top edge)
            if abs(scan_dir[1]) > 1e-10:
                t = (y_max - reference_point[1]) / scan_dir[1]
                x = reference_point[0] + t * scan_dir[0]
                if x_min <= x <= x_max:
                    intersections.append((x, y_max, t))

            # Edge 3: x = x_min (left edge)
            if abs(scan_dir[0]) > 1e-10:
                t = (x_min - reference_point[0]) / scan_dir[0]
                y = reference_point[1] + t * scan_dir[1]
                if y_min <= y <= y_max:
                    intersections.append((x_min, y, t))

            # Edge 4: x = x_max (right edge)
            if abs(scan_dir[0]) > 1e-10:
                t = (x_max - reference_point[0]) / scan_dir[0]
                y = reference_point[1] + t * scan_dir[1]
                if y_min <= y <= y_max:
                    intersections.append((x_max, y, t))

            # We should have exactly 2 intersections (entry and exit)
            num_intersections = len(intersections)
            if num_intersections < 2:
                continue
            elif num_intersections > 2:
                raise ValueError("more than two intersections")

            # Sort by parameter t to get the correct order
            intersections.sort(key=lambda pt: pt[2])

            # Remove duplicate intersections (can happen at corners)
            # Keep only intersections that are sufficiently distinct
            unique_intersections = [intersections[0]]
            for pt in intersections[1:]:
                # Check if this point is distinct from the last unique point
                last_pt = unique_intersections[-1]
                dist = np.sqrt(
                    (pt[0] - last_pt[0]) ** 2 + (pt[1] - last_pt[1]) ** 2
                )
                if dist > 1e-9:  # Tolerance for distinct points
                    unique_intersections.append(pt)

            # Only proceed if we have at least 2 distinct intersections
            if len(unique_intersections) < 2:
                continue

            start_pt = unique_intersections[0]
            end_pt = unique_intersections[-1]

            # Determine scan direction for this hatch and apply offsets
            if bidirectional and i % 2 == 1:
                start_pos = (
                    np.array([end_pt[0], end_pt[1]]) + start_offset * scan_dir
                )
                end_pos = (
                    np.array([start_pt[0], start_pt[1]]) - end_offset * scan_dir
                )

                scan_vector = start_pos - end_pos

            else:
                start_pos = (
                    np.array([start_pt[0], start_pt[1]])
                    - start_offset * scan_dir
                )
                end_pos = (
                    np.array([end_pt[0], end_pt[1]]) + end_offset * scan_dir
                )

                scan_vector = end_pos - start_pos

            expected_projection = np.dot(scan_vector, scan_dir)
            if expected_projection < 0:
                # Points are in wrong order due to excessive negative offsets
                # Skip this scan entirely
                continue

            start = [start_pos[0], start_pos[1], z_height]
            end = [end_pos[0], end_pos[1], z_height]

            # Only add segment if it has non-zero length
            seg_length = np.linalg.norm(np.array(end) - np.array(start))
            if seg_length > 1e-10:
                segments.append(
                    {"start": start, "end": end, "velocity": velocity}
                )

        if len(segments) == 0:
            raise ValueError(
                f"No valid scan segments could be generated. This can occur when "
                f"hatch_spacing ({hatch_spacing}) is too large relative to the "
                f"domain size (x: {x_bounds}, y: {y_bounds}), or when the domain "
                f"is degenerate. Consider reducing hatch_spacing or increasing the "
                f"domain size."
            )

        return cls.from_segments(segments, start_time=start_time)

    def __repr__(self):
        return (
            f"LaserPath(n_scans={self.n_scans}, " f"total_time={self.total_time:.6f}s)"
        )
