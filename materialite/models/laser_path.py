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

    def __init__(self, start_positions, end_positions, start_times, end_times, powers):
        self.start_positions = np.atleast_2d(np.array(start_positions))
        self.end_positions = np.atleast_2d(np.array(end_positions))
        self.start_times = np.atleast_1d(np.array(start_times))
        self.end_times = np.atleast_1d(np.array(end_times))
        self.powers = np.atleast_1d(np.array(powers))

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
        if len(self.start_times) != self.n_scans:
            raise ValueError(
                f"start_times must have length {self.n_scans}, got {len(self.start_times)}"
            )
        if len(self.end_times) != self.n_scans:
            raise ValueError(
                f"end_times must have length {self.n_scans}, got {len(self.end_times)}"
            )
        if len(self.powers) != self.n_scans:
            raise ValueError(
                f"powers must have length {self.n_scans}, got {len(self.powers)}"
            )
        if np.any(self.end_times <= self.start_times):
            raise ValueError("end_times must be greater than start_times for all scans")
        if np.any(self.start_times[1:] < self.end_times[:-1]):
            raise ValueError(
                "scans must be sequential (i.e., start time of a scan should be after "
                "end time of a previous scan)"
            )

    def get_position(self, time):
        """
        Get laser position at given time(s).

        Parameters
        ----------
        time : float or array_like
            The time(s) at which to compute the laser position.

        Returns
        -------
        ndarray
            If time is scalar:
                The [x, y, z] position of the laser at the given time. Returns [nan, nan, nan]
                if the laser is not active at the given time.
            If time is array_like:
                Array of shape (n_times, 3) containing positions. For times when the
                laser is not active, the position will be [nan, nan, nan].
        """
        time_array = np.atleast_1d(np.asarray(time))
        is_scalar = np.ndim(time) == 0

        # Find which scan segment each time belongs to
        indices = np.searchsorted(self.start_times, time_array, side="right") - 1

        # Check validity: index in range AND time within scan window
        valid = np.zeros(len(time_array), dtype=bool)
        valid_idx_mask = (indices >= 0) & (indices < self.n_scans)

        # Check if times at valid indices are within scan windows
        if np.any(valid_idx_mask):
            valid_idx = indices[valid_idx_mask]
            valid_times = time_array[valid_idx_mask]

            time_in_window = (valid_times >= self.start_times[valid_idx]) & (
                valid_times <= self.end_times[valid_idx]
            )

            valid[valid_idx_mask] = time_in_window

        positions = np.full((len(time_array), 3), np.nan)

        # Interpolate to find positions at valid times
        if np.any(valid):
            valid_indices = indices[valid]
            valid_times = time_array[valid]

            t0 = self.start_times[valid_indices]
            t1 = self.end_times[valid_indices]
            fractions = (valid_times - t0) / (t1 - t0)

            p0 = self.start_positions[valid_indices]
            p1 = self.end_positions[valid_indices]
            positions[valid] = p0 + fractions[:, np.newaxis] * (p1 - p0)

        if is_scalar:
            return positions[0]
        else:
            return positions

    def get_scan_indices(self, time):
        """
        Get the scan segment index/indices at given time(s).

        This method returns the index of the active scan segment for each
        given time. For times when the laser is not active (before the first
        scan, after the last scan, or between scans), it returns -1.

        Parameters
        ----------
        time : float or array_like
            The time(s) at which to determine the active scan segment.

        Returns
        -------
        int or ndarray
            If time is scalar:
                The scan segment index (0 to n_scans-1), or -1 if laser is
                not active at the given time.
            If time is array_like:
                Array of shape (n_times,) containing scan indices. Returns -1
                for times when the laser is not active.

        Examples
        --------
        >>> path = LaserPath.single_line_scan(
        ...     start=[0, 0, 0], end=[0.01, 0, 0],
        ...     velocity=0.5, power=100
        ... )
        >>> # Single time
        >>> idx = path.get_scan_indices(0.01)
        >>> # Multiple times
        >>> indices = path.get_scan_indices([0.0, 0.01, 0.1])
        >>> # Access scan properties
        >>> idx = path.get_scan_indices(0.01)
        >>> if idx >= 0:
        ...     power = path.powers[idx]
        ...     start = path.start_positions[idx]
        ...     end = path.end_positions[idx]
        """
        time_array = np.atleast_1d(np.asarray(time))
        is_scalar = np.ndim(time) == 0

        # Find which scan segment each time belongs to
        indices = np.searchsorted(self.start_times, time_array, side="right") - 1

        # Check validity: index in range AND time within scan window
        valid_idx_mask = (indices >= 0) & (indices < self.n_scans)
        scan_indices = np.full(len(time_array), -1, dtype=np.int32)

        # Check if times at valid indices are within scan windows
        if np.any(valid_idx_mask):
            valid_idx = indices[valid_idx_mask]
            valid_times = time_array[valid_idx_mask]

            time_in_window = (valid_times >= self.start_times[valid_idx]) & (
                valid_times <= self.end_times[valid_idx]
            )

            scan_indices[valid_idx_mask] = np.where(time_in_window, valid_idx, -1)

        if is_scalar:
            return int(scan_indices[0])
        else:
            return scan_indices

    @property
    def total_time(self):
        """Total time spanned by all scan segments."""
        return np.max(self.end_times)

    @classmethod
    def from_list(cls, paths):
        """
        Combine multiple LaserPath objects into a single LaserPath.

        This method concatenates multiple laser path objects by merging their
        scan segments, start positions, end positions, and time arrays. The
        resulting LaserPath contains all scan segments from all input paths
        in the order they appear in the input list.

        Parameters
        ----------
        paths : list of LaserPath
            List of LaserPath objects to combine. Must contain at least one
            LaserPath object.

        Returns
        -------
        LaserPath
            A new LaserPath object containing all scan segments from the
            input paths.

        Raises
        ------
        IndexError
            If the paths list is empty.

        Examples
        --------
        Combine two horizontal scans at different heights:

        >>> path1 = LaserPath.single_line_scan(
        ...     start=[0, 0, 0],
        ...     end=[0.01, 0, 0],
        ...     velocity=0.5
        ... )
        >>> path2 = LaserPath.single_line_scan(
        ...     start=[0, 0, 0.001],
        ...     end=[0.01, 0, 0.001],
        ...     velocity=0.5,
        ...     start_time=0.02
        ... )
        >>> combined_path = LaserPath.from_list([path1, path2])
        >>> print(combined_path.n_scans)
        2

        Combine multiple raster scans with different angles:

        >>> path1 = LaserPath.angled_raster_scan(
        ...     angle=0, hatch_spacing=0.001,
        ...     x_bounds=(0, 0.01), y_bounds=(0, 0.01),
        ...     velocity=0.5
        ... )
        >>> path2 = LaserPath.angled_raster_scan(
        ...     angle=90, hatch_spacing=0.001,
        ...     x_bounds=(0, 0.01), y_bounds=(0, 0.01),
        ...     velocity=0.5,
        ...     start_time=path1.total_time
        ... )
        >>> combined_path = LaserPath.from_list([path1, path2])
        """
        start_positions = paths[0].start_positions
        end_positions = paths[0].end_positions
        start_times = paths[0].start_times
        end_times = paths[0].end_times
        powers = paths[0].powers
        for p in paths[1:]:
            start_positions = np.concatenate(
                [start_positions, p.start_positions], axis=0
            )
            end_positions = np.concatenate([end_positions, p.end_positions], axis=0)
            start_times = np.concatenate([start_times, p.start_times], axis=0)
            end_times = np.concatenate([end_times, p.end_times], axis=0)
            powers = np.concatenate([powers, p.powers], axis=0)
        return cls(start_positions, end_positions, start_times, end_times, powers)

    @classmethod
    def single_line_scan(cls, start, end, velocity, power, start_time=0.0):
        """
        Create a single linear scan segment.

        Parameters
        ----------
        start : array_like
            Starting position [x, y, z].
        end : array_like
            Ending position [x, y, z].
        velocity : float
            Laser velocity.
        start_time : float, default 0.0
            Time at which the scan begins.

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
            powers=[power],
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
            - "velocity": scanning velocity
            - "power": laser power
            - "time_delay": delay between the end of the previous segment and the
              start of the current scan
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
        powers = []

        current_time = start_time

        for seg in segments:
            start = np.array(seg["start"])
            end = np.array(seg["end"])
            velocity = seg["velocity"]
            current_time += seg.get("delay", 0.0)

            distance = np.linalg.norm(end - start)
            duration = distance / velocity

            start_positions.append(start)
            end_positions.append(end)
            start_times.append(current_time)
            end_times.append(current_time + duration)
            powers.append(seg["power"])

            current_time += duration

        return cls(start_positions, end_positions, start_times, end_times, powers)

    @classmethod
    def raster_scan(
        cls,
        x_start,
        x_end,
        y_start,
        y_end,
        num_passes,
        velocity,
        power,
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
            Laser velocity.
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

            segments.append(
                {"start": start, "end": end, "velocity": velocity, "power": power}
            )

        return cls.from_segments(segments, start_time=start_time)

    @classmethod
    def angled_raster_scan(
        cls,
        angle,
        hatch_spacing,
        x_bounds,
        y_bounds,
        velocity,
        power,
        z_height=0.0,
        start_time=0.0,
        bidirectional=True,
        start_offset=0.0,
        end_offset=0.0,
        domain_offset=0.0,
        delay_between_scans=None,
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
            Perpendicular distance between parallel scan lines.
        x_bounds : tuple of float
            (x_min, x_max) bounds of the rectangular domain.
        y_bounds : tuple of float
            (y_min, y_max) bounds of the rectangular domain.
        velocity : float
            Laser scanning velocity.
        z_height : float, default 0.0
            Height (z-coordinate) at which all scans occur.
        start_time : float, default 0.0
            Time at which the first scan begins.
        bidirectional : bool, default True
            If True, alternates scan direction for consecutive hatches (typical
            raster pattern). If False, all scans proceed in the same direction.
        start_offset : float, default 0.0
            Distance to extend the start of each scan beyond the domain boundary.
            Positive values extend the scan backward (before the
            intersection), negative values shorten it (start inside domain).
        end_offset : float, default 0.0
            Distance to extend the end of each scan beyond the domain boundary.
            Positive values extend the scan forward (beyond the
            intersection), negative values shorten it (end inside domain).
        domain_offset : float, default 0.0
            Distance from the provided bounds to limit the scans to. Positive
            values extend the scan outside the bounds, negative values
            shorten scans to inside the bounds.
        delay_between_scans : float or None, default None
            Delay between consecutive scans. If None, the delay is
            automatically calculated based on the time required to move the laser
            from the end of one scan to the start of the next scan at the given
            velocity. If specified, the same delay is used between all scans.

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
        x_bound_low, x_bound_high = x_bounds
        y_bound_low, y_bound_high = y_bounds

        x_min = x_bound_low - domain_offset
        x_max = x_bound_high + domain_offset
        y_min = y_bound_low - domain_offset
        y_max = y_bound_high + domain_offset

        angle_rad = np.deg2rad(angle)
        scan_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        perp_dir = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

        corners = np.array(
            [
                [x_bound_low, y_bound_low],
                [x_bound_high, y_bound_low],
                [x_bound_high, y_bound_high],
                [x_bound_low, y_bound_high],
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

            # Sort by parameter t (time) to get the correct order
            intersections.sort(key=lambda pt: pt[2])

            # Remove duplicate intersections (can happen at corners)
            # Keep only intersections that are sufficiently distinct
            unique_intersections = [intersections[0]]
            for pt in intersections[1:]:
                last_pt = unique_intersections[-1]
                dist = np.sqrt((pt[0] - last_pt[0]) ** 2 + (pt[1] - last_pt[1]) ** 2)
                if dist > 1e-9:
                    unique_intersections.append(pt)

            # Only proceed if we have at least 2 distinct intersections
            if len(unique_intersections) < 2:
                continue

            start_pt = unique_intersections[0]
            end_pt = unique_intersections[-1]

            # Determine scan direction for this hatch and apply offsets
            if bidirectional and i % 2 == 1:
                start_pos = np.array([end_pt[0], end_pt[1]]) + start_offset * scan_dir
                end_pos = np.array([start_pt[0], start_pt[1]]) - end_offset * scan_dir

                scan_vector = start_pos - end_pos

            else:
                start_pos = (
                    np.array([start_pt[0], start_pt[1]]) - start_offset * scan_dir
                )
                end_pos = np.array([end_pt[0], end_pt[1]]) + end_offset * scan_dir

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
                # Calculate delay from previous segment
                if len(segments) > 0:
                    if delay_between_scans is not None:
                        delay = delay_between_scans
                    else:
                        prev_end = np.array(segments[-1]["end"])
                        curr_start = np.array(start)
                        travel_distance = np.linalg.norm(curr_start - prev_end)
                        delay = travel_distance / velocity
                else:
                    delay = 0.0

                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "velocity": velocity,
                        "power": power,
                        "delay": delay,
                    }
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
