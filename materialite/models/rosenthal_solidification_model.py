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
from numba import njit
from tqdm import trange

from scipy.spatial import KDTree

from materialite import Material
from materialite.models import Model
from materialite.models.laser_path import LaserPath
from materialite.tensor import Scalar, Vector, Orientation


@njit
def _potts_monte_carlo_flips(
    grain_ids, voxel_status, neighbors, mobility_values, selected_voxels
):
    """
    Numba-compiled function to perform Potts Monte Carlo flips.

    Parameters
    ----------
    grain_ids : ndarray
        Grain ID array (will be modified in-place)
    voxel_status : ndarray
        Voxel status array (0=unaffected, 1=mushy, 2=liquid, 3=solid)
    neighbors : ndarray
        Neighbor indices (shape: num_points x 27)
    mobility_values : ndarray
        Mobility values for each voxel
    selected_voxels : ndarray
        Indices of voxels selected for flip attempts

    Returns
    -------
    None
        grain_ids is modified in-place
    """
    for voxel_idx in selected_voxels:
        # Skip if voxel is not solid
        if voxel_status[voxel_idx] != 3:
            continue

        # Get valid neighbors for this voxel
        neighbor_list = neighbors[voxel_idx]
        valid_neighbors = neighbor_list[neighbor_list != -1]

        if len(valid_neighbors) == 0:
            continue

        # Randomly select one neighbor and get its grain ID
        selected_neighbor = valid_neighbors[np.random.randint(0, len(valid_neighbors))]
        proposed_grain_id = grain_ids[selected_neighbor]

        # Skip if neighbor has same grain ID (no change)
        current_grain_id = grain_ids[voxel_idx]
        if proposed_grain_id == current_grain_id:
            continue

        # Calculate current energy (number of dissimilar neighbors)
        current_energy = 0
        for neighbor_idx in valid_neighbors:
            if grain_ids[neighbor_idx] != current_grain_id:
                current_energy += 1

        # Calculate new energy if flip were to occur
        new_energy = 0
        for neighbor_idx in valid_neighbors:
            if grain_ids[neighbor_idx] != proposed_grain_id:
                new_energy += 1

        # Energy change
        delta_energy = new_energy - current_energy

        # Accept/reject decision
        if delta_energy > 0:
            # Energy increases: reject the flip
            continue
        else:
            # Energy decreases or stays same: accept with probability = mobility
            if np.random.rand() < mobility_values[voxel_idx]:
                grain_ids[voxel_idx] = proposed_grain_id
    return grain_ids


class RosenthalSolidificationModel(Model):
    """
    Temperature history model using the Rosenthal analytical solution.

    This model simulates the temperature evolution in a material during a moving
    heat source process (e.g., welding, laser scanning) using the Rosenthal
    equation for a point heat source. The model computes temperature fields at
    discrete time steps as a laser moves through the material domain.

    The Rosenthal equation provides an analytical solution for the temperature
    field around a moving point heat source in a semi-infinite medium, accounting
    for heat conduction and the velocity of the heat source.

    Parameters
    ----------
    initial_temperature : float, default 300
        Base temperature of the material in Kelvin.
    laser_power : float, default 37
        Laser power in Watts.
    laser_velocity : float, default 0.5
        Scanning velocity in meters per second.
    diffusivity : float, default 6e-6
        Material thermal diffusivity in m²/s.
    conductivity : float, default 10
        Material thermal conductivity in W/(m·K).
    absorptivity : float, default 0.5
        Laser absorption coefficient (dimensionless, 0-1).
    melt_temperature : float, default 1500
        Maximum temperature cap in Kelvin (typically the melt temperature).
    laser_path : LaserPath, optional
        LaserPath object defining the laser scanning trajectory. If not provided,
        a simple linear scan is created from laser_start_position, laser_velocity,
        and laser_path_direction (legacy parameters).
    laser_start_position : list of float, default [0.0, 0.0, 0.0]
        (Legacy) Initial position [x, y, z] of the laser in meters. Only used if
        laser_path is not provided.
    laser_velocity : float, default 0.5
        (Legacy) Scanning velocity in m/s. Only used if laser_path is not provided.
    laser_path_direction : str, default "x"
        (Legacy) Direction of laser movement ("x", "y", or "z"). Only used if
        laser_path is not provided.
    time_steps : int, optional
        Number of time steps to simulate. If not provided, automatically calculated
        from laser_path.total_time and time_step_duration.
    time_step_duration : float, default 0.001
        Duration of each time step in seconds.
    save_history : bool, default False
        Whether to save temperature at each time step. If True, the temperature
        field will be a Scalar with dims="pt" (shape: num_points x num_time_steps)
        containing the full history. If False, the temperature field will be a
        Scalar with dims="p" containing only the final temperature. Time stamps
        are stored in material.state["time_history"].

    Examples
    --------
    >>> from materialite import Material
    >>> from materialite.models import RosenthalSolidificationModel
    >>>
    >>> # Create a material
    >>> material = Material(dimensions=[50, 50, 50], spacing=[0.001, 0.001, 0.001])
    >>>
    >>> # Create and run the model
    >>> model = RosenthalSolidificationModel(
    ...     laser_power=50,
    ...     laser_velocity=0.5,
    ...     time_steps=100,
    ...     save_history=True
    ... )
    >>> new_material = model.run(material)
    >>>
    >>> # Access final temperature
    >>> final_temp = new_material.extract("temperature")
    >>>
    >>> # Access temperature
    >>> final_temp = new_material.extract("temperature")  # Scalar with dims="p" or "pt"
    >>>
    >>> # If save_history=True, access full history
    >>> if model.save_history:
    ...     temp_history = new_material.extract("temperature")  # Scalar with dims="pt"
    ...     time_history = new_material.state["time_history"]  # Array of times

    Notes
    -----
    The Rosenthal equation used here is:

    .. math::

        T(x, y, z, t) = T_0 + \\frac{\\eta Q}{2\\pi R k} \\exp\\left(\\frac{-v(x' + R)}{2\\alpha}\\right)

    where:
    - :math:`T_0` is the initial temperature
    - :math:`\\eta` is the power absorptivity
    - :math:`Q` is the laser power
    - :math:`R` is the distance from the laser position
    - :math:`k` is the thermal conductivity
    - :math:`v` is the laser velocity
    - :math:`\\alpha` is the thermal diffusivity
    - :math:`x'` is the position in the moving coordinate system

    The model uses Materialite's Scalar and Vector tensor objects for efficient
    computation and clear type semantics.
    """

    def __init__(
        self,
        laser_path: LaserPath,
        initial_temperature: float = 300,
        thermophysical_properties: dict = {
            "conductivity": 10.0,
            "diffusivity": 5.0e-6,
            "absorptivity": 0.5,
            "melt_temperature": 1500.0,
        },
        solidification_properties: dict = {"prefactor": 1.0e-5, "exponent": 2.0},
        potts_properties: dict = {"Q": 285000, "K_0": 0.2994, "K_MC": 0.27695},
        time_step_duration: float = 0.001,
        time_steps: int = None,
        output_times: list = None,
        smoothing_steps: int = 3,
        z_scale_factor: float = 0.6,
    ):
        self.initial_temperature = initial_temperature
        self.diffusivity = thermophysical_properties["diffusivity"]
        self.conductivity = thermophysical_properties["conductivity"]
        self.absorptivity = thermophysical_properties["absorptivity"]
        self.melt_temperature = thermophysical_properties["melt_temperature"]
        self.a = solidification_properties["prefactor"]
        self.b = solidification_properties["exponent"]
        self.time_step_duration = time_step_duration
        self.output_times = output_times
        self.laser_path = laser_path
        self.smoothing_steps = smoothing_steps
        self.potts_properties = potts_properties
        self.z_scale_factor = z_scale_factor
        # Determine time_steps if not provided
        if time_steps is None:
            self.time_steps = (
                int(np.ceil(self.laser_path.total_time / time_step_duration)) + 1
            )
        else:
            self.time_steps = time_steps

        if self.output_times is not None:
            end_time = self.laser_path.total_time
            if self.output_times[-1] < end_time - 1.0e-9:
                self.output_times = np.append(
                    self.output_times, self.laser_path.total_time
                )

    def run(
        self,
        material: Material,
        grain_id_label: str = "grain",
    ):
        """
        Run the Rosenthal temperature simulation.

        Parameters
        ----------
        material : Material
            The material object containing the spatial domain.
        grain_id_label : str, default "grain"
            Label for the grain ID field in the material.
        Returns
        -------
        Material
            New material with temperature field. If save_history is True, the
            temperature field is a Scalar with dims="pt" containing full history.
            If False, it's a Scalar with dims="p" containing final temperature.
            Time history array is stored in material.state["time_history"].
        """
        # Extract positions as a Vector (shape: num_points x 3)
        positions = Vector(material.extract(["x", "y", "z"]), dims="p")
        # Extract other fields from Material
        grain_ids = material.extract(
            grain_id_label
        )  # numpy array of shape (num_points,)

        # Get neighbors for solidification and Potts model calculations
        neighbors, num_neighbors, distances = _get_neighbors(material)
        capture_distances = np.zeros(material.num_points)

        # Initialize indices for liquid and mushy zone sites
        voxel_status = np.ones(material.num_points, dtype=np.int8) * 3

        time_array = np.arange(self.time_steps) * self.time_step_duration

        (
            laser_positions_array,
            scan_indices,
            scan_velocities,
            scan_directions,
            powers,
        ) = self._get_scan_information(time_array)

        if self.output_times is not None:
            temperature_history_array = np.zeros(
                (material.num_points, len(self.output_times)), dtype=np.float64
            )
            grain_history_array = np.zeros(
                (material.num_points, len(self.output_times)), dtype=np.float64
            )
        # Time loop
        old_scan_idx = -1
        output_idx = 0
        for time_step in trange(self.time_steps):
            scan_idx = scan_indices[time_step]

            if scan_idx < 0:
                # Laser is not active at this time - use ambient temperature
                temperature = Scalar(
                    self.initial_temperature * np.ones(material.num_points), dims="p"
                )
                active_mask = np.zeros(material.num_points, dtype=bool)
            else:
                laser_position = laser_positions_array[time_step]

                # Create mask for voxels at or below current laser z-position
                laser_z = laser_position.components[2]
                active_mask = positions.components[:, 2] <= laser_z + 1.0e-9

                if scan_idx != old_scan_idx:
                    scan_direction = scan_directions[scan_idx]
                    perp_direction = Vector([0, 0, 1]).cross(scan_direction)
                    scan_orientation = Orientation(
                        np.array(
                            [
                                scan_direction.components,
                                perp_direction.components,
                                [0, 0, 1],
                            ]
                        )
                    )
                    velocity = scan_velocities[scan_idx]
                    old_scan_idx = scan_idx

                # Calculate signed distance along travel direction
                displacement = (positions - laser_position).components
                displacement[:, -1] = displacement[:, -1] * self.z_scale_factor
                displacement = Vector(displacement, dims="p")

                x_prime = displacement * scan_direction
                y_prime = displacement * perp_direction
                z_prime = Scalar(displacement.components[:, -1], dims="p")
                distance = displacement.norm

                temperature = self.rosenthal_temperature(
                    x_prime=x_prime,
                    y_prime=y_prime,
                    z_prime=z_prime,
                    R=distance,
                    Qp=powers[scan_idx],
                    v=velocity,
                )
                # temperature_gradient = temperature_gradient.to_specimen_frame(
                #    scan_orientation
                # )

                # Apply z-mask: set temperature to ambient for voxels above scan
                temperature[~active_mask] = self.initial_temperature

                voxel_status = self._update_liquid_and_mushy_voxels(
                    voxel_status, temperature, active_mask
                )
                voxel_status, grain_ids, capture_distances, new_solid_indices = (
                    self._solidify_voxels(
                        voxel_status,
                        grain_ids,
                        temperature,
                        capture_distances,
                        neighbors,
                        distances,
                        active_mask,
                    )
                )
                for _ in range(self.smoothing_steps):
                    grain_ids = _potts_monte_carlo_flips(
                        grain_ids,
                        voxel_status,
                        neighbors,
                        np.ones(len(grain_ids)),
                        new_solid_indices,
                    )

                eligible_mask = (voxel_status == 3) & active_mask
                grain_ids = self._coarsen_microstructure(
                    voxel_status, grain_ids, temperature, neighbors, eligible_mask
                )

            if self.output_times is not None:
                if time_array[time_step] >= self.output_times[output_idx]:
                    temperature_history_array[:, output_idx] = temperature.components
                    grain_history_array[:, output_idx] = grain_ids
                    output_idx += 1

        if self.output_times is not None:
            temperature_field = Scalar(temperature_history_array, dims="pt")
            grain_field = Scalar(grain_history_array, dims="pt")
            new_fields = {
                "temperature": temperature_field,
                grain_id_label: grain_ids,
                grain_id_label + "_history": grain_field,
            }
        else:
            temperature_field = temperature
            new_fields = {
                "temperature": temperature_field,
                grain_id_label: grain_ids,
            }

        new_material = material.create_fields(new_fields)

        if self.output_times is not None:
            new_material.state["time_history"] = self.output_times

        return new_material

    def _coarsen_microstructure(
        self, voxel_status, grain_ids, temperature, neighbors, active_mask
    ):
        """
        Apply Potts Monte Carlo algorithm to simulate grain coarsening.

        Parameters
        ----------
        voxel_status : ndarray
            Status array (0=unaffected, 1=mushy, 2=liquid, 3=solid)
        grain_ids : ndarray
            Grain ID for each voxel
        temperature : Scalar
            Current temperature field
        neighbors : ndarray
            Neighbor indices for each voxel (shape: num_points x 27)
        active_mask : ndarray
            Boolean mask for voxels at or below current laser z-position

        Returns
        -------
        grain_ids : ndarray
            Updated grain IDs after attempted flips
        """
        Q = self.potts_properties["Q"]
        K_0 = self.potts_properties["K_0"]
        K_MC = self.potts_properties["K_MC"]
        mobility = (
            Q
            * (temperature - self.melt_temperature)
            / (8.314 * temperature * self.melt_temperature)
        ).apply(np.exp)
        num_flips = int(
            (5.0e-6) ** 2
            * K_MC
            / K_0
            * np.exp(Q / 8.314 / self.melt_temperature)
            / self.time_step_duration
        )

        # Get indices of active voxels
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            return grain_ids

        # Randomly select voxels from active voxels only
        selected_voxels = active_indices[
            np.random.randint(0, len(active_indices), size=num_flips)
        ]

        # Call numba-compiled function for efficient flip attempts
        grain_ids = _potts_monte_carlo_flips(
            grain_ids, voxel_status, neighbors, mobility.components, selected_voxels
        )

        return grain_ids

    def rosenthal_temperature(
        self,
        x_prime: Scalar,
        y_prime: Scalar,
        z_prime: Scalar,
        R: Scalar,
        Qp: float,
        v: float,
    ) -> Scalar:
        """
        Calculate temperature using the Rosenthal analytical solution.

        Parameters
        ----------
        x_prime : Scalar
            Position in the moving coordinate system (along travel direction).
        R : Scalar
            Distance from each point to the laser beam.
        alpha : float
            Thermal diffusivity [m²/s].
        cond : float
            Thermal conductivity [W/(m·K)].
        absorp : float
            Power absorptivity coefficient [dimensionless].
        Qp : float
            Laser power [W].
        v : float
            Laser velocity [m/s].
        T_0 : float
            Initial/ambient temperature [K].
        T_melt : float
            Maximum temperature cap (melt temperature) [K].

        Returns
        -------
        Scalar
            Temperature at each point [K], capped at T_melt.

        Notes
        -----
        The Rosenthal equation for a point heat source:

        .. math::

            T = T_0 + \\frac{\\eta Q}{2\\pi R k} \\exp\\left(\\frac{-v(x' + R)}{2\\alpha}\\right)

        Temperature values exceeding T_melt are capped at T_melt to represent
        the energy consumed by phase change (melting).
        """
        M = v / (2 * self.diffusivity)
        N = (
            2
            * np.pi
            * self.conductivity
            * (self.melt_temperature - self.initial_temperature)
            / (self.absorptivity * Qp)
        )

        # Avoid division by zero by setting a minimum value for R
        idx = np.argmin(R.components)
        if R.components[idx] == 0:
            R[idx] = 1.0e-15
        R_inv = R ** (-1)
        R_inv_factor = 1.0 + R_inv / M

        temperature_norm = R_inv / N * (-M * (x_prime + R)).apply(np.exp)
        temperature = (
            temperature_norm * (self.melt_temperature - self.initial_temperature)
            + self.initial_temperature
        )
        # dTdx = -temperature_norm * (1 + x_prime * R_inv * R_inv_factor)
        # dTdy = -temperature_norm * y_prime * R_inv * R_inv_factor
        # dTdz = -temperature_norm * z_prime * R_inv * R_inv_factor
        # temperature_gradient = Vector(
        #    np.c_[dTdx.components, dTdy.components, dTdz.components], dims="p"
        # ).unit

        temperature[temperature.components > self.melt_temperature] = (
            self.melt_temperature
        )

        return temperature

    def _update_liquid_and_mushy_voxels(self, voxel_status, temperature, active_mask):
        """
        Update voxel status based on temperature.

        Parameters
        ----------
        voxel_status : ndarray
            Status array (0=unaffected, 1=mushy, 2=liquid, 3=solid)
        temperature : Scalar
            Current temperature field
        active_mask : ndarray
            Boolean mask for voxels at or below current laser z-position

        Returns
        -------
        voxel_status : ndarray
            Updated status array
        """
        # mushy: 1, liquid: 2, solid: 3
        # Only update voxels within the active mask
        liquid_idx = (temperature.components >= self.melt_temperature) & active_mask
        new_mushy_idx = (
            (temperature.components < self.melt_temperature)
            & (voxel_status == 2)
            & active_mask
        )
        voxel_status[liquid_idx] = 2
        voxel_status[new_mushy_idx] = 1
        return voxel_status

    def _solidify_voxels(
        self,
        voxel_status,
        grain_ids,
        temperature,
        capture_distances,
        neighbors,
        distances,
        active_mask,
    ):
        """
        Solidify voxels in the mushy zone based on undercooling and neighbor proximity.

        Parameters
        ----------
        voxel_status : ndarray
            Status array (0=unaffected, 1=mushy, 2=liquid, 3=solid)
        grain_ids : ndarray
            Grain ID for each voxel
        temperature : Scalar
            Current temperature field
        capture_distances : ndarray
            Current capture distance for each voxel (accumulated over time)
        neighbors : ndarray
            Neighbor indices for each voxel (shape: num_points x 27)
        distances : ndarray
            Distances to neighbors for each voxel
        active_mask : ndarray
            Boolean mask for voxels at or below current laser z-position

        Returns
        -------
        voxel_status : ndarray
            Updated voxel status array
        grain_ids : ndarray
            Updated grain IDs
        capture_distances : ndarray
            Updated capture distances
        """
        # Find voxels in mushy zone that are also in active region
        mushy_mask = (voxel_status == 1) & active_mask
        mushy_indices = np.where(mushy_mask)[0]

        if len(mushy_indices) == 0:
            return voxel_status, grain_ids, capture_distances, mushy_indices

        # Compute undercooling for mushy voxels (melt temp - current temp)
        undercooling = self.melt_temperature - temperature.components[mushy_mask]

        # Compute change in capture distance for this time step
        delta_capture = (
            self.a * (undercooling**self.b) * self.time_step_duration / 5.0e-6
        )

        # Update capture distances for mushy voxels
        capture_distances[mushy_mask] += delta_capture

        # Vectorized neighbor processing for all mushy voxels
        # Get all neighbor data for mushy voxels at once
        mushy_neighbors = neighbors[mushy_indices]  # (num_mushy, 27)
        mushy_distances = distances[mushy_indices]  # (num_mushy, 27)

        # Create combined eligibility mask
        valid_mask = mushy_neighbors != -1
        within_capture = mushy_distances <= capture_distances[mushy_indices, np.newaxis]

        # Safe indexing for solid status (replace -1 with 0 to avoid index errors)
        safe_neighbors = np.where(mushy_neighbors >= 0, mushy_neighbors, 0)
        is_solid = voxel_status[safe_neighbors] == 3
        is_active = active_mask[
            safe_neighbors
        ]  # Neighbor must also be in active region

        # Combine all conditions to find eligible neighbors
        eligible_mask = valid_mask & within_capture & is_solid & is_active

        # Random selection via argmax trick (one neighbor per mushy voxel)
        random_vals = np.random.rand(*eligible_mask.shape)
        random_vals[~eligible_mask] = -np.inf
        selected_idx = np.argmax(random_vals, axis=1)

        # Update only voxels that have at least one eligible neighbor
        has_eligible = eligible_mask.any(axis=1)
        selected_neighbors = safe_neighbors[np.arange(len(mushy_indices)), selected_idx]

        # Update grain IDs and voxel status for voxels with eligible neighbors
        grain_ids[mushy_indices[has_eligible]] = grain_ids[
            selected_neighbors[has_eligible]
        ]
        voxel_status[mushy_indices[has_eligible]] = 3
        capture_distances[mushy_indices[has_eligible]] = 0.0

        return voxel_status, grain_ids, capture_distances, mushy_indices[has_eligible]

    def _get_scan_information(self, time_array):
        laser_positions_array = Vector(
            self.laser_path.get_position(time_array), dims="t"
        )  # (time_steps, 3)
        scan_indices = self.laser_path.get_scan_indices(time_array)  # (time_steps,)

        scan_vectors = Vector(
            self.laser_path.end_positions - self.laser_path.start_positions, dims="a"
        )
        durations = Scalar(
            self.laser_path.end_times - self.laser_path.start_times, dims="a"
        )
        scan_velocities = scan_vectors.norm / durations
        scan_directions = scan_vectors.unit
        powers = self.laser_path.powers[scan_indices]
        return (
            laser_positions_array,
            scan_indices,
            scan_velocities,
            scan_directions,
            powers,
        )


def _get_neighbors(material, neighborhood_distance=np.sqrt(3)):
    points = material.extract(["x_id", "y_id", "z_id"])
    epsilon = 1.0e-8
    distances, neighbors = KDTree(points).query(
        points,
        k=27,
        distance_upper_bound=(neighborhood_distance + epsilon),
        workers=-1,
    )
    neighbors[distances == 0] = -1
    neighbors[np.isinf(distances)] = -1
    neighbors = np.roll(neighbors, -1)
    distances = np.roll(distances, -1)

    num_neighbors = np.sum(neighbors != -1, axis=1)
    return neighbors, num_neighbors, distances
