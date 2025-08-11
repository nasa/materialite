# Copyright 2025 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.
#
# The Materialite platform is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import random

import cupy as cp
import cupyx.scipy.ndimage
import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy import spatial
from scipy.interpolate import RegularGridInterpolator
from tqdm import trange

from materialite import Material
from materialite.models import Model


class ConvolutionMicrostructureGPUModel(Model):
    def __init__(
        self,
        auto_assign_fields=False,
        save_history=True,
        adjust_xy_radii_multiple=2,
        adjust_xy_extra_voxels=0,
        z_range=30,
        save_frequency=10,
        time_steps=10,
        initial_temperature=300,
        thermophysical_props={
            "thermal_diffusivity": 12e-6,
            "melt_temperature": 1500,
            "volumetric_specific_heat": 678.6 * 4252,
            "power_absorptivity": 0.5,
        },
        enthalpy_props={
            "vapor_temperature": 1500,
            "heat_of_fusion": 0,
            "heat_of_vapor": 0,
            "melt_range": 0,
            "vapor_range": 0,
        },
        max_spin=20,
        neighborhood_distance=np.sqrt(3),
        seed=None,
        record_frequency=100_000,
        kbTs=0.00000001,
        Q=128312.1,
        orientations=[],
        a_dendrite=1.09 * 10 ** (-10),
        b_dendrite=5.23,
        K_MC=0.27695,
        K_0=0.2994,
        number_additional_flips=1,
        progress_bar=False,
    ):
        self.save_frequency = save_frequency
        self.auto_assign_fields = auto_assign_fields
        self.initial_temperature = initial_temperature  # [K]
        self.save_history = save_history
        self.time_steps = time_steps
        self.melt_temperature = thermophysical_props["melt_temperature"]  # [K]
        self.thermal_diffusivity = thermophysical_props["thermal_diffusivity"]  # [m2/s]
        self.power_absorptivity = thermophysical_props["power_absorptivity"]  # [-]
        self.volumetric_specific_heat = thermophysical_props["volumetric_specific_heat"]
        # [J/m^3/K]
        self.vapor_temperature = enthalpy_props["vapor_temperature"]
        self.heat_of_fusion = enthalpy_props["heat_of_fusion"]
        self.heat_of_vapor = enthalpy_props["heat_of_vapor"]
        self.melt_range = enthalpy_props["melt_range"]
        self.vapor_range = enthalpy_props["vapor_range"]
        self.enthalpy_method = False

        self.adjust_xy_radii_multiple = adjust_xy_radii_multiple
        self.adjust_xy_extra_voxels = adjust_xy_extra_voxels
        self.z_range = int(z_range)

        self.max_spin = max_spin
        self.neighborhood_distance = neighborhood_distance
        self.seed = seed
        self.record_frequency = record_frequency
        self.kbTs = kbTs
        self.Q = Q
        self.orientations = orientations
        self.K_MC = K_MC
        self.K_0 = K_0
        self.number_additional_flips = number_additional_flips
        self.a_dendrite = a_dendrite
        self.b_dendrite = b_dendrite

        self.progress_bar = progress_bar

    def run(self, material, laser, temperature_label, spin_label):
        self._clear_gpu_fields()
        self._initiate_temperature_simulation(material, laser, temperature_label)
        self._initiate_enthalpy_method()
        self._initiate_microstructure_simulation(material, spin_label)
        self._save_fields()
        self._update_z_domain(laser)

        progress_range = trange if self.progress_bar else range
        if self.enthalpy_method:
            for i in progress_range(self.time_steps):
                # Check and update the z-domain details
                self._check_and_update_layer(laser)
                self._take_enthalpy_time_step(laser)
                self._update_liquid_and_mushy_sites()
                self._take_microstructure_time_step()
                self._update_time(laser)
                self._save_fields()
        else:
            for i in progress_range(self.time_steps):
                # Check and update the z-domain details
                self._check_and_update_layer(laser)
                self._take_temperature_time_step(laser)
                self._update_liquid_and_mushy_sites()
                self._take_microstructure_time_step()
                self._update_time(laser)
                self._save_fields()

        new_material = self._finalize_material(material, temperature_label, spin_label)
        self._clear_gpu_fields()

        return new_material

    def _initiate_microstructure_simulation(self, material, spin_label):
        if hasattr(material, "neighbors") and hasattr(material, "num_neighbors"):
            self.neighbors = material.neighbors
            self.num_neighbors = material.num_neighbors
        else:
            self.neighbors, self.num_neighbors, self.neighbor_distances = (
                _get_neighbors(material, self.neighborhood_distance)
            )

        # Create mobility array
        mobility_temp = cp.copy(self._temperature.super_set)
        mobility_temp = self._update_mobility_field(
            self._temperature.super_set, self.melt_temperature, self.Q
        )
        self._mobility = Field(self.domain, mobility_temp)

        # Create spin array
        original_spin_field = (
            material.extract(spin_label).astype(int).reshape(material.dimensions)
        )

        # New random field for super_set
        material_temp = Material(
            dimensions=self._x.super_set.shape, spacing=material.spacing
        )
        spin_temp = (
            material_temp.create_random_integer_field("spin", 0, self.max_spin)
            .extract(spin_label)
            .astype(int)
            .reshape(material_temp.dimensions)
        )

        # Create field, and insert original set back in
        self._spin = Field(self.domain, spin_temp)
        self._spin._update_original_set(original_spin_field)

        # Create fields for grab radius field and
        self._grab_distance = Field(
            self.domain, np.zeros(self._temperature.super_set.shape)
        )

        # 0 = solid, 1 = mushy, 2 = liquid
        # Site needs to be a liquid to become mushy
        self._solidification = Field(
            self.domain, cp.zeros(self._temperature.super_set.shape)
        )

        # Get neighbors for super-set
        (
            self.super_neighbors,
            self.super_num_neighbors,
            self.super_neighbor_distances,
        ) = _get_neighbors(material_temp, self.neighborhood_distance)

        self.euler_angles = self.orientations.euler_angles
        return None

    def _take_microstructure_time_step(self):
        # Update mobility
        self._mobility.sub_set[:, :, :] = self._update_mobility_field(
            self._temperature.sub_set[:, :, :], self.melt_temperature, self.Q
        )

        num_flips = int(
            self.dx**2
            * self.K_MC
            / self.K_0
            * np.exp(self.Q / 8.314 / self.melt_temperature)
            * self._mobility.sub_set.size
        )

        # Flip sites based on number of steps
        domain_size = self._spin.sub_set.size
        new_spin_vector, successful_flip_attempts, recorded_energy = (
            self._do_neighbor_flips(
                self._spin.sub_vector,
                cp.asnumpy(self._mobility.sub_set).reshape(domain_size),
                self.sub_neighbors,
                self.sub_num_neighbors,
                num_flips,
                domain_size,
                self.seed,
                self.record_frequency,
                self.kbTs,
            )
        )
        self._spin.sub_vector[:] = new_spin_vector

        return None

    @staticmethod
    @numba.jit(nopython=True)
    def _do_specified_site_flips(
        spin_field,
        mobility_field,
        neighbors,
        num_neighbors,
        num_points,
        seed,
        record_frequency,
        kbTs,
        site_ids,
        num_attempts,
    ):
        if seed is not None:
            random.seed(seed)
        successful_flip_attempts = 0

        # Loop over sites that need flip attempts
        for current_site in range(site_ids.size):

            # Do number of attempts based on specified values
            for current_flip in range(num_attempts):

                sample_point = site_ids[current_site]

                neighbor_range = num_neighbors[sample_point]
                # Select neighbors that are not liquid
                eligible_spins = []
                for n in range(neighbor_range):
                    potential_spin = spin_field[neighbors[sample_point, n]]
                    if potential_spin != 0:
                        eligible_spins.append(potential_spin)

                test_spin_idx = random.randrange(0, len(eligible_spins))
                test_spin = eligible_spins[test_spin_idx]

                # Check if new spin lowers energy and accept it if so
                current_energy = 0
                test_energy = 0
                for n in range(neighbor_range):
                    spin_to_check = spin_field[neighbors[sample_point, n]]
                    current_energy += spin_to_check != spin_field[sample_point]
                    test_energy += spin_to_check != test_spin

                if test_energy <= current_energy:
                    spin_field[sample_point] = test_spin
                    successful_flip_attempts += 1
                else:
                    pass

            return spin_field, successful_flip_attempts

    @staticmethod
    @numba.jit(nopython=True)
    def _do_neighbor_flips(
        spin_field,
        mobility_field,
        neighbors,
        num_neighbors,
        num_monte_carlo_attempts,
        num_points,
        seed,
        record_frequency,
        kbTs,
    ):
        if seed is not None:
            random.seed(seed)
        successful_flip_attempts = 0
        recorded_energy = []

        for current_flip in range(num_monte_carlo_attempts):

            if current_flip % record_frequency == 0:
                recorded_energy.append(
                    _get_energy(num_points, neighbors, num_neighbors, spin_field)
                )

            sample_point = random.randrange(0, num_points, 1)

            # If site being attempted is liquid, skip
            if spin_field[sample_point] == 0:
                continue

            # Check if mobility is greater than random value. If not, will
            # not accept so skip
            probability_check = random.random()
            if mobility_field[sample_point] < probability_check:
                continue

            # Select random neighbor
            neighbor_range = num_neighbors[sample_point]
            eligible_spins = [
                spin_field[neighbors[sample_point, n]] for n in range(neighbor_range)
            ]
            test_spin_idx = random.randrange(0, neighbor_range)
            test_spin = eligible_spins[test_spin_idx]

            # If test site being attempted is liquid, skip
            if test_spin == 0:
                continue

            # If new spin is previous spin, energy will be the same
            if test_spin == spin_field[sample_point]:
                successful_flip_attempts += 1  # Not clear if this should count
                continue

            current_energy = 0
            test_energy = 0
            for n in range(neighbor_range):
                spin_to_check = spin_field[neighbors[sample_point, n]]
                current_energy += spin_to_check != spin_field[sample_point]
                test_energy += spin_to_check != test_spin

            if test_energy <= current_energy:
                spin_field[sample_point] = test_spin
                successful_flip_attempts += 1
            else:
                dE = 0.5 * (test_energy - current_energy)
                if (
                    np.exp(-dE / kbTs) * mobility_field[sample_point]
                    > probability_check
                ):
                    spin_field[sample_point] = test_spin
                    successful_flip_attempts += 1

        return spin_field, successful_flip_attempts, recorded_energy

    def _update_liquid_and_mushy_sites(self):
        domain_size = self._temperature.sub_set.size
        domain_shape = self._temperature.sub_set.shape
        melt_temp = self.melt_temperature

        rotation_matrix = np.zeros([3, 3])
        Axr = np.zeros([3, 3])
        Ax = np.identity(3)

        # Set new liquid sites to 2
        new_liquid_sites = cp.where(self._temperature.sub_set >= self.melt_temperature)
        self._solidification.sub_set[new_liquid_sites] = 2

        # Set new mushy sites to 1
        new_mushy_sites = cp.where(
            (self._temperature.sub_set < melt_temp)
            & (self._solidification.sub_set == 2)
        )
        self._solidification.sub_set[new_mushy_sites] = 1

        # Get all current mushy sites
        current_mushy_sites = cp.where(self._solidification.sub_set == 1)

        # Get 1D positions of mushy and liquid sites
        current_mushy_1D_sites = cp.asnumpy(
            indices_3d_to_1d(current_mushy_sites, domain_shape)
        )
        new_liquid_1D_sites = cp.asnumpy(
            indices_3d_to_1d(new_liquid_sites, domain_shape)
        )

        # Get temperature and temperature gradients of 1D sites
        dT = cp.gradient(self._temperature.sub_set, self.dx, edge_order=2)
        dTdx = dT[0]
        dTdy = dT[1]
        dTdz = dT[2]

        dTdx_mushy_1D = cp.asnumpy(dTdx.reshape(domain_size)[current_mushy_1D_sites])
        dTdy_mushy_1D = cp.asnumpy(dTdy.reshape(domain_size)[current_mushy_1D_sites])
        dTdz_mushy_1D = cp.asnumpy(dTdz.reshape(domain_size)[current_mushy_1D_sites])

        T_mushy_1D = cp.asnumpy(
            self._temperature.sub_set.reshape(domain_size)[current_mushy_1D_sites]
        )

        # Calculate undercooling at local 1D sites
        undercooling = self.melt_temperature - T_mushy_1D

        # Update sites that are newly liquid to have a spin of 0
        self._spin.sub_vector[new_liquid_1D_sites] = 0

        new_spins, new_grab_distances, new_solid_status, successfully_solidified = (
            self._determine_solidifying_sites(
                current_mushy_1D_sites,
                self.sub_num_neighbors,
                self.sub_neighbors,
                np.copy(self._spin.sub_vector),
                self.sub_neighbor_distances,
                np.copy(self._grab_distance.sub_vector),
                undercooling,
                self.euler_angles,
                dTdx_mushy_1D,
                dTdy_mushy_1D,
                dTdz_mushy_1D,
                Axr,
                Ax,
                rotation_matrix,
                self.a_dendrite,
                self.b_dendrite,
                self.dx,
            )
        )

        self._spin.sub_vector[current_mushy_1D_sites] = new_spins
        self._grab_distance.sub_vector[current_mushy_1D_sites] = new_grab_distances
        self._solidification.sub_set[current_mushy_sites] = new_solid_status

        # Flip the new sites to help smooth domain
        site_ids = current_mushy_1D_sites[successfully_solidified]
        domain_size = self._spin.sub_set.size

        if (site_ids.size > 0) and (self.number_additional_flips != 0):
            new_spin_vector, successful_flip_attempts = self._do_specified_site_flips(
                np.copy(self._spin.sub_vector),
                cp.asnumpy(self._mobility.sub_set).reshape(domain_size),
                self.sub_neighbors,
                self.sub_num_neighbors,
                domain_size,
                self.seed,
                self.record_frequency,
                self.kbTs,
                site_ids,
                self.number_additional_flips,
            )

            if successful_flip_attempts > 0:
                self._spin.sub_vector[:] = new_spin_vector

        return None

    @staticmethod
    @numba.jit(nopython=True)
    def _determine_solidifying_sites(
        current_mushy_1D_sites,
        sub_num_neighbors,
        sub_neighbors,
        spin_vector,
        sub_neighbor_distances,
        grab_vector,
        undercooling,
        all_euler_angles,
        dTdx_mushy_1D,
        dTdy_mushy_1D,
        dTdz_mushy_1D,
        Axr,
        Ax,
        rotation_matrix,
        a_dendrite,
        b_dendrite,
        grab_distance,
    ):

        max_size = 27

        new_spins = np.zeros(current_mushy_1D_sites.size, dtype=numba.int32)
        new_grab_distances = np.zeros(current_mushy_1D_sites.size)
        new_solidification_status = np.zeros(
            current_mushy_1D_sites.size, dtype=numba.int32
        )

        successfully_solidified = np.empty(current_mushy_1D_sites.size, dtype=np.bool_)
        for i in range(current_mushy_1D_sites.size):
            successfully_solidified[i] = False

        # i = current site, j = current neighbor
        for i in range(current_mushy_1D_sites.size):

            current_site = current_mushy_1D_sites[i]
            current_num_neighbors = sub_num_neighbors[current_site]
            current_neighbors = sub_neighbors[current_site][0:current_num_neighbors]
            current_neighbors_spins = spin_vector[current_neighbors]
            current_neighbors_distances = sub_neighbor_distances[current_site][
                0:current_num_neighbors
            ]

            # If a neighbor is not zero, will be more than 0
            if np.sum(current_neighbors_spins) > 0:
                grab_vector[current_site] += a_dendrite * (
                    undercooling[i] ** b_dendrite
                )
                current_distance = grab_vector[current_site]

                if current_distance > grab_distance:
                    relative_probability = np.zeros(max_size)
                    eligible_new_spins = np.zeros(max_size)
                    count = 0

                    # Then loop through neighbors
                    for j in range(current_neighbors.size):
                        neighbor_spin = current_neighbors_spins[j]
                        # if neighbor is liquid, break
                        if neighbor_spin == 0:
                            continue
                        else:
                            euler_angles = all_euler_angles[neighbor_spin]
                            misorientation_distance = (
                                _calculate_misorientation_vs_temperature(
                                    euler_angles,
                                    rotation_matrix,
                                    Ax,
                                    Axr,
                                    dTdx_mushy_1D[i],
                                    dTdy_mushy_1D[i],
                                    dTdz_mushy_1D[i],
                                )
                            )

                            # Adjust current distance to remove sites that are too far
                            current_neighbor_distance = current_neighbors_distances[j]
                            adjusted_distance = (
                                current_distance
                                / current_neighbor_distance
                                * misorientation_distance
                            )
                            if adjusted_distance < grab_distance:
                                continue

                            # Append to possible spins and relative probabilities
                            # which is weighted by misorientation distance
                            eligible_new_spins[count] = neighbor_spin
                            relative_probability[count] = misorientation_distance**1
                            count += 1

                    if count == 0:
                        # Site did not solidify
                        new_spins[i] = 0
                        new_grab_distances[i] = current_distance
                        new_solidification_status[i] = 1
                    else:
                        # Site successfully solidified
                        # Can uses relative probability below
                        solidified_spin = np.random.choice(eligible_new_spins[:count])

                        new_spins[i] = solidified_spin
                        successfully_solidified[i] = True
                        new_grab_distances[i] = 0.0
                        new_solidification_status[i] = 0

                # Site does not have sufficient distance
                else:
                    new_spins[i] = 0
                    new_grab_distances[i] = current_distance
                    new_solidification_status[i] = 1

            # Site does not have any neighbors
            else:
                new_spins[i] = 0
                new_grab_distances[i] = 0.0
                new_solidification_status[i] = 1

        return (
            new_spins,
            new_grab_distances,
            new_solidification_status,
            successfully_solidified,
        )

    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_misorientation_vs_temperature(
        euler_angles, rotation_matrix, Ax, Axr, dTdx, dTdy, dTdz
    ):
        # z1 = phi_1, x2 = Phi, z3 = phi_2
        z1 = euler_angles[0]
        x2 = euler_angles[1]
        z3 = euler_angles[2]
        c1, c2, c3 = np.cos(z1), np.cos(x2), np.cos(z3)
        s1, s2, s3 = np.sin(z1), np.sin(x2), np.sin(z3)

        # Rotation matrix generation
        rotation_matrix[0, 0] = c1 * c3 - c2 * s1 * s3
        rotation_matrix[1, 0] = c3 * s1 + c1 * c2 * s3
        rotation_matrix[2, 0] = s2 * s3
        rotation_matrix[0, 1] = -c1 * s3 - c2 * c3 * s1
        rotation_matrix[1, 1] = c1 * c2 * c3 - s1 * s3
        rotation_matrix[2, 1] = c3 * s2
        rotation_matrix[0, 2] = s1 * s2
        rotation_matrix[1, 2] = -c1 * s2
        rotation_matrix[2, 2] = c2

        for i in range(3):
            for j in range(3):
                Axr[i, j] = (
                    Ax[i, 0] * rotation_matrix[j, 0]
                    + Ax[i, 1] * rotation_matrix[j, 1]
                    + Ax[i, 2] * rotation_matrix[j, 2]
                )

        dT_mag = np.sqrt(dTdx**2 + dTdy**2 + dTdz**2)

        dxj = dTdx / dT_mag
        dyj = dTdy / dT_mag
        dzj = dTdz / dT_mag

        c_theta = 0
        for i in range(3):
            c_temp = np.abs(dxj * Axr[i, 0] + dyj * Axr[i, 1] + dzj * Axr[i, 2])
            if c_temp > c_theta:
                c_theta = c_temp

        # misorientation is arccos(c_theta)
        return c_theta

    @staticmethod
    def _update_mobility_field(temperature_field, max_temperature, Q):
        return np.exp(
            Q
            * (temperature_field - max_temperature)
            / (8.314 * temperature_field * max_temperature)
        )

    def _clear_gpu_fields(self):
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        self.temperature_field = None

        self._x = None
        self._y = None
        self._z = None
        self._temperature = None
        self._phase = None
        self._enthalpy = None
        self._spin = None
        self._grab_distance = None
        self._solidification = None

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return None

    def _check_and_update_layer(self, laser):
        self.current_layer = laser.get_current_beam_conditions()["layer_number"]
        if self.current_layer != self.previous_layer:
            print("\n---- New Layer ----")
            self._update_z_domain(laser)
        self.previous_layer = self.current_layer
        return None

    def _initiate_temperature_simulation(self, material, laser, temperature_label):
        if self.auto_assign_fields:
            material = material.create_uniform_field(
                temperature_label, self.initial_temperature
            )
        else:
            material = self.verify_material(material, [temperature_label])

        self.domain = Domain(
            material, laser, self.adjust_xy_radii_multiple, self.adjust_xy_extra_voxels
        )

        self._x, self._y, self._z, self._temperature = (
            self.domain._generate_input_fields(temperature_label)
        )

        phase_temp = cp.copy(self._temperature.super_set)
        phase_temp = self._update_phase_field(
            self._temperature.super_set, self.melt_temperature
        )
        self._phase = Field(self.domain, phase_temp)

        self.dx = material.spacing[0]
        self.time = 0.0
        self.previous_time = 0.0
        self.time_step = 0
        self.dt = 0

        # Updating the z-domain details
        conditions = laser.get_current_beam_conditions()
        self.current_layer = conditions["layer_number"]
        self.previous_layer = self.current_layer

        maximum_time_steps = laser.x_pos.size - 2
        if maximum_time_steps < self.time_steps:
            print("Too many steps. Adjusting total time steps to ", maximum_time_steps)
            self.time_steps = maximum_time_steps

        if self.save_history:
            self.temperature_history = {}
            self.phase_history = {}
            self.spin_history = {}
            self.dt_history = {}
            self.time_history = {}

        return None

    def _initiate_enthalpy_method(self):
        if self.heat_of_fusion <= 0 and self.heat_of_vapor <= 0:
            self.enthalpy_method = False
        else:
            self.enthalpy_method = True
            if self.melt_range <= 0:
                self.melt_range = 0.1

            if self.vapor_range <= 0:
                self.vapor_range = 0.1

            if self.vapor_temperature <= self.melt_temperature:
                self.vapor_temperature = self.melt_temperature + 0.1

            self.temp_interp = cp.array(
                [
                    0.0,
                    self.melt_temperature - self.melt_range,
                    self.melt_temperature,
                    self.vapor_temperature - self.vapor_range,
                    self.vapor_temperature,
                    1000000000.0,
                ]
            )

            self.enth_interp = cp.array(
                [
                    self.temp_interp[0] * self.volumetric_specific_heat,
                    self.temp_interp[1] * self.volumetric_specific_heat,
                    self.temp_interp[2] * self.volumetric_specific_heat
                    + self.heat_of_fusion,
                    self.temp_interp[3] * self.volumetric_specific_heat
                    + self.heat_of_fusion,
                    self.temp_interp[4] * self.volumetric_specific_heat
                    + self.heat_of_fusion
                    + self.heat_of_vapor,
                    self.temp_interp[5] * self.volumetric_specific_heat
                    + self.heat_of_fusion
                    + self.heat_of_vapor,
                ]
            )

            # Since temperature/ phase relationship is monotonic, can interp with floor
            # Corresponds to solid, mushy, liquid, mushy/vapor, vapor, not physical
            self.phase_interp = cp.array([0, 1, 2, 3, 4, 5])
            enthalpy_temp = cp.interp(
                cp.array(self._temperature.sub_set), self.temp_interp, self.enth_interp
            )
            self._enthalpy = Field(self.domain, enthalpy_temp)

        return None

    def _update_time(self, laser):
        laser.increment_time_step()
        self.time = laser.get_time()
        self.dt = laser.get_dt_of_step()
        self.previous_time = self.time
        self.time_step += 1

    def _take_temperature_time_step(self, laser):

        if self.dt != 0:
            standard_deviation = (
                np.sqrt(2 * self.thermal_diffusivity * np.round(self.dt, decimals=12))
                / self.dx
            )

            if laser.get_power() > 0.0001:
                dtemp_beam = (
                    self.dt
                    / self.volumetric_specific_heat
                    * self.power_absorptivity
                    * laser.heat_input(
                        self._x.sub_set, self._y.sub_set, self._z.sub_set
                    )
                )
            else:
                dtemp_beam = 0.0

            dtemp_diffusion = cupyx.scipy.ndimage.gaussian_filter(
                self._temperature.sub_set,
                standard_deviation,
                mode=["constant", "constant", "reflect"],
                cval=self.initial_temperature,
            )

            self._temperature.sub_set[:, :, :] = dtemp_beam + dtemp_diffusion

        self._phase.sub_set[:, :, :] = self._update_phase_field(
            self._temperature.sub_set[:, :, :], self.melt_temperature
        )

        return None

    def _take_enthalpy_time_step(self, laser):

        if self.dt != 0:

            standard_deviation = (
                np.sqrt(2 * self.thermal_diffusivity * np.round(self.dt, decimals=12))
                / self.dx
            )

            if laser.get_power() > 0.0001:
                dtemp_beam = (
                    self.dt
                    / self.volumetric_specific_heat
                    * self.power_absorptivity
                    * laser.heat_input(
                        self._x.sub_set, self._y.sub_set, self._z.sub_set
                    )
                )
            else:
                dtemp_beam = 0.0

            dtemp_diffusion = cupyx.scipy.ndimage.gaussian_filter(
                self._temperature.sub_set,
                standard_deviation,
                mode=["constant", "constant", "reflect"],
                cval=self.initial_temperature,
            )
            dtemp_diffusion = dtemp_diffusion - self._temperature.sub_set

            dtemp = dtemp_beam + dtemp_diffusion

            self._enthalpy.sub_set[:, :, :] += dtemp * self.volumetric_specific_heat
            self._temperature.sub_set[:, :, :] = cp.interp(
                cp.array(self._enthalpy.sub_set[:, :, :]),
                self.enth_interp,
                self.temp_interp,
            )

        self._phase.sub_set[:, :, :] = self._update_enthalpy_phase_field(
            self._temperature.sub_set[:, :, :],
            self.phase_interp,
            self.temp_interp,
        )

        return None

    def _save_fields(self):
        if self.save_history:
            if (self.time_step % self.save_frequency) == 0:
                self.temperature_history.update(
                    {self.time_step: np.copy(cp.asnumpy(self._temperature.super_set))}
                )
                self.phase_history.update(
                    {self.time_step: np.copy(cp.asnumpy(self._phase.super_set))}
                )
                self.spin_history.update(
                    {self.time_step: np.copy(cp.asnumpy(self._spin.super_set))}
                )
                self.dt_history.update({self.time_step: self.dt})
                self.time_history.update({self.time_step: self.time})
        return None

    def _finalize_material(self, material, temperature_label, spin_label):
        new_material = material.create_fields(
            {"phase": cp.asnumpy(self._phase.original_set).reshape(material.num_points)}
        )
        new_material = new_material.create_fields(
            {
                temperature_label: cp.asnumpy(self._temperature.original_set).reshape(
                    material.num_points
                )
            }
        )

        new_material = new_material.create_fields(
            {
                spin_label: cp.asnumpy(self._spin.original_set).reshape(
                    material.num_points
                )
            }
        )

        new_material.state["temperature_history"] = self.temperature_history
        new_material.state["phase_history"] = self.phase_history
        new_material.state["dt_history"] = self.dt_history
        new_material.state["spin_history"] = self.spin_history
        new_material.state["time_history"] = self.time_history
        new_material.state["orientations"] = self.orientations

        return new_material

    @staticmethod
    def _update_phase_field(temperature_field, melt_temperature):
        updated_phase_field = cp.where(temperature_field < melt_temperature, 0, 1)
        return updated_phase_field

    @staticmethod
    def _update_enthalpy_phase_field(temperature_field, phase_interp, temp_interp):
        updated_phase_field = cp.interp(
            cp.array(temperature_field), temp_interp, phase_interp
        )
        updated_phase_field = cp.floor(updated_phase_field)
        return updated_phase_field

    def plot_temperature_enthalpy_relationship(self):
        if self.enthalpy_method:
            plt.plot(cp.asnumpy(self.temp_interp), cp.asnumpy(self.enth_interp))
            plt.xlim(0, self.vapor_temperature + self.vapor_range + 100)
            plt.ylim(0, cp.asnumpy(self.enth_interp[4] * 1.2))
            plt.xlabel("Temperature (K)")
            plt.ylabel("Enthalpy")
            plt.show()
        else:
            print("Cannot plot. Enthalpy method not in use or initialized.")

    def _update_z_domain(self, laser):
        conditions = laser.get_current_beam_conditions()
        self.z_upper = int(np.round(conditions["z"] / self.dx))
        self.z_lower = int(self.z_upper - self.z_range)
        self.domain._set_sub_z_range(self.z_lower, self.z_upper)
        self._x._update_sub_set()
        self._y._update_sub_set()
        self._z._update_sub_set()
        self._temperature._update_sub_set()
        self._phase._update_sub_set()
        self._mobility._update_sub_set()
        self._spin._update_sub_set()
        self._solidification._update_sub_set()
        self._grab_distance._update_sub_set()
        if self.enthalpy_method:
            self._enthalpy._update_sub_set()

        # For neighbors
        self.sub_material = Material(self._x.sub_set.shape)
        self.sub_neighbors, self.sub_num_neighbors, self.sub_neighbor_distances = (
            _get_neighbors(self.sub_material, self.neighborhood_distance)
        )

        # Relate 1D index in 3D coordinates
        self.index_neighbor_relation = cp.arange(
            0, self.sub_num_neighbors.shape[0], 1
        ).reshape(self._x.sub_set.shape)

        self._spin.sub_vector = self._spin.sub_set.reshape(self._x.sub_set.size)
        self._grab_distance.sub_vector = self._grab_distance.sub_set.reshape(
            self._x.sub_set.size
        )

        return None


def _get_neighbors(material, neighborhood_distance):
    points = material.extract(["x_id", "y_id", "z_id"])
    epsilon = 10 ** (-8)
    distances, neighbors = spatial.KDTree(points).query(
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


@numba.jit(nopython=True)
def _get_energy(num_points, neighbors, num_neighbors, spin_field):
    current_energy = 0
    for sample_point in range(num_points):
        neighbor_range = num_neighbors[sample_point]
        for n in range(neighbor_range):
            spin_to_check = spin_field[neighbors[sample_point, n]]
            current_energy += spin_to_check != spin_field[sample_point]

    return current_energy


@numba.jit(nopython=True)
def _calculate_misorientation_vs_temperature(
    euler_angles, rotation_matrix, Ax, Axr, dTdx, dTdy, dTdz
):
    # z1 = phi_1, x2 = Phi, z3 = phi_2
    z1 = euler_angles[0]
    x2 = euler_angles[1]
    z3 = euler_angles[2]
    c1, c2, c3 = np.cos(z1), np.cos(x2), np.cos(z3)
    s1, s2, s3 = np.sin(z1), np.sin(x2), np.sin(z3)

    # Rotation matrix generation
    rotation_matrix[0, 0] = c1 * c3 - c2 * s1 * s3
    rotation_matrix[1, 0] = c3 * s1 + c1 * c2 * s3
    rotation_matrix[2, 0] = s2 * s3
    rotation_matrix[0, 1] = -c1 * s3 - c2 * c3 * s1
    rotation_matrix[1, 1] = c1 * c2 * c3 - s1 * s3
    rotation_matrix[2, 1] = c3 * s2
    rotation_matrix[0, 2] = s1 * s2
    rotation_matrix[1, 2] = -c1 * s2
    rotation_matrix[2, 2] = c2

    for i in range(3):
        for j in range(3):
            Axr[i, j] = (
                Ax[i, 0] * rotation_matrix[j, 0]
                + Ax[i, 1] * rotation_matrix[j, 1]
                + Ax[i, 2] * rotation_matrix[j, 2]
            )

    dT_mag = np.sqrt(dTdx**2 + dTdy**2 + dTdz**2)

    dxj = dTdx / dT_mag
    dyj = dTdy / dT_mag
    dzj = dTdz / dT_mag

    misorientation_distance = 0
    for i in range(3):
        misorientation_temp = np.abs(
            dxj * Axr[i, 0] + dyj * Axr[i, 1] + dzj * Axr[i, 2]
        )
        if misorientation_temp > misorientation_distance:
            misorientation_distance = misorientation_temp

    # arccos(misorientation_distance) returns the misorientation angle
    return misorientation_distance


@numba.jit(nopython=True)
def numba_weighted_random_choice(array, weights, size, seed):
    # Create result array with explicit dtype
    result = np.zeros(size, dtype=array.dtype)

    # Normalize weights to probabilities
    weights_sum = np.sum(weights)
    p = np.zeros(len(weights))
    for i in range(len(weights)):
        p[i] = weights[i] / weights_sum

    # Calculate cumulative probabilities
    cdf = np.zeros(len(p))
    cdf[0] = p[0]
    for i in range(1, len(p)):
        cdf[i] = cdf[i - 1] + p[i]

    # Sample using inverse CDF method
    for i in range(size):
        u = np.random.random()
        idx = 0
        for j in range(len(cdf)):
            if u <= cdf[j]:
                idx = j
                break
        result[i] = array[idx]

    return result


def indices_3d_to_1d(input_array, domain_shape):
    i = input_array[0]
    j = input_array[1]
    k = input_array[2]
    Ny = domain_shape[1]
    Nz = domain_shape[2]

    return i * (Ny * Nz) + j * Nz + k


class Domain:
    def __init__(self, material, laser, xy_radii_multi, xy_extra_voxels):

        self.material = material
        self.original_dimensions = material.dimensions

        self._determine_xy_adjustments(material, laser, xy_radii_multi, xy_extra_voxels)

        self.super_dimensions = material.dimensions
        self.super_dimensions[0] = material.dimensions[0] + sum(
            self.x_adjustment_amount
        )
        self.super_dimensions[1] = material.dimensions[1] + sum(
            self.y_adjustment_amount
        )
        self.super_dimensions[2] = material.dimensions[2]

        self.x_super_to_orig_indices = [
            self.x_adjustment_amount[0],
            int(self.super_dimensions[0] - self.x_adjustment_amount[1]),
        ]

        self.y_super_to_orig_indices = [
            self.y_adjustment_amount[0],
            int(self.super_dimensions[1] - self.y_adjustment_amount[1]),
        ]

        self.sub_dimensions = np.copy(self.super_dimensions)

    def _generate_input_fields(self, temperature_label):
        # These should act on fields that have initial values, rather than
        # resultant fields (like phase or enthalpy). Those can be handled
        # with the field class by passing in the domain

        # the data to interpolate
        x = self.material.fields["x"].to_numpy().reshape(self.material.dimensions)
        y = self.material.fields["y"].to_numpy().reshape(self.material.dimensions)
        z = self.material.fields["z"].to_numpy().reshape(self.material.dimensions)
        temp = (
            self.material.fields[temperature_label]
            .to_numpy()
            .reshape(self.material.dimensions)
        )

        # the inputs / x, y, z positional data
        x_id = np.unique(self.material.fields["x_id"].to_numpy())
        y_id = np.unique(self.material.fields["y_id"].to_numpy())
        z_id = np.unique(self.material.fields["z_id"].to_numpy())

        x_interp = RegularGridInterpolator(
            (x_id, y_id, z_id), x, method="linear", bounds_error=False, fill_value=None
        )

        y_interp = RegularGridInterpolator(
            (x_id, y_id, z_id), y, method="linear", bounds_error=False, fill_value=None
        )

        z_interp = RegularGridInterpolator(
            (x_id, y_id, z_id), z, method="linear", bounds_error=False, fill_value=None
        )

        temp_interp = RegularGridInterpolator(
            (x_id, y_id, z_id),
            temp,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        x_new_id = np.concatenate(
            (
                np.arange(x_id[0] - self.x_adjustment_amount[0], x_id[0]),
                x_id,
                np.arange(x_id[-1] + 1, x_id[-1] + 1 + self.x_adjustment_amount[1]),
            ),
            axis=0,
        )

        y_new_id = np.concatenate(
            (
                np.arange(y_id[0] - self.y_adjustment_amount[0], y_id[0]),
                y_id,
                np.arange(y_id[-1] + 1, y_id[-1] + 1 + self.y_adjustment_amount[1]),
            ),
            axis=0,
        )

        z_new_id = z_id

        xv, yv, zv = np.meshgrid(x_new_id, y_new_id, z_new_id, indexing="ij")
        points = np.column_stack((xv.flatten(), yv.flatten(), zv.flatten()))

        x_new = cp.array(x_interp(points).reshape(self.super_dimensions))
        y_new = cp.array(y_interp(points).reshape(self.super_dimensions))
        z_new = cp.array(z_interp(points).reshape(self.super_dimensions))
        temp_new = cp.array(temp_interp(points).reshape(self.super_dimensions))

        x_field = Field(self, x_new)
        y_field = Field(self, y_new)
        z_field = Field(self, z_new)
        temp_field = Field(self, temp_new)

        return x_field, y_field, z_field, temp_field

    def _determine_xy_adjustments(
        self, material, laser, xy_radii_multi, xy_extra_voxels
    ):

        x = material.fields["x"].to_numpy()
        y = material.fields["y"].to_numpy()
        dx = material.spacing[0]

        x_laser_max = np.max(laser.x_pos) + laser.beam_x_radius * xy_radii_multi
        x_laser_min = np.min(laser.x_pos) - laser.beam_x_radius * xy_radii_multi

        x_max_adjustment = (
            int(np.round((x_laser_max - np.max(x)) / dx)) + xy_extra_voxels
        )
        x_min_adjustment = (
            int(np.round((np.min(x) - x_laser_min) / dx)) + xy_extra_voxels
        )

        y_laser_max = np.max(laser.y_pos) + laser.beam_y_radius * xy_radii_multi
        y_laser_min = np.min(laser.y_pos) - laser.beam_y_radius * xy_radii_multi
        y_max_adjustment = (
            int(np.round((y_laser_max - np.max(y)) / dx)) + xy_extra_voxels
        )
        y_min_adjustment = (
            int(np.round((np.min(y) - y_laser_min) / dx)) + xy_extra_voxels
        )

        # Prevents domain from shrinking (only grows)
        self.x_adjustment_amount = np.maximum([x_min_adjustment, x_max_adjustment], 0)
        self.y_adjustment_amount = np.maximum([y_min_adjustment, y_max_adjustment], 0)

        return None

    def _set_sub_z_range(self, z_lower, z_upper):

        if z_lower < 0:
            z_lower = 0

        if z_upper > self.super_dimensions[2]:
            z_upper = self.super_dimensions[2]

        # Note - this is different than how adjustment is defined
        self.z_sub_adjustment = [z_lower, z_upper]
        self.sub_dimensions[2] = z_upper - z_lower
        return None


class Field:
    # The super, sub, and original sets all share memory with one another
    # If one changes values, they all change
    def __init__(self, domain, super_set):
        self.domain = domain
        self.super_set = super_set

        # Can the original set point to the super set?
        self.original_set = self.super_set[
            self.domain.x_super_to_orig_indices[
                0
            ] : self.domain.x_super_to_orig_indices[1],
            self.domain.y_super_to_orig_indices[
                0
            ] : self.domain.y_super_to_orig_indices[1],
            :,
        ]

        self.sub_set = super_set[:, :, :]

    def _update_sub_set(self):
        z_lower = self.domain.z_sub_adjustment[0]
        z_upper = self.domain.z_sub_adjustment[1]
        self.sub_set = self.super_set[:, :, z_lower:z_upper]
        return None

    def return_save_array(self):
        return None

    def _update_original_set(self, new_original_set):
        self.original_set[:, :, :] = new_original_set
        return None


class LaserBeam:
    def __init__(
        self,
        material,
        beam_x_radius=10.0e-6,
        beam_y_radius=10.0e-6,
        beam_z_radius=10.0e-6,
        laser_power=50,
        laser_velocity=1.0,
        hatch_spacing=10.0,  # [Voxels]
        layer_thickness=5,  # [Voxels]
        num_layers=1,
        z_start=35,  # [Voxels]
        rotation_angle_increment=90.0 * np.pi / 180,
        first_rotation_angle=0.0,
        scan_strategy="serpentine",
        scan_offset_start=-5,  # [Voxels]
        scan_offset_end=0,  # [Voxels]
        hatch_offset_start=5,  # [Voxels]
        hatch_offset_end=-5,  # [Voxels]
        pore_prob=0.000,
        dist_type="normal",
        mu=4.0,  # Voxels
        sigma=1.0,  # Voxels
        T_ref=300,  # Kelvin
        time_between_scans=0.01,
        time_between_layers=0.0,
        minimum_time_step=20e-6,
    ):
        self.beam_x_radius = beam_x_radius
        self.beam_y_radius = beam_y_radius
        self.beam_z_radius = beam_z_radius
        self.laser_power = laser_power
        self.laser_velocity = laser_velocity
        self.layer_thickness = layer_thickness
        self.hatch_spacing = hatch_spacing

        self.num_layers = num_layers
        self.z_start = z_start
        self.rotation_angle_increment = rotation_angle_increment
        self.first_rotation_angle = first_rotation_angle
        self.scan_strategy = scan_strategy

        self.scan_offset_start = scan_offset_start
        self.scan_offset_end = scan_offset_end
        self.hatch_offset_start = hatch_offset_start
        self.hatch_offset_end = hatch_offset_end

        self.pore_prob = pore_prob
        self.dist_type = dist_type
        self.mu = mu
        self.sigma = sigma

        self.T_ref = T_ref

        self.time_between_scans = time_between_scans
        self.time_between_layers = time_between_layers
        self.minimum_time_step = minimum_time_step

        self.dx = material.spacing[0]
        self.x_voxels = material.dimensions[0]
        self.y_voxels = material.dimensions[1]
        self.z_voxels = material.dimensions[2]

        self.voxel_velocity = self.laser_velocity / self.dx

        self.dt = self.dx / self.laser_velocity * 0.9
        self.dt_between_lines = self.dt * 0
        self.dt_short = self.dt_between_lines

        self.time_step = 0

        self.goldak_prefactor = (
            6
            * np.sqrt(3)
            / (
                np.pi
                * np.sqrt(np.pi)
                * self.beam_x_radius
                * self.beam_y_radius
                * self.beam_z_radius
            )
        )

    def create_build_path(self, material, adjust_domain=False, domain_epsilon=0.0):
        build = Build(self.dx)
        time_start_i = 0.0  # track start time for each layer

        for i in range(self.num_layers):
            # height and rotation angle of current layer
            z_layer_i = i * self.layer_thickness + self.z_start
            rotation_angle_i = (
                i * self.rotation_angle_increment + self.first_rotation_angle
            )

            # Create Layer instance (will contain a list of scans and
            # transitions between the scans that comprise the layer)
            layer_i = AutoLayer(
                self.x_voxels,
                self.y_voxels,
                z_layer_i,
                time_start_i,
                self.scan_strategy,
                self.hatch_spacing,
                self.laser_power,
                self.voxel_velocity,
                rotation_angle_i,
                self.dt,
                self.T_ref,
            )

            # Add scans to layer based on scan parameters
            layer_i.auto_generate_scans(
                self.scan_offset_start,
                self.scan_offset_end,
                self.hatch_offset_start,
                self.hatch_offset_end,
                self.dt_short,
            )
            layer_i.insert_transitions(self.T_ref)  # moving laser between scans
            time_start_i = layer_i.time_end

            # Add layer to Build instance
            build.add_layer_to_build(layer_i)

        build.keyhole_porosity(self.pore_prob, self.dist_type, self.mu, self.sigma)

        # Get points for directing beam around
        x_pos, y_pos, z_pos, time, beam_power, scan_number, layer_number = (
            build.get_all_real_path_points()
        )

        self.x_pos = x_pos
        self.y_pos = y_pos
        self.z_pos = z_pos
        self.time = time
        self.beam_power = beam_power
        self.scan_number = scan_number
        self.layer_number = layer_number

        if adjust_domain:
            self.adjust_beam_pathing_to_domain(material, domain_epsilon)

    def adjust_beam_pathing_to_domain(self, material, domain_epsilon=0.0):
        dt = np.copy(self.dt)

        # Mask of all of the steps outside in x or y
        mask = (
            (self.x_pos <= (np.max(material.extract("x"))) + domain_epsilon)
            * (self.y_pos <= (np.max(material.extract("y"))) + domain_epsilon)
            * (self.x_pos >= (np.min(material.extract("x"))) - domain_epsilon)
            * (self.y_pos >= (np.min(material.extract("y"))) - domain_epsilon)
        ) * self.beam_power != 0

        x_adjust = np.copy(self.x_pos[mask])
        y_adjust = np.copy(self.y_pos[mask])
        z_adjust = np.copy(self.z_pos[mask])
        scan_adjust = np.copy(self.scan_number[mask])
        layer_adjust = np.copy(self.layer_number[mask])
        power_adjust = np.copy(self.beam_power[mask])

        power_new = []
        x_new = []
        y_new = []
        z_new = []
        layer_new = []
        scan_new = []
        time_step_new = []

        scan_list = np.unique(scan_adjust)

        time_between_tracks = self.time_between_scans
        extra_scan_steps = int(np.round(time_between_tracks / self.minimum_time_step))

        time_between_layers = self.time_between_layers
        extra_layer_steps = int(np.round(time_between_layers / self.minimum_time_step))

        for i in range(scan_list.size):

            # go down scan list and pull off each one
            x_scan = x_adjust[np.where(scan_adjust == scan_list[i])]
            y_scan = y_adjust[np.where(scan_adjust == scan_list[i])]
            z_scan = z_adjust[np.where(scan_adjust == scan_list[i])]
            power_scan = power_adjust[np.where(scan_adjust == scan_list[i])]
            layer_scan = layer_adjust[np.where(scan_adjust == scan_list[i])]
            time_step_scan = np.ones(x_scan.size) * dt

            x_between_scans = np.ones(extra_scan_steps) * x_scan[-1]
            y_between_scans = np.ones(extra_scan_steps) * y_scan[-1]
            z_between_scans = np.ones(extra_scan_steps) * z_scan[-1]
            power_between_scans = np.ones(extra_scan_steps) * 0.0
            layer_between_scans = np.ones(extra_scan_steps) * layer_scan[-1]
            time_step_between_scans = np.ones(extra_scan_steps) * self.minimum_time_step

            x_between_layers = np.ones(extra_layer_steps) * x_scan[-1]
            y_between_layers = np.ones(extra_layer_steps) * y_scan[-1]
            z_between_layers = np.ones(extra_layer_steps) * z_scan[-1]
            power_between_layers = np.ones(extra_layer_steps) * 0.0
            layer_between_layers = np.ones(extra_layer_steps) * layer_scan[-1]
            time_step_between_layers = (
                np.ones(extra_layer_steps) * self.minimum_time_step
            )

            next_point = np.max(np.where(scan_adjust == scan_list[i])) + 1
            if (next_point < layer_adjust.size) and (
                layer_adjust[next_point] != layer_scan[0]
            ):
                x_combined = np.concatenate((x_scan, x_between_scans, x_between_layers))
                y_combined = np.concatenate((y_scan, y_between_scans, y_between_layers))
                z_combined = np.concatenate((z_scan, z_between_scans, z_between_layers))
                power_combined = np.concatenate(
                    (power_scan, power_between_scans, power_between_layers)
                )
                layer_combined = np.concatenate(
                    (layer_scan, layer_between_scans, layer_between_layers)
                )
                time_step_combined = np.concatenate(
                    (time_step_scan, time_step_between_scans, time_step_between_layers)
                )
            else:
                x_combined = np.concatenate((x_scan, x_between_scans))
                y_combined = np.concatenate((y_scan, y_between_scans))
                z_combined = np.concatenate((z_scan, z_between_scans))
                power_combined = np.concatenate((power_scan, power_between_scans))
                layer_combined = np.concatenate((layer_scan, layer_between_scans))
                time_step_combined = np.concatenate(
                    (time_step_scan, time_step_between_scans)
                )

            x_new = np.append(x_new, x_combined)
            y_new = np.append(y_new, y_combined)
            z_new = np.append(z_new, z_combined)
            power_new = np.append(power_new, power_combined)
            layer_new = np.append(layer_new, layer_combined)
            time_step_new = np.append(time_step_new, time_step_combined)

            scan_tmp = np.zeros(x_combined.size) + scan_list[i]
            scan_new = np.append(scan_new, scan_tmp)

        self.x_pos = x_new
        self.y_pos = y_new
        self.z_pos = z_new
        self.time = np.cumsum(time_step_new)
        self.dt_step = time_step_new
        self.beam_power = power_new
        self.scan_number = scan_new
        self.layer_number = layer_new

        return None

    def plot_beam_path(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(
            [0, self.x_voxels, self.x_voxels, 0, 0],
            [0, 0, self.y_voxels, self.y_voxels, 0],
            "k-",
            linewidth=5,
        )
        plt.plot(self.x_pos / self.dx, self.y_pos / self.dx, "ro")
        plt.plot(self.x_pos / self.dx, self.y_pos / self.dx, "b--")
        ax.set_aspect(1.0)
        plt.show()

    def get_current_beam_conditions(self):
        conditions = {
            "x": self.x_pos[self.time_step],
            "y": self.y_pos[self.time_step],
            "z": self.z_pos[self.time_step],
            "time": self.time[self.time_step],
            "beam_power": self.beam_power[self.time_step],
            "scan_number": self.scan_number[self.time_step],
            "layer_number": self.layer_number[self.time_step],
        }

        return conditions

    def get_time(self):
        return self.time[self.time_step]

    def get_power(self):
        return self.beam_power[self.time_step]

    def get_dt_of_step(self):
        return self.dt_step[self.time_step]

    def increment_time_step(self):
        self.time_step += 1

    def distance_to_beam(self, x, y, z):
        return cp.sqrt(
            (x - self.x_pos[self.time_step]) ** 2
            + (y - self.y_pos[self.time_step]) ** 2
            + (z - self.z_pos[self.time_step]) ** 2
        )

    def _gaussian_beam(self, x, y, z):
        return cp.exp(
            -2 * ((x - self.x_pos[self.time_step]) ** 2) / self.beam_x_radius**2
            + -2 * ((y - self.y_pos[self.time_step]) ** 2) / self.beam_y_radius**2
            + -2 * ((z - self.z_pos[self.time_step]) ** 2) / self.beam_z_radius**2
        )

    def heat_input(self, x, y, z):
        q_beam = self.goldak_prefactor * self._gaussian_beam(x, y, z)
        normalization_coeff = cp.sum(q_beam * self.dx**3)
        if normalization_coeff == 0:
            q_beam = 0
        else:
            q_beam = q_beam * self.beam_power[self.time_step] / normalization_coeff
        return q_beam

    def plot_1d_beam_shape(self, direction="x"):
        if direction == "x":
            x = np.linspace(-self.beam_x_radius * 3, self.beam_x_radius * 3, 200)
            p = np.exp(-2 * ((x - 0) ** 2) / self.beam_x_radius**2)
            plt.plot(x * 1e6, p)
            plt.xlabel("X-Position (µm)")
            plt.ylabel("Normalized Beam Power")
            plt.show()

        elif direction == "y":
            y = np.linspace(-self.beam_x_radius * 3, self.beam_y_radius * 3, 200)
            p = np.exp(-2 * ((y - 0) ** 2) / self.beam_y_radius**2)
            plt.plot(y * 1e6, p)
            plt.xlabel("Y-Position (µm)")
            plt.ylabel("Normalized Beam Power")
            plt.show()

        elif direction == "z":
            z = np.linspace(0, self.beam_z_radius * 3, 200)
            p = np.exp(-2 * ((z - 0) ** 2) / self.beam_z_radius**2)
            plt.plot(z * 1e6, p)
            plt.xlabel("Z-Position (µm)")
            plt.ylabel("Normalized Beam Power")
            plt.show()

    def import_beam_path(self, x, y, z, time, power, scan, layer):
        "Not going to do any sanitization; user beware"
        arrays = [x, y, z, time, power, scan, layer]
        first_shape = arrays[0].shape
        if np.all([arr.shape == first_shape for arr in arrays]):
            if np.all(np.diff(time) >= 0):
                self.x_pos = x
                self.y_pos = y
                self.z_pos = z
                self.time = time
                self.beam_power = power
                self.scan_number = scan
                self.layer_number = layer
            else:
                print("Time does not advance monotonically")

        else:
            print("Array size mismatch")

        return None


class Build:
    def __init__(self, voxel_spacing):
        self.voxel_spacing = voxel_spacing

        self.total_time = 0.0
        self.layer_list = []

    def write_to_spparks_path_file(
        self,
        fname,
        fmt=["%g", "%g", "%g", "%g", "%g", "%.0f", "%g", "%g"],
        num_keyhole_pores=0,
        **kwargs,
    ):
        all_data = []
        for layer in self.layer_list:
            for scan in layer.scan_list:
                scan_data = [
                    scan.x_points,
                    scan.y_points,
                    scan.z_points,
                    scan.power_points,
                    scan.velocity_points,
                    scan.time_points,
                    scan.temperature_points,
                    scan.keyhole_points,
                ]
                all_data.append(scan_data)

        data_to_write = np.concatenate(all_data, axis=1).T
        data_to_write[:, 4] = self._convert_velocity_to_real_units(data_to_write[:, 4])
        data_to_write[:, 5] = self._convert_time_to_nanoseconds(data_to_write[:, 5])

        if num_keyhole_pores > 0:
            all_points = np.arange(len(data_to_write))
            x_bounds = kwargs.get("x_bounds", None)
            y_bounds = kwargs.get("y_bounds", None)
            z_bounds = kwargs.get("z_bounds", None)
            keyhole_radius = kwargs.get("keyhole_radius", None)
            region_idx = self._get_points_in_region(
                data_to_write[:, 0],
                data_to_write[:, 1],
                data_to_write[:, 2],
                x_bounds,
                y_bounds,
                z_bounds,
            )
            allowed_points = all_points[region_idx]
            keyhole_points = random.sample(list(allowed_points), num_keyhole_pores)
            data_to_write[keyhole_points, -1] = keyhole_radius

        np.savetxt(fname, data_to_write, fmt=fmt, delimiter=",")

    def get_all_real_path_points(self):
        scan_count = 0
        layer_count = 0

        all_data = []
        for layer in self.layer_list:
            for scan in layer.scan_list:
                scan_track = np.zeros(scan.x_points.size) + scan_count
                layer_track = np.zeros(scan.x_points.size) + layer_count

                scan_data = [
                    scan.x_points,
                    scan.y_points,
                    scan.z_points,
                    scan.time_points,
                    scan.power_points,
                    scan_track,
                    layer_track,
                ]
                all_data.append(scan_data)

                scan_count += 1

            layer_count += 1

        all_data = np.concatenate(all_data, axis=1).T
        x = all_data[:, 0] * self.voxel_spacing
        y = all_data[:, 1] * self.voxel_spacing
        z = all_data[:, 2] * self.voxel_spacing
        time = all_data[:, 3]
        power = all_data[:, 4]
        scan_number = all_data[:, 5]
        layer_number = all_data[:, 6]
        return x, y, z, time, power, scan_number, layer_number

    def get_layer_real_path_points(self, layer_number):
        layer_data = []
        for scan in self.layer_list[layer_number].scan_list:
            scan_data = [
                scan.x_points,
                scan.y_points,
                scan.z_points,
                scan.time_points,
                scan.power_points,
            ]
            layer_data.append(scan_data)

        layer_data = np.concatenate(layer_data, axis=1).T
        x = layer_data[:, 0] * self.voxel_spacing
        y = layer_data[:, 1] * self.voxel_spacing
        z = layer_data[:, 2] * self.voxel_spacing
        time = layer_data[:, 3]
        power = layer_data[:, 4]
        return x, y, z, time, power

    def add_layer_to_build(self, layer):
        self.layer_list.append(layer)
        self.total_time = layer.time_end

    def keyhole_porosity(self, pore_prob, dist_type, *args, **kwargs):
        for layer in self.layer_list:
            for scan in layer.scan_list:
                scan.insert_keyhole_porosity(pore_prob, dist_type, *args, **kwargs)

    def get_number_of_points_in_region(self, x_bounds, y_bounds, z_bounds):
        num_points = 0
        for layer in self.layer_list:
            for scan in layer.scan_list:
                z_check = scan.z_points[0]
                if np.logical_or(z_check < z_bounds[0], z_check > z_bounds[1]):
                    continue
                scan_points, _ = scan.get_points_in_region(x_bounds, y_bounds)
                num_points = num_points + scan_points
        return num_points

    def _convert_velocity_to_real_units(self, data):
        real_velocity_flag = data >= 0
        data[real_velocity_flag] = data[real_velocity_flag] * self.voxel_spacing
        return data

    @staticmethod
    def _get_points_in_region(
        x_points, y_points, z_points, x_bounds, y_bounds, z_bounds
    ):
        region_idx_x = Build._get_region_indices(x_points, x_bounds)
        region_idx_y = Build._get_region_indices(y_points, y_bounds)
        region_idx_z = Build._get_region_indices(z_points, z_bounds)
        region_idx_xy = np.logical_and(region_idx_x, region_idx_y)
        region_idx = np.logical_and(region_idx_xy, region_idx_z)
        return region_idx

    @staticmethod
    def _get_region_indices(points, bounds):
        if bounds is not None:
            return np.logical_and(points >= bounds[0], points <= bounds[1])
        else:
            return np.ones(len(points), dtype=bool)

    @staticmethod
    def _convert_time_to_nanoseconds(data):
        data = data * 10**9
        return data


class Layer:
    def __init__(self, x_size, y_size, z, time_start):
        self.x_size = x_size
        self.y_size = y_size
        self.z = z
        self.time_start = time_start
        self.time_end = time_start

        self.scan_list = []

    def insert_transitions(self, reference_temperature):
        idx = 1
        scan_list_tmp = self.scan_list.copy()
        for scan_end, scan_start in zip(scan_list_tmp[:-1], scan_list_tmp[1:]):
            x = scan_start.x_points[0]
            y = scan_start.y_points[0]
            z = scan_start.z_points[0]
            v = scan_start.velocity_points[0]
            t = scan_end.time_points[-1]
            rotation_angle = scan_start.rotation_angle
            transition = Transition(
                x, y, z, v, t, reference_temperature, rotation_angle
            )
            self.scan_list.insert(idx, transition)
            idx = idx + 2

        first_scan = self.scan_list[0]
        x_start = first_scan.x_points[0]
        y_start = first_scan.y_points[0]
        velocity_start = first_scan.velocity_points[0]
        rotation_angle_start = first_scan.rotation_angle
        transition_start = Transition(
            x_start,
            y_start,
            self.z,
            velocity_start,
            self.time_start,
            reference_temperature,
            rotation_angle_start,
        )
        self.scan_list.insert(0, transition_start)

    def add_scan_to_layer(self, scan):
        self.scan_list.append(scan)
        self.time_end = scan.time_points[-1]


class AutoLayer(Layer):
    def __init__(
        self,
        x_size,
        y_size,
        z,
        time_start,
        path_type,
        hatch_spacing,
        laser_power,
        velocity,
        rotation_angle,
        time_step,
        reference_temperature,
    ):
        super().__init__(x_size, y_size, z, time_start)
        self.path_type = path_type
        self.hatch_spacing = hatch_spacing
        self.laser_power = laser_power
        self.velocity = velocity
        self.rotation_angle = rotation_angle
        self.reference_temperature = reference_temperature
        self._time_step = time_step

        self._quadrant = None
        self._num_scans = None
        self._rotation_matrix_passive = None
        self._build_rotation_matrices()

    def auto_generate_scans(
        self,
        offset_start,
        offset_end,
        hatch_offset_start,
        hatch_offset_end,
        time_offset,
        *args,
        **kwargs,
    ):
        (
            x_points_rotated,
            y_points_rotated,
            rotation_angle_list,
            scan_length,
        ) = self._get_rotated_scan_points(
            offset_start, offset_end, hatch_offset_start, hatch_offset_end
        )
        total_time = self.time_start + time_offset

        for x_start, y_start, rotation_angle in zip(
            x_points_rotated, y_points_rotated, rotation_angle_list
        ):
            scan = LaserScan(
                self.velocity,
                self.laser_power,
                self._time_step,
                scan_length,
                rotation_angle,
                self.reference_temperature,
            )
            scan.generate_path_points(
                x_start, y_start, self.z, total_time, xy_coord_type="local"
            )
            scan.generate_power_and_velocity_points(*args, **kwargs)
            self.add_scan_to_layer(scan)
            total_time = scan.time_points[-1] + time_offset

    def _get_rotated_scan_points(
        self, offset_start, offset_end, hatch_offset_start, hatch_offset_end
    ):
        corners = self._domain_corners()
        scan_start_corner_rotated = self._passive_rotation(corners[self._quadrant])
        scan_end_corner_rotated = self._passive_rotation(
            corners[(self._quadrant + 2) % 4]
        )
        start_corner_rotated = self._passive_rotation(corners[self._start_corner])
        end_corner_rotated = self._passive_rotation(corners[self._end_corner])
        layer_bounds = [
            start_corner_rotated[1] + hatch_offset_start,
            end_corner_rotated[1] + hatch_offset_end,
        ]
        scan_bounds = [
            scan_start_corner_rotated[0] - offset_start,
            scan_end_corner_rotated[0] + offset_start,
        ]
        layer_length = layer_bounds[1] - layer_bounds[0]
        scan_length = scan_bounds[1] - scan_bounds[0] + offset_end
        self._num_scans = int(np.ceil(layer_length / self.hatch_spacing) + 1)
        x_points_rotated = (scan_start_corner_rotated[0] - offset_start) * np.ones(
            (self._num_scans,)
        )
        y_points_rotated = (
            np.arange(self._num_scans) * self.hatch_spacing + layer_bounds[0]
        )
        rotation_angle_list = self.rotation_angle * np.ones((self._num_scans,))
        if self.path_type == "serpentine":
            ref_corner_serpentine = corners[(self._quadrant + 2) % 4]
            x_points_rotated[1::2] = (
                -self._passive_rotation(ref_corner_serpentine)[0] - offset_start
            )
            y_points_rotated[1::2] = -y_points_rotated[1::2]
            rotation_angle_list[1::2] = rotation_angle_list[1::2] + np.pi
        return x_points_rotated, y_points_rotated, rotation_angle_list, scan_length

    def _domain_corners(self):
        corners = {}
        corners[0] = [0, 0]
        corners[1] = [self.x_size, 0]
        corners[2] = [self.x_size, self.y_size]
        corners[3] = [0, self.y_size]
        return corners

    def _passive_rotation(self, points_to_rotate):
        return np.matmul(self._rotation_matrix_passive, points_to_rotate)

    def _build_rotation_matrices(self):
        c = np.cos(self.rotation_angle)
        s = np.sin(self.rotation_angle)
        tol = 1.0e-10
        if np.abs(c) < tol:
            c = 0.0
        if np.abs(s) < tol:
            s = 0.0
        if c > 0 and s >= 0:
            self._quadrant = 0
        elif c <= 0 and s > 0:
            self._quadrant = 1
        elif c < 0 and s <= 0:
            self._quadrant = 2
        else:
            self._quadrant = 3
        self._start_corner = (self._quadrant + 1) % 4
        self._end_corner = (self._quadrant + 3) % 4
        self._rotation_matrix_passive = np.array([[c, s], [-s, c]])


class LaserScan:
    def __init__(
        self,
        velocity,
        laser_power,
        time_step,
        distance,
        rotation_angle,
        reference_temperature,
    ):
        self._velocity = velocity
        self._laser_power = laser_power
        self._time_step = time_step
        self._distance = distance
        self.rotation_angle = rotation_angle

        self._path_step = time_step * velocity
        self.x_points = None
        self.y_points = None
        self.z_points = None
        self.power_points = None
        self.velocity_points = None
        self.time_points = None
        self.num_points = int(np.ceil(distance / self._path_step) + 1)
        self.keyhole_points = np.zeros(self.num_points)
        self.temperature_points = reference_temperature * np.ones(self.num_points)

    def generate_path_points(
        self, x_start, y_start, z, time_start, xy_coord_type="global"
    ):
        path_points = np.arange(0, self._distance + self._path_step, self._path_step)
        self._rotate_and_shift_path_points(path_points, x_start, y_start, xy_coord_type)
        self._generate_time_points(time_start)
        self.z_points = z * np.ones((self.num_points,))

    def generate_power_and_velocity_points(self, power_flag=None, *args, **kwargs):
        self.power_points = self._laser_power * np.ones((self.num_points,))
        self.velocity_points = self._velocity * np.ones((self.num_points,))

    def insert_keyhole_porosity(
        self,
        pore_prob,
        dist_type,
        *args,
        x_bounds=None,
        y_bounds=None,
        z_bounds=None,
        **kwargs,
    ):
        if z_bounds is not None:
            if np.logical_or(
                self.z_points[0] < z_bounds[0], self.z_points[0] > z_bounds[1]
            ):
                return
        num_eligible_points, eligible_idx = self.get_points_in_region(
            x_bounds, y_bounds
        )
        rng = np.random.default_rng()
        if dist_type == "normal":
            pore_dist = rng.normal
        pore_check = np.ones((self.num_points,))
        pore_check[eligible_idx] = rng.random(num_eligible_points)
        pore_idx = np.nonzero(pore_check < pore_prob)[0]
        num_pores = len(pore_idx)
        if num_pores > 0:
            pore_sizes = np.array([pore_dist(*args, size=num_pores)])
            pore_sizes = np.reshape(pore_sizes, (np.size(pore_sizes),))
            pore_sizes[pore_sizes < 0] = 0
            self.keyhole_points[pore_idx] = pore_sizes

    def get_points_in_region(self, x_bounds, y_bounds):
        if x_bounds is not None:
            region_idx_x = np.logical_and(
                self.x_points >= x_bounds[0], self.x_points <= x_bounds[1]
            )
        else:
            region_idx_x = np.ones((self.num_points), dtype=bool)
        if y_bounds is not None:
            region_idx_y = np.logical_and(
                self.y_points >= y_bounds[0], self.y_points <= y_bounds[1]
            )
        else:
            region_idx_y = np.ones((self.num_points), dtype=bool)
        region_idx = np.logical_and(region_idx_x, region_idx_y)
        num_region_points = len(np.nonzero(region_idx)[0])
        return num_region_points, region_idx

    def _rotate_and_shift_path_points(
        self, path_points, x_start, y_start, xy_coord_type
    ):
        c = np.cos(self.rotation_angle)
        s = np.sin(self.rotation_angle)
        tol = 1.0e-10
        if np.abs(c) < tol:
            c = 0.0
        if np.abs(s) < tol:
            s = 0.0
        if xy_coord_type == "local":
            self.x_points = c * path_points + c * x_start - s * y_start
            self.y_points = s * path_points + s * x_start + c * y_start
        else:
            self.x_points = c * path_points + x_start
            self.y_points = s * path_points + y_start
        max_x_jump = np.max(self.x_points[1:] - self.x_points[:-1])
        max_y_jump = np.max(self.y_points[1:] - self.y_points[:-1])
        if max_x_jump > 1 or max_y_jump > 1:
            raise ValueError(
                "Reduce time step: laser jumps >1 voxel in one step! "
                f"global x-direction jump = {max_x_jump}, "
                f"global y-direction jump = {max_y_jump}, "
                f"time step = {self._time_step}"
            )

    def _generate_time_points(self, time_start):
        time_stop = time_start + (self.num_points - 1) * self._time_step
        self.time_points = np.linspace(time_start, time_stop, self.num_points)


class Transition:
    def __init__(self, x, y, z, velocity, t, reference_temperature, rotation_angle):
        c = np.cos(rotation_angle)
        if np.abs(c) < 1e-10:
            c = 0.0
        s = np.sin(rotation_angle)
        if np.abs(s) < 1e-10:
            s = 0.0

        self.x_points = np.array([x, x]) - c * 0.02
        self.y_points = np.array([y, y]) - s * 0.02
        self.z_points = np.array([z, z])
        self.time_points = np.array([t, t])
        self.power_points = np.array([0.0, 0.0])
        self.velocity_points = np.array([-1.0, velocity])
        self.keyhole_points = np.array([0.0, 0.0])
        self.temperature_points = np.array(
            [reference_temperature, reference_temperature]
        )

    def insert_keyhole_porosity(self, *args, **kwargs):
        pass

    def get_points_in_region(self, x_bounds, y_bounds):
        return 0, np.array([False, False])
