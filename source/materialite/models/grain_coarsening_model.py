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

import numpy as np
from numba import jit
from scipy import spatial

from materialite.models import Model


class GrainCoarseningModel(Model):
    def __init__(
        self,
        max_grain_id=20,
        num_flip_attempts=100,
        neighborhood_distance=np.sqrt(3),
        seed=None,
        only_neighbors=True,
        auto_assign_fields=False,
        temperature_mobility_relationship=None,
        record_frequency=100000,
        kbTs=0.00000001,
    ):
        self.max_grain_id = max_grain_id
        self.num_flip_attempts = num_flip_attempts
        self.neighborhood_distance = neighborhood_distance
        self.seed = seed
        self.only_neighbors = only_neighbors
        self.record_frequency = record_frequency
        self.kbTs = kbTs
        self.temperature_mobility_relationship = temperature_mobility_relationship
        self.auto_assign_fields = auto_assign_fields

    def run(
        self,
        material,
        grain_label="grain",
        mobility_label="mobility",
        temperature_label="temperature",
    ):
        try:
            spin_field = material.extract(grain_label).astype(int)
        except KeyError:
            rng = np.random.default_rng()
            spin_field = rng.integers(0, self.max_grain_id, size=material.num_points)
        try:
            mobility_field = material.extract(mobility_label).astype(np.float32)
        except KeyError:
            mobility_field = np.ones(material.num_points, dtype=np.float32)
        try:
            temperature_field = material.extract(temperature_label).astype(np.float32)
        except KeyError:
            temperature_field = np.ones(material.num_points, dtype=np.float32)

        if hasattr(material, "neighbors") and hasattr(material, "num_neighbors"):
            neighbors = material.neighbors
            num_neighbors = material.num_neighbors
        else:
            neighbors, num_neighbors = _get_neighbors(
                material, self.neighborhood_distance
            )

        if callable(self.temperature_mobility_relationship):
            mobility_field = self.temperature_mobility_relationship(temperature_field)

        if self.only_neighbors:
            new_spin_field, successful_flip_attempts, recorded_energy = (
                self._do_neighbor_flips(
                    spin_field,
                    mobility_field,
                    neighbors,
                    num_neighbors,
                    self.num_flip_attempts,
                    material.num_points,
                    self.seed,
                    self.record_frequency,
                    self.kbTs,
                )
            )
        else:
            new_spin_field, successful_flip_attempts, recorded_energy = self._do_flips(
                spin_field,
                mobility_field,
                neighbors,
                num_neighbors,
                self.num_flip_attempts,
                self.max_spin,
                material.num_points,
                self.seed,
                self.record_frequency,
                self.kbTs,
            )

        new_material = material.create_fields(
            {
                grain_label: new_spin_field,
                mobility_label: mobility_field,
                temperature_label: temperature_field,
            }
        )

        new_material.state["successful_flip_attempts"] = successful_flip_attempts

        # Not a recommended way of attaching data to a Material - should be fields or state instead
        new_material.neighbors = neighbors
        new_material.num_neighbors = num_neighbors
        new_material.recorded_energy = recorded_energy
        new_material.successful_flip_attempts = successful_flip_attempts

        return new_material

    @staticmethod
    @jit(nopython=True)
    def _do_flips(
        spin_field,
        mobility_field,
        neighbors,
        num_neighbors,
        num_flip_attempts,
        max_spin,
        num_points,
        seed,
        record_frequency,
        kbTs,
    ):
        if seed is not None:
            random.seed(seed)
        successful_flip_attempts = 0
        recorded_energy = []

        for current_flip in range(num_flip_attempts):
            if current_flip % record_frequency == 0:
                recorded_energy.append(
                    _get_energy(num_points, neighbors, num_neighbors, spin_field)
                )

            sample_point = random.randrange(0, num_points, 1)

            # Check if mobility is greater than random value. If not, will
            # not accept so skip
            probability_check = random.random()
            if mobility_field[sample_point] < probability_check:
                continue

            test_spin = random.randint(0, max_spin)
            neighbor_range = num_neighbors[sample_point]
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
                dE = 0.5 * (current_energy - test_energy)
                if (
                    np.exp(-dE / kbTs) * mobility_field[sample_point]
                    > probability_check
                ):
                    spin_field[sample_point] = test_spin
                    successful_flip_attempts += 1

        return spin_field, successful_flip_attempts, recorded_energy

    @staticmethod
    @jit(nopython=True)
    def _do_neighbor_flips(
        spin_field,
        mobility_field,
        neighbors,
        num_neighbors,
        num_flip_attempts,
        num_points,
        seed,
        record_frequency,
        kbTs,
    ):
        if seed is not None:
            random.seed(seed)
        successful_flip_attempts = 0
        recorded_energy = []

        for current_flip in range(num_flip_attempts):

            if current_flip % record_frequency == 0:
                recorded_energy.append(
                    _get_energy(num_points, neighbors, num_neighbors, spin_field)
                )

            sample_point = random.randrange(0, num_points, 1)

            # Check if mobility is greater than random value. If not, will
            # not accept so skip
            probability_check = random.random()
            if mobility_field[sample_point] < probability_check:
                continue

            neighbor_range = num_neighbors[sample_point]
            eligible_spins = [
                spin_field[neighbors[sample_point, n]] for n in range(neighbor_range)
            ]
            test_spin_idx = random.randrange(0, neighbor_range)
            test_spin = eligible_spins[test_spin_idx]

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
    distances = np.roll(distances, -1)  # Not needed for this application

    num_neighbors = np.sum(neighbors != -1, axis=1)
    return neighbors, num_neighbors


# This assumes that material has a grain, and will break if it does not
def calculate_potts_energy(material, model: GrainCoarseningModel):
    if hasattr(material, "neighbors") and hasattr(material, "num_neighbors"):
        neighbors = material.neighbors
        num_neighbors = material.num_neighbors
    else:
        neighbors, num_neighbors = _get_neighbors(material, model.neighborhood_distance)

    num_points = material.num_points
    spin_field = material.extract("grain").astype(int)
    current_energy = _get_energy(num_points, neighbors, num_neighbors, spin_field)

    return current_energy


@jit(nopython=True)
def _get_energy(num_points, neighbors, num_neighbors, spin_field):
    current_energy = 0
    for sample_point in range(num_points):
        neighbor_range = num_neighbors[sample_point]
        for n in range(neighbor_range):
            spin_to_check = spin_field[neighbors[sample_point, n]]
            current_energy += spin_to_check != spin_field[sample_point]

    return current_energy
