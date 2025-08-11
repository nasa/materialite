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

import numpy as np
from materialite.tensor import Order2SymmetricTensor


class LoadSchedule:
    def __init__(self, strain_increment, stress_increment, stress, stress_mask):
        self.strain_increment = strain_increment
        self.stress_increment = stress_increment
        self.stress = stress
        self.stress_mask = stress_mask
        if np.any(np.logical_and(stress_mask != 1, stress_mask != 0)):
            raise ValueError("stress mask can only have zeros and ones")

    @staticmethod
    def apply_stress_mask(strain, stress, stress_mask):
        return Order2SymmetricTensor(
            strain.components * (1 - stress_mask)
        ), Order2SymmetricTensor(stress.components * stress_mask)

    @classmethod
    def from_constant_rates(cls, strain_rate, stress_rate, stress_mask, start_time=0):
        strain_rate, stress_rate = cls.apply_stress_mask(
            strain_rate, stress_rate, stress_mask
        )
        f_dstrain = lambda t, dt: strain_rate * dt
        f_dstress = lambda t, dt: stress_rate * dt
        f_stress = lambda t, dt: stress_rate * (t + dt - start_time)
        return cls(f_dstrain, f_dstress, f_stress, stress_mask)

    @classmethod
    def from_constant_uniaxial_strain_rate(
        cls,
        magnitude=1.0,
        direction="z",
        start_time=0,
    ):
        strain_rates = {
            "x": [magnitude, 0, 0],
            "y": [0, magnitude, 0],
            "z": [0, 0, magnitude],
        }
        stress_masks = {"x": [0, 1, 1], "y": [1, 0, 1], "z": [1, 1, 0]}
        stress_mask = np.concatenate([stress_masks[direction], np.ones(3)])
        strain_rate = Order2SymmetricTensor(
            np.concatenate([strain_rates[direction], np.zeros(3)])
        )
        return cls.from_constant_rates(
            strain_rate, Order2SymmetricTensor.zero(), stress_mask, start_time
        )

    @classmethod
    def from_ramp(cls, times, strains, stresses, stress_mask):
        start_time = times[0]
        time_change = times[1] - start_time
        strains, stresses = cls.apply_stress_mask(strains, stresses, stress_mask)
        strain_rate = (strains[1] - strains[0]) / time_change
        stress_rate = (stresses[1] - stresses[0]) / time_change

        f_dstrain = lambda t, dt: strain_rate * dt
        f_dstress = lambda t, dt: stress_rate * dt
        f_stress = lambda t, dt: stress_rate * (t + dt - start_time) + stresses[0]

        return cls(f_dstrain, f_dstress, f_stress, stress_mask)

    @classmethod
    def from_cyclic_stress(
        cls,
        stress_amplitude,
        frequency,
        stress_mask=np.ones(6),
        start_time=0,
        ramp_rate=0,
    ):
        stress_amplitude = Order2SymmetricTensor(
            stress_amplitude.components * stress_mask
        )
        amplitude_function = (
            lambda t: stress_amplitude
            * (1 + ramp_rate * (t - start_time))
            * np.sin(2 * np.pi * frequency * (t - start_time))
        )
        f_dstrain = lambda t, dt: Order2SymmetricTensor.zero()
        f_dstress = lambda t, dt: amplitude_function(t + dt) - amplitude_function(t)
        f_stress = lambda t, dt: amplitude_function(t + dt)
        return cls(f_dstrain, f_dstress, f_stress, stress_mask)

    @classmethod
    def from_cyclic_strain(
        cls,
        strain_amplitude,
        frequency,
        stress_mask,
        start_time=0,
        ramp_rate=0,
    ):
        strain_amplitude = Order2SymmetricTensor(
            strain_amplitude.components * (1 - stress_mask)
        )
        amplitude_function = (
            lambda t: strain_amplitude
            * (1 + ramp_rate * (t - start_time))
            * np.sin(2 * np.pi * frequency * (t - start_time))
        )
        f_dstrain = lambda t, dt: amplitude_function(t + dt) - amplitude_function(t)
        f_dstress = lambda t, dt: Order2SymmetricTensor.zero()
        f_stress = lambda t, dt: Order2SymmetricTensor.zero()
        return cls(f_dstrain, f_dstress, f_stress, stress_mask)
