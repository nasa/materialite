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
from materialite.models import Model
from materialite.tensor import Order2SymmetricTensor


class TaylorModel(Model):
    def __init__(
        self, applied_strain, critical_resolved_shear_stress=1.0, mask_label=None
    ):
        self.applied_strain = self._verify_applied_strain(applied_strain)
        self.critical_resolved_shear_stress = critical_resolved_shear_stress
        self.mask_label = mask_label
        self._vertices = self._get_yield_vertices()

        self._orientations = None
        self._mask = None

    def run(self, material, orientation_label="orientation"):
        # Input
        material = self.verify_material(material, [orientation_label])
        self._orientations = material.extract(orientation_label)
        self._mask = (
            np.ones(material.num_points)
            if self.mask_label is None
            else material.extract(self.mask_label)
        )

        # Calculations
        crystal_strains = self.applied_strain.to_crystal_frame(self._orientations)
        crystal_stresses, taylor_factors = (
            self._get_crystal_stresses_and_taylor_factors(crystal_strains)
        )
        stresses = crystal_stresses.to_specimen_frame(self._orientations)
        stresses, taylor_factors = self._apply_mask(stresses, taylor_factors)
        stresses = stresses * self.critical_resolved_shear_stress

        # Output
        new_material = material.create_fields(
            {
                "stress": list(stresses),
                "taylor_factor": taylor_factors,
            }
        )
        new_material.state["average_taylor_factor"] = np.mean(taylor_factors)
        return new_material

    def _get_crystal_stresses_and_taylor_factors(self, crystal_strains):
        work = crystal_strains * self._vertices
        max_work_idx = np.argmax(work.abs.components, axis=1)
        max_work = np.take_along_axis(
            work.components, max_work_idx[:, np.newaxis], axis=1
        ).squeeze()
        crystal_stresses = Order2SymmetricTensor(
            self._vertices[max_work_idx].components, "p"
        )
        negatives = max_work < 0.0
        crystal_stresses[negatives] = -crystal_stresses[negatives]
        effective_applied_strain = (
            np.sqrt(2.0 / 3.0) * self.applied_strain.norm.components
        )
        taylor_factors = np.abs(max_work) / effective_applied_strain
        return crystal_stresses, taylor_factors

    def _apply_mask(self, stresses, taylor_factors):
        not_mask = np.logical_not(self._mask)
        stresses[not_mask] = 0.0
        taylor_factors[not_mask] = 0.0
        return stresses, taylor_factors

    @staticmethod
    def _get_yield_vertices():
        p13 = 1.0 / 3.0
        m13 = -p13
        p23 = 2.0 / 3.0
        m16 = -1.0 / 6.0
        p12 = 0.5
        m12 = -p12
        vertices = np.array(
            [
                [p23, m13, m13, 0.0, 0.0, 0.0],
                [m13, p23, m13, 0.0, 0.0, 0.0],
                [m13, m13, p23, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, p12, p12, p12],
                [0.0, 0.0, 0.0, m12, p12, p12],
                [0.0, 0.0, 0.0, p12, m12, p12],
                [0.0, 0.0, 0.0, p12, p12, m12],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [p13, m16, m16, 0.0, p12, p12],
                [p13, m16, m16, 0.0, m12, m12],
                [p13, m16, m16, 0.0, p12, m12],
                [p13, m16, m16, 0.0, m12, p12],
                [m16, p13, m16, p12, 0.0, p12],
                [m16, p13, m16, m12, 0.0, m12],
                [m16, p13, m16, m12, 0.0, p12],
                [m16, p13, m16, p12, 0.0, m12],
                [m16, m16, p13, p12, p12, 0.0],
                [m16, m16, p13, m12, m12, 0.0],
                [m16, m16, p13, p12, m12, 0.0],
                [m16, m16, p13, m12, p12, 0.0],
                [0.0, p12, m12, p12, 0.0, 0.0],
                [0.0, p12, m12, m12, 0.0, 0.0],
                [m12, 0.0, p12, 0.0, p12, 0.0],
                [m12, 0.0, p12, 0.0, m12, 0.0],
                [p12, m12, 0.0, 0.0, 0.0, p12],
                [p12, m12, 0.0, 0.0, 0.0, m12],
            ]
        ) * np.sqrt(6)
        return Order2SymmetricTensor.from_stress_voigt(vertices, "s")

    @staticmethod
    def _verify_applied_strain(applied_strain):
        if not isinstance(applied_strain, Order2SymmetricTensor):
            raise ValueError("applied strain must be a SymmetricTensor")
        if not np.allclose(applied_strain.trace.components, 0):
            raise ValueError("applied strain must be traceless")
        return applied_strain
