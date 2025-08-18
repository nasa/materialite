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

from materialite.models import Model


class DecoupledCrystalElasticity(Model):
    def __init__(self, applied_strain=None):
        self.applied_strain = applied_strain

    def run(
        self, material, stiffness_label="stiffness", orientation_label="orientation"
    ):
        # Inputs
        material = self.verify_material(material, [stiffness_label, orientation_label])
        stiffnesses = material.extract(stiffness_label)
        orientations = material.extract(orientation_label)

        # Do calculations
        stresses = self._calculate_stresses(stiffnesses, orientations)

        # Output
        return material.create_fields({"stress": stresses})

    def _calculate_stresses(self, stiffnesses, orientations):
        crystal_strains = self.applied_strain.to_crystal_frame(orientations)
        crystal_stresses = stiffnesses @ crystal_strains
        return crystal_stresses.to_specimen_frame(orientations)
