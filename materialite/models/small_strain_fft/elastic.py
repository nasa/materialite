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


class Elastic:
    def __init__(self, stiffness, thermal_expansion_coefficients=None):
        self.stiffness = stiffness
        self.thermal_expansion_coefficients = thermal_expansion_coefficients
        self.state_variables = dict()
        self.available_state_variables = None
        self._state_variable_info = dict()

    def initialize(self, orientations):
        self.state_variables["stiffnesses"] = self.stiffness.to_specimen_frame(
            orientations
        )
        if self.thermal_expansion_coefficients is not None:
            self.state_variables["thermal_expansion_coefficients"] = (
                self.thermal_expansion_coefficients.to_specimen_frame(orientations)
            )
        return self.state_variables["stiffnesses"]

    def calculate_stress_and_tangent(self, strain, guess_stress, time_increment):
        stress = self.state_variables["stiffnesses"] @ strain
        return stress, self.state_variables["stiffnesses"], 1

    def update_state_variables(self):
        pass

    def postprocess(self, output_variables=None):
        outputs = self.generate_outputs(output_variables)
        self.state_variables = dict()
        return outputs

    def generate_outputs(self, output_variables=None):
        if output_variables is not None:
            outputs = {k: self.state_variables.get(k, None) for k in output_variables}
        else:
            outputs = dict()
        return outputs

    def calculate_thermal_strain(self, temperature_increment):
        if self.thermal_expansion_coefficients is None:
            raise ValueError(
                "Thermal expansion coefficients not defined for this material."
            )
        else:
            coefficients = self.state_variables["thermal_expansion_coefficients"]
            return coefficients * temperature_increment

    def __repr__(self):
        return f"Elastic(stiffness={np.round(self.stiffness.voigt, 3)})"
