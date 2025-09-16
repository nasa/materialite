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

from materialite import Order2SymmetricTensor, Order4SymmetricTensor


class Multiphase:
    def __init__(self, phases, models, indices, num_points):
        self.phases = phases
        self.models = models
        self.indices = indices
        self.num_points = num_points
        self.available_state_variables = []
        for m in self.models:
            new_state_variables = m.available_state_variables
            if new_state_variables is not None:
                self.available_state_variables += m.available_state_variables
        self._output_variable_info = None

    def initialize(self, orientations):
        tangent = Order4SymmetricTensor.zero().repeat(self.num_points)
        for m, i in zip(self.models, self.indices):
            tangent_i = m.initialize(orientations[i])
            tangent[i] = tangent_i
        return tangent

    def calculate_stress_and_tangent(self, strain, guess_stress, time_increment):
        stress = Order2SymmetricTensor.zero().repeat(self.num_points)
        tangent = Order4SymmetricTensor.zero().repeat(self.num_points)
        iterations = []
        for m, i in zip(self.models, self.indices):
            stress_i, tangent_i, converged = m.calculate_stress_and_tangent(
                strain[i], guess_stress[i], time_increment
            )
            if not converged:
                return None, None, converged
            stress[i] = stress_i
            tangent[i] = tangent_i
            iterations.append(converged)
        return stress, tangent, np.max(iterations)

    def update_state_variables(self):
        for m in self.models:
            m.update_state_variables()

    def postprocess(self, output_variables):
        outputs = self.generate_outputs(output_variables)
        for m in self.models:
            m.state_variables = dict()
        return outputs

    def generate_outputs(self, output_variables=None):
        outputs = dict()
        if output_variables is None:
            return outputs
        if self._output_variable_info is None:
            self._output_variable_info = dict()
            for v in output_variables:
                if v not in self.available_state_variables:
                    raise ValueError(
                        f"output variable {v} is not available in any constitutive models in this simulation"
                    )
                shape_v = 0
                for m in self.models:
                    type_vm, shape_vm = m._state_variable_info.get(v, (None, None))
                    if type_vm is None:
                        continue
                    shape_v = shape_vm if shape_vm > shape_v else shape_v
                    type_v = type_vm
                shape_v = [self.num_points, shape_v] if shape_v > 0 else self.num_points
                self._output_variable_info[v] = (type_v, shape_v)
        for v in output_variables:
            type_v, shape_v = self._output_variable_info[v]
            data_v = type_v.zero().repeat(shape_v)
            for m, i in zip(self.models, self.indices):
                data_vm = m.state_variables.get(v, None)
                if data_vm is None:
                    continue
                try:
                    data_v[i] = data_vm
                except ValueError:
                    data_v[i, : data_vm.shape[-1]] = data_vm
            outputs[v] = data_v
        return outputs
