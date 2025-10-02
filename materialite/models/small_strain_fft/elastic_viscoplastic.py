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

import logging

import numpy as np

from materialite import Order2SymmetricTensor, Order4SymmetricTensor, Scalar, Vector


class ElasticViscoplastic:
    def __init__(
        self,
        stiffness,
        slip_systems,
        reference_slip_rate,
        rate_exponent,
        slip_resistance,
        hardening_function,
        hardening_properties,
    ):
        self.stiffness = stiffness
        self._ref_modulus = float(
            stiffness.directional_modulus(Vector([1, 0, 0])).components
        )
        self.slip_systems = slip_systems
        self.reference_slip_rate = Scalar(reference_slip_rate)
        self.rate_exponent = Scalar(rate_exponent)
        self.initial_slip_resistance = slip_resistance
        self.hardening_function = hardening_function
        self.hardening_properties = hardening_properties

        self.state_variables = dict()
        self.available_state_variables = [
            "slip_resistances",
            "plastic_strains",
            "slip_system_shear_strains",
        ]
        self._state_variable_info = None
        self._orientations = None
        self._logger = logging.getLogger("ElasticViscoplastic")

    def initialize(self, orientations):
        schmid_tensor = self.slip_systems.schmid_tensor.sym
        num_points = orientations.shape[0]
        num_slip_systems = schmid_tensor.shape[0]
        self._state_variable_info = dict(
            zip(
                self.available_state_variables,
                [
                    (Scalar, num_slip_systems),
                    (Order2SymmetricTensor, 0),
                    (Scalar, num_slip_systems),
                ],
            )
        )
        slip_resistances = (
            Scalar.zero().repeat((num_points, num_slip_systems))
            + self.initial_slip_resistance
        )
        plastic_strains = Order2SymmetricTensor.zero().repeat(num_points)
        slip_system_shear_strains = Scalar.zero().repeat((num_points, num_slip_systems))
        accumulated_slip = Scalar.zero().repeat(num_points)
        self.state_variables["orientations"] = orientations
        self.state_variables["stiffnesses"] = self.stiffness
        self.state_variables["schmid_tensors"] = schmid_tensor
        self.state_variables["schmid_outer_schmid"] = schmid_tensor.outer(schmid_tensor)
        self.state_variables["slip_resistances"] = slip_resistances
        self.state_variables["old_slip_resistances"] = slip_resistances.copy()
        self.state_variables["plastic_strains"] = plastic_strains
        self.state_variables["_plastic_strains"] = plastic_strains.copy()
        self.state_variables["_old_plastic_strains"] = plastic_strains.copy()
        self.state_variables["slip_system_shear_strains"] = slip_system_shear_strains
        self.state_variables["old_slip_system_shear_strains"] = (
            slip_system_shear_strains.copy()
        )
        self.state_variables["accumulated_slip"] = accumulated_slip
        self.state_variables["old_accumulated_slip"] = accumulated_slip.copy()

        return self.stiffness.to_specimen_frame(orientations)

    def calculate_stress_and_tangent(self, strains, guess_stresses, time_increment):
        TOLERANCE = 2.0e-5
        MAX_ITERATIONS = 60
        orientations = self.state_variables["orientations"]
        guess_stress_norms = guess_stresses.norm
        stresses = guess_stresses.to_crystal_frame(orientations)
        strains = strains.to_crystal_frame(orientations)

        old_plastic_strains = self.state_variables["_old_plastic_strains"]
        old_slip_resistances = self.state_variables["old_slip_resistances"]
        old_slip_system_shear_strains = self.state_variables[
            "old_slip_system_shear_strains"
        ]
        old_accumulated_slip = self.state_variables["old_accumulated_slip"]
        slip_resistances = self.state_variables["slip_resistances"]
        schmid_tensors = self.state_variables["schmid_tensors"]
        stiffnesses = self.state_variables["stiffnesses"]
        schmid_outer_schmid = self.state_variables["schmid_outer_schmid"]

        converged = False
        iteration = 0
        num_points = len(guess_stress_norms)
        converged_percent = 0.0
        while not converged:
            iteration += 1
            if converged_percent >= 0.5:
                not_converged = np.logical_not(conv_check)
                (
                    stresses[not_converged],
                    stress_diff[not_converged],
                    slip_resistances[not_converged],
                    plastic_strains[not_converged],
                    slip_increments[not_converged],
                    jacobians_inv[not_converged],
                ) = self._run_iteration(
                    stiffnesses,
                    slip_resistances[not_converged],
                    schmid_tensors,
                    schmid_outer_schmid,
                    stresses[not_converged],
                    old_plastic_strains[not_converged],
                    time_increment,
                    strains[not_converged],
                    old_accumulated_slip[not_converged],
                    old_slip_resistances[not_converged],
                )
            else:
                if iteration > 1:
                    del stress_diff, plastic_strains, slip_increments, jacobians_inv
                (
                    stresses,
                    stress_diff,
                    slip_resistances,
                    plastic_strains,
                    slip_increments,
                    jacobians_inv,
                ) = self._run_iteration(
                    stiffnesses,
                    slip_resistances,
                    schmid_tensors,
                    schmid_outer_schmid,
                    stresses,
                    old_plastic_strains,
                    time_increment,
                    strains,
                    old_accumulated_slip,
                    old_slip_resistances,
                )

            # Check convergence
            conv_check = np.logical_or(
                (stress_diff / guess_stress_norms).components <= TOLERANCE,
                stress_diff.components / self._ref_modulus < 1.0e-12,
            )
            converged_percent = len(np.nonzero(conv_check)[0]) / num_points
            converged = np.all(conv_check)

            if iteration == MAX_ITERATIONS and not converged:
                self._logger.info("Too many iterations in material loop")
                return None, None, converged

        d_sigma_d_epsilon = jacobians_inv @ stiffnesses
        self._logger.debug(f"material iterations: {iteration}")

        self.state_variables["_plastic_strains"] = plastic_strains
        self.state_variables["slip_resistances"] = slip_resistances
        self.state_variables["slip_system_shear_strains"] = (
            old_slip_system_shear_strains + slip_increments
        )
        self.state_variables["accumulated_slip"] = (
            old_accumulated_slip + slip_increments.abs.sum("s")
        )

        return (
            stresses.to_specimen_frame(orientations),
            d_sigma_d_epsilon.to_specimen_frame(orientations),
            iteration,
        )

    def _run_iteration(
        self,
        stiffnesses,
        slip_resistances,
        schmid_tensors,
        schmid_outer_schmid,
        stresses,
        old_plastic_strains,
        time_increment,
        strains,
        old_accumulated_slip,
        old_slip_resistances,
    ):
        prev_stresses = stresses.copy()

        # Resolved shear stress
        resolved_shear_stresses = stresses * schmid_tensors

        # Plastic slip rate
        prefactor = (
            self.reference_slip_rate
            * (resolved_shear_stresses / slip_resistances).abs
            ** (self.rate_exponent - 1)
            / slip_resistances
        )

        plastic_slip_rates = prefactor * resolved_shear_stresses

        # Plastic strain rate
        plastic_strains = (
            old_plastic_strains
            + (plastic_slip_rates * schmid_tensors).sum("s") * time_increment
        )

        # Derivative for N-R Jacobian
        jacobians_inv = (
            Order4SymmetricTensor.identity()
            + time_increment
            * stiffnesses
            @ (self.rate_exponent * prefactor * schmid_outer_schmid).sum("s")
        ).inv

        # Stress update
        stresses = prev_stresses - jacobians_inv @ (
            stresses - stiffnesses @ (strains - plastic_strains)
        )
        slip_resistances = self.hardening_function(
            self.hardening_properties,
            old_accumulated_slip,
            old_slip_resistances,
            plastic_slip_rates,
            time_increment,
        )
        stress_diff = (stresses - prev_stresses).norm
        slip_increments = plastic_slip_rates * time_increment

        return (
            stresses,
            stress_diff,
            slip_resistances,
            plastic_strains,
            slip_increments,
            jacobians_inv,
        )

    def update_state_variables(self):
        crystal_plastic_strains = self.state_variables["_plastic_strains"]
        self.state_variables["_old_plastic_strains"] = crystal_plastic_strains.copy()
        self.state_variables["plastic_strains"] = (
            crystal_plastic_strains.to_specimen_frame(
                self.state_variables["orientations"]
            )
        )
        self.state_variables["old_slip_resistances"] = self.state_variables[
            "slip_resistances"
        ].copy()
        self.state_variables["old_slip_system_shear_strains"] = self.state_variables[
            "slip_system_shear_strains"
        ].copy()
        self.state_variables["old_accumulated_slip"] = self.state_variables[
            "accumulated_slip"
        ].copy()

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

    def __repr__(self):
        return (
            f"ElasticViscoplastic(stiffness={self.stiffness.voigt}, \n"
            + f"slip_systems={self.slip_systems}, \n"
            + f"reference_slip_rate={self.reference_slip_rate.components}, "
            + f"rate_exponent={self.rate_exponent.components}, "
            + f"initial_slip_resistance={self.initial_slip_resistance}, "
            + f"hardening_function={self.hardening_function.__name__}, "
            + f"hardening_properties={self.hardening_properties})"
        )
