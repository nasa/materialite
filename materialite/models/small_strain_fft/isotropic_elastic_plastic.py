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
from materialite import Order2SymmetricTensor, Order4SymmetricTensor, Scalar


class IsotropicElasticPlastic:
    def __init__(
        self,
        modulus,
        shear_modulus,
        yield_stress,
        hardening_function,
        hardening_properties,
    ):
        self.modulus = modulus
        self.shear_modulus = shear_modulus
        self.bulk_modulus = (
            shear_modulus * modulus / (3 * (3 * shear_modulus - modulus))
        )
        self.stiffness = Order4SymmetricTensor.from_isotropic_constants(
            modulus, shear_modulus
        )

        self.yield_stress = Scalar(yield_stress)
        self.hardening_function = hardening_function
        self.hardening_properties = hardening_properties
        self.state_variables = dict()
        self.available_state_variables = [
            "yield_stresses",
            "plastic_strains",
            "eq_plastic_strains",
        ]
        self._state_variable_info = dict(
            zip(
                self.available_state_variables,
                [
                    (Scalar, 0),
                    (Order2SymmetricTensor, 0),
                    (Scalar, 0),
                ],
            )
        )

    def initialize(self, orientations):
        num_points = len(orientations)
        stiffnesses = Order4SymmetricTensor(
            np.tile(self.stiffness.components, (num_points, 1, 1))
        )
        self.state_variables["stiffnesses"] = stiffnesses
        self.state_variables["plastic_strains"] = Order2SymmetricTensor.zero().repeat(
            num_points
        )
        self.state_variables["old_plastic_strains"] = (
            Order2SymmetricTensor.zero().repeat(num_points)
        )
        self.state_variables["eq_plastic_strains"] = Scalar(0.0).repeat(num_points)
        self.state_variables["old_eq_plastic_strains"] = Scalar(0.0).repeat(num_points)
        self.state_variables["yield_stresses"] = self.yield_stress
        self.state_variables["old_yield_stresses"] = self.yield_stress
        return stiffnesses

    def calculate_stress_and_tangent(self, strains, guess_stresses, time_increment):
        TOLERANCE = 1.0e-7
        MAX_ITERATIONS = 100
        old_plastic_strains = self.state_variables["old_plastic_strains"]
        old_eq_plastic_strains = self.state_variables["old_eq_plastic_strains"]
        old_yield_stresses = self.state_variables["old_yield_stresses"]
        yield_stresses = old_yield_stresses.copy()
        stiffnesses = self.state_variables["stiffnesses"]

        elastic_predictor_strains = strains - old_plastic_strains
        dev_elastic_predictor_strains = elastic_predictor_strains.dev
        dev_predictor_stresses = (
            2.0 * self.shear_modulus * dev_elastic_predictor_strains
        )
        mean_stresses = self.bulk_modulus * elastic_predictor_strains.trace
        mises_predictor_stresses = np.sqrt(1.5) * dev_predictor_stresses.norm
        stress_ratios = dev_predictor_stresses / mises_predictor_stresses
        jacobians_inv = -1.0 / (3.0 * self.shear_modulus)
        elastic_idx = mises_predictor_stresses.components <= yield_stresses.components
        if np.all(elastic_idx):
            stresses = (
                dev_predictor_stresses
                + mean_stresses * Order2SymmetricTensor.identity()
            )
            return stresses, stiffnesses, 1

        eq_plastic_strain_increments = Scalar(0.0)

        converged = False
        iteration = 0
        while not converged:
            iteration += 1
            residuals = (
                mises_predictor_stresses
                - 3.0 * self.shear_modulus * eq_plastic_strain_increments
                - yield_stresses
            )
            residuals[residuals.components <= 0.0] = Scalar(0.0)
            prev_eq_plastic_strain_increments = eq_plastic_strain_increments.copy()
            eq_plastic_strain_increments = (
                eq_plastic_strain_increments - jacobians_inv * residuals
            )
            yield_stresses = self.hardening_function(
                self.hardening_properties,
                old_eq_plastic_strains,
                old_yield_stresses,
                eq_plastic_strain_increments / time_increment,
                time_increment,
            )
            strain_diff = (
                eq_plastic_strain_increments - prev_eq_plastic_strain_increments
            )
            converged = np.all(strain_diff.abs.components < TOLERANCE)

            if iteration == MAX_ITERATIONS and not converged:
                print("Too many iterations in material loop")
                return None, None, converged
        plastic_strains = (
            old_plastic_strains + 1.5 * stress_ratios * eq_plastic_strain_increments
        )
        self.state_variables["eq_plastic_strains"] = (
            old_eq_plastic_strains + eq_plastic_strain_increments
        )
        self.state_variables["plastic_strains"] = plastic_strains
        self.state_variables["yield_stresses"] = yield_stresses

        ratio = 3.0 * self.shear_modulus / mises_predictor_stresses
        dev_stresses = (
            1.0 - ratio * eq_plastic_strain_increments
        ) * dev_predictor_stresses
        stresses = dev_stresses + mean_stresses * Order2SymmetricTensor.identity()

        i_outer_i = Order2SymmetricTensor.identity().outer(
            Order2SymmetricTensor.identity()
        )
        dev_identity = Order4SymmetricTensor.identity() - i_outer_i / 3.0
        stress_outer_stress = dev_predictor_stresses.outer(dev_predictor_stresses)
        term1 = 2.0 * self.shear_modulus * ratio * eq_plastic_strain_increments
        term2 = ratio**2 * (
            eq_plastic_strain_increments / mises_predictor_stresses
            - 1 / (3 * self.shear_modulus)
        )
        d_sigma_d_epsilon = (
            stiffnesses - term1 * dev_identity + term2 * stress_outer_stress
        )
        d_sigma_d_epsilon[elastic_idx] = stiffnesses[elastic_idx]

        return stresses, d_sigma_d_epsilon, iteration

    def update_state_variables(self):
        self.state_variables["old_plastic_strains"] = self.state_variables[
            "plastic_strains"
        ].copy()
        self.state_variables["old_eq_plastic_strains"] = self.state_variables[
            "eq_plastic_strains"
        ].copy()
        self.state_variables["old_yield_stresses"] = self.state_variables[
            "yield_stresses"
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
            f"IsotropicElasticPlastic(modulus={self.modulus}, "
            + f"shear_modulus={self.shear_modulus}, "
            + f"yield_stress={self.yield_stress.components}, "
            + f"hardening_function={self.hardening_function.__name__}, "
            + f"hardening_properties={self.hardening_properties})"
        )
