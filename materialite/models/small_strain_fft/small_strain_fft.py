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
#
# This model is based on the following references:
# [1]: T.W.J. de Geus, J. Vondrejc, J. Zeman, R.H.J. Peerlings, M.G.D. Geers. Finite strain FFT-based non-linear solvers made simple. Computer Methods in Applied Mechanics and Engineering, 2017, 318:412–430. doi: 10.1016/j.cma.2016.12.032, arXiv: 1603.08893
# [2]: J. Zeman, T.W.J. de Geus, J. Vondrejc, R.H.J. Peerlings, M.G.D. Geers. A finite element perspective on nonlinear FFT-based micromechanical simulations. International Journal for Numerical Methods in Engineering, 2017, 111(10):903–926. doi: 10.1002/nme.5481, arXiv: 1601.05970
# [3]: T.W.J. de Geus, J. Vondrejc. FFT-based non-linear solvers made simple (v0.1.0). Zenodo, 2019. doi:10.5281/zenodo.3550748. url:https://github.com/tdegeus/GooseFFT

import logging

import numpy as np
import scipy.sparse.linalg as sp
from materialite.models import Model
from materialite.models.small_strain_fft import Multiphase
from materialite.tensor import Order2SymmetricTensor, Order4SymmetricTensor
from numpy.fft import fftfreq, fftn, ifftn


class SmallStrainFFT(Model):
    def __init__(
        self,
        load_schedule,
        end_time=1,
        initial_time_increment=None,
        max_time_increment=None,
        min_time_increment=1.0e-9,
        start_time=0.0,
        constitutive_model=None,
    ):
        self.load_schedule = load_schedule
        self.end_time = end_time
        self.initial_time_increment = (
            self.end_time if initial_time_increment is None else initial_time_increment
        )
        self.max_time_increment = (
            self.initial_time_increment
            if max_time_increment is None
            else max_time_increment
        )
        self.min_time_increment = min_time_increment
        self.start_time = start_time
        self._constitutive_model = constitutive_model
        self._logger = logging.getLogger("SmallStrainFFT")

        self._sizes = None
        self._dimensions = None
        self._num_points = None
        self._projection = None

    def run(
        self,
        material,
        orientation_label="orientation",
        phase_label=None,
        output_variables=None,
        output_times=None,
        global_tolerance=1.0e-4,
        postprocessor=None,
    ):
        self._sizes = material.sizes
        self._dimensions = material.dimensions
        self._num_points = material.num_points
        time = self.start_time
        time_tolerance = self.min_time_increment / 2.0
        if output_times is not None:
            time_idx = 0
            output_times = np.append(output_times, np.inf)
            next_output_time = output_times[time_idx]
        else:
            next_output_time = np.inf
        new_state = dict()

        # Values that stay constant
        orientations = material.extract(orientation_label)
        self._projection = self._get_projection_operator()
        ndof = self._num_points * 6
        constitutive_model = self._get_constitutive_model(
            material, phase_label, self._constitutive_model
        )

        # Initialize
        tangent = constitutive_model.initialize(orientations)
        stress = Order2SymmetricTensor.zeros(self._num_points)
        strain = Order2SymmetricTensor.zeros(self._num_points)
        old_tangent = tangent.copy()
        old_strain = Order2SymmetricTensor.zeros(self._num_points)
        old_stress = Order2SymmetricTensor.zeros(self._num_points)
        mean_strain = Order2SymmetricTensor.zero()
        mean_applied_stress = Order2SymmetricTensor.zero()
        time_step_id = 0
        time_increment = self.initial_time_increment
        old_max_constit_iters = 1

        while time < (self.end_time - time_tolerance):
            self._logger.info(f"New increment: time {time} + {time_increment}")
            strain_increment = self.load_schedule.strain_increment(time, time_increment)
            stress_increment = self.load_schedule.stress_increment(time, time_increment)
            mean_applied_stress = self.load_schedule.stress(time, time_increment)
            strain = old_strain + strain_increment
            if not np.all(stress_increment.components < 1.0e-14):
                strain += tangent.mean().inv @ stress_increment
            mean_strain = strain.mean()
            mean_von_mises_strain = np.sqrt(2 / 3) * mean_strain.dev.norm.components
            if time_step_id == 0:
                guess_stress = stress_increment + tangent @ strain_increment
            else:
                guess_stress = old_stress + stress_increment

            TOLERANCE = global_tolerance
            MAX_ITERATIONS = 100
            converged = False
            iteration = 0
            cgtol = 1.0e-3
            all_constit_iters = []
            while not converged:
                iteration += 1
                if iteration == 1:
                    stress, tangent, constit_iters = (
                        constitutive_model.calculate_stress_and_tangent(
                            strain, guess_stress, time_increment
                        )
                    )
                    if not constit_iters:
                        tangent = old_tangent.copy()
                        break
                    b = self._apply_projection_tensor(mean_applied_stress - stress)
                    guess_stress = stress.copy()
                    all_constit_iters.append(constit_iters)
                self._logger.debug(f"global iteration {iteration}")
                # Solve A * x = b for fluctuation strains in Fourier space
                fluctuation_strain = self._get_fluctuation_strain(
                    tangent, ndof, b, cgtol
                )
                strain += fluctuation_strain

                stress, tangent, constit_iters = (
                    constitutive_model.calculate_stress_and_tangent(
                        strain, guess_stress, time_increment
                    )
                )
                if not constit_iters:
                    tangent = old_tangent.copy()
                    break
                all_constit_iters.append(constit_iters)

                # Update right-hand-side with new stresses
                b = self._apply_projection_tensor(mean_applied_stress - stress)

                # Check convergence
                strain_err = (
                    np.mean(fluctuation_strain.norm.components) / mean_von_mises_strain
                )
                mean_stress = stress.mean()
                mean_von_mises_stress = mean_stress.dev.norm.components * np.sqrt(3 / 2)
                stress_err = (
                    np.mean((stress - guess_stress).norm.components)
                    / mean_von_mises_stress
                )
                if strain_err < TOLERANCE and iteration > 1 and stress_err < TOLERANCE:
                    converged = True

                max_err = np.max([stress_err, strain_err])
                if max_err < 5.0e-7:
                    cgtol = 1.0e-12
                elif max_err < 1.0e-6:
                    cgtol = 1.0e-9
                elif max_err < 1.0e-2:
                    cgtol = 1.0e-5
                else:
                    cgtol = 1.0e-3

                guess_stress = stress.copy()
                self._logger.debug(f"stress err: {stress_err}")
                self._logger.debug(f"strain err: {strain_err}")

                if iteration == MAX_ITERATIONS and not converged:
                    self._logger.info("Too many iterations in macroloading loop")
                    break

            if converged:
                constitutive_model.update_state_variables()
                old_stress = stress.copy()
                old_strain = strain.copy()
                old_tangent = tangent.copy()
                time_step_id += 1
                time += time_increment
                if time >= (next_output_time - time_tolerance):
                    outputs = constitutive_model.generate_outputs(output_variables)
                    outputs.update({"stress": stress, "strain": strain, "time": time})
                    if postprocessor is not None:
                        outputs = postprocessor(outputs)
                    new_state[f"output_time_{time_idx}"] = outputs
                    time_idx += 1
                    next_output_time = output_times[time_idx]
                max_constit_iters = np.max(all_constit_iters)
                time_increment = self._get_new_time_increment(
                    time,
                    time_increment,
                    next_output_time,
                    max_constit_iters,
                    old_max_constit_iters,
                )
                old_max_constit_iters = np.max(
                    [old_max_constit_iters, max_constit_iters]
                )
            else:
                time_increment = time_increment / 2.0
                old_max_constit_iters = 1
                if time_increment < self.min_time_increment:
                    raise ValueError("min time increment reached")

        outputs = constitutive_model.postprocess(output_variables)
        outputs.update({"stress": stress, "strain": strain})
        new_material = material.create_fields(outputs)
        new_material.state.update(new_state)
        return new_material

    def _get_new_time_increment(
        self,
        time,
        time_increment,
        next_output_time,
        max_constit_iters,
        old_max_constit_iters,
    ):
        fixed_time_increment = np.min([next_output_time - time, self.end_time - time])
        eligible_increments = [self.max_time_increment, fixed_time_increment]
        if (
            max_constit_iters < 10 and max_constit_iters < 0.6 * old_max_constit_iters
        ) or max_constit_iters <= 6:
            eligible_increments.append(time_increment * 2.0)
        else:
            eligible_increments.append(time_increment)
        return np.min(eligible_increments)

    def _get_fluctuation_strain(self, tangent, ndof, b, cgtol=1.0e-5):
        cg_iters = 0

        def count_iters(arr):
            nonlocal cg_iters
            cg_iters += 1

        Ax = lambda deps: self._left_hand_side(deps, tangent)
        deps_vector, _ = sp.cg(
            rtol=cgtol,
            atol=1.0e-10,
            A=sp.LinearOperator(shape=(ndof, ndof), matvec=Ax, dtype="float"),
            b=b,
            callback=count_iters,
        )
        self._logger.debug(f"cg iterations: {cg_iters}")
        fluctuation_strain = Order2SymmetricTensor(
            deps_vector.reshape(self._num_points, 6), "p"
        )
        return fluctuation_strain

    def _get_projection_operator(self):
        Nx, Ny, Nz = self._dimensions
        qx = fftfreq(Nx)
        qy = fftfreq(Ny)
        qz = fftfreq(Nz)
        frequencies = np.array(np.meshgrid(qx, qy, qz, indexing="ij"))
        frequency_norms = np.linalg.norm(frequencies, axis=0)
        normalized_frequencies = np.divide(
            frequencies,
            frequency_norms,
            out=np.zeros_like(frequencies),
            where=frequency_norms != 0,
        )
        q = normalized_frequencies
        A = np.einsum("im, jxyz, lxyz -> xyzijlm", np.eye(3), q, q, optimize=True)
        Ghat4 = 0.5 * (
            A
            + np.einsum("xyzijlm -> xyzijml", A, optimize=True)
            + np.einsum("xyzijlm -> xyzjilm", A, optimize=True)
            + np.einsum("xyzijlm -> xyzjiml", A, optimize=True)
        ) - np.einsum("ixyz, jxyz, lxyz, mxyz -> xyzijlm", q, q, q, q, optimize=True)
        if Nx % 2 == 0:
            Ghat4[Nx // 2, :, :, :, :, :, :] = 0.0
        if Ny % 2 == 0:
            Ghat4[:, Ny // 2, :, :, :, :, :] = 0.0
        if Nz % 2 == 0:
            Ghat4[:, :, Nz // 2, :, :, :, :] = 0.0
        tensor = Order4SymmetricTensor.from_cartesian(
            Ghat4.reshape((self._num_points, 3, 3, 3, 3)), "p"
        )
        if not np.array_equal(self.load_schedule.stress_mask, np.zeros(6)):
            stress_bc = np.zeros((6, 6))
            indices = np.nonzero(self.load_schedule.stress_mask)[0]
            stress_bc[indices, indices] = 1
            tensor[0] = Order4SymmetricTensor(stress_bc)
        return tensor.components.reshape((*self._dimensions, 6, 6))

    def _left_hand_side(self, deps, d_sigma_d_epsilon):
        stress_guess = d_sigma_d_epsilon @ Order2SymmetricTensor(
            deps.reshape((self._num_points, 6)), "p"
        )
        return self._apply_projection_tensor(stress_guess)

    def _apply_projection_tensor(self, tensor):
        tensor_grid = tensor.components.reshape((*self._dimensions, 6))
        fourier_tensor = fftn(tensor_grid, axes=(0, 1, 2))
        fourier_product = np.einsum(
            "xyzij, xyzj -> xyzi", self._projection, fourier_tensor, optimize=True
        )
        return ifftn(fourier_product, axes=(0, 1, 2)).real.ravel()

    def _get_constitutive_model(self, material, phase_label, constitutive_model):
        if phase_label is not None and constitutive_model is not None:
            raise ValueError("cannot specify phase and constitutive model")
        if phase_label is None and constitutive_model is None:
            raise ValueError("must specify phase label or provide a constitutive model")
        if phase_label is not None:
            phase_fields = material.extract_regional_field(phase_label)
            phases = phase_fields[phase_label].to_list()
            models = phase_fields["constitutive_model"].to_list()
            phase_indices = material.get_region_indices(region_label=phase_label)
            return Multiphase(
                phases, models, [phase_indices[p] for p in phases], self._num_points
            )
        else:
            return constitutive_model
