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

from materialite import Scalar


def _get_slip_increments(plastic_slip_rates, time_increment, properties):
    if "s" in plastic_slip_rates.dims_str:
        hardening_matrix = properties.get("hardening_matrix", None)
        if hardening_matrix is None:
            slip_rates = plastic_slip_rates.abs.sum("s")
        else:
            hardening_matrix = Scalar(hardening_matrix, "sa")
            slip_rates = (hardening_matrix * plastic_slip_rates.abs).sum("s")
            slip_rates = Scalar(slip_rates, "ps")
    else:
        slip_rates = plastic_slip_rates.abs
    return slip_rates * time_increment


def perfect_plasticity(
    properties,
    old_accumulated_slip,
    old_slip_resistances,
    plastic_slip_rates,
    time_increment,
):
    return old_slip_resistances


def linear(
    properties,
    old_accumulated_slip,
    old_slip_resistances,
    plastic_slip_rates,
    time_increment,
):
    slip_increments = _get_slip_increments(
        plastic_slip_rates, time_increment, properties
    )
    return old_slip_resistances + properties["hardening_rate"] * slip_increments


def voce(
    properties,
    old_accumulated_slip,
    old_slip_resistances,
    plastic_slip_rates,
    time_increment,
):
    slip_increments = _get_slip_increments(
        plastic_slip_rates, time_increment, properties
    )
    theta0 = properties["theta_0"]
    theta1 = properties["theta_1"]
    tau1 = properties["tau_1"]
    ratio = np.abs(theta0 / tau1)
    exp_initial = Scalar(
        np.exp(-ratio * old_accumulated_slip.components), old_accumulated_slip.dims_str
    )
    exp_change = Scalar(
        np.exp(-ratio * slip_increments.components), slip_increments.dims_str
    )
    slip_resistances = (
        old_slip_resistances
        + theta1 * slip_increments
        - (ratio * tau1 - theta1) / ratio * exp_initial * (exp_change - 1)
        - theta1
        / ratio
        * exp_initial
        * (
            exp_change * (1 + ratio * (old_accumulated_slip + slip_increments))
            - 1
            - ratio * old_accumulated_slip
        )
    )
    return slip_resistances


def armstrong_frederick(
    properties,
    old_accumulated_slip,
    old_slip_resistances,
    plastic_slip_rates,
    time_increment,
):
    slip_increments = _get_slip_increments(
        plastic_slip_rates, time_increment, properties
    )
    hardening = properties["direct_hardening"]
    dynamic_recovery = properties["dynamic_recovery"]
    return old_slip_resistances + (
        hardening - dynamic_recovery * old_slip_resistances
    ) * slip_increments / (1 + dynamic_recovery * slip_increments)
