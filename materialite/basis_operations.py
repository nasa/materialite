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


# Not particularly liking calling this the natural basis but it is here for now
def natural_basis():
    r2 = 1 / np.sqrt(2)
    r3 = 1 / np.sqrt(3)
    r6 = 1 / np.sqrt(6)
    return np.array(
        [
            r3 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            r6 * np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]]),
            r2 * np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]]),
            r2 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
            r2 * np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
            r2 * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        ]
    )


def natural_product_basis():
    return np.einsum("mij, nkl -> mnijkl", natural_basis(), natural_basis())


def mandel_basis():
    r2 = 1 / np.sqrt(2)
    return np.array(
        [
            [[1.0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1.0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 1.0]],
            [[0, 0, 0], [0, 0, r2], [0, r2, 0]],
            [[0, 0, r2], [0, 0, 0], [r2, 0, 0]],
            [[0, r2, 0], [r2, 0, 0], [0, 0, 0]],
        ]
    )


def mandel_product_basis():
    return np.einsum("mij, nkl -> mnijkl", mandel_basis(), mandel_basis())


def mises_basis():
    r2 = 1 / np.sqrt(2)
    r3 = 1 / np.sqrt(3)
    r6 = 1 / np.sqrt(6)
    return np.array(
        [
            r2 * np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]]),
            r6 * np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]]),
            r2 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
            r2 * np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
            r2 * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
            r3 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ]
    )


def mises_product_basis():
    return np.einsum("mij, nkl -> mnijkl", mises_basis(), mises_basis())


def strain_voigt_basis():
    return np.array(
        [
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
            0.5 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
            0.5 * np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
            0.5 * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        ]
    )


def stress_voigt_basis():
    return np.array(
        [
            [[1.0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1.0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 1.0]],
            [[0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]],
            [[0, 0, 1.0], [0, 0, 0], [1.0, 0, 0]],
            [[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]],
        ]
    )


def strain_voigt_dual_basis():
    return stress_voigt_basis()


def stress_voigt_dual_basis():
    return strain_voigt_basis()


def voigt_product_basis():
    return np.einsum(
        "mij, nkl -> mnijkl", stress_voigt_basis(), strain_voigt_dual_basis()
    )


def voigt_dual_product_basis():
    return np.einsum(
        "mij, nkl -> mnijkl", stress_voigt_dual_basis(), strain_voigt_basis()
    )
