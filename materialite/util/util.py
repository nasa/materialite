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
from numpy.linalg import norm


def get_random_unit_vector(dimension=3, rng=np.random.default_rng()):
    v = rng.normal(size=dimension)
    return v / norm(v)


def get_random_unit_vector_batch(
    num_vectors=100, dimension=3, rng=np.random.default_rng()
):
    v = rng.normal(size=(num_vectors, dimension))
    return v / norm(v, axis=1).reshape(num_vectors, 1)


def camel_to_snake(s):
    return "".join(["_" + c.lower() if c.isupper() else c for c in s]).lstrip("_")


def power_of_two_below(value):
    power = 0
    while value >= 2**power and power < 100:
        power += 1
    return 2 ** (power - 1)


def cartesian_grid(dimensions):
    x, y, z = np.mgrid[0 : dimensions[0], 0 : dimensions[1], 0 : dimensions[2]]
    return np.c_[np.ravel(x), np.ravel(y), np.ravel(z)]


def get_random_orthogonal_matrix(dimension=3, rng=np.random.default_rng()):
    basis = []
    for _ in range(dimension):
        e = get_random_unit_vector(dimension=dimension, rng=rng)
        for b in basis:
            e -= np.dot(b, e) * b

        basis.append(e / norm(e))

    return np.array(basis)


def repeat_data(data, x_size, y_size, z_size):
    ones = np.ones((data.shape[0], 3))
    data_periodic = [
        data,
        data + np.array([-x_size, -y_size, -z_size]) * ones,
        data + np.array([0, -y_size, -z_size]) * ones,
        data + np.array([x_size, -y_size, -z_size]) * ones,
        data + np.array([-x_size, 0, -z_size]) * ones,
        data + np.array([0, 0, -z_size]) * ones,
        data + np.array([x_size, 0, -z_size]) * ones,
        data + np.array([-x_size, y_size, -z_size]) * ones,
        data + np.array([0, y_size, -z_size]) * ones,
        data + np.array([x_size, y_size, -z_size]) * ones,
        data + np.array([-x_size, -y_size, 0]) * ones,
        data + np.array([0, -y_size, 0]) * ones,
        data + np.array([x_size, -y_size, 0]) * ones,
        data + np.array([-x_size, 0, 0]) * ones,
        data + np.array([x_size, 0, 0]) * ones,
        data + np.array([-x_size, y_size, 0]) * ones,
        data + np.array([0, y_size, 0]) * ones,
        data + np.array([x_size, y_size, 0]) * ones,
        data + np.array([-x_size, -y_size, z_size]) * ones,
        data + np.array([0, -y_size, z_size]) * ones,
        data + np.array([x_size, -y_size, z_size]) * ones,
        data + np.array([-x_size, 0, z_size]) * ones,
        data + np.array([0, 0, z_size]) * ones,
        data + np.array([x_size, 0, z_size]) * ones,
        data + np.array([-x_size, y_size, z_size]) * ones,
        data + np.array([0, y_size, z_size]) * ones,
        data + np.array([x_size, y_size, z_size]) * ones,
    ]
    return np.concatenate(data_periodic, axis=0)
