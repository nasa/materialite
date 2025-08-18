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
from materialite.tensor import Vector
from numpy.linalg import norm


class SlipSystem:
    def __init__(self, normals, directions):
        self.normal = normals.unit
        self.direction = directions.unit

    def __len__(self):
        return len(self.normal)

    def __repr__(self):
        return (
            "SlipSystem(normals: "
            + str(np.round(self.normal.components, 3))
            + ",\n directions: "
            + str(np.round(self.direction.components, 3))
            + ")"
        )

    @classmethod
    def octahedral(cls):
        normals = [
            [1, 1, -1],
            [1, 1, -1],
            [1, 1, -1],
            [1, -1, -1],
            [1, -1, -1],
            [1, -1, -1],
            [1, -1, 1],
            [1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        directions = [
            [0, 1, 1],
            [1, 0, 1],
            [1, -1, 0],
            [0, 1, -1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, -1],
            [1, 1, 0],
            [0, 1, -1],
            [1, 0, -1],
            [1, -1, 0],
        ]
        return cls(Vector(normals, "s"), Vector(directions, "s"))

    @classmethod
    def basal(cls):
        normals = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
        normals = convert_hexagonal_planes_to_cartesian(normals)
        directions = [[2, -1, -1, 0], [-1, 2, -1, 0], [-1, -1, 2, 0]]
        directions = convert_hexagonal_directions_to_cartesian(directions)
        return cls(Vector(normals, "s"), Vector(directions, "s"))

    @classmethod
    def prismatic(cls):
        normals = [[1, 0, -1, 0], [0, -1, 1, 0], [-1, 1, 0, 0]]
        normals = convert_hexagonal_planes_to_cartesian(normals)
        directions = [[-1, 2, -1, 0], [2, -1, -1, 0], [-1, -1, 2, 0]]
        directions = convert_hexagonal_directions_to_cartesian(directions)
        return cls(Vector(normals, "s"), Vector(directions, "s"))

    @classmethod
    def pyramidal_a(cls, c_over_a=1.633):
        normals = [
            [1, 0, -1, 1],
            [0, -1, 1, 1],
            [-1, 1, 0, 1],
            [-1, 0, 1, 1],
            [0, 1, -1, 1],
            [1, -1, 0, 1],
        ]
        normals = convert_hexagonal_planes_to_cartesian(normals, c_over_a)
        directions = [
            [-1, 2, -1, 0],
            [2, -1, -1, 0],
            [-1, -1, 2, 0],
            [-1, 2, -1, 0],
            [2, -1, -1, 0],
            [1, 1, -2, 0],
        ]
        directions = convert_hexagonal_directions_to_cartesian(directions, c_over_a)
        return cls(Vector(normals, "s"), Vector(directions, "s"))

    @classmethod
    def pyramidal_ca(cls, c_over_a=1.633):
        normals = [
            [1, 0, -1, 1],
            [1, 0, -1, 1],
            [0, -1, 1, 1],
            [0, -1, 1, 1],
            [-1, 1, 0, 1],
            [-1, 1, 0, 1],
            [-1, 0, 1, 1],
            [-1, 0, 1, 1],
            [0, 1, -1, 1],
            [0, 1, -1, 1],
            [1, -1, 0, 1],
            [1, -1, 0, 1],
        ]
        normals = convert_hexagonal_planes_to_cartesian(normals, c_over_a)
        directions = [
            [-1, -1, 2, 3],
            [-2, 1, 1, 3],
            [1, 1, -2, 3],
            [-1, 2, -1, 3],
            [2, -1, -1, 3],
            [1, -2, 1, 3],
            [2, -1, -1, 3],
            [1, 1, -2, 3],
            [-1, -1, 2, 3],
            [1, -2, 1, 3],
            [-2, 1, 1, 3],
            [-1, 2, -1, 3],
        ]
        directions = convert_hexagonal_directions_to_cartesian(directions, c_over_a)
        return cls(Vector(normals, "s"), Vector(directions, "s"))

    @property
    def schmid_tensor(self):
        return self.direction.outer(self.normal)

    def max_schmid_factor(self, load_direction=Vector([0, 0, 1.0])):
        return (
            self.schmid_tensor * load_direction.unit.outer(load_direction.unit)
        ).abs.max("s")

    def concatenate(self, other):
        normals = np.concatenate(
            [self.normal.components, other.normal.components], axis=0
        )
        directions = np.concatenate(
            [self.direction.components, other.direction.components], axis=0
        )
        return SlipSystem(Vector(normals, "s"), Vector(directions, "s"))


def convert_hexagonal_directions_to_cartesian(directions, c_over_a=1.633):
    directions = np.atleast_2d(directions)
    u = directions[:, 0] * 3 / 2
    v = (directions[:, 0] / 2 + directions[:, 1]) * np.sqrt(3)
    w = directions[:, 3] * c_over_a
    directions_cartesian = np.c_[u, v, w]
    return directions_cartesian / norm(directions_cartesian, axis=1, keepdims=True)


def convert_hexagonal_planes_to_cartesian(planes, c_over_a=1.633):
    planes = np.atleast_2d(planes)
    u = planes[:, 0]
    v = (planes[:, 0] + 2 * planes[:, 1]) / np.sqrt(3)
    w = planes[:, 3] / c_over_a
    planes_cartesian = np.c_[u, v, w]
    return planes_cartesian / norm(planes_cartesian, axis=1, keepdims=True)
