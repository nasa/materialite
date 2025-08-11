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


def add_ipf_colors_field(
    material,
    orientation_label="orientation",
    specimen_frame_direction=Vector([0, 0, 1]),
    unit_cell="cubic",
    ipf_color_label="ipf_color",
):
    orientations = material.extract(orientation_label)
    ipf_colors = get_ipf_colors(specimen_frame_direction, orientations, unit_cell)
    return material.create_fields({ipf_color_label: ipf_colors.tolist()})


def get_ipf_colors(specimen_frame_direction, orientations, unit_cell):
    specimen_frame_direction = Vector(specimen_frame_direction)
    direction = specimen_frame_direction / specimen_frame_direction.norm
    crystal_directions = direction.to_crystal_frame(orientations)
    ipf_points = _convert_to_fundamental_region(crystal_directions, unit_cell)
    return _get_ipf_colors_from_fundamental_points(ipf_points, unit_cell)


def get_ipf(specimen_frame_direction, orientations, unit_cell):
    specimen_frame_direction = Vector(specimen_frame_direction)
    direction = specimen_frame_direction / specimen_frame_direction.norm
    crystal_directions = direction.to_crystal_frame(orientations)
    ipf_points_3D = _convert_to_fundamental_region(crystal_directions, unit_cell)
    ipf_points = _get_stereographic_projection(ipf_points_3D)
    ipf_boundary = _get_ipf_boundary(unit_cell)
    return ipf_points, ipf_boundary


def _convert_to_fundamental_region(directions, unit_cell):
    if unit_cell.lower() in ["cubic", "fcc", "bcc"]:
        p = np.atleast_2d(directions.components)
        p[p[:, 0] < 0, 0] *= -1
        p[p[:, 1] < 0, 1] *= -1
        p[p[:, 2] < 0, 2] *= -1

        cond = p[:, 0] < p[:, 1]
        p[cond, 0], p[cond, 1] = p[cond, 1], p[cond, 0].copy()

        cond = p[:, 2] < p[:, 0]
        p[cond, 2], p[cond, 0] = p[cond, 0], p[cond, 2].copy()

        cond = p[:, 0] < p[:, 1]
        p[cond, 0], p[cond, 1] = p[cond, 1], p[cond, 0].copy()
    elif unit_cell.lower() in ["hcp", "hexagonal"]:
        p = np.atleast_2d(directions.components).copy()
        p[p[:, 0] < 0, 0] *= -1
        p[p[:, 1] < 0, 1] *= -1
        p[p[:, 2] < 0, 2] *= -1
        angles = np.arctan2(p[:, 1], p[:, 0])
        reduce_cond = angles >= np.pi / 3
        c60 = 0.5
        s60 = np.sqrt(3) / 2
        reduce_R = np.array([[c60, s60], [-s60, c60]])
        p[reduce_cond, :2] = np.einsum("ij, bj -> bi", reduce_R, p[reduce_cond, :2])
        angles[reduce_cond] -= np.pi / 3

        reflect_cond = angles >= np.pi / 6
        reflect_R = np.array([[c60, s60], [s60, -c60]])
        p[reflect_cond, :2] = np.einsum("ij, bj -> bi", reflect_R, p[reflect_cond, :2])
        np.arctan2(p[:, 1], p[:, 0]) * 180 / np.pi

    return p


def _get_ipf_colors_from_fundamental_points(points, unit_cell):
    if unit_cell.lower() in ["cubic", "fcc", "bcc"]:
        u = points[:, 2] - points[:, 0]
        v = np.sqrt(2) * (points[:, 0] - points[:, 1])
        w = np.sqrt(3) * points[:, 1]
        uvw = np.c_[u, v, w]
    elif unit_cell.lower() in ["hexagonal", "hcp"]:
        u = points[:, 2]
        v = points[:, 0] - np.sqrt(3) * points[:, 1]
        w = 2 * points[:, 1]
        uvw = np.c_[u, v, w]
    return _adjust_colors(uvw)


def _adjust_colors(uvw):
    uvw = uvw / np.max(np.abs(uvw), axis=1, keepdims=True)
    return np.sqrt(uvw)


def _get_stereographic_projection(ipf_points):
    azimuth, elevation = _cartesian_to_spherical(ipf_points)
    P = 1.0 / np.cos(elevation / 2.0)
    x_projected = P * np.sin(elevation / 2) * np.cos(azimuth)
    y_projected = P * np.sin(elevation / 2) * np.sin(azimuth)
    return np.c_[x_projected, y_projected]


def _cartesian_to_spherical(xyz):
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    elevation = np.arctan2(np.sqrt(xy), xyz[:, 2])
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    return azimuth, elevation


def _get_ipf_boundary(unit_cell):
    num_points = 1001
    if unit_cell.lower() in ["cubic", "fcc", "bcc"]:
        great_circle_xz = np.ones(num_points)
        great_circle_points_3D = np.c_[
            great_circle_xz, np.linspace(0, 1, num_points), great_circle_xz
        ]
        great_circle_points_3D = great_circle_points_3D / norm(
            great_circle_points_3D, axis=1, keepdims=True
        )
        great_circle_points = _get_stereographic_projection(great_circle_points_3D)
        ipf_boundary = np.concatenate([[[0, 0]], great_circle_points, [[0, 0]]], axis=0)
    elif unit_cell.lower() in ["hexagonal", "hcp"]:
        pole_1 = np.array([1, 0, 0])
        pole_2 = np.array([np.sqrt(3) / 2, 0.5, 0])
        great_circle_direction = pole_2 - pole_1
        increments = np.linspace(0, 1, num_points)
        great_circle_points_3D = (
            increments[:, np.newaxis] * great_circle_direction + pole_1
        )
        great_circle_points_3D = great_circle_points_3D / norm(
            great_circle_points_3D, axis=1, keepdims=True
        )
        great_circle_points = _get_stereographic_projection(great_circle_points_3D)
        ipf_boundary = np.concatenate([[[0, 0]], great_circle_points, [[0, 0]]], axis=0)
    return ipf_boundary


def get_symmetry_operators(unit_cell):
    # source: Rollett lecture notes on symmetry; corrected from Kocks, Tome,
    #    and Wenk, Texture and Anisotropy
    i = np.array([1, 0, 0])
    j = np.array([0, 1, 0])
    k = np.array([0, 0, 1])
    if unit_cell.lower() in ["hexagonal", "hcp"]:
        a = np.sqrt(3) / 2
        p5 = 0.5
        m5 = -0.5
        symmetry_matrices = [0] * 12
        symmetry_matrices[0] = np.eye(3)
        symmetry_matrices[1] = np.array([[m5, a, 0], [-a, m5, 0], k])
        symmetry_matrices[2] = np.array([[m5, -a, 0], [a, m5, 0], k])
        symmetry_matrices[3] = np.array([[p5, a, 0], [-a, p5, 0], k])
        symmetry_matrices[4] = np.array([-i, -j, k])
        symmetry_matrices[5] = np.array([[p5, -a, 0], [a, p5, 0], k])
        symmetry_matrices[6] = np.array([[m5, -a, 0], [-a, p5, 0], -k])
        symmetry_matrices[7] = np.array([i, -j, -k])
        symmetry_matrices[8] = np.array([[m5, a, 0], [a, p5, 0], -k])
        symmetry_matrices[9] = np.array([[p5, a, 0], [a, m5, 0], -k])
        symmetry_matrices[10] = np.array([-i, j, -k])
        symmetry_matrices[11] = np.array([[p5, -a, 0], [-a, m5, 0], -k])

    elif unit_cell.lower() in ["cubic", "fcc", "bcc"]:
        symmetry_matrices = [0] * 24
        symmetry_matrices[0] = np.eye(3)
        symmetry_matrices[1] = np.array([k, i, j])
        symmetry_matrices[2] = np.array([j, k, i])
        symmetry_matrices[3] = np.array([-j, k, -i])
        symmetry_matrices[4] = np.array([-j, -k, i])
        symmetry_matrices[5] = np.array([j, -k, -i])
        symmetry_matrices[6] = np.array([-k, i, -j])
        symmetry_matrices[7] = np.array([-k, -i, j])
        symmetry_matrices[8] = np.array([k, -i, -j])
        symmetry_matrices[9] = np.array([-i, j, -k])
        symmetry_matrices[10] = np.array([-i, -j, k])
        symmetry_matrices[11] = np.array([i, -j, -k])
        symmetry_matrices[12] = np.array([-k, -j, -i])
        symmetry_matrices[13] = np.array([k, -j, i])
        symmetry_matrices[14] = np.array([k, j, -i])
        symmetry_matrices[15] = np.array([-k, j, i])
        symmetry_matrices[16] = np.array([-i, -k, -j])
        symmetry_matrices[17] = np.array([i, -k, j])
        symmetry_matrices[18] = np.array([i, k, -j])
        symmetry_matrices[19] = np.array([-i, k, j])
        symmetry_matrices[20] = np.array([-j, -i, -k])
        symmetry_matrices[21] = np.array([j, -i, k])
        symmetry_matrices[22] = np.array([j, i, -k])
        symmetry_matrices[23] = np.array([-j, i, k])

    else:
        raise ValueError(f"{unit_cell} is not a valid unit cell")
    return np.array(symmetry_matrices)
