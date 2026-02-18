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

import h5py
import numpy as np
import pandas as pd
from materialite import Material, Orientation
from materialite.util import camel_to_snake


def import_dream3d(
    file,
    simpl_geometry_path,
    field_paths=None,
    region_id_path=None,
    region_field_paths=None,
):
    if region_id_path is None and region_field_paths is not None:
        raise ValueError(
            "region_id_path must be specified if region_field_paths are provided"
        )
    d3d = h5py.File(file, mode="r")

    simpl_geometry_data = d3d.get(simpl_geometry_path)

    dimensions = list(simpl_geometry_data.get("DIMENSIONS"))
    num_points = np.prod(dimensions)
    origin = list(simpl_geometry_data.get("ORIGIN"))
    spacing = list(simpl_geometry_data.get("SPACING"))

    fields = _initialize_fields(dimensions, origin, spacing)

    field_paths = [] if field_paths is None else field_paths
    for field_path in field_paths:
        label = camel_to_snake(field_path.split("/")[-1])
        field = d3d.get(field_path)[:]
        field = field.reshape((num_points, field.shape[-1]))
        num_components = field.shape[1]
        for ii in range(num_components):
            suffix = "" if num_components == 1 else "_" + str(ii + 1)
            fields[label + suffix] = field[:, ii]

    regional_fields = {}
    if region_id_path is not None:
        region_id_label = camel_to_snake(region_id_path.split("/")[-1])
        region_id_field = d3d.get(region_id_path)[:].reshape(-1)
        fields[region_id_label] = region_id_field
        if region_field_paths is not None:
            unique_region_ids = np.unique(region_id_field)
            regional_fields[region_id_label] = unique_region_ids
            start_idx = 0 if 0 in unique_region_ids else 1

    region_field_paths = [] if region_field_paths is None else region_field_paths
    for region_field_path in region_field_paths:
        region_label = camel_to_snake(region_field_path.split("/")[-1])
        region_field = d3d.get(region_field_path)[:]
        num_components = region_field.shape[1]
        for ii in range(num_components):
            suffix = "" if num_components == 1 else "_" + str(ii + 1)
            regional_fields[region_label + suffix] = region_field[start_idx:, ii]

    material = Material(
        dimensions=dimensions, origin=origin, spacing=spacing, fields=fields
    )
    if regional_fields:
        material = material.create_regional_fields(
            region_label=region_id_label, regional_fields=regional_fields
        )
    return material


def _initialize_fields(dimensions, origin, spacing):
    grid = np.indices(dimensions).T.reshape(-1, 3)
    return pd.DataFrame(
        {
            "x": grid[:, 0] * spacing[0] + origin[0],
            "y": grid[:, 1] * spacing[1] + origin[1],
            "z": grid[:, 2] * spacing[2] + origin[2],
        }
    )


def import_spparks(
    file,
    usecols=[1, 2, 3, 4],
    names=["feature", "x", "y", "z"],
    skiprows=9,
    voxel_size=1,
):
    fields = pd.read_csv(
        file,
        sep=" ",
        header=None,
        skiprows=skiprows,
        usecols=usecols,
        names=names,
    )
    fields.sort_values(by=["z", "y", "x"], inplace=True)

    dimensions = [fields[dim].max() - fields[dim].min() + 1 for dim in ["x", "y", "z"]]

    origin = [0, 0, 0]
    fields[["x", "y", "z"]] = voxel_size * fields[["x", "y", "z"]]
    spacing = [voxel_size, voxel_size, voxel_size]

    return Material(
        dimensions=dimensions, origin=origin, spacing=spacing, fields=fields
    )


def import_vgstudio(
    file,
    dimensions,
    label="xct",
    spacing=[1, 1, 1],
    origin=[0, 0, 0],
    dtype=np.float32,
    x_range=None,
    y_range=None,
    z_range=None,
):
    if x_range is None:
        x_range = [0, dimensions[0]]
    if y_range is None:
        y_range = [0, dimensions[1]]
    if z_range is None:
        z_range = [0, dimensions[2]]

    field = np.fromfile(file, dtype)
    field = field.reshape(np.array(dimensions)[::-1])
    field = field[
        z_range[0] : z_range[1], y_range[0] : y_range[1], x_range[0] : x_range[1]
    ]
    dimensions = field.shape[::-1]

    return Material(
        dimensions=dimensions,
        spacing=spacing,
        origin=origin,
        fields={label: field.ravel()},
    )


def import_evpfft(
    file,
    spacing=[1, 1, 1],
    origin=[0, 0, 0],
    grain_label="grain",
    phase_label="phase",
    orientation_label="orientation",
    euler_angles_to_radians=True,
):

    fields = pd.read_csv(
        file,
        header=None,
        delimiter=" ",
        names=[
            "euler_angle_1",
            "euler_angle_2",
            "euler_angle_3",
            "x_id",
            "y_id",
            "z_id",
            grain_label,
            phase_label,
        ],
    ).sort_values(by=["x_id", "y_id", "z_id"])

    if euler_angles_to_radians:
        orientations = Orientation.from_euler_angles(
            fields[["euler_angle_1", "euler_angle_2", "euler_angle_3"]].to_numpy(),
            in_degrees=True,
        )
    else:
        orientations = Orientation.from_euler_angles(
            fields[["euler_angle_1", "euler_angle_2", "euler_angle_3"]].to_numpy()
        )

    dimensions = fields[["x_id", "y_id", "z_id"]].max().to_numpy()

    fields = fields.assign(**{orientation_label: orientations}).drop(
        labels=["euler_angle_1", "euler_angle_2", "euler_angle_3"], axis=1
    )

    return Material(
        dimensions=dimensions,
        spacing=spacing,
        origin=origin,
        fields=fields,
    )
