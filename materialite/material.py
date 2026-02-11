# Copyright 2025 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.
#
# The Materialite platform is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd
import pyvista as pv
from materialite.tensor import Order2SymmetricTensor, Orientation, Scalar, Vector
from materialite.util import cartesian_grid, power_of_two_below, repeat_data
from matplotlib import pyplot as plt
from scipy import spatial


class Material:
    """
    A 3D point grid representation for materials modeling.

    The Material class provides a unified data structure for representing
    materials as 3D grids of points with associated spatial numerical or
    tensorial fields.

    Parameters
    ----------
    dimensions : list, default [16, 16, 16]
        Number of points in each direction [x, y, z].
    origin : list, default [0, 0, 0]
        Coordinates of the domain origin [x, y, z].
    spacing : list, default [1, 1, 1]
        Distance between points in each direction [x, y, z].
    sizes : list, optional
        Total size of the domain in each direction. If provided, overrides spacing.
    fields : dict or DataFrame, optional
        Initial field data. If None, creates coordinate fields only.
    """

    def __init__(
        self,
        dimensions=[16, 16, 16],
        origin=[0, 0, 0],
        spacing=[1, 1, 1],
        sizes=None,
        fields=None,
    ):
        self._dimensions = np.array(dimensions)
        self._origin = np.array(origin)
        spacing = np.array(spacing)
        sizes = np.array(sizes) if sizes is not None else None
        self._spacing, self._sizes = self._get_spacing_and_sizes(
            spacing, sizes, self.dimensions
        )

        self.fields = self._initialize_fields(fields)
        self.state = dict()
        self._regional_fields = dict()

    @classmethod
    def from_image(cls, image, spacing=1, label="intensity", flip_y=True):
        """
        Create a Material from an image by adding a singleton z-dimension.
        Parameters
        ----------
        image : 2D array-like
            Input image data.
        spacing : float or list, default 1
            Spacing in x and y directions.
        label : str, default 'intensity'
            Label for the image intensity field.
        flip_y : bool, default False
            Whether to flip the image in the y-direction.
        Returns
        -------
        Material
            New Material instance with singleton z-dimension.
        """

        if isinstance(spacing, (int, float)):
            spacing = [spacing, spacing]
        elif len(spacing) == 2:
            spacing = [spacing[0], spacing[1], 1]

        image = np.array(image)

        # Images typically have a row-column format, so change to x-y
        if flip_y:
            image = np.flipud(image).T

        return cls(
            dimensions=[image.shape[0], image.shape[1], 1],
            spacing=[spacing[0], spacing[1], 1],
            fields={label: image.flatten()},
        )

    @property
    def origin(self):
        """Origin coordinates of the domain."""
        return self._origin.copy()

    @property
    def dimensions(self):
        """Number of points in each direction."""
        return self._dimensions.copy()

    @property
    def spacing(self):
        """Spacing between points in each direction."""
        return self._spacing.copy()

    @property
    def sizes(self):
        """Total size of the domain in each direction."""
        return self._sizes.copy()

    @property
    def center(self):
        """Center of the domain."""
        return self.origin + self.sizes / 2

    @property
    def center_id(self):
        """Index of the center point of the domain."""
        return (self.dimensions - 1) // 2

    @property
    def far_corner(self):
        """Far corner of the domain."""
        return self.origin + self.sizes

    @property
    def corners(self):
        """All corners of the domain indexed by multi-dimensional index."""
        patterns = np.stack(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij"), axis=-1)
        return self.origin + patterns * self.sizes

    @property
    def corner_ids(self):
        """All corner indices of the domain."""
        return np.stack(
            np.meshgrid(
                [0, self.dimensions[0] - 1],
                [0, self.dimensions[1] - 1],
                [0, self.dimensions[2] - 1],
                indexing="ij",
            ),
            axis=-1,
        )

    def _get_spacing_and_sizes(self, spacing, sizes, dimensions):
        """Calculate spacing and sizes from given parameters."""
        if sizes is not None:
            spacing = np.divide(
                sizes,
                dimensions - 1,
                out=np.zeros_like(sizes, dtype=float),
                where=dimensions != 1,
            )
            sizes[np.where(dimensions == 1)[0]] = 0
        else:
            sizes = spacing * (dimensions - 1)
        return spacing, sizes

    def _initialize_fields(self, fields):
        """Initialize the fields DataFrame with coordinate information."""
        grid = cartesian_grid(self.dimensions)
        if fields is None:
            fields = pd.DataFrame(
                {
                    "x": grid[:, 0] * self.spacing[0] + self.origin[0],
                    "y": grid[:, 1] * self.spacing[1] + self.origin[1],
                    "z": grid[:, 2] * self.spacing[2] + self.origin[2],
                }
            )
        else:
            fields = self._initialize_fields_from_user_input(fields, grid)

        return fields.assign(x_id=grid[:, 0], y_id=grid[:, 1], z_id=grid[:, 2])

    def _initialize_fields_from_user_input(self, fields, grid):
        """Process user-provided fields and validate against grid dimensions."""
        fields = pd.DataFrame(fields)
        if len(fields) != self.num_points:
            raise ValueError(
                "length of fields does not match number of points from dimensions"
            )
        provided_labels = fields.columns
        if np.any(
            ["x" in provided_labels, "y" in provided_labels, "z" in provided_labels]
        ):
            # user provided at least one of x, y, and z
            # fill singleton dimensions with ones in fields
            singleton_dimensions = np.where(self.dimensions == 1)[0]
            xyz = np.array(["x", "y", "z"])
            fields = fields.assign(
                **dict(
                    zip(xyz[singleton_dimensions], self.origin[singleton_dimensions])
                )
            ).sort_values(by=["x", "y", "z"], ignore_index=True)
            inferred_dims = [len(fields[s].unique()) for s in ["x", "y", "z"]]
            if not np.array_equal(inferred_dims, self.dimensions):
                raise ValueError(
                    "provided x, y, and z fields do not match provided dimensions"
                )
            inferred_sizes = (
                fields[["x", "y", "z"]].max() - fields[["x", "y", "z"]].min()
            )
            if not np.allclose(inferred_sizes, self.sizes):
                raise ValueError(
                    "provided x, y, and z fields do not match provided sizes"
                )
            return fields
        else:
            # x, y, and z were all not provided by the user
            return fields.assign(
                x=grid[:, 0] * self.spacing[0] + self.origin[0],
                y=grid[:, 1] * self.spacing[1] + self.origin[1],
                z=grid[:, 2] * self.spacing[2] + self.origin[2],
            )

    @property
    def num_points(self):
        """Total number of points in the material."""
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

    def run(self, model, **kwargs):
        """
        Run a model on this material.

        Parameters
        ----------
        model : callable
            A model object or function that takes a Material as first argument.
        **kwargs
            Additional arguments passed to the model.

        Returns
        -------
        Result of the model evaluation.
        """
        return model(self, **kwargs)

    def copy(self):
        """Create a deep copy of the material."""
        return deepcopy(self)

    def get_fields(self):
        """
        Get all fields including regional fields merged in.

        Returns
        -------
        pandas.DataFrame
            Combined field data with regional fields merged.
        """
        fields = self.fields.copy()
        for k, v in self._regional_fields.items():
            try:
                fields = fields.merge(
                    v, on=k, how="left", suffixes=(False, False), validate="many_to_one"
                )
            except ValueError:
                raise ValueError(
                    "Error when merging regional fields. Two different regional fields defined values for the same field."
                )
        return fields

    def extract(self, labels):
        """
        Extract field data as arrays or tensor objects.

        Parameters
        ----------
        labels : str or list
            Field name(s) to extract.

        Returns
        -------
        numpy.ndarray or tensor object
            Extracted field data.
        """
        if isinstance(labels, str):
            data = self.get_fields()[labels].to_list()
            try:
                result = type(data[0]).from_list(data)
            except AttributeError:
                result = np.array(data)
        else:
            result = self.get_fields()[labels].to_numpy()
        return result

    def create_fields(self, fields):
        """
        Create a new material with additional fields.

        Parameters
        ----------
        fields : dict
            Dictionary of field_name: field_data pairs.

        Returns
        -------
        Material
            New material with the added fields.
        """
        material = self.copy()
        material.fields = material.fields.assign(**fields)
        return material

    def create_random_integer_field(
        self, label, low, high=None, rng=np.random.default_rng()
    ):
        """
        Create a field with random integer values.

        Parameters
        ----------
        label : str
            Name of the new field.
        low : int
            Lower bound (inclusive).
        high : int, optional
            Upper bound (exclusive).
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        Material
            New material with the random field added.
        """
        values = rng.integers(
            low=low, high=high, size=self.num_points
        )  # Excludes high value
        return self.create_fields({label: values})

    def create_uniform_field(self, label, value):
        """
        Create a field with uniform values at all points.

        Parameters
        ----------
        label : str
            Name of the new field.
        value : any
            Value to assign to all points.

        Returns
        -------
        Material
            New material with the uniform field added.
        """
        field = {label: [value] * self.num_points}
        return self.create_fields(field)

    def segment(self, label, threshold, low=0, high=1, new_label=None):
        """
        Create a binary segmentation of a field based on a threshold.

        Parameters
        ----------
        label : str
            Name of the field to segment.
        threshold : float
            Threshold value for segmentation.
        low : any, default 0
            Value assigned to points below threshold.
        high : any, default 1
            Value assigned to points above threshold.
        new_label : str, optional
            Name for the new field. Defaults to "segmented_{label}".

        Returns
        -------
        Material
            New material with the segmented field added.
        """
        if new_label is None:
            new_label = "segmented_" + label
        return self.create_fields(
            {new_label: np.where(self.extract(label) > threshold, high, low)}
        )

    def create_voronoi(
        self,
        num_regions=10,
        label="region",
        rng=np.random.default_rng(),
        periodic=False,
    ):
        """
        Create Voronoi regions in the material.

        Parameters
        ----------
        num_regions : int, default 10
            Number of Voronoi regions to create.
        label : str, default "region"
            Name for the region field.
        rng : numpy.random.Generator
            Random number generator for seed points.
        periodic : bool, default False
            Whether to enforce periodic boundary conditions.

        Returns
        -------
        Material
            New material with Voronoi regions added.
        """
        voronoi_points = rng.random((num_regions, 3)) * self.sizes + self.origin
        material_points = self.fields[["x", "y", "z"]].to_numpy()

        if periodic:
            voronoi_ref = np.tile(np.arange(num_regions), 27)
            voronoi_all = np.arange(num_regions * 27)
            voronoi_dict = dict(zip(voronoi_all, voronoi_ref))
            voronoi_points_periodic = repeat_data(voronoi_points, *self.sizes)
            material_points_periodic = repeat_data(material_points, *self.sizes)
            _, point_regions_periodic = spatial.cKDTree(voronoi_points_periodic).query(
                material_points_periodic, k=1
            )
            point_regions = [
                voronoi_dict[p] for p in point_regions_periodic[: self.num_points]
            ]
        else:
            _, point_regions = spatial.cKDTree(voronoi_points).query(
                material_points, k=1
            )

        return self.create_fields({label: point_regions})

    def assign_random_orientations(
        self,
        region_label="region",
        orientation_label="orientation",
        rng=np.random.default_rng(),
    ):
        """
        Assign random crystallographic orientations to regions.

        Parameters
        ----------
        region_label : str, default "region"
            Name of the field containing region IDs.
        orientation_label : str, default "orientation"
            Name for the new orientation field.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        Material
            New material with random orientations assigned to regions.
        """
        unique_regions = self.fields[region_label].unique()
        num_regions = len(unique_regions)
        orientations = Orientation.random(num_regions, rng=rng)
        if num_regions == 1:
            orientations = Orientation([orientations.rotation_matrix], dims="p")
        regional_field = {region_label: unique_regions, orientation_label: orientations}
        return self.create_regional_fields(
            region_label,
            pd.DataFrame(regional_field).sort_values(
                by=region_label, ignore_index=True
            ),
        )

    def insert_feature(self, feature, fields):
        """
        Insert field values into points inside a geometric feature.

        Parameters
        ----------
        feature : Feature
            Geometric feature defining the region of interest
        fields : dict
            Dictionary mapping field names to values to insert

        Returns
        -------
        Material
            New material with updated field values
        """
        points = Vector(self.fields[["x", "y", "z"]].to_numpy())
        is_inside = feature.check_inside(points)
        num_points = self.num_points

        new_fields = {}

        for label, values in fields.items():
            # Create default field if it doesn't exist, otherwise copy existing
            if label in self.fields:
                new_values = self.extract(label)
            else:
                if hasattr(values, "zero"):
                    new_values = values.zero().repeat(num_points)
                elif hasattr(values, "identity"):
                    new_values = values.identity().repeat(num_points)
                else:
                    new_values = np.zeros(num_points)

            indices = np.where(is_inside)
            rows = indices[0]  # Point indices

            # Check if single value applies to all points vs multiple features
            is_uniform = np.isscalar(values) or (
                hasattr(values, "dims_str") and not values.dims_str
            )

            if is_uniform:
                new_values[rows] = values
            else:
                # Map each point to its corresponding feature's value
                feature_indices = indices[1:]
                cols = np.ravel_multi_index(feature_indices, is_inside.shape[1:])

                # Should be a better way to get numpy array indexing to work here
                try:
                    # Try tensor-style indexing first
                    new_values[rows] = values[cols]
                except (TypeError, IndexError):
                    # Fall back to numpy array indexing for lists
                    values_array = np.asarray(values)
                    new_values[rows] = values_array[cols]

            new_fields[label] = new_values

        return self.create_fields(new_fields)

    def remove_field(self, field_label, in_regional_field=None):
        """
        Remove a field from the material.

        Parameters
        ----------
        field_label : str
            Name of the field to remove.
        in_regional_field : str, optional
            Name of regional field if removing from regional fields.

        Returns
        -------
        Material
            New material with the specified field removed.
        """
        material = self.copy()
        if in_regional_field is not None:
            material._regional_fields[in_regional_field] = material._regional_fields[
                in_regional_field
            ].drop(field_label, axis=1)
        elif field_label in list(self.fields):
            material.fields = self.fields.drop(field_label, axis=1)
            if field_label in self._regional_fields.keys():
                del material._regional_fields[field_label]
        return material

    def create_regional_fields(self, region_label, regional_fields):
        """
        Create fields that vary by region rather than by point.

        Parameters
        ----------
        region_label : str
            Name of the field containing region IDs.
        regional_fields : dict or DataFrame
            Regional field data with region_label as key column.

        Returns
        -------
        Material
            New material with regional fields added.
        """
        regional_fields = pd.DataFrame(regional_fields)
        if regional_fields[region_label].duplicated().any():
            raise ValueError(
                f"Could not create regional field: {region_label} has non-unique values in regional field"
            )
        material = self.copy()
        if region_label in self._regional_fields:
            # merge with existing regional fields and overwrite any duplicated fields
            old_df = material._regional_fields[region_label].copy()
            old_regions = np.sort(old_df[region_label].to_numpy())
            new_regions = np.sort(regional_fields[region_label].to_numpy())
            if not np.array_equal(old_regions, new_regions):
                raise ValueError(
                    f"Values of {region_label} do not match between new and existing regional field"
                )
            regional_fields = (
                old_df[[region_label]]
                .merge(regional_fields, on=region_label, how="left")
                .combine_first(old_df)
            )
        elif region_label not in list(self.fields):
            raise ValueError(
                f"Could not create regional field: {region_label} is not a field"
            )
        # make sure merges won't produce any NaNs
        try:
            data_check = material.fields.merge(
                regional_fields,
                on=region_label,
                how="left",
                suffixes=(False, False),
                indicator=True,
            )
        except ValueError:
            raise ValueError(
                f"Could not create regional field: the provided regional fields include a field that already exists in the Material"
            )
        if (data_check["_merge"] == "left_only").any():
            missing_keys = (
                data_check.loc[data_check["_merge"] == "left_only", region_label]
                .unique()
                .tolist()
            )
            raise ValueError(
                f"Values of {region_label} not found in regional field: {missing_keys}"
            )
        material._regional_fields[region_label] = regional_fields
        return material

    def update_regional_field(self, region_label, regional_field_update):
        regional_field_update = pd.DataFrame(regional_field_update)
        material = self.copy()
        regional_field = material._regional_fields[region_label]
        material._regional_fields[region_label] = pd.concat(
            [regional_field, regional_field_update], ignore_index=True
        )
        return material

    def extract_regional_field(self, region_label, field_label=None):
        """
        Extract regional field data.

        Parameters
        ----------
        region_label : str
            Name of the regional field.
        field_label : str, optional
            Specific field within the regional data to extract.

        Returns
        -------
        DataFrame or array
            Regional field data.
        """
        if field_label is None:
            return self._regional_fields[region_label]
        else:
            data = self._regional_fields[region_label][field_label].to_list()
            try:
                result = type(data[0]).from_list(data)
            except AttributeError:
                result = data
            return result

    def get_region_volume_fractions(self, region_label="region"):
        """
        Calculate volume fraction of each region.

        Parameters
        ----------
        region_label : str, default "region"
            Name of the field containing region IDs.

        Returns
        -------
        dict
            Dictionary mapping region ID to volume fraction.
        """
        features = self.fields[region_label].to_numpy()
        unique_features, points_per_feature = np.unique(features, return_counts=True)
        return dict(zip(unique_features, points_per_feature / self.num_points))

    def get_region_indices(self, region_label="region"):
        """
        Get point indices for each region.

        Parameters
        ----------
        region_label : str, default "region"
            Name of the field containing region IDs.

        Returns
        -------
        dict
            Dictionary mapping region ID to list of point indices.
        """
        regions = self.fields[region_label]
        return {
            region: indices.tolist()
            for region, indices in regions.groupby(regions).groups.items()
        }

    def plot(
        self,
        label,
        component=None,
        kind="voxel",
        colormap="coolwarm",
        show_grid=False,
        show_edges=False,
        color_lims=None,
        opacity=1.0,
        mask=None,
    ):
        """
        Plot a field using PyVista visualization.

        Parameters
        ----------
        label : str
            Name of the field to plot.
        component : int or list, optional
            Component(s) to plot for multi-component fields.
        kind : str, default "voxel"
            Plot type: "voxel", "voxelnode", "point", or "ipf_map".
        colormap : str, default "coolwarm"
            Colormap name.
        show_grid : bool, default False
            Whether to show grid lines.
        show_edges : bool, default False
            Whether to show voxel edges.
        color_lims : tuple, optional
            Color scale limits (min, max).
        opacity : float, default 1.0
            Transparency level.
        """
        fields = self.get_fields().sort_values(by=["z", "y", "x"])

        # Determine slices to grab components of the field
        slices = [slice(None)]
        if component is not None:
            component = [component] if not isinstance(component, list) else component
            for c in component:
                slices.append(slice(c, c + 1))
        slices = tuple(slices)

        if label not in list(fields):
            raise TypeError(f"{label} not found in fields")

        data = fields[label].to_list()
        if isinstance(data[0], Order2SymmetricTensor):
            tensor = type(data[0]).from_list(data)
            if "t" in tensor.dims_str:
                result = tensor[:, -1].stress_voigt
            else:
                result = tensor.stress_voigt
        elif isinstance(data[0], Scalar) or isinstance(data[0], Vector):
            tensor = type(data[0]).from_list(data)
            if "t" in tensor.dims_str:
                result = tensor[:, -1].components
            else:
                result = tensor.components
        else:
            result = np.array(data)
        plot_array = np.squeeze(result[slices])
        if len(plot_array.shape) > 1 and kind.lower() != "ipf_map":
            raise ValueError(
                "Tried to plot a field with multiple dimensions. "
                + "You may need to specify a component."
            )

        if mask is not None:
            if mask not in list(fields):
                raise TypeError(f"{mask} not found in fields")

            mask_array = np.bool_(fields[mask].values)

        # Material points are displayed as voxel centroid (cell) values
        if kind.lower() == "voxel":
            grid = pv.ImageData()
            grid.dimensions = self.dimensions + 1
            grid.origin = list(self.origin - self.spacing / 2)
            grid.spacing = self.spacing
            grid.cell_data[label] = plot_array
            if mask is not None:
                grid = grid.extract_cells(mask_array)
            grid.plot(
                cmap=colormap,
                clim=color_lims,
                show_edges=show_edges,
                show_grid=show_grid,
                scalar_bar_args=dict(vertical=True, interactive=True),
                opacity=opacity,
            )

        # Material points are displayed as voxel node values
        elif kind.lower() == "voxelnode":
            grid = pv.ImageData()
            grid.dimensions = self.dimensions
            grid.origin = self.origin
            grid.spacing = self.spacing
            grid.point_data[label] = plot_array
            grid.plot(
                cmap=colormap,
                clim=color_lims,
                show_edges=show_edges,
                show_grid=show_grid,
                scalar_bar_args=dict(vertical=True, interactive=True),
                opacity=opacity,
            )

        # Material points are displayed as a point cloud
        elif kind.lower() == "point":
            point_cloud = pv.PolyData(
                fields[["x", "y", "z"]].to_numpy(dtype=np.float32)
            )
            point_cloud[label] = plot_array
            point_cloud.plot(
                render_points_as_spheres=True,
                cmap=colormap,
                clim=color_lims,
                show_grid=show_grid,
                scalar_bar_args=dict(vertical=True, interactive=True),
                opacity=opacity,
            )

        elif kind.lower() == "ipf_map":
            grid = pv.ImageData()
            grid.dimensions = self.dimensions + 1
            grid.origin = list(self.origin - self.spacing / 2)
            grid.spacing = self.spacing
            grid["colors"] = plot_array
            grid.plot(scalars="colors", rgb=True)
        else:
            raise ValueError(f"{kind} is not a valid plot kind")

    def plot_slice(
        self,
        label,
        component=None,
        colormap="coolwarm",
        color_lims=None,
        figsize=(10, 8),
        title=None,
        ax=None,
        by_id=False,
    ):
        """
        Plot 2D slice from Material with one singleton dimension.

        Parameters
        ----------
        label : str
            Field name to plot
        component : int or list, optional
            Component(s) to plot for multi-component fields
        colormap : str, default "coolwarm"
            Colormap name
        color_lims : tuple, optional
            Color scale limits (min, max)
        figsize : tuple, default (10, 8)
            Figure size (only used if ax is None)
        title : str, optional
            Plot title
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, creates new figure and axis.
        by_id : bool, default False
            Whether to plot by material point ID

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
            If ax was provided, fig will be None
        """

        # Check for exactly one singleton dimension
        if len(np.where(self.dimensions == 1)[0]) != 1:
            raise ValueError(
                f"plot_slice requires exactly one singleton dimension, "
                f"but dimensions are {self.dimensions}"
            )

        fields = self.get_fields()

        # Determine slices to grab components of the field
        slices = [slice(None)]
        if component is not None:
            component = [component] if not isinstance(component, list) else component
            for c in component:
                slices.append(slice(c, c + 1))
        slices = tuple(slices)

        if label not in list(fields):
            raise TypeError(f"{label} not found in fields")

        # Handle different data types (tensors, scalars, vectors)
        data = fields[label].to_list()
        if isinstance(data[0], Order2SymmetricTensor):
            tensor = type(data[0]).from_list(data)
            result = (
                tensor[:, -1].stress_voigt
                if "t" in tensor.dims_str
                else tensor.stress_voigt
            )
        elif isinstance(data[0], Scalar) or isinstance(data[0], Vector):
            tensor = type(data[0]).from_list(data)
            result = (
                tensor[:, -1].components
                if "t" in tensor.dims_str
                else tensor.components
            )
        else:
            result = np.array(data)

        plot_array = np.squeeze(result[slices])

        if len(plot_array.shape) != 1:
            raise ValueError(
                "Tried to plot a field with multiple dimensions. "
                "You may need to specify a component."
            )

        # Reshape to 2D based on material dimensions
        plot_data = plot_array.reshape([d for d in self.dimensions if d != 1])

        # Determine active axes
        axis_names = ["x", "y", "z"]
        active_indices = [i for i in range(3) if self.dimensions[i] != 1]
        active_axes = [axis_names[i] for i in active_indices]

        # Get coordinate extents for active axes (cell-centered like plot() voxel mode)
        extent = (
            [
                self.origin[active_indices[0]] - self.spacing[active_indices[0]] / 2,
                self.origin[active_indices[0]]
                + self.sizes[active_indices[0]]
                + self.spacing[active_indices[0]] / 2,
                self.origin[active_indices[1]] - self.spacing[active_indices[1]] / 2,
                self.origin[active_indices[1]]
                + self.sizes[active_indices[1]]
                + self.spacing[active_indices[1]] / 2,
            ]
            if not by_id
            else None
        )

        # Determine color limits
        color_lims = (
            color_lims if color_lims is not None else (plot_data.min(), plot_data.max())
        )

        # Create figure/axis if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            show_plot = True
        else:
            fig = None
            show_plot = False

        # Plot with extent (transpose to match [y, x] for imshow)
        im = ax.imshow(
            plot_data.T,
            cmap=colormap,
            vmin=color_lims[0],
            vmax=color_lims[1],
            origin="lower",
            aspect="auto",
            extent=extent,
        )

        xlabel = f"{active_axes[0]} (ID)" if by_id else f"{active_axes[0]}"
        ylabel = f"{active_axes[1]} (ID)" if by_id else f"{active_axes[1]}"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        plt.colorbar(im, ax=ax, label=label)
        plt.tight_layout()
        if show_plot:
            plt.show()

        return fig, ax

    def compare_slice_plots(
        self,
        label1,
        label2,
        component=None,
        colormap="coolwarm",
        color_lims=None,
        figsize=(12, 6),
    ):
        """
        Compare 2D slices from Material with one singleton dimension.

        Parameters
        ----------
        label1 : str
            First field name to plot
        label2 : str
            Second field name to plot
        component : int or list, optional
            Component(s) to plot for multi-component fields
        colormap : str, default "coolwarm"
            Colormap name
        color_lims : tuple, optional
            Color scale limits (min, max)
        figsize : tuple, default (12, 6)
            Figure size

        Returns
        -------
        fig, axes : matplotlib Figure and Axes array
        """

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        self.plot_slice(
            label1,
            component=component,
            colormap=colormap,
            color_lims=color_lims,
            ax=axes[0],
        )

        self.plot_slice(
            label2,
            component=component,
            colormap=colormap,
            color_lims=color_lims,
            ax=axes[1],
        )

        for ax in axes:
            ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.show()

        return fig, axes

    def apply(self, func, *func_args, out=None, adapter=None, **func_kwargs):
        """
        Apply function with automatic field extraction and optional field creation.

        Parameters:
        -----------
        func : callable
            Function to apply to the extracted fields
        *func_args : str or any
            Field labels (str) to extract or raw values to pass to function
        out : str or iterable of str, optional
            If provided, create field(s) with these labels containing the result(s)
        adapter : str or None, optional
            How to reshape fields before/after function application:
            - None: No reshaping, pass flat arrays (default)
            - '3d': Reshape to (D, H, W) for volumetric operations
            - 'tomopy': Reshape to (1, H, W) for tomopy-style operations
            - 'image': Reshape to (H, W) for 2D image operations
        **func_kwargs : str or any
            Field labels (str) to extract or raw values to pass as keyword arguments

        Returns:
        --------
        Material or result
            New Material with added field(s) if `out` is specified, otherwise the raw result

        Examples:
        ---------
        >>> # 3D volumetric smoothing
        >>> mat = material.apply(filters.gaussian, 'attenuation', sigma=2.0,
        ...                      adapter='3d', out='smooth')

        >>> # 2D image processing on slice
        >>> mat = slice_material.apply(filters.median, 'attenuation',
        ...                            adapter='image', out='filtered')

        >>> # TomoPy operations
        >>> mat = slice_material.apply(tomopy.remove_ring, 'attenuation',
        ...                            adapter='tomopy', out='clean')
        """

        # Map string shortcuts to adapter instances
        adapter_map = {
            None: NoReshape(),
            "3d": Reshape3D(),
            "3D": Reshape3D(),
            "tomopy": ReshapeTomopy(),
            "image": ReshapeImage(),
        }

        # Validate and get adapter instance
        if adapter not in adapter_map:
            raise ValueError(
                f"Unknown adapter '{adapter}'. "
                f"Valid options: {list(adapter_map.keys())}"
            )

        adapter_obj = adapter_map[adapter]
        field_labels = self.get_fields().columns

        # Extract fields and track which args/kwargs are fields vs raw values
        # Format: each becomes (value, is_field_boolean)
        extracted_args = [
            (
                (self.extract(arg), True)
                if isinstance(arg, str) and arg in field_labels
                else (arg, False)
            )
            for arg in func_args
        ]

        extracted_kwargs = {
            k: (
                (self.extract(v), True)
                if isinstance(v, str) and v in field_labels
                else (v, False)
            )
            for k, v in func_kwargs.items()
        }

        # Apply adapter to reshape only the extracted fields (not raw values)
        args = [
            adapter_obj.reshape_field(val, self.dimensions) if is_field else val
            for val, is_field in extracted_args
        ]
        kwargs = {
            k: adapter_obj.reshape_field(val, self.dimensions) if is_field else val
            for k, (val, is_field) in extracted_kwargs.items()
        }

        # Run the function
        result = func(*args, **kwargs)

        # Early return if no output field specified
        if out is None:
            return result

        # Reshape result back to flat arrays for Material storage
        # Handle both single outputs and tuple/list outputs
        if isinstance(result, tuple):
            result = tuple(
                adapter_obj.reshape_result(r) if isinstance(r, np.ndarray) else r
                for r in result
            )
        elif isinstance(result, np.ndarray):
            result = adapter_obj.reshape_result(result)

        # Normalize out to list for uniform handling
        out_labels = [out] if isinstance(out, str) else list(out)

        # Create fields dictionary
        if len(out_labels) == 1:
            # Single output: assign entire result to the field
            fields = {out_labels[0]: result}
        else:
            # Multiple outputs: unpack result and match to labels
            try:
                fields = dict(zip(out_labels, result))
            except TypeError as e:
                raise ValueError(
                    f"Cannot unpack result into {len(out_labels)} fields. "
                    f"Result must be iterable with matching length. "
                    f"Error: {e}"
                )

        return self.create_fields(fields)

    def pipe(self, func, *args, **kwargs):
        """
        Apply a function that receives the material instance as its first argument.

        Enables chaining of operations that need access to material attributes
        or methods by passing the current material as the first argument to func.

        Parameters
        ----------
        func : callable
            Function that takes a Material instance as its first argument.
        *args, **kwargs : any
            Additional arguments passed to the function.

        Returns
        -------
        any
            Result of func(self, *args, **kwargs).
        """
        return func(self, *args, **kwargs)

    @staticmethod
    def _get_cropped_fields(fields, points_above):
        """Filter fields to keep only points below specified indices."""
        return fields.query(
            f"x_id <= {points_above[0]} and y_id <= {points_above[1]} and z_id <= {points_above[2]}"
        )

    def crop_by_range(self, x_range=None, y_range=None, z_range=None):
        """
        Crop the material by coordinate ranges. The returned material will include points at both
        endpoints of the provided ranges (i.e., the endpoints are inclusive).

        Parameters
        ----------
        x_range, y_range, z_range : tuple, optional
            Coordinate ranges (min, max) for each direction.

        Returns
        -------
        Material
            Cropped material.
        """
        fields = self.fields.copy()
        x_range = (self.origin[0], np.inf) if x_range is None else x_range
        y_range = (self.origin[1], np.inf) if y_range is None else y_range
        z_range = (self.origin[2], np.inf) if z_range is None else z_range
        fields = fields.query(
            f"x >= {x_range[0]} and x <= {x_range[1]} and y >= {y_range[0]} and y <= {y_range[1]} and z >= {z_range[0]} and z <= {z_range[1]}"
        )
        dimensions = [
            fields.x_id.max() - fields.x_id.min() + 1,
            fields.y_id.max() - fields.y_id.min() + 1,
            fields.z_id.max() - fields.z_id.min() + 1,
        ]
        origin = np.array([fields.x.min(), fields.y.min(), fields.z.min()])
        new_material = Material(
            dimensions=dimensions,
            origin=origin,
            spacing=self.spacing,
            fields=fields.drop(columns=["x_id", "y_id", "z_id"]).reset_index(drop=True),
        )
        for k, v in self._regional_fields.items():
            new_material = new_material.create_regional_fields(k, v)
        return new_material

    def crop_by_id_range(self, x_id_range=None, y_id_range=None, z_id_range=None):
        """
        Crop the material by point ID ranges. The returned material will include points at both
        endpoints of the provided ranges (i.e., the endpoints are inclusive).

        Parameters
        ----------
        x_id_range, y_id_range, z_id_range : tuple, optional
            Point ID ranges (min, max) for each direction.

        Returns
        -------
        Material
            Cropped material.
        """
        fields = self.fields.copy()
        x_id_range = (fields.x_id.min(), np.inf) if x_id_range is None else x_id_range
        y_id_range = (fields.y_id.min(), np.inf) if y_id_range is None else y_id_range
        z_id_range = (fields.z_id.min(), np.inf) if z_id_range is None else z_id_range
        fields = fields.query(
            f"x_id >= {x_id_range[0]} and x_id <= {x_id_range[1]} and y_id >= {y_id_range[0]} and y_id <= {y_id_range[1]} and z_id >= {z_id_range[0]} and z_id <= {z_id_range[1]}"
        )
        dimensions = [
            fields.x_id.max() - fields.x_id.min() + 1,
            fields.y_id.max() - fields.y_id.min() + 1,
            fields.z_id.max() - fields.z_id.min() + 1,
        ]
        origin = np.array([fields.x.min(), fields.y.min(), fields.z.min()])
        new_material = Material(
            dimensions=dimensions,
            origin=origin,
            spacing=self.spacing,
            fields=fields.drop(columns=["x_id", "y_id", "z_id"]).reset_index(drop=True),
        )
        for k, v in self._regional_fields.items():
            new_material = new_material.create_regional_fields(k, v)
        return new_material

    def chop_by_point_count(self, x=None, y=None, z=None):
        """
        Create a subset by removing points from domain boundaries.

        Parameters
        ----------
        x, y, z : tuple, optional
            Number of points to remove (min_side, max_side) in each direction.

        Returns
        -------
        Material
            New material with boundary points removed.
        """

        # Filter the fields in prescribed coordinate ranges
        fields = deepcopy(self.fields)

        # Set the number of points to chop to zero if arguments are None
        x = (0, 0) if x is None else x
        y = (0, 0) if y is None else y
        z = (0, 0) if z is None else z
        chopped_range = {
            "x_id": (fields.x_id.min() + x[0], fields.x_id.max() - x[1]),
            "y_id": (fields.y_id.min() + y[0], fields.y_id.max() - y[1]),
            "z_id": (fields.z_id.min() + z[0], fields.z_id.max() - z[1]),
        }
        mask = np.array(
            [True] * self.num_points
        )  # repeated to handle the no argument case
        for label, chopped_range in chopped_range.items():
            if chopped_range is not None:
                mask = (
                    mask
                    & (fields[label] >= chopped_range[0])
                    & (fields[label] <= chopped_range[1])
                )

        fields = fields.loc[mask].reset_index(drop=True)

        # Generate a new fieldless submaterial
        material = Material(
            dimensions=list(
                fields[["x_id", "y_id", "z_id"]].max()
                - fields[["x_id", "y_id", "z_id"]].min()
                + 1
            ),
            origin=list(fields[["x", "y", "z"]].min()),
            spacing=self.spacing,
            fields=fields.drop(columns=["x", "y", "z", "x_id", "y_id", "z_id"]),
        )
        return material

    def export_to_vtk(self, output="fields.vtk", labels=None):
        """
        Export material data to VTK format file.

        Parameters
        ----------
        output : str, default "fields.vtk"
            Output file path.
        labels : list, optional
            Specific field labels to export. If None, exports all fields.
        """
        if labels is None:
            fields = self.get_fields().sort_values(by=["z", "y", "x"])
        else:
            fields = self.get_fields()[labels].sort_values(by=["z", "y", "x"])

        output_text = (
            "# vtk DataFile Version 2.0\n"
            + "Material Export\n"
            + "ASCII\n"
            + "DATASET STRUCTURED_POINTS\n"
            + f"DIMENSIONS {self.dimensions[0] + 1} {self.dimensions[1] + 1} {self.dimensions[2] + 1}\n"
            + "ASPECT_RATIO 1 1 1\n"
            + "ORIGIN 0 0 0\n"
            + f"CELL_DATA {len(fields)}\n"
        )

        for label in fields.columns:
            output_text += (
                f"SCALARS {label} float\n"
                + "LOOKUP_TABLE default\n"
                + "\n".join(fields[label].astype(str))
                + "\n"
            )

        with open(output, "w") as output_file:
            output_file.write(output_text)

    def export_to_evpfft(
        self,
        orientation_label="orientation",
        grain_label="grain",
        phase_label="phase",
        output="fields.txt",
        euler_angles_to_degrees=True,
    ):
        """
        Export material data for EVP-FFT crystal plasticity solver.

        Parameters
        ----------
        orientation_label : string, default "orientation"
            Name of field containing Materialite Orientations of the points.
        grain_label : str, default "grain"
            Name of field containing grain IDs.
        phase_label : str, default "phase"
            Name of field containing phase IDs.
        output : str, default "fields.txt"
            Output file path.
        euler_angles_to_degrees : bool, default True
            Whether to write Euler angles in degrees (True) or radians (False).
        """

        valid_dimensions = np.array(
            [power_of_two_below(dim) for dim in self.dimensions]
        )

        if not np.array_equal(valid_dimensions, self.dimensions):
            warnings.warn(
                "Outputting to EVP-FFT with dimensions that are not powers of 2. Use Material.crop_by_range() or Material.crop_by_id_range() if the dimensions must be powers of 2."
            )
        if euler_angles_to_degrees:
            euler_angles = self.extract(orientation_label).euler_angles_in_degrees
        else:
            euler_angles = self.extract(orientation_label).euler_angles
        x_id = self.extract("x_id") + 1
        y_id = self.extract("y_id") + 1
        z_id = self.extract("z_id") + 1
        grains = self.extract(grain_label)
        phases = self.extract(phase_label)

        pd.DataFrame(
            {
                "euler_angle_1": euler_angles[:, 0],
                "euler_angle_2": euler_angles[:, 1],
                "euler_angle_3": euler_angles[:, 2],
                "x": x_id,
                "y": y_id,
                "z": z_id,
                "grain": grains,
                "phase": phases,
            }
        ).sort_values(by=["z", "y", "x"]).to_csv(
            output, header=False, sep=" ", index=False, float_format="%.5f"
        )


class Feature:
    """
    Abstract base class for geometric features used to define regions in materials.

    Features are used with Material.insert_feature() to assign properties to
    specific geometric regions.
    """

    @abstractmethod
    def check_inside(self):
        """Check if points are inside the feature geometry."""
        raise NotImplementedError

    def _ensure_compatible_dims(self, tensor):
        """Ensure tensor doesn't use 'p' dimension which conflicts with material points."""
        if "p" not in tensor.dims_str:
            return tensor
        return tensor.with_dims(tensor.dims_str.replace("p", "r"))


class Sphere(Feature):
    """
    Spherical feature for defining spherical regions.

    Parameters
    ----------
    radius : float or array-like
        Radius of the sphere(s).
    centroid : array-like
        Center coordinates [x, y, z] of the sphere(s).
    """

    def __init__(self, radius, centroid):
        self.radius = self._ensure_compatible_dims(Scalar(radius))
        self.centroid = self._ensure_compatible_dims(Vector(centroid))
        if self.radius.shape != self.centroid.shape:
            raise ValueError(
                f"Different numbers of radii and centroids were provided to Sphere"
            )

    def check_inside(self, points):
        return (points - self.centroid).norm.components <= self.radius.components


class Superellipsoid(Feature):
    """
    Superellipsoid feature for defining various rounded shapes.

    Parameters
    ----------
    x_radius : float or array-like
        Semi-axis length in the x direction.
    y_radius : float or array-like
        Semi-axis length in the y direction.
    z_radius : float or array-like
        Semi-axis length in the z direction.
    shape_exponent : float or array-like
        Exponent controlling shape (2.0 = ellipsoid, >2 = box-like, <2 = diamond-like).
    centroid : array-like
        Center coordinates [x, y, z] of the superellipsoid(s).
    """

    def __init__(self, x_radius, y_radius, z_radius, shape_exponent, centroid):
        self.x_radius = self._ensure_compatible_dims(Scalar(x_radius))
        self.y_radius = self._ensure_compatible_dims(Scalar(y_radius))
        self.z_radius = self._ensure_compatible_dims(Scalar(z_radius))
        self.shape_exponent = self._ensure_compatible_dims(Scalar(shape_exponent))
        self.centroid = self._ensure_compatible_dims(Vector(centroid))
        shapes = [
            self.x_radius.shape,
            self.y_radius.shape,
            self.z_radius.shape,
            self.shape_exponent.shape,
            self.centroid.shape,
        ]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(
                f"Different numbers of x radii, y radii, z radii, shape exponents, and/or centroids were provided to Superellipsoid"
            )

    def check_inside(self, points):
        # Stack radii into a single tensor for vectorized computation
        radius = Scalar.from_stack(
            [self.x_radius, self.y_radius, self.z_radius],
            new_dim="b",
        )

        return (
            (((points - self.centroid) * Vector.basis()).abs / radius)
            ** self.shape_exponent
        ).sum("b").components <= 1


class Box(Feature):
    """
    Rectangular box feature for defining box-shaped regions.

    Parameters
    ----------
    min_corner : array-like, default [-inf, -inf, -inf]
        Minimum corner coordinates [x, y, z]. Use None for -inf.
    max_corner : array-like, default [inf, inf, inf]
        Maximum corner coordinates [x, y, z]. Use None for +inf.
    """

    def __init__(
        self,
        min_corner=[-np.inf, -np.inf, -np.inf],
        max_corner=[np.inf, np.inf, np.inf],
    ):
        self.min_corner = self._ensure_compatible_dims(
            self._remove_nones(min_corner, -np.inf)
        )
        self.max_corner = self._ensure_compatible_dims(
            self._remove_nones(max_corner, np.inf)
        )
        if self.max_corner.shape != self.min_corner.shape:
            raise ValueError(
                f"Different numbers of min_corners and max_corners were provided to Box"
            )

    def _remove_nones(self, corner, default_for_none):
        """Convert iterable to Vector, replacing None values with default."""
        if isinstance(corner, Vector):
            corner = corner.components
        corner = np.array(corner)
        corner_shape = corner.shape
        corner = corner.ravel()

        # Replace None values with the appropriate infinity
        processed = [default_for_none if x is None else x for x in corner]
        return Vector(np.reshape(processed, corner_shape))

    def check_inside(self, points):
        above_min = (points - self.min_corner).components >= 0
        below_max = -(points - self.max_corner).components >= 0

        # Point is inside if all coordinates are between min and max
        return np.logical_and(np.all(above_min, axis=-1), np.all(below_max, axis=-1))


class ReshapeAdapter(ABC):
    """
    Abstract base class for field reshaping strategies.

    Strategies define how to transform flat C-ordered field arrays into
    formats expected by different processing libraries, and how to convert
    results back to flat arrays for Material storage.
    """

    @abstractmethod
    def reshape_field(self, field, dimensions):
        """
        Reshape a flat field array to the required format.

        Parameters
        ----------
        field : ndarray
            Flat 1D array to reshape
        dimensions : ndarray
            Material dimensions (nx, ny, nz)

        Returns
        -------
        ndarray
            Reshaped array in the format needed by the target function
        """
        pass

    @abstractmethod
    def reshape_result(self, result):
        """
        Reshape function result back to flat 1D array.

        Parameters
        ----------
        result : ndarray
            Result from the applied function

        Returns
        -------
        ndarray
            Flattened result for Material storage
        """
        pass


class NoReshape(ReshapeAdapter):
    """Pass fields as flat arrays (default behavior)."""

    def reshape_field(self, field, dimensions):
        return field

    def reshape_result(self, result):
        return result


class Reshape3D(ReshapeAdapter):
    """
    Reshape to full 3D volume (D, H, W).

    Used for volumetric operations on 3D data (e.g., scipy.ndimage,
    skimage filters on volumes).
    """

    def reshape_field(self, field, dimensions):
        return field.reshape(dimensions)

    def reshape_result(self, result):
        return result.ravel()


class ReshapeTomopy(ReshapeAdapter):
    """
    Reshape to tomopy-style (1, H, W) for single-layer operations.

    TomoPy expects a batch dimension even for single slices. Requires
    exactly one singleton dimension in the Material.
    """

    def reshape_field(self, field, dimensions):
        singleton_dims = np.where(dimensions == 1)[0]
        if len(singleton_dims) != 1:
            raise ValueError(
                f"ReshapeTomopy requires one singleton dimension, got {dimensions}"
            )
        squeezed = field.reshape(dimensions).squeeze()
        return squeezed[np.newaxis, :, :]  # Add batch dimension

    def reshape_result(self, result):
        return result.squeeze().ravel()


class ReshapeImage(ReshapeAdapter):
    """
    Reshape to 2D image (H, W) for skimage-style operations.

    Most skimage functions expect pure 2D arrays. Requires exactly one
    singleton dimension in the Material.
    """

    def reshape_field(self, field, dimensions):
        singleton_dims = np.where(dimensions == 1)[0]
        if len(singleton_dims) != 1:
            raise ValueError(
                f"ReshapeImage requires one singleton dimension, got {dimensions}"
            )
        return field.reshape(dimensions).squeeze()

    def reshape_result(self, result):
        return result.ravel()
