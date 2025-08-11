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

from abc import ABC, abstractmethod
from copy import deepcopy
from numbers import Number

import numpy as np
from numpy.linalg import inv

from materialite.basis_operations import (
    mandel_basis,
    mandel_product_basis,
    natural_basis,
    natural_product_basis,
    strain_voigt_basis,
    strain_voigt_dual_basis,
    stress_voigt_basis,
    stress_voigt_dual_basis,
    voigt_dual_product_basis,
    voigt_product_basis,
)

# dimension descriptions:
# p: points
# s: slip systems (or other per-point stuff)
# i, j: Cartesian indices
# m, n: Mandel basis indices

DIM_NAMES = {
    "p": "points",
    "s": "slip systems",
    "i": "Cartesian components (i)",
    "j": "Cartesian components (j)",
    "m": "Mandel basis components (m)",
    "n": "Mandel basis components (n)",
}
DIM_ORDER = {"p": 1, "s": 2, "i": 3, "j": 4, "m": 3, "n": 4}
ORDER_FUNC = lambda x: DIM_ORDER[x]
INNER_PRODUCT_INDICES = {
    "j": "j",
    "n": "n",
    "ij": "ij",
    "mn": "mn",
    "jj": "",
    "nij": "",
}


def order_dims(dims):
    return "".join(sorted(dims, key=ORDER_FUNC))


def cartesian_to_reduced(matrices, basis):
    matrices = np.asarray(matrices)
    dims = "abij"
    actual_dims = dims[-len(matrices.shape) :]
    return np.einsum(f"nij, {actual_dims}", basis, matrices, optimize=True)


def reduced_to_cartesian(vectors, basis):
    dims = "abn"
    actual_dims = dims[-len(vectors.shape) :]
    return np.einsum(f"nij, {actual_dims}", basis, vectors, optimize=True)


def reduced_matrix_to_cartesian(matrices, basis):
    dims = "abmn"
    actual_dims = dims[-len(matrices.shape) :]
    return np.einsum(f"mnijkl, {actual_dims}", basis, matrices, optimize=True)


def cartesian_to_reduced_matrix(cartesians, basis):
    cartesians = np.asarray(cartesians)
    dims = "abijkl"
    actual_dims = dims[-len(cartesians.shape) :]
    return np.einsum(f"mnijkl, {actual_dims}", basis, cartesians, optimize=True)


def convert_vector_basis(vector, from_basis, to_basis):
    vector = np.asarray(vector)
    dims = "abn"
    actual_dims = dims[-len(vector.shape) :]
    return np.einsum(
        f"{actual_dims}, nij, qij", vector, from_basis, to_basis, optimize=True
    )


def convert_matrix_basis(matrix, from_basis, to_basis):
    matrix = np.asarray(matrix)
    dims = "abmn"
    actual_dims = dims[-len(matrix.shape) :]
    return np.einsum(
        f"{actual_dims}, mnijkl, pqijkl", matrix, from_basis, to_basis, optimize=True
    )


def _check_consistent_dims(tensor):
    obj = type(tensor)
    num_dims = len(tensor.indices_set)
    shape = tensor.components.shape
    if num_dims != len(shape):
        raise ValueError(
            f"tried to create {obj} with dimensions {tensor.indices_str} but components have shape {shape}"
        )


def _check_allowed_dims(dims, allowed_dims):
    if not dims in allowed_dims:
        raise ValueError(
            f"dimensions passed to tensors must be points (p) and/or slip systems (s); provided dimensions were {dims}"
        )


def _broadcast_components(tensor1, tensor2):
    if tensor1.dims_str == "ps" and tensor2.dims_str == "p":
        components1 = tensor1.components
        components2 = tensor2.components[:, np.newaxis, ...]
    elif tensor1.dims_str == "p" and tensor2.dims_str == "ps":
        components1 = tensor1.components[:, np.newaxis, ...]
        components2 = tensor2.components
    else:
        components1 = tensor1.components
        components2 = tensor2.components
    return components1, components2


def _broadcast_cartesian(tensor1, tensor2):
    if tensor1.dims_str == "ps" and tensor2.dims_str == "p":
        components1 = tensor1.cartesian
        components2 = tensor2.cartesian[:, np.newaxis, ...]
    elif tensor1.dims_str == "p" and tensor2.dims_str == "ps":
        components1 = tensor1.cartesian[:, np.newaxis, ...]
        components2 = tensor2.cartesian
    else:
        components1 = tensor1.cartesian
        components2 = tensor2.cartesian
    return components1, components2


class Orientation:
    _allowed_dims = ["ps", "p", "s", ""]
    _dim_lookup = {0: "", 1: "p", 2: "ps"}
    __array_ufunc__ = None

    def __init__(self, rotation_matrix, dims=None):
        if isinstance(rotation_matrix, Orientation):
            self.rotation_matrix = rotation_matrix.rotation_matrix.copy()
            self.indices_str = rotation_matrix.indices_str
            self.dims_str = rotation_matrix.dims_str
            self.indices_set = rotation_matrix.indices_set
            self.dims_set = rotation_matrix.dims_set
            return
        self.rotation_matrix = np.asarray(rotation_matrix)
        if dims is None:
            num_indices = len(self.rotation_matrix.shape) - 2
            dims = self._dim_lookup[num_indices]
        self.dims_str = dims
        self.indices_str = dims + "ij"
        self.dims_set = set(self.dims_str)
        self.indices_set = set(self.indices_str)
        _check_allowed_dims(dims, self._allowed_dims)
        matrix_shape = self.rotation_matrix.shape
        if len(self.indices_set) != len(matrix_shape):
            raise ValueError(
                f"tried to create Orientation with dimensions {self.indices_str} but rotation matrix has shape {matrix_shape}"
            )

    @property
    def rotation_matrix_mandel(self):
        R_mandel = np.zeros((*self.shape, 6, 6))
        R = self.rotation_matrix
        r2 = np.sqrt(2)
        R_mandel[..., :3, :3] = R**2
        R_mandel[..., 0, 3] = r2 * R[..., 0, 1] * R[..., 0, 2]
        R_mandel[..., 0, 4] = r2 * R[..., 0, 0] * R[..., 0, 2]
        R_mandel[..., 0, 5] = r2 * R[..., 0, 0] * R[..., 0, 1]
        R_mandel[..., 1, 3] = r2 * R[..., 1, 1] * R[..., 1, 2]
        R_mandel[..., 1, 4] = r2 * R[..., 1, 0] * R[..., 1, 2]
        R_mandel[..., 1, 5] = r2 * R[..., 1, 0] * R[..., 1, 1]
        R_mandel[..., 2, 3] = r2 * R[..., 2, 1] * R[..., 2, 2]
        R_mandel[..., 2, 4] = r2 * R[..., 2, 0] * R[..., 2, 2]
        R_mandel[..., 2, 5] = r2 * R[..., 2, 0] * R[..., 2, 1]
        R_mandel[..., 3, 0] = r2 * R[..., 1, 0] * R[..., 2, 0]
        R_mandel[..., 3, 1] = r2 * R[..., 1, 1] * R[..., 2, 1]
        R_mandel[..., 3, 2] = r2 * R[..., 1, 2] * R[..., 2, 2]
        R_mandel[..., 4, 0] = r2 * R[..., 0, 0] * R[..., 2, 0]
        R_mandel[..., 4, 1] = r2 * R[..., 0, 1] * R[..., 2, 1]
        R_mandel[..., 4, 2] = r2 * R[..., 0, 2] * R[..., 2, 2]
        R_mandel[..., 5, 0] = r2 * R[..., 0, 0] * R[..., 1, 0]
        R_mandel[..., 5, 1] = r2 * R[..., 0, 1] * R[..., 1, 1]
        R_mandel[..., 5, 2] = r2 * R[..., 0, 2] * R[..., 1, 2]
        R_mandel[..., 3, 3] = R[..., 1, 1] * R[..., 2, 2] + R[..., 1, 2] * R[..., 2, 1]
        R_mandel[..., 3, 4] = R[..., 1, 0] * R[..., 2, 2] + R[..., 1, 2] * R[..., 2, 0]
        R_mandel[..., 3, 5] = R[..., 1, 0] * R[..., 2, 1] + R[..., 1, 1] * R[..., 2, 0]
        R_mandel[..., 4, 3] = R[..., 0, 1] * R[..., 2, 2] + R[..., 0, 2] * R[..., 2, 1]
        R_mandel[..., 4, 4] = R[..., 0, 0] * R[..., 2, 2] + R[..., 0, 2] * R[..., 2, 0]
        R_mandel[..., 4, 5] = R[..., 0, 0] * R[..., 2, 1] + R[..., 0, 1] * R[..., 2, 0]
        R_mandel[..., 5, 3] = R[..., 0, 1] * R[..., 1, 2] + R[..., 0, 2] * R[..., 1, 1]
        R_mandel[..., 5, 4] = R[..., 0, 0] * R[..., 1, 2] + R[..., 0, 2] * R[..., 1, 0]
        R_mandel[..., 5, 5] = R[..., 0, 0] * R[..., 1, 1] + R[..., 0, 1] * R[..., 1, 0]
        return np.squeeze(R_mandel)

    @property
    def shape(self):
        num_dims = len(self.dims_set)
        if num_dims == 0:
            return ()
        return self.rotation_matrix.shape[:num_dims]

    def __len__(self):
        if len(self.dims_set) == 0:
            return None
        else:
            return len(self.rotation_matrix)

    def __iter__(self):
        return OrientationIterator(self.rotation_matrix)

    def __getitem__(self, slice_):
        if len(self.dims_str) is None:
            raise ValueError("can't index")
        return Orientation(self.rotation_matrix[slice_])

    def __setitem__(self, key, item):
        if not isinstance(item, Orientation):
            raise ValueError(f"tried to set Orientation with {type(item)}")
        self.rotation_matrix[key] = item.rotation_matrix

    @classmethod
    def identity(cls):
        return cls(np.eye(3))

    @classmethod
    def from_miller_indices(cls, plane, direction, dims=None):
        plane = np.asarray(plane)
        plane = plane / np.linalg.norm(plane, axis=-1)[..., np.newaxis]
        direction = np.asarray(direction)
        direction = direction / np.linalg.norm(direction, axis=-1)[..., np.newaxis]
        td = np.cross(plane, direction)
        rotation_matrix = np.array([direction, td, plane])
        rotation_matrix = np.moveaxis(rotation_matrix, 0, -1)
        return cls(rotation_matrix, dims)

    @classmethod
    def from_rotation_matrix(cls, rotation_matrix, dims=None):
        return cls(rotation_matrix, dims)

    @classmethod
    def from_euler_angles(cls, euler_angles, in_degrees=False, dims=None):
        """
        Bunge Euler Angle Convention

        The rotation matrix R formed from these Euler angles is used to take a vector's
        components relative to a specimen reference frame (v_i) and transform them to that same
        vector's components relative to the crystal reference frame (v'_i).

        v'_i = R_ij * v_j

        R can also be used to construct the crystal basis *vectors* (e'_i) as a linear combination
        of specimen basis *vectors* (e_i).

        e'_i = R_ij * e_j

        R can equivalently be written in terms of dot products of the basis vectors.

        R_ij = e'_i . e_j

        """
        euler_angles = np.asarray(euler_angles, dtype=np.float64)
        if in_degrees:
            euler_angles *= np.pi / 180.0

        z1 = euler_angles[..., 0]
        x2 = euler_angles[..., 1]
        z3 = euler_angles[..., 2]
        c1, c2, c3 = np.cos(z1), np.cos(x2), np.cos(z3)
        s1, s2, s3 = np.sin(z1), np.sin(x2), np.sin(z3)

        rotation_matrix = np.zeros((*euler_angles.shape[:-1], 3, 3))
        rotation_matrix[..., 0, 0] = c1 * c3 - c2 * s1 * s3
        rotation_matrix[..., 0, 1] = c3 * s1 + c1 * c2 * s3
        rotation_matrix[..., 0, 2] = s2 * s3
        rotation_matrix[..., 1, 0] = -c1 * s3 - c2 * c3 * s1
        rotation_matrix[..., 1, 1] = c1 * c2 * c3 - s1 * s3
        rotation_matrix[..., 1, 2] = c3 * s2
        rotation_matrix[..., 2, 0] = s1 * s2
        rotation_matrix[..., 2, 1] = -c1 * s2
        rotation_matrix[..., 2, 2] = c2

        return cls(np.squeeze(rotation_matrix), dims)

    @classmethod
    def random(cls, num=100, rng=np.random.default_rng()):
        z1 = rng.random(num) * 2.0 * np.pi
        cos_x2 = rng.random(num) * 2.0 - 1.0
        x2 = np.arccos(cos_x2)
        z3 = rng.random(num) * 2.0 * np.pi
        return cls.from_euler_angles(np.c_[z1, x2, z3])

    @classmethod
    def from_list(cls, orientations):
        rotation_matrices = [o.rotation_matrix for o in orientations]
        return cls(rotation_matrices)

    @property
    def euler_angles(self):
        # Source: "Euler Angle Formulas", David Eberly
        R = self.rotation_matrix
        n = self.shape

        R22_less_than_one = R[..., 2, 2] < 1.0
        R22_equals_one = np.logical_not(R22_less_than_one)

        R22_greater_than_negative_one = R[..., 2, 2] > -1.0
        R22_equals_negative_one = np.logical_not(R22_greater_than_negative_one)

        R22_default = np.logical_and(R22_less_than_one, R22_greater_than_negative_one)
        z1 = np.arctan2(R[..., 2, 0], -R[..., 2, 1])
        x2 = np.arccos(R[..., 2, 2])
        z3 = np.arctan2(R[..., 0, 2], R[..., 1, 2])
        eulers_default = np.moveaxis(np.array([z1, x2, z3]), 0, -1)

        if np.all(R22_default):
            return np.squeeze(eulers_default)

        eulers_negative_one = np.array(
            [np.arctan2(R[..., 1, 0], R[..., 0, 0]), np.pi * np.ones(n), np.zeros(n)]
        )
        eulers_negative_one = np.moveaxis(eulers_negative_one, 0, -1)
        eulers_one = np.array(
            [np.arctan2(-R[..., 1, 0], R[..., 0, 0]), np.zeros(n), np.zeros(n)]
        )
        eulers_one = np.moveaxis(eulers_one, 0, -1)

        # Three conditions rolled into a messy operation
        return np.squeeze(
            np.einsum("..., ...j -> ...j", R22_default, eulers_default)
            + np.einsum(
                "..., ...j -> ...j", R22_equals_negative_one, eulers_negative_one
            )
            + np.einsum("..., ...j -> ...j", R22_equals_one, eulers_one)
        )

    @property
    def euler_angles_in_degrees(self):
        return self.euler_angles * 180.0 / np.pi

    @property
    def trace(self):
        return Scalar(np.einsum("...ii -> ...", self.rotation_matrix), self.dims_str)

    def __repr__(self):
        dimensions = ", ".join([DIM_NAMES[i] for i in self.dims_str])
        return (
            f"{type(self).__name__}("
            + str(np.round(self.euler_angles, 3))
            + f", dims: ({dimensions}), Euler angles shape: {self.euler_angles.shape})"
        )

    def __matmul__(self, orientation):
        if not isinstance(orientation, Orientation):
            return NotImplemented
        u = order_dims(self.dims_set.union(orientation.dims_set))
        other_indices = orientation.dims_str + "jk"
        output_indices = u + "ik"
        return Orientation(
            np.einsum(
                f"{self.indices_str}, {other_indices} -> {output_indices}",
                self.rotation_matrix,
                orientation.rotation_matrix,
                optimize=True,
            ),
            u,
        )


class OrientationIterator:
    def __init__(self, rotation_matrix):
        self.idx = 0
        self.rotation_matrix = rotation_matrix

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        try:
            return Orientation(self.rotation_matrix[self.idx - 1])
        except IndexError:
            self.idx = 0
            raise StopIteration


class Tensor(ABC):
    _allowed_dims = ["ps", "p", "s", ""]
    _dim_lookup = {0: "", 1: "p", 2: "ps"}
    __array_ufunc__ = None

    def __init__(self, components, dims):
        if isinstance(components, Tensor):
            self.components = components.components.copy()
            self.indices_str = components.indices_str
            self.dims_str = components.dims_str
            self.indices_set = components.indices_set
            self.dims_set = components.dims_set
            self.axis_dict = components.axis_dict
            return
        self.components = np.asarray(components)
        if dims is None:
            num_indices = len(self.components.shape) - self._component_dims
            dims = self._dim_lookup[num_indices]
        self.indices_str = dims + self._component_indices
        self.dims_str = dims
        self.indices_set = set(self.indices_str)
        self.dims_set = set(dims)
        self.axis_dict = dict(zip(dims, [0, 1]))
        _check_allowed_dims(dims, self._allowed_dims)
        _check_consistent_dims(self)

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        dimensions = ", ".join([DIM_NAMES[i] for i in self.dims_str])
        return (
            f"{type(self).__name__}("
            + str(self.components)
            + f", dims: ({dimensions}), components shape: {self.components.shape})"
        )

    def __len__(self):
        if len(self.dims_set) == 0:
            return None
        else:
            return len(self.components)

    def __iter__(self):
        return TensorIterator(self)

    def __neg__(self):
        return type(self)(-self.components, self.dims_str)

    def __getitem__(self, slice_):
        self._check_valid_slice(slice_)
        components = self.components[slice_]
        num_components = len(components.shape)
        if num_components == self._component_dims:
            dims = ""
        elif num_components == len(self.indices_set):
            dims = self.dims_str
        else:
            dims = (
                "s"
                if (isinstance(slice_, Number) or isinstance(slice_[0], Number))
                else "p"
            )
        return type(self)(components, dims)

    def _check_valid_slice(self, slice_):
        if len(self.dims_set) == 0:
            raise ValueError(f"Tried to index {type(self).__name__} with no dimensions")
        elif (
            isinstance(slice_, Number)
            or isinstance(slice_, slice)
            or isinstance(slice_, list)
            or isinstance(slice_, np.ndarray)
        ):
            return
        elif len(slice_) > len(self.dims_set):
            dimensions = ", ".join([DIM_NAMES[i] for i in self.dims_str])
            raise ValueError(
                f"Provided {len(slice_)} indices to {type(self).__name__} with dimensions ({dimensions})"
            )
        elif slice_[0] == ...:
            raise ValueError(
                f"Tried to index {type(self).__name__} with ellipses and an index"
            )
        return

    def sum(self, dim=None):
        axis, str_dims = self._dims_for_max_sum_mean(dim, "sum")
        return type(self)(np.sum(self.components, axis=axis), str_dims)

    def mean(self, dim=None):
        if dim is None and self.dims_str == "":
            return self
        axis, str_dims = self._dims_for_max_sum_mean(dim, "mean")
        return type(self)(np.mean(self.components, axis=axis), str_dims)

    def _dims_for_max_sum_mean(self, dim, sum_or_mean):
        if dim is None:
            dim = "s" if "s" in self.dims_str else "p"
        axis = self.axis_dict.get(dim, None)
        if axis is None:
            raise ValueError(
                f"tried to take {sum_or_mean} over {dim} dim in {type(self).__name__} with dims {self.dims_str}"
            )
        str_dims = order_dims(self.dims_set - {dim})
        return axis, str_dims

    @property
    def shape(self):
        num_dims = len(self.dims_set)
        if num_dims == 0:
            return ()
        return self.components.shape[:num_dims]

    @classmethod
    def from_list(cls, tensor, dims=None):
        components = np.array([t.components for t in tensor])
        return cls(components, dims)

    @abstractmethod
    def __mul__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __rmul__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __matmul__(self, *args, **kwargs):
        raise NotImplementedError


class TensorIterator:
    def __init__(self, tensor):
        self.idx = 0
        self.tensor = tensor

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        try:
            return self.tensor[self.idx - 1]
        except IndexError:
            self.idx = 0
            raise StopIteration


class Scalar(Tensor):
    _component_dims = 0
    _component_indices = ""

    def __init__(self, components, dims=None):
        super().__init__(components, dims)

    @classmethod
    def random(cls, num, rng=np.random.default_rng()):
        components = rng.random(num)
        return cls(components)

    @classmethod
    def zero(cls):
        return cls(0)

    @classmethod
    def zeros(cls, shape, dims=None):
        return cls(np.zeros(shape), dims)

    @property
    def abs(self):
        return Scalar(np.abs(self.components), self.dims_str)

    @property
    def sqrt(self):
        return Scalar(np.sqrt(self.components), self.dims_str)

    @property
    def cosh(self):
        return Scalar(np.cosh(self.components), self.dims_str)

    def max(self, dim=None):
        axis, str_dims = self._dims_for_max_sum_mean(dim, "max")
        return Scalar(np.max(self.components, axis=axis), str_dims)

    def apply(self, function):
        return Scalar(function(self.components), self.dims_str)

    def __setitem__(self, key, item):
        if isinstance(item, Number):
            self.components[key] = item
        elif not isinstance(item, Scalar):
            raise ValueError(f"tried to set Scalar with {type(item)}")
        else:
            self.components[key] = item.components

    def __pow__(self, tensor):
        if isinstance(tensor, Number):
            return Scalar(self.components**tensor, self.dims_str)
        elif isinstance(tensor, Scalar):
            components1, components2 = _broadcast_components(self, tensor)
            return Scalar(components1**components2, self.dims_str)
        return NotImplemented

    def __add__(self, tensor):
        if isinstance(tensor, Number):
            return Scalar(self.components + tensor, self.dims_str)
        dims = order_dims(self.dims_set.union(tensor.dims_set))
        if isinstance(tensor, Scalar):
            components1, components2 = _broadcast_components(self, tensor)
            return Scalar(components1 + components2, dims)
        return NotImplemented

    def __radd__(self, tensor):
        return self.__add__(tensor)

    def __sub__(self, tensor):
        if isinstance(tensor, Number):
            return Scalar(self.components - tensor, self.dims_str)
        dims = order_dims(self.dims_set.union(tensor.dims_set))
        if isinstance(tensor, Scalar):
            components1, components2 = _broadcast_components(self, tensor)
            return Scalar(components1 - components2, dims)
        return NotImplemented

    def __rsub__(self, tensor):
        return -self + tensor

    def __truediv__(self, tensor):
        if isinstance(tensor, Number):
            return Scalar(self.components / tensor, self.indices_str)
        elif isinstance(tensor, Scalar):
            components1, components2 = _broadcast_components(self, tensor)
            return Scalar(components1 / components2, self.indices_str)
        return NotImplemented

    def __rtruediv__(self, scalar):
        if isinstance(scalar, Number):
            return Scalar(scalar / self.components, self.indices_str)
        return NotImplemented

    def __mul__(self, tensor):
        if isinstance(tensor, Number):
            return Scalar(tensor * self.components, self.indices_str)

        u = order_dims(self.dims_set.union(tensor.dims_set))
        output_indices = u + tensor._component_indices
        new_type = type(tensor)
        return new_type(
            np.einsum(
                f"{self.indices_str}, {tensor.indices_str} -> {output_indices}",
                self.components,
                tensor.components,
                optimize=True,
            ),
            u,
        )

    def __rmul__(self, tensor):
        return self.__mul__(tensor)

    def __matmul__(self, tensor):
        raise ValueError("cannot matmul with Scalar")

    def __rmatmul__(self, tensor):
        raise ValueError("cannot matmul with Scalar")


class Vector(Tensor):
    _component_dims = 1
    _component_indices = "j"

    def __init__(self, components, dims=None):
        super().__init__(components, dims)

    @classmethod
    def random(cls, num, rng=np.random.default_rng()):
        components = np.squeeze(rng.random((num, 3)))
        return cls(components)

    @classmethod
    def random_unit(cls, num=1, rng=np.random.default_rng()):
        components = np.squeeze(rng.normal(size=(num, 3)))
        return cls(components).unit

    @classmethod
    def zero(cls):
        return cls(np.zeros(3))

    @classmethod
    def zeros(cls, shape, dims=None):
        if isinstance(shape, Number):
            return cls(np.zeros((shape, 3)), dims)
        else:
            return cls(np.zeros((*shape, 3)), dims)

    @property
    def cartesian(self):
        return self.components

    @property
    def norm(self):
        components = np.einsum("...i, ...i", self.components, self.components)
        return Scalar(np.sqrt(components), self.dims_str)

    @property
    def unit(self):
        return self / self.norm

    def __setitem__(self, key, item):
        if isinstance(item, Number):
            self.components[key] = item
        elif not isinstance(item, Vector):
            raise ValueError(f"tried to set Vector with {type(item)}")
        else:
            self.components[key] = item.components

    def __add__(self, tensor):
        if isinstance(tensor, Number):
            return Vector(self.components + tensor, self.dims_str)
        dims = order_dims(self.dims_set.union(tensor.dims_set))
        if isinstance(tensor, Vector):
            components1, components2 = _broadcast_components(self, tensor)
            return Vector(components1 + components2, dims)
        return NotImplemented

    def __sub__(self, tensor):
        if isinstance(tensor, Number):
            return Vector(self.components - tensor, self.dims_str)
        dims = order_dims(self.dims_set.union(tensor.dims_set))
        if isinstance(tensor, Vector):
            components1, components2 = _broadcast_components(self, tensor)
            return Vector(components1 - components2, dims)
        return NotImplemented

    def __truediv__(self, tensor):
        if isinstance(tensor, Scalar):
            return Vector(
                self.components / tensor.components[..., np.newaxis],
                self.dims_str,
            )
        elif isinstance(tensor, Number):
            return Vector(self.components / tensor, self.dims_str)
        return NotImplemented

    def __mul__(self, tensor):
        if isinstance(tensor, Number):
            return Vector(tensor * self.components, self.dims_str)

        u = self.dims_set.union(tensor.dims_set)

        if isinstance(tensor, Vector):
            output_indices = order_dims(u)
            return Scalar(
                np.einsum(
                    f"{self.indices_str}, {tensor.indices_str} -> {output_indices}",
                    self.components,
                    tensor.components,
                    optimize=True,
                ),
                output_indices,
            )

        return NotImplemented

    def __rmul__(self, tensor):
        return self.__mul__(tensor)

    def __matmul__(self, tensor):
        return NotImplemented

    def outer(self, tensor):
        if not isinstance(tensor, Vector):
            raise ValueError(
                f"tried to do outer product of Vector with {type(tensor).__name__}"
            )
        self_dims = self.dims_str + "i"
        output_indices = order_dims(self.dims_set.union(tensor.dims_set)) + "ij"
        return Order2Tensor(
            np.einsum(
                f"{self_dims}, {tensor.indices_str} -> {output_indices}",
                self.components,
                tensor.components,
                optimize=True,
            ),
            output_indices[:-2],
        )

    def to_crystal_frame(self, orientations):
        output_dims, output_indices = self._get_transformation_indices(orientations)
        orientation_indices = orientations.dims_str + "mj"
        components = np.einsum(
            f"{orientation_indices}, {self.indices_str} -> {output_indices}",
            orientations.rotation_matrix,
            self.components,
            optimize=True,
        )
        return Vector(components, output_dims)

    def to_specimen_frame(self, orientations):
        output_dims, output_indices = self._get_transformation_indices(orientations)
        orientation_indices = orientations.dims_str + "jm"
        components = np.einsum(
            f"{orientation_indices}, {self.indices_str} -> {output_indices}",
            orientations.rotation_matrix,
            self.components,
            optimize=True,
        )
        return Vector(components, output_dims)

    def _get_transformation_indices(self, orientations):
        output_dims = order_dims(self.dims_set.union(orientations.dims_set))
        output_indices = output_dims + "m"
        return output_dims, output_indices


class Order2Tensor(Tensor):
    _component_dims = 2
    _component_indices = "ij"
    _mul_lookup = {"n": "ij", "ij": "ij"}
    _matmul_lookup = {"n": ["jk", "ik"], "ij": ["jk", "ik"], "j": ["j", "i"]}

    def __init__(self, components, dims=None):
        super().__init__(components, dims)

    @classmethod
    def identity(cls):
        return cls(np.identity(3))

    @classmethod
    def zero(cls):
        return cls(np.zeros((3, 3)))

    @classmethod
    def zeros(cls, shape, dims=None):
        if isinstance(shape, Number):
            return cls(np.zeros((shape, 3, 3)), dims)
        else:
            return cls(np.zeros((*shape, 3, 3)), dims)

    @classmethod
    def random(cls, num, rng=np.random.default_rng()):
        components = np.squeeze(rng.random((num, 3, 3)))
        return cls(components)

    @classmethod
    def from_tensor_product(cls, vector1, vector2):
        dims = order_dims(vector1.dims_set.union(vector2.dims_set))
        op = np.einsum(
            f"{vector1.dims_str + 'i'}, {vector2.dims_str + 'j'} -> {dims + 'ij'}",
            vector1.components,
            vector2.components,
            optimize=True,
        )
        return Order2Tensor(op, dims)

    @property
    def cartesian(self):
        return self.components

    @property
    def T(self):
        return Order2Tensor(np.swapaxes(self.components, -1, -2), self.dims_str)

    @property
    def transpose(self):
        return self.T

    @property
    def inverse(self):
        return Order2Tensor(inv(self.components), self.dims_str)

    @property
    def inv(self):
        return self.inverse

    @property
    def sym(self):
        return Order2SymmetricTensor.from_cartesian(
            0.5 * (self.components + self.T.components), self.dims_str
        )

    @property
    def norm(self):
        components = np.einsum("...ij, ...ij", self.components, self.components)
        return Scalar(np.sqrt(components), self.dims_str)

    @property
    def trace(self):
        return Scalar(np.einsum("...ii -> ...", self.components), self.dims_str)

    @property
    def dev(self):
        volumetric = 1 / 3 * self.trace * Order2Tensor.identity()
        return self - volumetric

    def __setitem__(self, key, item):
        if isinstance(item, Number):
            self.components[key] = item
        elif not isinstance(item, Order2Tensor):
            raise ValueError(f"tried to set Order2Tensor with {type(item)}")
        else:
            self.components[key] = item.components

    def __add__(self, tensor):
        if isinstance(tensor, Number):
            return Order2Tensor(self.components + tensor, self.dims_str)
        dims = order_dims(self.dims_set.union(tensor.dims_set))
        if isinstance(tensor, Order2Tensor) or isinstance(
            tensor, Order2SymmetricTensor
        ):
            components1, components2 = _broadcast_cartesian(self, tensor)
            return Order2Tensor(components1 + components2, dims)
        return NotImplemented

    def __sub__(self, tensor):
        if isinstance(tensor, Number):
            return Order2Tensor(self.components - tensor, self.dims_str)
        dims = order_dims(self.dims_set.union(tensor.dims_set))
        if isinstance(tensor, Order2Tensor) or isinstance(
            tensor, Order2SymmetricTensor
        ):
            components1, components2 = _broadcast_cartesian(self, tensor)
            return Order2Tensor(components1 - components2, dims)
        return NotImplemented

    def __truediv__(self, tensor):
        if isinstance(tensor, Scalar):
            return Order2Tensor(
                self.components / tensor.components[..., np.newaxis, np.newaxis],
                self.dims_str,
            )
        elif isinstance(tensor, Number):
            return Order2Tensor(self.components / tensor, self.dims_str)
        return NotImplemented

    def __mul__(self, tensor):
        if isinstance(tensor, Number):
            return Order2Tensor(tensor * self.components, self.dims_str)

        output_indices = order_dims(self.dims_set.union(tensor.dims_set))
        other_indices = self._mul_lookup.get(tensor._component_indices)
        if other_indices is None:
            return NotImplemented
        other_indices = tensor.dims_str + other_indices
        return Scalar(
            np.einsum(
                f"{self.indices_str}, {other_indices} -> {output_indices}",
                self.cartesian,
                tensor.cartesian,
            ),
            output_indices,
        )

    def __rmul__(self, tensor):
        return self.__mul__(tensor)

    def __matmul__(self, tensor):
        u = order_dims(self.dims_set.union(tensor.dims_set))
        other_indices, output_indices = self._matmul_lookup.get(
            tensor._component_indices
        )
        if other_indices is None:
            return NotImplemented
        output_type = Order2Tensor if len(output_indices) == 2 else Vector
        other_indices = tensor.dims_str + other_indices
        output_indices = u + output_indices
        return output_type(
            np.einsum(
                f"{self.indices_str}, {other_indices} -> {output_indices}",
                self.cartesian,
                tensor.cartesian,
                optimize=True,
            ),
            u,
        )

    def to_crystal_frame(self, orientations):
        output_dims, output_indices = self._get_transformation_indices(orientations)
        orientation_indices_1 = orientations.dims_str + "mi"
        orientation_indices_2 = orientations.dims_str + "nj"
        components = np.einsum(
            f"{orientation_indices_1}, {orientation_indices_2}, {self.indices_str} -> {output_indices}",
            orientations.rotation_matrix,
            orientations.rotation_matrix,
            self.components,
            optimize=True,
        )
        return Order2Tensor(components, output_dims)

    def to_specimen_frame(self, orientations):
        output_dims, output_indices = self._get_transformation_indices(orientations)
        orientation_indices_1 = orientations.dims_str + "im"
        orientation_indices_2 = orientations.dims_str + "jn"
        components = np.einsum(
            f"{orientation_indices_1}, {orientation_indices_2}, {self.indices_str} -> {output_indices}",
            orientations.rotation_matrix,
            orientations.rotation_matrix,
            self.components,
            optimize=True,
        )
        return Order2Tensor(components, output_dims)

    def _get_transformation_indices(self, orientations):
        output_dims = order_dims(self.dims_set.union(orientations.dims_set))
        output_indices = output_dims + "mn"
        return output_dims, output_indices


class Order2SymmetricTensor(Tensor):
    _basis = mandel_basis()
    _component_dims = 1
    _component_indices = "n"
    _mul_lookup = {"n": "n", "ij": "ij"}
    _matmul_lookup = {"n": ["jk", "ik"], "ij": ["jk", "ik"], "j": ["j", "i"]}

    def __init__(self, components, dims=None):
        super().__init__(components, dims)

    @classmethod
    def identity(cls):
        return cls.from_cartesian(np.identity(3))

    @classmethod
    def zero(cls):
        return cls(np.zeros(6))

    @classmethod
    def zeros(cls, shape, dims=None):
        if isinstance(shape, Number):
            return cls(np.zeros((shape, 6)), dims)
        else:
            return cls(np.zeros((*shape, 6)), dims)

    @classmethod
    def random(cls, num, rng=np.random.default_rng()):
        components = np.squeeze(rng.random((num, 6)))
        return cls(components)

    @classmethod
    def from_tensor_product(cls, vector1, vector2):
        return Order2Tensor.from_tensor_product(vector1, vector2).sym

    @classmethod
    def from_cartesian(cls, matrices, dims=None):
        if not np.allclose(matrices, np.einsum("...ij -> ...ji", matrices), atol=1e-14):
            raise ValueError(
                "tried to create Order2SymmetricTensor using non-symmetric input"
            )
        return cls(cartesian_to_reduced(matrices, cls._basis), dims)

    @classmethod
    def from_strain_voigt(cls, vectors, dims=None):
        return cls(
            convert_vector_basis(vectors, strain_voigt_basis(), cls._basis),
            dims,
        )

    @classmethod
    def from_stress_voigt(cls, vectors, dims=None):
        return cls(
            convert_vector_basis(vectors, stress_voigt_basis(), cls._basis),
            dims,
        )

    @classmethod
    def from_natural(cls, vectors, dims=None):
        return cls(
            convert_vector_basis(vectors, natural_basis(), cls._basis),
            dims,
        )

    @classmethod
    def from_mandel(cls, components, dims=None):
        return cls(components, dims)

    @property
    def cartesian(self):
        return reduced_to_cartesian(self.components, self._basis)

    @property
    def strain_voigt(self):
        return convert_vector_basis(
            self.components, self._basis, strain_voigt_dual_basis()
        )

    @property
    def stress_voigt(self):
        return convert_vector_basis(
            self.components, self._basis, stress_voigt_dual_basis()
        )

    @property
    def natural(self):
        return convert_vector_basis(self.components, self._basis, natural_basis())

    @property
    def mandel(self):
        return self.components

    @property
    def T(self):
        return self

    @property
    def transpose(self):
        return self

    @property
    def inverse(self):
        return Order2SymmetricTensor.from_cartesian(inv(self.cartesian), self.dims_str)

    @property
    def inv(self):
        return self.inverse

    @property
    def norm(self):
        components = np.einsum("...n, ...n", self.components, self.components)
        return Scalar(np.sqrt(components), self.dims_str)

    @property
    def trace(self):
        components = np.sum(self.components[..., :3], axis=-1)
        return Scalar(components, self.dims_str)

    @property
    def dev(self):
        volumetric = 1 / 3 * self.trace * Order2SymmetricTensor.identity()
        return self - volumetric

    def __setitem__(self, key, item):
        if isinstance(item, Number):
            self.components[key] = item
        elif not isinstance(item, Order2SymmetricTensor):
            raise ValueError(f"tried to set Order2SymmetricTensor with {type(item)}")
        else:
            self.components[key] = item.components

    def __add__(self, tensor):
        if isinstance(tensor, Number):
            return Order2SymmetricTensor(self.components + tensor, self.dims_str)
        dims = order_dims(self.dims_set.union(tensor.dims_set))
        if isinstance(tensor, Order2Tensor):
            components1, components2 = _broadcast_cartesian(self, tensor)
            return Order2Tensor(components1 + components2, dims)
        elif isinstance(tensor, Order2SymmetricTensor):
            components1, components2 = _broadcast_components(self, tensor)
            return Order2SymmetricTensor(components1 + components2, dims)
        return NotImplemented

    def __sub__(self, tensor):
        if isinstance(tensor, Number):
            return Order2SymmetricTensor(self.components - tensor, self.dims_str)
        dims = order_dims(self.dims_set.union(tensor.dims_set))
        if isinstance(tensor, Order2Tensor):
            components1, components2 = _broadcast_cartesian(self, tensor)
            return Order2Tensor(components1 - components2, dims)
        elif isinstance(tensor, Order2SymmetricTensor):
            components1, components2 = _broadcast_components(self, tensor)
            return Order2SymmetricTensor(components1 - components2, dims)
        return NotImplemented

    def __truediv__(self, tensor):
        if isinstance(tensor, Scalar):
            return Order2SymmetricTensor(
                self.components / tensor.components[..., np.newaxis],
                self.dims_str,
            )
        elif isinstance(tensor, Number):
            return Order2SymmetricTensor(self.components / tensor, self.dims_str)
        return NotImplemented

    def __mul__(self, tensor):
        if isinstance(tensor, Number):
            return Order2SymmetricTensor(tensor * self.components, self.dims_str)

        output_indices = order_dims(self.dims_set.union(tensor.dims_set))
        indices = self._mul_lookup.get(tensor._component_indices)
        if indices is None:
            return NotImplemented
        elif indices == "ij":
            self_values = self.cartesian
        else:
            self_values = self.components
        self_indices = self.dims_str + indices
        other_indices = tensor.dims_str + indices
        return Scalar(
            np.einsum(
                f"{self_indices}, {other_indices} -> {output_indices}",
                self_values,
                tensor.components,
            ),
            output_indices,
        )

    def __rmul__(self, tensor):
        return self.__mul__(tensor)

    def __matmul__(self, tensor):
        u = order_dims(self.dims_set.union(tensor.dims_set))
        other_indices, output_indices = self._matmul_lookup.get(
            tensor._component_indices
        )
        if other_indices is None:
            return NotImplemented
        output_type = (
            Order2SymmetricTensor.from_cartesian if len(output_indices) == 2 else Vector
        )
        self_indices = self.dims_str + "ij"
        other_indices = tensor.dims_str + other_indices
        output_indices = u + output_indices
        components = np.einsum(
            f"{self_indices}, {other_indices} -> {output_indices}",
            self.cartesian,
            tensor.cartesian,
            optimize=True,
        )
        try:
            return output_type(components, u)
        except ValueError:
            return Order2Tensor(components, u)

    def outer(self, tensor):
        if not isinstance(tensor, Order2SymmetricTensor):
            raise ValueError(
                f"tried to do outer product of Order2SymmetricTensor with {type(tensor).__name__}"
            )
        self_dims = self.dims_str + "m"
        output_indices = order_dims(self.dims_set.union(tensor.dims_set)) + "mn"
        return Order4SymmetricTensor(
            np.einsum(
                f"{self_dims}, {tensor.indices_str} -> {output_indices}",
                self.components,
                tensor.components,
                optimize=True,
            ),
            output_indices[:-2],
        )

    def to_crystal_frame(self, orientations):
        output_dims, output_indices, self_indices = self._get_transformation_indices(
            orientations
        )
        orientation_indices_1 = orientations.dims_str + "mi"
        orientation_indices_2 = orientations.dims_str + "nj"
        components = np.einsum(
            f"{orientation_indices_1}, {orientation_indices_2}, {self_indices} -> {output_indices}",
            orientations.rotation_matrix,
            orientations.rotation_matrix,
            self.cartesian,
            optimize=True,
        )
        return Order2SymmetricTensor.from_cartesian(components, output_dims)

    def to_specimen_frame(self, orientations):
        output_dims, output_indices, self_indices = self._get_transformation_indices(
            orientations
        )
        orientation_indices_1 = orientations.dims_str + "im"
        orientation_indices_2 = orientations.dims_str + "jn"
        components = np.einsum(
            f"{orientation_indices_1}, {orientation_indices_2}, {self_indices} -> {output_indices}",
            orientations.rotation_matrix,
            orientations.rotation_matrix,
            self.cartesian,
            optimize=True,
        )
        return Order2SymmetricTensor.from_cartesian(components, output_dims)

    def _get_transformation_indices(self, orientations):
        output_dims = order_dims(self.dims_set.union(orientations.dims_set))
        output_indices = output_dims + "mn"
        self_indices = self.dims_str + "ij"
        return output_dims, output_indices, self_indices


class Order4SymmetricTensor(Tensor):
    _basis = mandel_product_basis()
    _component_dims = 2
    _component_indices = "mn"
    _mul_lookup = {"mn": "mn"}
    _matmul_lookup = {
        "n": ["mn", "n", "m", Order2SymmetricTensor],
        "ij": ["ijkl", "kl", "ij", Order2SymmetricTensor.from_cartesian],
        "mn": ["mn", "no", "mo", None],
    }

    def __init__(self, components, dims=None):
        super().__init__(components, dims)

    @classmethod
    def identity(cls):
        return cls(np.eye(6))

    @classmethod
    def zero(cls):
        return cls(np.zeros((6, 6)))

    @classmethod
    def zeros(cls, shape, dims=None):
        if isinstance(shape, Number):
            return cls(np.zeros((shape, 6, 6)), dims)
        else:
            return cls(np.zeros((*shape, 6, 6)), dims)

    @classmethod
    def from_voigt(cls, matrix, dims=None):
        return cls(
            convert_matrix_basis(matrix, voigt_product_basis(), cls._basis),
            dims,
        )

    @classmethod
    def from_cartesian(cls, array_4d, dims=None):
        return cls(cartesian_to_reduced_matrix(array_4d, cls._basis), dims)

    @classmethod
    def from_mandel(cls, components, dims=None):
        return cls(components, dims)

    @classmethod
    def from_cubic_constants(cls, C11, C12, C44):
        return cls.from_voigt(
            np.array(
                [
                    [C11, C12, C12, 0, 0, 0],
                    [C12, C11, C12, 0, 0, 0],
                    [C12, C12, C11, 0, 0, 0],
                    [0, 0, 0, C44, 0, 0],
                    [0, 0, 0, 0, C44, 0],
                    [0, 0, 0, 0, 0, C44],
                ]
            )
        )

    @classmethod
    def from_transverse_isotropic_constants(cls, C11, C12, C13, C33, C44):
        C66 = (C11 - C12) / 2.0
        return cls.from_voigt(
            np.array(
                [
                    [C11, C12, C13, 0, 0, 0],
                    [C12, C11, C13, 0, 0, 0],
                    [C13, C13, C33, 0, 0, 0],
                    [0, 0, 0, C44, 0, 0],
                    [0, 0, 0, 0, C44, 0],
                    [0, 0, 0, 0, 0, C66],
                ]
            )
        )

    @classmethod
    def from_isotropic_constants(cls, modulus, shear_modulus):
        E, G = modulus, shear_modulus
        L = G * (E - 2 * G) / (3 * G - E)
        return cls.from_voigt(
            np.array(
                [
                    [L + 2 * G, L, L, 0, 0, 0],
                    [L, L + 2 * G, L, 0, 0, 0],
                    [L, L, L + 2 * G, 0, 0, 0],
                    [0, 0, 0, G, 0, 0],
                    [0, 0, 0, 0, G, 0],
                    [0, 0, 0, 0, 0, G],
                ]
            )
        )

    @property
    def cartesian(self):
        return reduced_matrix_to_cartesian(self.components, self._basis)

    @property
    def voigt(self):
        return convert_matrix_basis(
            self.components, self._basis, voigt_dual_product_basis()
        )

    @property
    def natural(self):
        return convert_matrix_basis(
            self.components, self._basis, natural_product_basis()
        )

    @property
    def mandel(self):
        return self.components

    @property
    def inverse(self):
        return Order4SymmetricTensor(inv(self.components), self.dims_str)

    @property
    def inv(self):
        return self.inverse

    def __setitem__(self, key, item):
        if isinstance(item, Number):
            self.components[key] = item
        elif not isinstance(item, Order4SymmetricTensor):
            raise ValueError(f"tried to set Order4SymmetricTensor with {type(item)}")
        else:
            self.components[key] = item.components

    def __add__(self, tensor):
        if isinstance(tensor, Number):
            return Order4SymmetricTensor(self.components + tensor, self.dims_str)
        dims = order_dims(self.dims_set.union(tensor.dims_set))
        if isinstance(tensor, Order4SymmetricTensor):
            return Order4SymmetricTensor(self.components + tensor.components, dims)
        return NotImplemented

    def __sub__(self, tensor):
        if isinstance(tensor, Number):
            return Order4SymmetricTensor(self.components - tensor, self.dims_str)
        dims = order_dims(self.dims_set.union(tensor.dims_set))
        if isinstance(tensor, Order4SymmetricTensor):
            return Order4SymmetricTensor(self.components - tensor.components, dims)
        return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, Number):
            return Order4SymmetricTensor(self.components / scalar)
        return NotImplemented

    def __mul__(self, tensor):
        if isinstance(tensor, Number):
            return Order4SymmetricTensor(tensor * self.components, self.dims_str)

        output_indices = order_dims(self.dims_set.union(tensor.dims_set))
        indices = self._mul_lookup.get(tensor._component_indices)
        if indices is None:
            return NotImplemented
        other_indices = tensor.dims_str + indices
        return Scalar(
            np.einsum(
                f"{self.indices_str}, {other_indices} -> {output_indices}",
                self.components,
                tensor.components,
            ),
            output_indices,
        )

    def __rmul__(self, tensor):
        return self.__mul__(tensor)

    def __matmul__(self, tensor):
        u = order_dims(self.dims_set.union(tensor.dims_set))
        self_indices, other_indices, output_indices, output_type = (
            self._matmul_lookup.get(tensor._component_indices, [None] * 4)
        )
        if other_indices is None:
            return NotImplemented
        self_values = self.cartesian if self_indices == "ijkl" else self.components
        self_indices = self.dims_str + self_indices
        other_indices = tensor.dims_str + other_indices
        output_indices = u + output_indices
        output_type = Order4SymmetricTensor if output_type is None else output_type

        return output_type(
            np.einsum(
                f"{self_indices}, {other_indices} -> {output_indices}",
                self_values,
                tensor.components,
                optimize=True,
            ),
            u,
        )

    @property
    def norm(self):
        components = np.einsum("...mn, ...mn", self.components, self.components)
        return Scalar(np.sqrt(components), self.dims_str)

    def directional_modulus(self, direction):
        direction = direction / direction.norm
        direction_tensor = direction.outer(direction).sym
        return 1.0 / (direction_tensor * (self.inv @ direction_tensor))

    def directional_bulk_modulus(self, direction):
        direction = direction / direction.norm
        direction_tensor = direction.outer(direction).sym
        return 1.0 / (3 * (self.inv @ direction_tensor).trace)

    def directional_shear_modulus(self, normal, direction):
        direction = direction / direction.norm
        normal = normal / normal.norm
        if not np.allclose((direction * normal).components, 0.0):
            raise ValueError(
                "tried to get directional shear moduli with directions that are not perpendicular"
            )
        direction_tensor = normal.outer(direction).sym
        return 1.0 / (4.0 * (direction_tensor * (self.inv @ direction_tensor)))

    def directional_poissons_ratio(self, transverse_direction, axial_direction):
        axial = axial_direction / axial_direction.norm
        transverse = transverse_direction / transverse_direction.norm
        axial_tensor = axial.outer(axial).sym
        transverse_tensor = transverse.outer(transverse).sym
        if not np.allclose((axial * transverse).components, 0.0):
            raise ValueError(
                "tried to get directional Poisson's ratio with axial and transverse directions that are not perpendicular"
            )
        moduli = self.directional_modulus(axial)

        return -moduli * (axial_tensor * (self.inv @ transverse_tensor))

    def to_crystal_frame(self, orientations):
        output_dims, output_indices, self_indices = self._get_transformation_indices(
            orientations
        )
        orientation_indices_1 = orientations.dims_str + "ai"
        orientation_indices_2 = orientations.dims_str + "bj"
        R = orientations.rotation_matrix_mandel
        components = np.einsum(
            f"{orientation_indices_1}, {orientation_indices_2}, {self_indices} -> {output_indices}",
            R,
            R,
            self.components,
            optimize=True,
        )
        return Order4SymmetricTensor(components, output_dims)

    def to_specimen_frame(self, orientations):
        output_dims, output_indices, self_indices = self._get_transformation_indices(
            orientations
        )
        orientation_indices_1 = orientations.dims_str + "ia"
        orientation_indices_2 = orientations.dims_str + "jb"
        R = orientations.rotation_matrix_mandel
        components = np.einsum(
            f"{orientation_indices_1}, {orientation_indices_2}, {self_indices} -> {output_indices}",
            R,
            R,
            self.components,
            optimize=True,
        )
        return Order4SymmetricTensor(components, output_dims)

    def _get_transformation_indices(self, orientations):
        output_dims = order_dims(self.dims_set.union(orientations.dims_set))
        output_indices = output_dims + "ab"
        self_indices = self.dims_str + "ij"
        return output_dims, output_indices, self_indices

    def repeat(self, num_points):
        if "p" in self.indices_set:
            raise ValueError(
                "cannot repeat Order4SymmetricTensor that already has points dimension"
            )
        num_dims = len(self.indices_set)
        components = np.tile(self.components, [num_points] + [1] * num_dims)
        return Order4SymmetricTensor(components, "p")
