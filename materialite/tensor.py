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

import string
from abc import ABC, abstractmethod
from copy import deepcopy
from numbers import Number

import numpy as np
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
from numpy.linalg import inv


def DIM_NAMES(dim_char):

    # Predefined dimension names
    known_dims = {
        "p": "points",
        "s": "slip systems",
        "t": "time",
        "i": "Cartesian components (i)",
        "j": "Cartesian components (j)",
        "m": "Mandel basis components (m)",
        "n": "Mandel basis components (n)",
    }

    # Return predefined name if available
    if dim_char in known_dims:
        return known_dims[dim_char]

    return dim_char


INNER_PRODUCT_INDICES = {
    "j": "j",
    "n": "n",
    "ij": "ij",
    "mn": "mn",
    "jj": "",
    "nij": "",
}


RESERVED_DIMS_AND_INDICES = "pstijmn"


def order_dims(left_dims=None, right_dims=None):
    """
    Combine two dimension iterables using left-is-always-right precedence.

    The left dimensions maintain their exact order and act as "authorities"
    that decide where new dimensions from the right should be placed.

    Example: order_dims("ab", "bca") → "cab"
    - Start with "ab"
    - Authority 'a' sees 'c' comes before 'a' in "bca", so places 'c' before 'a'
    """

    # Handle edge cases: if either input is empty, return the other
    if not right_dims:
        return left_dims
    if not left_dims:
        return right_dims

    # Start with left dimensions as the foundation - they never reorder relatively
    result = list(left_dims)

    # List of right_dims that are not in left_dims
    only_right_dims = [dim for dim in right_dims if dim not in left_dims]

    # Add each new "only in right" dimension one at a time in order
    for new_dim in only_right_dims:

        # Default: place new dimension at the end if no authority has an opinion
        insert_index = len(result)

        # Ask each left dimension (authority) where this new dimension should go
        # Authorities are consulted left-to-right (leftmost = highest rank)
        # Only left_dims that also appear in right_dims can give opinions as authorities
        # (they need context about the new dimension's index relatively)
        authority_left_dims = [dim for dim in left_dims if dim in right_dims]
        for authority_dim in authority_left_dims:

            authority_index = result.index(authority_dim)

            if right_dims.index(new_dim) < right_dims.index(authority_dim):
                # Authority says: "This dimension comes before me in right_dims,
                # so place it directly before me in the intermediate result"
                insert_index = authority_index
                break  # Highest-ranking authority's "before" decision is final
            else:
                # Authority says: "This dimension comes after me in right_dims,
                # so place it right after me (or further after if other authorities disagree)"
                insert_index = max(insert_index, authority_index + 1)
                # Continue asking other authorities - they might push it further right

        # Place the new dimension at the determined index
        result.insert(insert_index, new_dim)

    return "".join(result)


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
    shape = tensor.components.shape
    if tensor.num_indices != len(shape):
        raise ValueError(
            f"tried to create {obj} with dimensions {tensor.indices_str} but components have shape {shape}"
        )


def _broadcast_tensor_data(tensor1, tensor2, attr_name="components"):
    """
    Broadcast tensor data by aligning dimensions and transposing data as needed.

    This ensures that tensors with different dimension orders (e.g., "ab" vs. "ba")
    have their data properly aligned so operations work correctly.
    """
    # Use left-is-always-right precedence to determine final dimension ordering
    final_dims = order_dims(tensor1.dims_str, tensor2.dims_str)

    # Extract the data arrays we want to broadcast
    data1 = getattr(tensor1, attr_name)  # e.g., tensor1.components
    data2 = getattr(tensor2, attr_name)

    # Transpose data so dimensions are in the same order
    aligned_data1 = _transpose_to_match_dims(data1, tensor1.dims_str, final_dims)
    aligned_data2 = _transpose_to_match_dims(data2, tensor2.dims_str, final_dims)

    # Handle missing dimensions and broadcast to compatible shapes
    result1, result2 = _add_missing_dims_and_broadcast(
        aligned_data1, aligned_data2, tensor1.dims_str, tensor2.dims_str, final_dims
    )

    return result1, result2, final_dims


def _transpose_to_match_dims(data, current_dims, target_dims):
    """Transpose data so dimensions align with target order."""
    if current_dims == target_dims:
        return data  # No transpose needed

    # Map target dimensions to their current axis positions
    transpose_axes = [
        current_dims.index(dim) for dim in target_dims if dim in current_dims
    ]
    # Preserve component dimensions at the end (e.g., vector [x,y,z] components)
    transpose_axes.extend(range(len(current_dims), data.ndim))

    return np.transpose(data, transpose_axes)


def _add_missing_dims_and_broadcast(data1, data2, dims1, dims2, final_dims):
    """Add missing dimensions as size-1 and broadcast to common shape."""

    # Build target shapes for each tensor in the final dimension ordering
    target_shape1 = []
    target_shape2 = []

    data1_dim_idx = 0
    data2_dim_idx = 0

    for dim in final_dims:
        if dim in dims1:
            # Tensor1 has this dimension
            target_shape1.append(data1.shape[data1_dim_idx])
            data1_dim_idx += 1
        else:
            # Tensor1 missing this dimension - add as size-1
            target_shape1.append(1)

        if dim in dims2:
            # Tensor2 has this dimension
            target_shape2.append(data2.shape[data2_dim_idx])
            data2_dim_idx += 1
        else:
            # Tensor2 missing this dimension - add as size-1
            target_shape2.append(1)

    comp_shape1 = data1.shape[data1_dim_idx:]
    comp_shape2 = data2.shape[data2_dim_idx:]

    # Combine dimensions with component dimensions
    full_shape1 = tuple(target_shape1) + comp_shape1
    full_shape2 = tuple(target_shape2) + comp_shape2

    # Reshape to add missing dimensions and broadcast to common shape
    reshaped1 = data1.reshape(full_shape1)
    reshaped2 = data2.reshape(full_shape2)

    # Calculate final broadcast shape (max size for each dimension)
    final_dims_shape = tuple(
        max(s1, s2) for s1, s2 in zip(target_shape1, target_shape2)
    )
    final_shape1 = final_dims_shape + comp_shape1
    final_shape2 = final_dims_shape + comp_shape2

    return np.broadcast_to(reshaped1, final_shape1), np.broadcast_to(
        reshaped2, final_shape2
    )


def _add_missing_dims_and_broadcast(data1, data2, dims1, dims2, final_dims):
    """Add missing dimensions as size-1 and broadcast to common shape."""

    # After transpose, both data arrays are aligned to final_dims order
    # We just need to pad with size-1 for missing dimensions

    target_shape1 = []
    target_shape2 = []

    data1_dim_idx = 0  # Track position in data1
    data2_dim_idx = 0  # Track position in data2

    for dim in final_dims:
        if dim in dims1:
            # This dimension exists in tensor1
            target_shape1.append(data1.shape[data1_dim_idx])
            data1_dim_idx += 1
        else:
            # Missing dimension - add as size-1
            target_shape1.append(1)

        if dim in dims2:
            # This dimension exists in tensor2
            target_shape2.append(data2.shape[data2_dim_idx])
            data2_dim_idx += 1
        else:
            # Missing dimension - add as size-1
            target_shape2.append(1)

    # Component shapes are everything after the batch dimensions
    comp_shape1 = data1.shape[data1_dim_idx:]
    comp_shape2 = data2.shape[data2_dim_idx:]

    # Rest of function stays the same...
    full_shape1 = tuple(target_shape1) + comp_shape1
    full_shape2 = tuple(target_shape2) + comp_shape2

    reshaped1 = data1.reshape(full_shape1)
    reshaped2 = data2.reshape(full_shape2)

    final_dims_shape = tuple(
        max(s1, s2) for s1, s2 in zip(target_shape1, target_shape2)
    )
    final_shape1 = final_dims_shape + comp_shape1
    final_shape2 = final_dims_shape + comp_shape2

    return np.broadcast_to(reshaped1, final_shape1), np.broadcast_to(
        reshaped2, final_shape2
    )


# Broadcasting wrappers
def _broadcast_components(tensor1, tensor2):
    return _broadcast_tensor_data(tensor1, tensor2, "components")


def _broadcast_cartesian(tensor1, tensor2):
    return _broadcast_tensor_data(tensor1, tensor2, "cartesian")


def _default_dims(num):
    if num == 0:
        return ""
    elif num == 1:
        return "p"
    elif num == 2:
        return "ps"
    elif num == 3:
        return "pst"
    else:
        # For 4+ dimensions, exclude reserved letters from the alphabet
        available_letters = [
            c for c in string.ascii_lowercase if c not in RESERVED_DIMS_AND_INDICES
        ]
        return "pst" + "".join(available_letters[: num - 3])


class Tensor(ABC):
    __array_ufunc__ = None
    _reserved_indices = "ijmn"

    def __init__(self, components, dims):

        # If components is already a Tensor, copy it
        if isinstance(components, Tensor):
            self.components = components.components.copy()
            self.indices_str = components.indices_str
            self.dims_str = components.dims_str
            return

        self.components = np.asarray(components)

        # Choose default dims from components if dims are not provided
        if dims is None:
            num_indices = len(self.components.shape) - self._component_dims
            dims = _default_dims(num_indices)

        if any(char in self._reserved_indices for char in dims):
            raise ValueError(
                f"Dimensions ({dims}) cannot overlap with indices reserved for tensor components ({self._reserved_indices})"
            )

        self.indices_str = dims + self._component_indices
        self.dims_str = dims

        _check_consistent_dims(self)

    def copy(self):
        return deepcopy(self)

    @property
    def dims(self):
        return self.dims_str[:]  # Slice is to make a copy

    def with_dims(self, dims):
        return type(self)(self.components, dims)

    def __repr__(self):
        dimensions = ", ".join([DIM_NAMES(i) for i in self.dims_str])
        return (
            f"{type(self).__name__}("
            + str(self.components)
            + f", dims: ({dimensions}), components shape: {self.components.shape})"
        )

    def __len__(self):
        if not self.dims_str:
            raise TypeError(f"{type(self).__name__} with no dimensions has no length")
        return len(self.components)

    def __iter__(self):
        return TensorIterator(self)

    def __neg__(self):
        return type(self)(-self.components, self.dims_str)

    def __getitem__(self, slice_):

        self._check_valid_slice(slice_)
        components = self.components[slice_]

        # Figure out which dimensions remain after slicing
        dims = self._get_remaining_dims(slice_)

        return type(self)(components, dims)

    def _get_remaining_dims(self, slice_):
        """
        Determine which dimensions survive the slicing operation.

        Rules: Integer indices remove dimensions, slices/lists/arrays keep them.
        """
        if isinstance(slice_, (int, np.integer)):
            # Single integer removes the first dimension
            return self.dims_str[1:]
        elif isinstance(slice_, slice):
            # Slice notation (e.g., [:]) keeps all dimensions
            return self.dims_str
        elif isinstance(slice_, (list, np.ndarray)):
            # Fancy indexing keeps the dimension structure
            return self.dims_str
        elif isinstance(slice_, tuple):
            # Multiple indices - check each one individually
            remaining_dims = []
            for i, s in enumerate(slice_):
                if i >= len(self.dims_str):
                    break  # Don't go beyond our named dimensions

                # Determine if this index removes or keeps the dimension
                if isinstance(s, (int, np.integer)):
                    # Integer removes the dimension (skip it)
                    pass
                elif isinstance(s, (slice, list, np.ndarray)):
                    # Slice/fancy indexing keeps the dimension
                    remaining_dims.append(self.dims_str[i])

            return "".join(remaining_dims)
        else:
            # Unknown slice type - assume it keeps dimensions
            return self.dims_str

    def _check_valid_slice(self, slice_):

        # Tensors with no dimensions can't be indexed
        if not self.dims_str:
            raise ValueError(f"Cannot index {type(self).__name__} with no dimensions")

        # Simple slice types are always valid
        if isinstance(slice_, (Number, slice, list, np.ndarray)):
            return

        # For tuple slices, check bounds
        if isinstance(slice_, tuple) and len(slice_) > len(self.dims_str):
            dims_desc = ", ".join([DIM_NAMES(d) for d in self.dims_str])
            raise ValueError(
                f"Provided {len(slice_)} indices to {type(self).__name__} "
                f"with only {len(self.dims_str)} dimensions ({dims_desc})"
            )

        # Ellipsis is not supported (would complicate dimension tracking)
        if slice_ is ...:
            raise ValueError(
                f"Ellipsis indexing is not supported for {type(self).__name__}"
            )

        # Check for ellipsis in tuple slices
        if isinstance(slice_, tuple):
            for element in slice_:
                if element is ...:
                    raise ValueError(
                        f"Ellipsis indexing is not supported for {type(self).__name__}"
                    )

        return

    def mean(self, dim=None):
        if dim is None and self.dims_str == "":
            return self
        axis, str_dims = self._dims_for_reduction(dim, "mean")
        return type(self)(np.mean(self.components, axis=axis), str_dims)

    def sum(self, dim=None):
        if dim is None and self.dims_str == "":
            return self
        axis, str_dims = self._dims_for_reduction(dim, "sum")
        return type(self)(np.sum(self.components, axis=axis), str_dims)

    def _dims_for_reduction(self, dim, reduction_type):
        if dim is None:
            # Default to first dimension if it exists
            if not self.dims_str:
                raise ValueError(
                    f"cannot {reduction_type} over tensor with no dimensions"
                )
            dim = self.dims_str[0]

        try:
            axis = self.dims_str.index(dim)
        except ValueError:
            raise ValueError(
                f"dimension '{dim}' not found in {type(self).__name__} with dims '{self.dims_str}'"
            )

        str_dims = self.dims_str.replace(dim, "")
        return axis, str_dims

    @property
    def num_dims(self):
        return len(self.dims_str)

    @property
    def num_indices(self):
        return len(self.indices_str)

    @property
    def shape(self):
        if self.num_dims == 0:
            return ()
        return self.components.shape[: self.num_dims]

    def reorder(self, dims):

        # If order is already correct, just return the tensor
        if dims == self.dims_str:
            return self

        # Confirm that the new dims are a permutation of the original
        if set(dims) != set(self.dims_str):
            raise ValueError(
                f"New dimension order '{dims}' must contain the same dimensions as "
                f"the original tensor's '{self.dims_str}'"
            )

        # Use einsum to reorder (ending ellipsis for component dimensions)
        new_components = np.einsum(f"{self.dims_str}... -> {dims}...", self.components)

        return type(self)(new_components, dims)

    @classmethod
    def from_list(cls, tensor):
        if "p" in tensor[0].dims_str:
            raise ValueError(
                "Cannot create list from tensors that already have a points dimension"
            )
        dims = "p" + tensor[0].dims_str
        components = np.array([t.components for t in tensor])
        return cls(components, dims)

    @classmethod
    def from_stack(cls, tensors, new_dim, axis=0):
        """
        Stack tensors by creating a new dimension.

        Parameters
        ----------
        tensors : list of Tensor
            List of tensors to stack. All tensors must have the same dimensions.
        new_dim : str
            New dimension name to create for stacking.
        axis : int
            Axis where the data will be stacked.

        Returns
        -------
        Tensor
            New tensor with stacked data along a new dimension.
        """
        if not tensors:
            raise ValueError("Cannot stack empty list of tensors")

        # Validate tensors have the same dimensions
        base_dims = tensors[0].dims_str
        for tensor in tensors:
            if tensor.dims_str != base_dims:
                raise ValueError(
                    f"All tensors must have the same dimensions. "
                    f"Found '{tensor.dims_str}' but expected '{base_dims}'."
                )

        # Validate the new dimension doesn't already exist
        if new_dim in base_dims:
            raise ValueError(
                f"Dimension '{new_dim}' already exists in tensor dimensions '{base_dims}'"
            )
        if axis > len(base_dims):
            raise ValueError(
                f"Axis {axis} is greater than the number of tensor dimensions `{base_dims}`"
            )

        # Stack components along a new axis (at position 0)
        stacked_components = np.stack([t.components for t in tensors], axis=axis)

        # Create new dimensions string with new dimension first
        new_dims = base_dims[:axis] + new_dim + base_dims[axis:]

        return cls(stacked_components, new_dims)

    @abstractmethod
    def __mul__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __rmul__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __matmul__(self, *args, **kwargs):
        raise NotImplementedError

    def repeat(self, shape, dims=None):

        # Only allow repeat on tensors with no dimensions
        if self.dims_str:
            raise ValueError(
                f"Cannot repeat {type(self).__name__} that already has dimensions '{self.dims_str}'. "
                f"Repeat only works on tensors with no existing dimensions."
            )

        # Check if iterable to allow the user to pass in an int
        shape = tuple(shape) if hasattr(shape, "__iter__") else (shape,)

        if dims is None:
            dims = _default_dims(len(shape))

        # Add singleton dimensions at the front
        expanded = self.components[(np.newaxis,) * len(shape)]

        # Broadcast to final shape
        final_shape = shape + self.components.shape

        # A copy of the broadcasted array is needed to avoid setitem issues for a view
        # There are ways to avoid this, but it's the simplest solution
        repeated_components = np.broadcast_to(expanded, final_shape).copy()

        return type(self)(repeated_components, dims)


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
    def random(cls, shape, rng=np.random.default_rng(), dims=None):
        components = rng.random(shape)
        return cls(components, dims)

    @classmethod
    def zero(cls):
        return cls(0.0)

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
        if dim is None and self.dims_str == "":
            return self
        axis, str_dims = self._dims_for_reduction(dim, "max")
        return Scalar(np.max(self.components, axis=axis), str_dims)

    def apply(self, function):
        return Scalar(function(self.components), self.dims_str)

    def __setitem__(self, key, item):
        if isinstance(item, Number):
            self.components[key] = item
        elif not isinstance(item, Scalar):
            raise ValueError(f"cannot set Scalar with {type(item)}")
        else:
            self.components[key] = item.components

    def __pow__(self, tensor):
        if isinstance(tensor, Number):
            return Scalar(self.components**tensor, self.dims_str)
        elif isinstance(tensor, Scalar):
            components1, components2, new_dims = _broadcast_components(self, tensor)
            return Scalar(components1**components2, new_dims)
        return NotImplemented

    def __add__(self, tensor):
        if isinstance(tensor, Number):
            return Scalar(self.components + tensor, self.dims_str)
        if isinstance(tensor, Scalar):
            components1, components2, new_dims = _broadcast_components(self, tensor)
            return Scalar(components1 + components2, new_dims)
        return NotImplemented

    def __radd__(self, tensor):
        return self.__add__(tensor)

    def __sub__(self, tensor):
        return self + -tensor

    def __rsub__(self, tensor):
        return -self + tensor

    def __truediv__(self, tensor):
        if isinstance(tensor, Number):
            return Scalar(self.components / tensor, self.indices_str)
        elif isinstance(tensor, Scalar):
            components1, components2, new_dims = _broadcast_components(self, tensor)
            return Scalar(components1 / components2, new_dims)
        return NotImplemented

    def __rtruediv__(self, scalar):
        if isinstance(scalar, Number):
            return Scalar(scalar / self.components, self.indices_str)
        return NotImplemented

    def __mul__(self, tensor):
        if isinstance(tensor, Number):
            return Scalar(tensor * self.components, self.indices_str)

        u = order_dims(self.dims_str, tensor.dims_str)
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
    def random(cls, shape=1, rng=np.random.default_rng(), dims=None):
        # Check if iterable to allow the user to pass in an int
        shape = tuple(shape) if hasattr(shape, "__iter__") else (shape,)
        components = np.squeeze(rng.random((*shape, 3)))

        return cls(components, dims)

    @classmethod
    def random_unit(cls, shape=1, rng=np.random.default_rng(), dims=None):
        # Check if iterable to allow the user to pass in an int
        shape = tuple(shape) if hasattr(shape, "__iter__") else (shape,)
        components = np.squeeze(rng.normal(size=(*shape, 3)))

        return cls(components, dims).unit

    @classmethod
    def zero(cls):
        return cls(np.zeros(3))

    @classmethod
    def basis(cls, dim="b"):
        return cls(np.eye(3), dim)

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
            raise ValueError(f"cannot set Vector with {type(item)}")
        else:
            self.components[key] = item.components

    def __add__(self, tensor):
        if isinstance(tensor, Number):
            return Vector(self.components + tensor, self.dims_str)
        if isinstance(tensor, Vector):
            components1, components2, new_dims = _broadcast_components(self, tensor)
            return Vector(components1 + components2, new_dims)
        return NotImplemented

    def __sub__(self, tensor):
        return self + -tensor

    def __truediv__(self, tensor):
        if isinstance(tensor, Number):
            return Vector(self.components / tensor, self.dims_str)
        elif isinstance(tensor, Scalar):
            components1, components2, new_dims = _broadcast_components(self, tensor)
            return Vector(components1 / components2[..., np.newaxis], new_dims)
        return NotImplemented

    def __mul__(self, tensor):

        if isinstance(tensor, Number):
            return Vector(tensor * self.components, self.dims_str)

        u = order_dims(self.dims_str, tensor.dims_str)

        if isinstance(tensor, Scalar):
            output_indices = u + self._component_indices
            return Vector(
                np.einsum(
                    f"{self.indices_str}, {tensor.indices_str} -> {output_indices}",
                    self.components,
                    tensor.components,
                    optimize=True,
                ),
                u,
            )

        if isinstance(tensor, Vector):
            output_indices = u
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
        output_indices = order_dims(self.dims_str, tensor.dims_str) + "ij"
        return Order2Tensor(
            np.einsum(
                f"{self_dims}, {tensor.indices_str} -> {output_indices}",
                self.components,
                tensor.components,
                optimize=True,
            ),
            output_indices[:-2],
        )

    def cross(self, tensor):
        if not isinstance(tensor, Vector):
            raise ValueError(
                f"Cannot do cross product between a Vector and {type(tensor)}"
            )
        permutation = np.zeros((3, 3, 3))
        permutation[0, 1, 2] = permutation[1, 2, 0] = permutation[2, 0, 1] = 1
        permutation[1, 0, 2] = permutation[0, 2, 1] = permutation[2, 1, 0] = -1
        self_dims = self.dims_str + "j"
        tensor_dims = tensor.dims_str + "k"
        output_dims = order_dims(self.dims_str, tensor.dims_str) + "i"
        components = np.einsum(
            f"ijk, {self_dims}, {tensor_dims} -> {output_dims}",
            permutation,
            self.components,
            tensor.components,
            optimize=True,
        )
        return Vector(components, dims=output_dims[:-1])

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
        output_dims = order_dims(self.dims_str, orientations.dims_str)
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
    def random(cls, shape=1, rng=np.random.default_rng(), dims=None):
        # Check if iterable to allow the user to pass in an int
        shape = tuple(shape) if hasattr(shape, "__iter__") else (shape,)
        components = np.squeeze(rng.random((*shape, 3, 3)))
        return cls(components, dims)

    @classmethod
    def from_tensor_product(cls, vector1, vector2):
        dims = order_dims(vector1.dims_str, vector2.dims_str)
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
            0.5 * (self + self.T).components, self.dims_str
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
        if isinstance(tensor, Order2Tensor) or isinstance(
            tensor, Order2SymmetricTensor
        ):
            components1, components2, new_dims = _broadcast_cartesian(self, tensor)
            return Order2Tensor(components1 + components2, new_dims)
        return NotImplemented

    def __sub__(self, tensor):
        return self + -tensor

    def __truediv__(self, tensor):
        if isinstance(tensor, Scalar):
            # Broadcast components
            components1, components2, new_dims = _broadcast_components(self, tensor)
            return Order2Tensor(
                components1 / components2[..., np.newaxis, np.newaxis],
                new_dims,
            )
        elif isinstance(tensor, Number):
            return Order2Tensor(self.components / tensor, self.dims_str)
        return NotImplemented

    def __mul__(self, tensor):
        if isinstance(tensor, Number):
            return Order2Tensor(tensor * self.components, self.dims_str)

        output_indices = order_dims(self.dims_str, tensor.dims_str)
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
        u = order_dims(self.dims_str, tensor.dims_str)
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
        output_dims = order_dims(self.dims_str, orientations.dims_str)
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
    def random(cls, shape=1, rng=np.random.default_rng(), dims=None):
        # Check if iterable to allow the user to pass in an int
        shape = tuple(shape) if hasattr(shape, "__iter__") else (shape,)
        components = np.squeeze(rng.random((*shape, 6)))

        return cls(components, dims)

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
        if isinstance(tensor, Order2Tensor):
            components1, components2, new_dims = _broadcast_cartesian(self, tensor)
            return Order2Tensor(components1 + components2, new_dims)
        elif isinstance(tensor, Order2SymmetricTensor):
            components1, components2, new_dims = _broadcast_components(self, tensor)
            return Order2SymmetricTensor(components1 + components2, new_dims)
        return NotImplemented

    def __sub__(self, tensor):
        return self + -tensor

    def __truediv__(self, tensor):
        if isinstance(tensor, Scalar):
            # Broadcast components
            components1, components2, new_dims = _broadcast_components(self, tensor)
            return Order2SymmetricTensor(
                components1 / components2[..., np.newaxis],
                new_dims,
            )
        elif isinstance(tensor, Number):
            return Order2SymmetricTensor(self.components / tensor, self.dims_str)
        return NotImplemented

    def __mul__(self, tensor):
        if isinstance(tensor, Number):
            return Order2SymmetricTensor(tensor * self.components, self.dims_str)

        output_indices = order_dims(self.dims_str, tensor.dims_str)
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
        u = order_dims(self.dims_str, tensor.dims_str)
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
        output_indices = order_dims(self.dims_str, tensor.dims_str) + "mn"
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
        output_dims = order_dims(self.dims_str, orientations.dims_str)
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
        if isinstance(tensor, Order4SymmetricTensor):
            dims = order_dims(self.dims_str, tensor.dims_str)
            return Order4SymmetricTensor(self.components + tensor.components, dims)
        return NotImplemented

    def __sub__(self, tensor):
        return self + -tensor

    def __truediv__(self, scalar):
        if isinstance(scalar, Number):
            return Order4SymmetricTensor(self.components / scalar)
        return NotImplemented

    def __mul__(self, tensor):
        if isinstance(tensor, Number):
            return Order4SymmetricTensor(tensor * self.components, self.dims_str)

        output_indices = order_dims(self.dims_str, tensor.dims_str)
        indices = self._mul_lookup.get(tensor._component_indices)
        if indices is None:
            return NotImplemented
        other_indices = tensor.dims_str + indices
        return Scalar(
            np.einsum(
                f"{self.indices_str}, {other_indices} -> {output_indices}",
                self.components,
                tensor.components,
                optimize=True,
            ),
            output_indices,
        )

    def __rmul__(self, tensor):
        return self.__mul__(tensor)

    def __matmul__(self, tensor):
        u = order_dims(self.dims_str, tensor.dims_str)
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
        unit_direction = Vector(direction).unit
        direction_tensor = unit_direction.outer(unit_direction).sym
        return 1.0 / (direction_tensor * (self.inv @ direction_tensor))

    def directional_bulk_modulus(self, direction):
        unit_direction = Vector(direction).unit
        direction_tensor = unit_direction.outer(unit_direction).sym
        return 1.0 / (3 * (self.inv @ direction_tensor).trace)

    def directional_shear_modulus(self, normal, direction):
        unit_normal = Vector(normal).unit
        unit_direction = Vector(direction).unit
        if not np.allclose((unit_direction * unit_normal).components, 0.0):
            raise ValueError(
                "tried to get directional shear moduli with directions that are not perpendicular"
            )
        direction_tensor = unit_normal.outer(unit_direction).sym
        return 1.0 / (4.0 * (direction_tensor * (self.inv @ direction_tensor)))

    def directional_poissons_ratio(self, transverse_direction, axial_direction):
        unit_transverse = Vector(transverse_direction).unit
        unit_axial = Vector(axial_direction).unit
        axial_tensor = unit_axial.outer(unit_axial).sym
        transverse_tensor = unit_transverse.outer(unit_transverse).sym
        if not np.allclose((unit_axial * unit_transverse).components, 0.0):
            raise ValueError(
                "tried to get directional Poisson's ratio with axial and transverse directions that are not perpendicular"
            )
        moduli = self.directional_modulus(unit_axial)

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
        output_dims = order_dims(self.dims_str, orientations.dims_str)
        output_indices = output_dims + "ab"
        self_indices = self.dims_str + "ij"
        return output_dims, output_indices, self_indices


class Orientation:
    __array_ufunc__ = None

    def __init__(self, rotation_matrix, dims=None):
        if isinstance(rotation_matrix, Orientation):
            self.rotation_matrix = rotation_matrix.rotation_matrix.copy()
            self.indices_str = rotation_matrix.indices_str
            self.dims_str = rotation_matrix.dims_str
            return
        self.rotation_matrix = np.asarray(rotation_matrix)
        if dims is None:
            num_indices = len(self.rotation_matrix.shape) - 2
            dims = _default_dims(num_indices)
        self.dims_str = dims
        self.indices_str = dims + "ij"
        matrix_shape = self.rotation_matrix.shape
        if self.num_indices != len(matrix_shape):
            raise ValueError(
                f"tried to create Orientation with dimensions {self.indices_str} but rotation matrix has shape {matrix_shape}"
            )

    def copy(self):
        return deepcopy(self)

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
    def num_dims(self):
        return len(self.dims_str)

    @property
    def num_indices(self):
        return len(self.indices_str)

    @property
    def shape(self):
        if self.num_dims == 0:
            return ()
        return self.rotation_matrix.shape[: self.num_dims]

    def reorder(self, dims):

        # If order is already correct, just return the orientation
        if dims == self.dims_str:
            return self

        # Confirm that the new dims are a permutation of the original
        if set(dims) != set(self.dims_str):
            raise ValueError(
                f"New dimension order '{dims}' must contain the same dimensions as "
                f"the original orientation's '{self.dims_str}'"
            )

        # Use einsum to reorder (ending ellipsis for component dimensions)
        new_components = np.einsum(
            f"{self.dims_str}... -> {dims}...", self.rotation_matrix
        )

        return type(self)(new_components, dims)

    def __len__(self):
        if self.num_dims == 0:
            return None
        else:
            return len(self.rotation_matrix)

    def __iter__(self):
        return OrientationIterator(self.rotation_matrix)

    # def __getitem__(self, slice_):
    #     if len(self.dims_str) is None:
    #         raise ValueError("can't index")
    #     return Orientation(self.rotation_matrix[slice_])

    def __getitem__(self, slice_):

        self._check_valid_slice(slice_)
        components = self.rotation_matrix[slice_]

        # Figure out which dimensions remain after slicing
        dims = self._get_remaining_dims(slice_)

        return type(self)(components, dims)

    def _get_remaining_dims(self, slice_):
        """
        Determine which dimensions survive the slicing operation.

        Rules: Integer indices remove dimensions, slices/lists/arrays keep them.
        """
        if isinstance(slice_, (int, np.integer)):
            # Single integer removes the first dimension
            return self.dims_str[1:]
        elif isinstance(slice_, slice):
            # Slice notation (e.g., [:]) keeps all dimensions
            return self.dims_str
        elif isinstance(slice_, (list, np.ndarray)):
            # Fancy indexing keeps the dimension structure
            return self.dims_str
        elif isinstance(slice_, tuple):
            # Multiple indices - check each one individually
            remaining_dims = []
            for i, s in enumerate(slice_):
                if i >= len(self.dims_str):
                    break  # Don't go beyond our named dimensions

                # Determine if this index removes or keeps the dimension
                if isinstance(s, (int, np.integer)):
                    # Integer removes the dimension (skip it)
                    pass
                elif isinstance(s, (slice, list, np.ndarray)):
                    # Slice/fancy indexing keeps the dimension
                    remaining_dims.append(self.dims_str[i])

            return "".join(remaining_dims)
        else:
            # Unknown slice type - assume it keeps dimensions
            return self.dims_str

    def _check_valid_slice(self, slice_):

        # Tensors with no dimensions can't be indexed
        if not self.dims_str:
            raise ValueError(f"Cannot index {type(self).__name__} with no dimensions")

        # Simple slice types are always valid
        if isinstance(slice_, (Number, slice, list, np.ndarray)):
            return

        # For tuple slices, check bounds
        if isinstance(slice_, tuple) and len(slice_) > len(self.dims_str):
            dims_desc = ", ".join([DIM_NAMES(d) for d in self.dims_str])
            raise ValueError(
                f"Provided {len(slice_)} indices to {type(self).__name__} "
                f"with only {len(self.dims_str)} dimensions ({dims_desc})"
            )

        # Ellipsis is not supported (would complicate dimension tracking)
        if slice_ is ...:
            raise ValueError(
                f"Ellipsis indexing is not supported for {type(self).__name__}"
            )

        # Check for ellipsis in tuple slices
        if isinstance(slice_, tuple):
            for element in slice_:
                if element is ...:
                    raise ValueError(
                        f"Ellipsis indexing is not supported for {type(self).__name__}"
                    )

        return

    def __setitem__(self, key, item):
        if not isinstance(item, Orientation):
            raise ValueError(f"tried to set Orientation with {type(item)}")
        self.rotation_matrix[key] = item.rotation_matrix

    @classmethod
    def identity(cls):
        return cls(np.eye(3))

    @classmethod
    def from_miller_indices(cls, plane, direction, dims=None):
        plane = Vector(plane).unit
        direction = Vector(direction).unit
        if plane.shape != direction.shape:
            raise ValueError("Must provide same number of plane(s) and direction(s) to construct Orientation(s) from Miller indices")
        td = plane.cross(direction)
        rotation_matrix = np.stack(
            [direction.components, td.components, plane.components], axis=-1
        )
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
    def random(cls, shape=1, rng=np.random.default_rng(), dims=None):
        # Check if iterable to allow the user to pass in an int
        shape = tuple(shape) if hasattr(shape, "__iter__") else (shape,)

        # Generate random Euler angles using uniform distribution on SO(3)
        z1 = rng.random(shape) * 2.0 * np.pi
        cos_x2 = rng.random(shape) * 2.0 - 1.0
        x2 = np.arccos(cos_x2)
        z3 = rng.random(shape) * 2.0 * np.pi

        # Stack Euler angles and create orientations
        euler_angles = np.stack([z1, x2, z3], axis=-1)

        return cls.from_euler_angles(euler_angles, dims=dims)

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
        dimensions = ", ".join([DIM_NAMES(i) for i in self.dims_str])
        return (
            f"{type(self).__name__}("
            + str(np.round(self.euler_angles, 3))
            + f", dims: ({dimensions}), Euler angles shape: {self.euler_angles.shape})"
        )

    def __matmul__(self, orientation):
        if not isinstance(orientation, Orientation):
            return NotImplemented
        u = order_dims(self.dims_str, orientation.dims_str)
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

    def repeat(self, shape, dims=None):

        # Only allow repeat on tensors with no dimensions
        if self.dims_str:
            raise ValueError(
                f"Cannot repeat {type(self).__name__} that already has dimensions '{self.dims_str}'. "
                f"Repeat only works on tensors with no existing dimensions."
            )

        # Check if iterable to allow the user to pass in an int
        shape = tuple(shape) if hasattr(shape, "__iter__") else (shape,)

        if dims is None:
            dims = _default_dims(len(shape))

        # Add singleton dimensions at the front
        expanded = self.rotation_matrix[(np.newaxis,) * len(shape)]

        # Broadcast to final shape
        final_shape = shape + self.rotation_matrix.shape

        # A copy of the broadcasted array is needed to avoid setitem issues for a view
        # There are ways to avoid this, but it's the simplest solution
        repeated_components = np.broadcast_to(expanded, final_shape).copy()

        return type(self)(repeated_components, dims)


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
