import numpy as np
from materialite.models import Model


class Crop(Model):
    """Model for extracting regions from materials"""

    def __init__(self, crop_fn):
        self.crop_fn = crop_fn

    def run(self, material):
        return self.crop_fn(material)

    @classmethod
    def from_slice_by_id(cls, axis="z", index=0):
        """Extract single slice at index along axis"""
        axis_map = {
            "x": "x_id_range",
            "y": "y_id_range",
            "z": "z_id_range",
            "X": "x_id_range",
            "Y": "y_id_range",
            "Z": "z_id_range",
            0: "x_id_range",
            1: "y_id_range",
            2: "z_id_range",
        }
        range_key = axis_map[axis]

        def crop(material):
            return material.crop_by_id_range(**{range_key: (index, index)})

        return cls(crop)

    @classmethod
    def from_box_by_id(
        cls, x_length=None, y_length=None, z_length=None, center_id=None
    ):
        """Extract box region with specified lengths centered at point."""

        def crop(material):

            # Default to material center if not specified
            center_id = material.center_id if center_id is None else np.array(center_id)

            ranges = {}

            if x_length is not None:
                x_start_id = center_id[0] - x_length // 2
                x_end_id = x_start_id + x_length - 1
                ranges["x_id_range"] = (x_start_id, x_end_id)

            if y_length is not None:
                y_start_id = center_id[1] - y_length // 2
                y_end_id = y_start_id + y_length - 1
                ranges["y_id_range"] = (y_start_id, y_end_id)

            if z_length is not None:
                z_start_id = center_id[2] - z_length // 2
                z_end_id = z_start_id + z_length - 1
                ranges["z_id_range"] = (z_start_id, z_end_id)

            return material.crop_by_id_range(**ranges) if ranges else material

        return cls(crop)

    @classmethod
    def from_ranges(cls, x_range=None, y_range=None, z_range=None):
        """Extract region by coordinate ranges (min, max)"""

        def crop(material):
            return material.crop_by_range(
                x_range=x_range, y_range=y_range, z_range=z_range
            )

        return cls(crop)

    @classmethod
    def from_ranges_by_id(cls, x_id_range=None, y_id_range=None, z_id_range=None):
        """Extract region by index ranges (start, end) inclusive"""

        def crop(material):
            return material.crop_by_id_range(
                x_id_range=x_id_range, y_id_range=y_id_range, z_id_range=z_id_range
            )

        return cls(crop)

    @classmethod
    def from_fraction_by_id(
        cls, x_fraction=1.0, y_fraction=1.0, z_fraction=1.0, centered=True
    ):
        """Extract fraction of domain by indices, centered or from origin"""

        def crop(material):
            dimensions = material.dimensions

            num_x_points = int(dimensions[0] * x_fraction)
            num_y_points = int(dimensions[1] * y_fraction)
            num_z_points = int(dimensions[2] * z_fraction)

            if centered:
                x_start = (dimensions[0] - num_x_points) // 2
                y_start = (dimensions[1] - num_y_points) // 2
                z_start = (dimensions[2] - num_z_points) // 2
            else:
                x_start = y_start = z_start = 0

            return material.crop_by_id_range(
                x_id_range=(x_start, x_start + num_x_points - 1),
                y_id_range=(y_start, y_start + num_y_points - 1),
                z_id_range=(z_start, z_start + num_z_points - 1),
            )

        return cls(crop)

    @classmethod
    def from_clip_by_id(cls, axis="z", index=0, keep="above"):
        """Clip material at orthogonal plane (remove everything on one side)."""
        axis_map = {"x": "x_id_range", "y": "y_id_range", "z": "z_id_range"}
        range_key = axis_map[axis.lower()]

        def crop(material):
            dimensions = material.dimensions
            axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
            max_index = dimensions[axis_index] - 1

            if keep == "above":
                id_range = (index, max_index)
            elif keep == "below":
                id_range = (0, index)
            else:
                raise ValueError(f"keep must be 'above' or 'below', got {keep}")

            return material.crop_by_id_range(**{range_key: id_range})

        return cls(crop)
