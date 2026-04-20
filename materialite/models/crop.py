import numpy as np
from materialite.models import Model

_AXIS_LABELS = ("x", "y", "z")


def _normalize_axis(axis):
    if axis in (0, "x", "X"):
        return 0
    elif axis in (1, "y", "Y"):
        return 1
    elif axis in (2, "z", "Z"):
        return 2
    raise ValueError(f"axis must be 0, 1, 2, 'x', 'y', or 'z', got {axis!r}")


class Crop(Model):
    def __init__(self, crop_fn):
        self.crop_fn = crop_fn

    def run(self, material):
        return self.crop_fn(material)

    @classmethod
    def from_slice_by_id(cls, axis="z", index=0):
        label = _AXIS_LABELS[_normalize_axis(axis)]

        def crop(material):
            return material.crop_by_id(**{label: (index, index)})

        return cls(crop)

    @classmethod
    def from_box_by_id(cls, dimensions=None, center_id=None):
        # dimensions: a 3-element iterable [x_size, y_size, z_size].
        # Any element that is None spans the full range for that axis.
        # If dimensions itself is None, the material is returned unchanged.
        # Captured before the closure to avoid UnboundLocalError.
        _center_id = center_id
        _dimensions = dimensions

        def crop(material):
            c = material.center_id if _center_id is None else np.array(_center_id)
            sizes = [None, None, None] if _dimensions is None else list(_dimensions)
            kwargs = {}
            for i, (size, label) in enumerate(zip(sizes, _AXIS_LABELS)):
                if size is not None:
                    start = c[i] - size // 2
                    kwargs[label] = (start, start + size - 1)
            return material.crop_by_id(**kwargs) if kwargs else material

        return cls(crop)

    @classmethod
    def from_crop(cls, x=None, y=None, z=None):
        def crop(material):
            return material.crop(x=x, y=y, z=z)

        return cls(crop)

    @classmethod
    def from_crop_by_id(cls, x=None, y=None, z=None):
        def crop(material):
            return material.crop_by_id(x=x, y=y, z=z)

        return cls(crop)

    @classmethod
    def from_fraction(cls, fractions=None, centered=True):
        # fractions: a 3-element iterable [x_fraction, y_fraction, z_fraction].
        # Each value in (0, 1] selects that fraction of the domain.
        # Uses "round half up" so that 0.5 on an odd-dimensional axis
        # produces a crop centered at center_id rather than one point off.
        _fractions = [1.0, 1.0, 1.0] if fractions is None else list(fractions)

        def crop(material):
            dims = material.dimensions
            counts = [
                max(1, min(dims[i], int(dims[i] * _fractions[i] + 0.5)))
                for i in range(3)
            ]
            starts = (
                [(dims[i] - counts[i]) // 2 for i in range(3)]
                if centered
                else [0, 0, 0]
            )
            return material.crop_by_id(
                x=(starts[0], starts[0] + counts[0] - 1),
                y=(starts[1], starts[1] + counts[1] - 1),
                z=(starts[2], starts[2] + counts[2] - 1),
            )

        return cls(crop)

    @classmethod
    def from_clip(cls, normal="z", value=0.0, keep="low"):
        # Physical-coordinate counterpart to from_clip_by_id.
        # Cuts at a physical coordinate value rather than a grid index.
        _KEEP_HIGH = {"above", "positive", "high"}
        _KEEP_LOW = {"below", "negative", "low"}
        if keep not in _KEEP_HIGH | _KEEP_LOW:
            raise ValueError(
                f"keep must be one of {sorted(_KEEP_HIGH | _KEEP_LOW)}, got {keep!r}"
            )

        keep_high = keep in _KEEP_HIGH
        label = _AXIS_LABELS[_normalize_axis(normal)]

        def crop(material):
            coord_range = (value, np.inf) if keep_high else (-np.inf, value)
            return material.crop(**{label: coord_range})

        return cls(crop)

    @classmethod
    def from_clip_by_id(cls, normal="z", index=0, keep="low"):
        # `normal` specifies the axis perpendicular to the cutting plane.
        # e.g. normal="z" cuts the xy-plane at the given z index.
        #
        # Synonyms accepted.
        # "above" / "positive" / "high"  → keep indices >= index (the high-index side)
        # "below" / "negative" / "low"   → keep indices <= index (the low-index side)
        _KEEP_HIGH = {"above", "positive", "high"}
        _KEEP_LOW = {"below", "negative", "low"}
        if keep not in _KEEP_HIGH | _KEEP_LOW:
            raise ValueError(
                f"keep must be one of {sorted(_KEEP_HIGH | _KEEP_LOW)}, got {keep!r}"
            )

        keep_high = keep in _KEEP_HIGH
        dim_index = _normalize_axis(normal)
        label = _AXIS_LABELS[dim_index]

        def crop(material):
            max_index = material.dimensions[dim_index] - 1
            id_range = (index, max_index) if keep_high else (0, index)
            return material.crop_by_id(**{label: id_range})

        return cls(crop)
