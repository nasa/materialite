import numpy as np
import pandas as pd
import pytest  # Includes: tmp_path
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from materialite import Box, Material, Sphere, Superellipsoid, import_dream3d
from materialite.util import power_of_two_below


@pytest.fixture
def seeded_rng():
    return np.random.default_rng(seed=12345)


@pytest.fixture
def material():
    return Material()


@pytest.fixture
def small_material():
    return Material(dimensions=[2, 3, 4])


@pytest.fixture
def expected_default_fields():
    xyz_range = np.arange(16)
    expected_x = np.repeat(xyz_range, 256)
    expected_y = np.tile(np.repeat(xyz_range, 16), 16)
    expected_z = np.tile(xyz_range, 256)
    return pd.DataFrame(
        {
            "x": expected_x,
            "y": expected_y,
            "z": expected_z,
            "x_id": expected_x,
            "y_id": expected_y,
            "z_id": expected_z,
        }
    )


@pytest.fixture
def initialized_material():
    dimensions = np.array([2, 3, 4])
    origin = np.array([0, 1, 2])
    spacing = np.array([1, 2, 3])
    material = Material(dimensions=dimensions, origin=origin, spacing=spacing)
    return material


@pytest.fixture
def expected_initialized_fields():
    dimensions = np.array([2, 3, 4])
    origin = np.array([0, 1, 2])
    spacing = np.array([1, 2, 3])
    dim_x, dim_y, dim_z = dimensions
    o_x, o_y, o_z = origin
    s_x, s_y, s_z = spacing
    x_range = np.arange(dim_x) * s_x + o_x
    y_range = np.arange(dim_y) * s_y + o_y
    z_range = np.arange(dim_z) * s_z + o_z
    expected_x = np.repeat(x_range, dim_y * dim_z)
    expected_y = np.tile(np.repeat(y_range, dim_z), dim_x)
    expected_z = np.tile(z_range, dim_x * dim_y)
    expected_fields = pd.DataFrame(
        {
            "x": expected_x,
            "y": expected_y,
            "z": expected_z,
            "x_id": (expected_x - o_x) / s_x,
            "y_id": (expected_y - o_y) / s_y,
            "z_id": (expected_z - o_z) / s_z,
        }
    )
    return expected_fields


@pytest.fixture
def expected_evpfft_file_contents():
    return (
        "99.80835 66.40346 28.98868 1 1 1 3 1\n"
        + "99.80835 66.40346 28.98868 2 1 1 3 1\n"
        + "99.80835 66.40346 28.98868 1 2 1 3 1\n"
        + "99.80835 66.40346 28.98868 2 2 1 3 1\n"
        + "202.08827 115.84247 321.15894 1 1 2 5 1\n"
        + "202.08827 115.84247 321.15894 2 1 2 5 1\n"
        + "149.57115 93.980354 66.8812 1 2 2 2 1\n"
        + "149.57115 93.980354 66.8812 2 2 2 2 1\n"
        + "124.95919 77.123436 3.692635 1 1 3 1 1\n"
        + "14.567412 69.05068 44.111614 2 1 3 9 1\n"
        + "124.95919 77.123436 3.692635 1 2 3 1 1\n"
        + "36.929478 123.74354 9.999895 2 2 3 7 1\n"
        + "124.95919 77.123436 3.692635 1 1 4 1 1\n"
        + "14.567412 69.05068 44.111614 2 1 4 9 1\n"
        + "149.27193 113.16453 307.57553 1 2 4 4 1\n"
        + "149.27193 113.16453 307.57553 2 2 4 4 1\n"
    )


def test_run(small_material, mocker):
    model = mocker.Mock()
    arg1 = 1
    new_material = small_material.run(model, arg1=arg1)
    model.assert_called_once_with(small_material, arg1=arg1)
    assert new_material == model.return_value


def test_num_points(small_material):
    assert small_material.num_points == 2 * 3 * 4


def test_init_default_material(material, expected_default_fields):
    expected_dimensions = np.array([16, 16, 16])
    expected_origin = np.array([0, 0, 0])
    expected_spacing = np.array([1, 1, 1])
    expected_sizes = expected_dimensions - 1
    assert_array_equal(material.dimensions, expected_dimensions)
    assert_array_equal(material.origin, expected_origin)
    assert_array_equal(material.spacing, expected_spacing)
    assert_array_equal(material.sizes, expected_sizes)
    assert_frame_equal(material.fields, expected_default_fields, check_dtype=False)
    assert material.state == {}


def test_init_material_with_inputs(initialized_material, expected_initialized_fields):
    expected_dimensions = np.array([2, 3, 4])
    expected_origin = np.array([0, 1, 2])
    expected_spacing = np.array([1, 2, 3])
    expected_sizes = np.array([1, 4, 9])
    assert_array_equal(initialized_material.dimensions, expected_dimensions)
    assert_array_equal(initialized_material.origin, expected_origin)
    assert_array_equal(initialized_material.spacing, expected_spacing)
    assert_array_equal(initialized_material.sizes, expected_sizes)
    assert_frame_equal(
        initialized_material.fields, expected_initialized_fields, check_dtype=False
    )


def test_choose_origin():
    material = Material(dimensions=[4, 4, 4], origin=[1, 2, 3])
    assert material.fields.x.max() == 4
    assert material.fields.y.max() == 5
    assert material.fields.z.max() == 6
    assert len(material.fields) == 64


def test_choose_spacing():
    material = Material(
        dimensions=[5, 5, 9], origin=[2, 4, 5], spacing=[0.5, 0.25, 0.125]
    )
    assert material.fields.x.max() == 4
    assert material.fields.y.max() == 5
    assert material.fields.z.max() == 6
    assert len(material.fields) == 225


def test_choose_sizes():
    material = Material(dimensions=[2, 3, 4], origin=[2, 4, 5], sizes=[2, 4, 6])
    expected_spacing = np.array([2, 2, 2])
    assert_array_equal(material.spacing, expected_spacing)
    assert material.fields.x.max() == 4
    assert material.fields.y.max() == 8
    assert material.fields.z.max() == 11


def test_initial_material_feature(material):
    material = material.create_fields(fields={"feature": 2})
    assert material.fields.feature.sum() == 2 * 16**3


def test_init_material_bad_fields():
    field_too_short = {"a": [1, 2]}
    with pytest.raises(ValueError):
        _ = Material(fields=field_too_short)


def test_init_material_no_xyz(expected_default_fields):
    field = pd.DataFrame({"a": np.arange(16**3)})
    expected_fields = pd.concat([field, expected_default_fields], axis=1)

    material = Material(fields=field)

    assert_frame_equal(material.fields, expected_fields, check_dtype=False)


def test_init_material_missing_field():
    dimensions = np.array([2, 2, 2])
    fields = {"x": np.tile(np.arange(2), 4)}
    with pytest.raises(KeyError):
        _ = Material(dimensions=dimensions, fields=fields)


def test_init_material_inconsistent_dimensions():
    from materialite.util import cartesian_grid

    dimensions = np.array([2, 2, 2])
    coords = cartesian_grid([2, 2, 3])
    fields = pd.DataFrame(data=coords, columns=["x", "y", "z"])
    with pytest.raises(ValueError):
        _ = Material(dimensions=dimensions, fields=fields)


def test_init_material_inconsistent_sizes():
    from materialite.util import cartesian_grid

    dimensions = np.array([2, 2, 2])
    coords = cartesian_grid([2, 2, 2])
    coords[:, 0] = coords[:, 0] * 2
    fields = pd.DataFrame(data=coords, columns=["x", "y", "z"])
    with pytest.raises(ValueError):
        _ = Material(dimensions=dimensions, fields=fields)


def test_init_material_all_xyz(small_material):
    dimensions = small_material.dimensions
    expected_fields = small_material.fields
    # keep x, y, and z, but pass them in unsorted
    fields = expected_fields[["x", "y", "z"]].copy().sort_values(by=["x", "y", "z"])

    material = Material(dimensions=dimensions, fields=fields)

    assert_frame_equal(material.fields, expected_fields, check_dtype=False)


def test_init_material_one_xyz():
    dimensions = np.array([2, 1, 1])
    origin = np.array([0, 1, 2])
    x = np.arange(2)
    yz_id = np.zeros(2)
    expected_y = yz_id + 1
    expected_z = (yz_id + 1) * 2
    fields = {"x": x}
    expected_fields = pd.DataFrame(
        {
            "x": x,
            "y": expected_y,
            "z": expected_z,
            "x_id": x,
            "y_id": yz_id,
            "z_id": yz_id,
        }
    )

    material = Material(dimensions=dimensions, origin=origin, fields=fields)

    assert_frame_equal(material.fields, expected_fields, check_dtype=False)


def test_initialize_sphere():
    sphere = Sphere(radius=1, centroid=[2, 2, 2])
    assert sphere.radius == 1 and sphere.centroid == [2, 2, 2]


def test_check_inside_box():
    x = np.array([-0.001, -1, 0, 0, 0, 0.5, 1, 1.001, 2, 1, 1])
    y = np.array([-0.001, 0, -1, 0, 0, 0.5, 1, 1.001, 1, 2, 1])
    z = np.array([-0.001, 0, 0, -1, 0, 0.5, 1, 1.001, 1, 1, 2])
    box = Box(min_corner=[0, 0, 0], max_corner=[1, 1, 1])
    expected_results = np.array(
        [False, False, False, False, True, True, True, False, False, False, False]
    )
    assert_array_equal(box.check_inside(x, y, z), expected_results)


def test_check_inside_semi_infinite_box_on_min_side():
    x = np.array([-0.001, -1, 0, 0, 0, 0.5, 1, 1.001, 2, 1, 1])
    y = np.array([-0.001, 0, -1, 0, 0, 0.5, 1, 1.001, 1, 2, 1])
    z = np.array([-0.001, 0, 0, -1, 0, 0.5, 1, 1.001, 1, 1, 2])
    box = Box(max_corner=[1, 1, 1])
    expected_results = np.array(
        [True, True, True, True, True, True, True, False, False, False, False]
    )
    assert_array_equal(box.check_inside(x, y, z), expected_results)


def test_check_inside_semi_infinite_box_on_max_side():
    x = np.array([-0.001, -1, 0, 0, 0, 0.5, 1, 1.001, 2, 1, 1])
    y = np.array([-0.001, 0, -1, 0, 0, 0.5, 1, 1.001, 1, 2, 1])
    z = np.array([-0.001, 0, 0, -1, 0, 0.5, 1, 1.001, 1, 1, 2])
    box = Box(min_corner=[0, 0, 0])
    expected_results = np.array(
        [False, False, False, False, True, True, True, True, True, True, True]
    )
    assert_array_equal(box.check_inside(x, y, z), expected_results)


def test_insert_feature():
    material = Material(dimensions=[4, 4, 4])
    material = material.create_fields(
        {"feature": np.ones(material.num_points)}
    ).insert_feature(Sphere(radius=1, centroid=[0, 0, 0]), fields={"feature": 2})

    assert material.fields.query("x < 0.1 and y < 0.1 and z < 0.1").feature.iat[0] == 2


def test_export_to_vtk(small_material, tmp_path):
    output_filename = tmp_path / "fields.vtk"
    small_material.export_to_vtk(output=output_filename)
    with open(output_filename) as output_file:
        dimensions = output_file.read().split("\n")[4].split()[1:]
    assert dimensions == ["3", "4", "5"]


def test_add_feature_with_field(small_material):
    sphere_1 = Sphere(radius=1, centroid=[0, 0, 0])
    sphere_2 = Sphere(radius=2, centroid=[1, 2, 3])

    material = (
        small_material.create_fields(
            fields={
                "feature": 1,
                "euler_angles_1": 0,
                "euler_angles_2": 0,
                "euler_angles_3": 0,
            }
        )
        .insert_feature(
            sphere_1,
            fields={
                "feature": 2,
                "euler_angles_1": 30,
                "euler_angles_2": 45,
                "euler_angles_3": 60,
            },
        )
        .insert_feature(
            sphere_2,
            fields={
                "feature": 3,
                "euler_angles_1": -30,
                "euler_angles_2": -45,
                "euler_angles_3": -60,
            },
        )
    )

    assert (
        material.get_fields()
        .query("x < 0.1 and y < 0.1 and z < 0.1")
        .euler_angles_1.iat[0]
        == 30.0
    )
    assert (
        material.get_fields()
        .query("x > 0.9 and y > 1.9 and z > 2.9")
        .euler_angles_3.iat[0]
        == -60
    )


def test_initialize_superellipsoid():
    superellipsoid = Superellipsoid(
        major_radius=3,
        intermediate_radius=2,
        minor_radius=1,
        shape_exponent=10,
        centroid=[2, 2, 2],
    )
    assert superellipsoid.major_radius == 3
    assert superellipsoid.intermediate_radius == 2
    assert superellipsoid.minor_radius == 1
    assert superellipsoid.shape_exponent == 10
    assert superellipsoid.centroid == [2, 2, 2]


def test_insert_superellipsoids(small_material):
    superellipsoid_1 = Superellipsoid(
        major_radius=3,
        intermediate_radius=2,
        minor_radius=1,
        shape_exponent=10,
        centroid=[0, 0, 0],
    )
    superellipsoid_2 = Superellipsoid(
        major_radius=3,
        intermediate_radius=2,
        minor_radius=1,
        shape_exponent=10,
        centroid=[1, 2, 3],
    )
    material = (
        small_material.create_fields(fields={"feature": 1})
        .insert_feature(superellipsoid_1, fields={"feature": 2})
        .insert_feature(superellipsoid_2, fields={"feature": 3})
    )

    assert (
        material.get_fields().query("x < 0.1 and y < 0.1 and z < 0.1").feature.iat[0]
        == 2
    )
    assert (
        material.get_fields().query("x > 0.9 and y > 1.9 and z > 2.9").feature.iat[0]
        == 3
    )


def test_export_to_evpfft(tmp_path, expected_evpfft_file_contents):
    data_container = "DataContainers/SyntheticVolumeDataContainer"
    material = import_dream3d(
        file="tests/Cylinder_Synthetic.dream3d",
        simpl_geometry_path=f"{data_container}/_SIMPL_GEOMETRY",
        region_id_path=f"{data_container}/CellData/FeatureIds",
        region_field_paths=[f"{data_container}/Grain Data/EulerAngles"],
    ).create_uniform_field("phase", 1)

    output_filename = tmp_path / "fields.txt"
    material.export_to_evpfft(
        feature_label="feature_ids",
        output=output_filename,
        euler_angles_to_degrees=True,
    )
    with open(output_filename) as output_file:
        contents = output_file.read()
    assert contents == expected_evpfft_file_contents


def test_insert_feature_with_phase(material):
    sphere = Sphere(radius=1, centroid=[0, 0, 0])
    material = material.create_fields(fields={"feature": 1, "phase": 1}).insert_feature(
        sphere, fields={"feature": 2, "phase": 0}
    )
    assert material.get_fields().phase.iloc[0] == 0
    assert material.get_fields().phase.iloc[-1] == 1


def test_export_to_evpfft_without_dream3d(tmp_path):
    sphere = Sphere(radius=1, centroid=[0, 0, 0])
    material = (
        Material(dimensions=[2, 2, 2])
        .create_fields(
            fields={
                "feature": 1,
                "phase": 1,
                "euler_angles_1": 0,
                "euler_angles_2": 0,
                "euler_angles_3": 0,
            }
        )
        .insert_feature(
            sphere,
            fields={
                "feature": 2,
                "euler_angles_1": 30,
                "euler_angles_2": 45,
                "euler_angles_3": 60,
            },
        )
    )
    output_filename = tmp_path / "small_model_fields.txt"
    material.export_to_evpfft(output=output_filename)
    with open(output_filename) as output_file:
        first_line = output_file.readline()
    assert "1 1 1 2 1" in first_line


def test_power_of_two_below():
    assert (
        power_of_two_below(129) == 128
        and power_of_two_below(1) == 1
        and power_of_two_below(8)
        and power_of_two_below(1000) == 512
        and power_of_two_below(99) == 64
    )


def test_create_single_field(material):
    field = np.ones(16**3)
    name = "ones"
    expected_fields = material.fields
    expected_fields[name] = field
    new_material = material.create_fields({name: field})
    assert_frame_equal(new_material.fields, expected_fields)


def test_create_uniform_number_field(material):
    field = np.ones(16**3)
    value = 1.0
    name = "ones"
    expected_fields = material.fields
    expected_fields[name] = field
    new_material = material.create_fields({name: value})
    assert_frame_equal(new_material.fields, expected_fields)


def test_create_uniform_array_field(material):
    value = np.arange(6).reshape((2, 3))
    name = "range"
    expected_fields = material.fields
    expected_fields[name] = [value] * 16**3
    new_material = material.create_uniform_field(name, value)
    assert_frame_equal(new_material.fields, expected_fields)


def test_create_fields(material):
    ones_field = np.ones(16**3)
    twos_field = 2 * np.ones(16**3)
    ones_name = "ones"
    twos_name = "twos"
    expected_fields = material.fields
    expected_fields[ones_name] = ones_field
    expected_fields[twos_name] = twos_field
    new_material = material.create_fields(
        {ones_name: ones_field, twos_name: twos_field}
    )
    assert_frame_equal(new_material.fields, expected_fields)


def test_remove_fields(small_material):
    material = small_material
    expected_fields = material.get_fields()
    material = material.create_uniform_field("a", 1).remove_field("a")
    assert_frame_equal(material.get_fields(), expected_fields)
    material = material.create_fields({"test_x": material.extract("x")})
    expected_fields_test_x = material.get_fields()
    expected_fields_test_x["data"] = np.repeat([100, 200], int(material.num_points / 2))
    regional_fields = pd.DataFrame(
        columns=["test_x", "data", "data2"],
        data=np.c_[[0, 1], [100, 200], [1000, 2000]],
    )
    material = material.create_regional_fields("test_x", regional_fields)
    assert_frame_equal(material.remove_field("test_x").get_fields(), expected_fields)
    assert_frame_equal(
        material.remove_field("data2", in_regional_field="test_x").get_fields(),
        expected_fields_test_x,
    )


def test_segment(small_material):
    threshold = 0
    material = small_material.segment("x", threshold)
    expected_fields = small_material.get_fields().assign(
        **{"segmented_x": np.repeat([0, 1], material.num_points / 2)}
    )
    assert_frame_equal(material.get_fields(), expected_fields)


def test_random_integer_field(material):
    field_label = "spin"
    rng = np.random.default_rng(seed=12345)
    expected_field = rng.integers(low=3, size=16**3)
    expected_fields = material.get_fields()
    expected_fields[field_label] = expected_field

    rng = np.random.default_rng(seed=12345)
    assert_frame_equal(
        material.create_random_integer_field(label=field_label, low=3, rng=rng).fields,
        expected_fields,
    )


def test_extract_field(small_material):
    expected_field = np.repeat(
        [0, 1], int(small_material.num_points) / small_material.dimensions[0]
    )
    assert_array_equal(small_material.extract("x"), expected_field)


def test_extract_multidimensional_field(small_material):
    n = int(small_material.num_points)
    expected_field = np.arange(n * 3).reshape((n, 3))
    small_material = small_material.create_fields({"field": list(expected_field)})
    assert_array_equal(small_material.extract("field"), expected_field)


def test_extract_multiple_fields(small_material):
    x = np.repeat([0, 1], int(small_material.num_points) / small_material.dimensions[0])
    z = np.tile(
        np.arange(small_material.dimensions[-1]),
        np.prod(small_material.dimensions[:-1]),
    )
    expected_fields = pd.DataFrame(data=np.c_[x, z], columns=["x", "z"])
    assert_array_equal(small_material.extract(["x", "z"]), expected_fields)


def test_extract_tensor(small_material):
    from materialite.tensor import Vector

    vectors = Vector.random(small_material.num_points)
    material = small_material.create_fields({"vector": vectors})
    extracted_vectors = material.extract("vector")
    assert_array_equal(extracted_vectors.components, vectors.components)

    material = small_material.create_uniform_field("vector", vectors[0])
    extracted_vectors = material.extract("vector")
    assert_array_equal(
        extracted_vectors.components, [vectors[0].components] * material.num_points
    )


def test_volume_fraction(material):
    features = np.ones(4096)
    features[-10:] = 2
    material = material.create_fields({"feature": features})
    expected_unique_features = np.array([1, 2])
    expected_volume_fractions = np.array([4086, 10]) / 4096

    volume_fractions = material.get_region_volume_fractions(region_label="feature")

    assert_array_equal(list(volume_fractions.keys()), expected_unique_features)
    assert_array_equal(list(volume_fractions.values()), expected_volume_fractions)


def test_region_indices(material):
    box = Box(max_corner=[7, np.inf, np.inf])
    material = material.create_uniform_field("phase", 1).insert_feature(
        box, fields={"phase": 2}
    )
    indices = material.get_region_indices("phase")
    assert_array_equal(indices[1], np.arange(2048) + 2048)
    assert_array_equal(indices[2], np.arange(2048))


def test_create_voronoi():
    expected_point_regions = np.array(
        [1, 1, 2, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 0, 1, 1, 1, 0]
    )
    material = Material(dimensions=[2, 3, 4], origin=[1, 1, 1]).create_voronoi(
        num_regions=3, rng=np.random.default_rng(12345)
    )

    assert_array_equal(material.fields.region.to_numpy(), expected_point_regions)


def test_create_voronoi_periodic():
    material = Material(dimensions=[2, 3, 4], origin=[1, 1, 1]).create_voronoi(
        num_regions=3, rng=np.random.default_rng(12345), periodic=True
    )
    expected_point_regions = np.array(
        [0, 1, 2, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 1, 2, 0]
    )

    assert_array_equal(material.fields.region.to_numpy(), expected_point_regions)


def test_assign_random_orientations(small_material, seeded_rng):
    xy_points = np.prod(small_material.dimensions[:-1])
    # Euler angles from seeded rng (range of z1 and z3 is (-pi, pi])
    z1 = 2 * np.pi * np.array([0.22733602, 0.31675834, 0.79736546 - 1, 0.67625467 - 1])
    x2 = np.arccos(2 * np.array([0.39110955, 0.33281393, 0.59830875, 0.18673419]) - 1)
    z3 = (
        2
        * np.pi
        * np.array([0.67275604 - 1, 0.94180287 - 1, 0.24824571, 0.94888115 - 1])
    )
    expected_eulers = np.c_[z1, x2, z3]
    expected_eulers = np.tile(expected_eulers, (xy_points, 1))
    new_material = small_material.assign_random_orientations(
        region_label="z", rng=seeded_rng
    )
    eulers = new_material.extract("orientation").euler_angles
    assert_allclose(eulers, expected_eulers)


# def test_plot_detects_nonnumeric_data(material):
#     material.create_uniform_field('text', 'text')
#     with pytest.raises(TypeError):
#         material.plot('text')


@pytest.mark.parametrize(
    "x_range, y_range, z_range, dimensions, origin",
    [
        ((2, 2), (6, 15), (20, 24), [1, 4, 2], [2, 6, 20]),
        (None, None, None, [16, 16, 16], [0, 0, 0]),
        ((-10, 2), (-10, 15), (-10, 24), [2, 6, 7], [0, 0, 0]),
        ((2, 100), (6, 100), (20, 100), [15, 14, 11], [2, 6, 20]),
    ],
)
def test_dimensions_and_origin_of_cropped_material(
    material, x_range, y_range, z_range, dimensions, origin
):
    material = Material(spacing=[2, 3, 4])
    submaterial = material.crop_by_range(
        x_range=x_range, y_range=y_range, z_range=z_range
    )
    assert_array_equal(submaterial.dimensions, dimensions)
    assert_array_equal(submaterial.origin, origin)


def test_material_is_unchanged_when_crop_has_no_input(material):
    material = material.create_random_integer_field("field", 0, 100)
    submaterial = material.crop_by_range()
    assert_array_equal(submaterial.dimensions, material.dimensions)
    assert_array_equal(submaterial.origin, material.origin)
    assert_frame_equal(material.fields, submaterial.fields, check_like=True)


@pytest.mark.parametrize(
    "x_id_range, y_id_range, z_id_range, dimensions, origin",
    [
        ((1, 1), (2, 5), (5, 6), [1, 4, 2], [2, 6, 20]),
        (None, None, None, [16, 16, 16], [0, 0, 0]),
        ((-10, 1), (-10, 5), (-10, 6), [2, 6, 7], [0, 0, 0]),
        ((1, 100), (2, 100), (5, 100), [15, 14, 11], [2, 6, 20]),
    ],
)
def test_dimensions_and_origin_of_cropped_by_id_material(
    x_id_range, y_id_range, z_id_range, dimensions, origin
):
    material = Material(spacing=[2, 3, 4])
    submaterial = material.crop_by_id_range(
        x_id_range=x_id_range, y_id_range=y_id_range, z_id_range=z_id_range
    )
    assert_array_equal(submaterial.dimensions, dimensions)
    assert_array_equal(submaterial.origin, origin)


def test_dimensions_and_origin_of_cropped_material(material):
    submaterial = material.crop_by_range(x_range=(1, 1), y_range=(2, 5), z_range=(5, 6))
    assert_array_equal(submaterial.dimensions, [1, 4, 2])
    assert_array_equal(submaterial.origin, [1, 2, 5])


def test_dimensions_and_origin_of_submaterial_chopped_by_point_count(material):
    submaterial = material.chop_by_point_count(x=(1, 14), y=(2, 10), z=(5, 9))
    assert_array_equal(submaterial.dimensions, [1, 4, 2])
    assert_array_equal(submaterial.origin, [1, 2, 5])


def test_material_is_unchanged_when_chopped_by_point_count_is_used_with_no_input(
    material,
):
    material = material.create_random_integer_field("field", 0, 100)
    submaterial = material.chop_by_point_count()
    assert_array_equal(submaterial.dimensions, material.dimensions)
    assert_array_equal(submaterial.origin, material.origin)
    assert_frame_equal(material.fields, submaterial.fields, check_like=True)


def test_create_regional_fields(small_material):
    labels = ["x", "new1", "new2"]
    regions = [0, 1]
    new1 = [1, 2]
    new2 = [3, 4]
    expected_fields = small_material.get_fields().assign(
        **{"new1": np.repeat(new1, 12), "new2": np.repeat(new2, 12)}
    )
    df = pd.DataFrame(columns=labels, data=np.c_[regions, new1, new2])
    new_material = small_material.create_regional_fields("x", df)
    assert_frame_equal(new_material.get_fields(), expected_fields, check_dtype=False)
    with pytest.raises(ValueError):
        _ = new_material.create_uniform_field("new1", 1).get_fields()


def test_overwrite_old_regional_fields(small_material):
    labels = ["x", "new1"]
    regions = [0, 1]
    new1 = [1, 2]
    new2 = [3, 4]
    expected_fields = small_material.get_fields().assign(
        **{"new1": np.repeat(new2, 12)}
    )
    df = pd.DataFrame(columns=labels, data=np.c_[regions, new1])
    df2 = pd.DataFrame(columns=labels, data=np.c_[regions, new2])
    new_material = small_material.create_regional_fields(
        "x", df
    ).create_regional_fields("x", df2)
    assert_frame_equal(new_material.get_fields(), expected_fields)


def test_extract_regional_field(small_material):
    labels = ["x", "new1", "new2"]
    regions = [0, 1]
    new1 = [1, 2]
    new2 = [3, 4]
    df = pd.DataFrame(columns=labels, data=np.c_[regions, new1, new2])
    new_material = small_material.create_regional_fields("x", df)
    extracted_df = new_material.extract_regional_field("x")
    assert_frame_equal(extracted_df, df)
    extracted_new1 = new_material.extract_regional_field("x", "new1")
    assert extracted_new1 == new1


def test_regional_field_error_if_duplicated_keys(small_material):
    label = "x"
    duplicated_categories = [0, 0, 1]
    values = [1, 2, 3]
    regional_fields = pd.DataFrame({label: duplicated_categories, "new": values})
    with pytest.raises(ValueError):
        _ = small_material.create_regional_fields(label, regional_fields)


def test_regional_field_error_if_field_doesnt_exist(small_material):
    bad_label = "x1"
    categories = [0, 1]
    values = [1, 2]
    regional_fields = pd.DataFrame({bad_label: categories, "new": values})
    with pytest.raises(ValueError):
        _ = small_material.create_regional_fields(bad_label, regional_fields)


def test_regional_field_error_if_field_exists(small_material):
    label = "x"
    existing_field_label = "y"
    categories = [0, 1]
    values = [1, 2]
    regional_fields = pd.DataFrame({label: categories, existing_field_label: values})
    with pytest.raises(ValueError):
        _ = small_material.create_regional_fields(label, regional_fields)


def test_regional_field_error_if_missing_keys(small_material):
    label = "y"
    categories = [0, 1]
    values = [1, 2]
    regional_fields = pd.DataFrame({label: categories, "new": values})
    with pytest.raises(ValueError):
        _ = small_material.create_regional_fields(label, regional_fields)


def test_regional_field_error_if_new_regional_field_does_not_have_same_keys(
    small_material,
):
    labels = ["x", "new1"]
    bad_labels = ["x", "new2"]
    categories = [0, 1]
    bad_categories = [0, 2]
    new1 = [1, 2]
    new2 = [3, 4]
    df = pd.DataFrame(columns=labels, data=np.c_[categories, new1])
    bad_df = pd.DataFrame(columns=bad_labels, data=np.c_[bad_categories, new2])
    with pytest.raises(ValueError):
        new_material = small_material.create_regional_fields(
            "x", df
        ).create_regional_fields("x", bad_df)
