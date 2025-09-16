import numpy as np
import pandas as pd
import pytest
from materialite import import_dream3d, import_spparks, import_vgstudio, import_evpfft
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

UTIL_PATH = "materialite.util"


def write_to_file(filename, data):
    content = "\n".join([" ".join(map(str, d)) for d in data])
    filename.write_text(content)


@pytest.fixture
def evpfft_file_contents():
    return [
        [99.80835, 66.40345, 28.98868, 1, 1, 1, 3, 1],
        [99.80835, 66.40345, 28.98868, 2, 1, 1, 3, 1],
        [99.80835, 66.40345, 28.98868, 1, 2, 1, 3, 1],
        [99.80835, 66.40345, 28.98868, 2, 2, 1, 3, 1],
        [-157.91174, 115.84247, -38.84106, 1, 1, 2, 5, 1],
        [-157.91174, 115.84247, -38.84106, 2, 1, 2, 5, 1],
        [149.57114, 93.98035, 66.88121, 1, 2, 2, 2, 1],
        [149.57114, 93.98035, 66.88121, 2, 2, 2, 2, 1],
        [124.95919, 77.12344, 3.69264, 1, 1, 3, 1, 1],
        [14.56741, 69.05068, 44.11161, 2, 1, 3, 9, 1],
        [124.95919, 77.12344, 3.69264, 1, 2, 3, 1, 1],
        [36.92948, 123.74353, 9.99989, 2, 2, 3, 7, 1],
        [124.95919, 77.12344, 3.69264, 1, 1, 4, 1, 1],
        [14.56741, 69.05068, 44.11161, 2, 1, 4, 9, 1],
        [149.27193, 113.16453, -52.42448, 1, 2, 4, 4, 1],
        [149.27193, 113.16453, -52.42448, 2, 2, 4, 4, 1],
    ]


@pytest.fixture
def spparks_file_contents():
    return [
        [1, 9026, 0, 0, 0],
        [2, 9026, 1, 0, 0],
        [3, 9026, 2, 0, 0],
        [4, 9026, 0, 1, 0],
        [5, 9026, 1, 1, 0],
        [6, 9026, 2, 1, 0],
        [7, 9026, 0, 2, 0],
        [8, 9026, 1, 2, 0],
        [9, 9026, 2, 2, 0],
        [10, 9026, 0, 0, 1],
        [11, 9026, 1, 0, 1],
        [12, 9026, 2, 0, 1],
        [13, 9026, 0, 1, 1],
        [14, 9026, 1, 1, 1],
        [15, 9026, 2, 1, 1],
        [16, 9026, 0, 2, 1],
        [17, 9026, 1, 2, 1],
        [18, 743, 2, 2, 1],
    ]


@pytest.fixture
def expected_dream3d_import():
    attributes = {}
    num_features = 9
    dimensions = [2, 3, 4]
    origin = [1, 1, 1]
    spacing = 2.0
    x_range = np.arange(2)
    y_range = np.arange(3)
    z_range = np.arange(4)
    zz, yy, xx = np.meshgrid(z_range, y_range, x_range, indexing="ij")
    grid = np.c_[np.ravel(xx), np.ravel(yy), np.ravel(zz)]
    feature_array = np.array(
        [3, 3, 3, 3, 3, 3, 5, 5, 2, 2, 6, 6, 1, 9, 1, 7, 8, 8, 1, 9, 4, 4, 8, 8]
    )
    unique_features = np.arange(num_features) + 1
    euler_angles = np.array(
        [
            [2.1809492, 1.3460579, 0.06444864],
            [2.610509, 1.6402665, 1.1672972],
            [1.7419844, 1.1589589, 0.5059479],
            [2.6052866, 1.9750936, 5.3682055],
            [3.5271056, 2.0218325, 5.605281],
            [1.9338474, 0.7969357, 3.21395],
            [0.64454097, 2.159732, 0.17453109],
            [2.9441507, 2.2782216, 0.95512784],
            [0.2542493, 1.2051617, 0.7698929],
        ]
    ).astype(np.float32)
    num_neighbors = np.array([6, 5, 3, 4, 4, 3, 5, 4, 4]).astype(np.int32)

    attributes["fields"] = pd.DataFrame(
        {
            "x": grid[:, 0] * spacing + origin[0],
            "y": grid[:, 1] * spacing + origin[1],
            "z": grid[:, 2] * spacing + origin[2],
            "phases": 1,
            "feature_ids": feature_array,
            "x_id": grid[:, 0],
            "y_id": grid[:, 1],
            "z_id": grid[:, 2],
        }
    )

    attributes["dimensions"] = dimensions
    attributes["origin"] = origin
    attributes["spacing"] = spacing
    attributes["sizes"] = None
    attributes["feature_fields"] = pd.DataFrame(
        {
            "euler_angles_1": euler_angles[:, 0],
            "euler_angles_2": euler_angles[:, 1],
            "euler_angles_3": euler_angles[:, 2],
            "num_neighbors": num_neighbors,
            "feature_ids": unique_features,
        }
    )

    attributes["fields"] = (
        attributes["fields"]
        .merge(attributes["feature_fields"], on="feature_ids")
        .sort_values(by=["x", "y", "z"], ignore_index=True)
    )
    return attributes


def test_dream3d_importer(expected_dream3d_import):
    data_container = "DataContainers/SyntheticVolumeDataContainer"
    material = import_dream3d(
        file="tests/Cylinder_Synthetic.dream3d",
        simpl_geometry_path=f"{data_container}/_SIMPL_GEOMETRY",
        region_id_path=f"{data_container}/CellData/FeatureIds",
        field_paths=[f"{data_container}/CellData/Phases"],
        region_field_paths=[
            f"{data_container}/Grain Data/EulerAngles",
            f"{data_container}/Grain Data/NumNeighbors",
        ],
    )
    assert_allclose(material.dimensions, expected_dream3d_import["dimensions"])
    assert_allclose(material.origin, expected_dream3d_import["origin"])
    assert_allclose(material.spacing, [expected_dream3d_import["spacing"]] * 3)
    assert_frame_equal(
        material.get_fields(), expected_dream3d_import["fields"], check_dtype=False
    )


def test_dream3d_importer_fields_only(expected_dream3d_import):
    data_container = "DataContainers/SyntheticVolumeDataContainer"
    material2 = import_dream3d(
        file="tests/Cylinder_Synthetic.dream3d",
        simpl_geometry_path=f"{data_container}/_SIMPL_GEOMETRY",
        region_id_path=f"{data_container}/CellData/FeatureIds",
        region_field_paths=[],
        field_paths=[f"{data_container}/CellData/EulerAngles"],
    )
    euler_labels = ["euler_angles_1", "euler_angles_2", "euler_angles_3"]
    assert_frame_equal(
        material2.get_fields()[euler_labels],
        expected_dream3d_import["fields"][euler_labels],
    )


def test_dream3d_importer_error_if_region_id_label_must_be_specified():
    data_container = "DataContainers/SyntheticVolumeDataContainer"
    with pytest.raises(ValueError):
        _ = import_dream3d(
            file="tests/Cylinder_Synthetic.dream3d",
            simpl_geometry_path=f"{data_container}/_SIMPL_GEOMETRY",
            region_field_paths=[f"{data_container}/Grain Data/NumNeighbors"],
        )


def test_import_vgstudio(mocker):
    file = "file.vol"
    dimensions = [2, 3, 4]
    num_points = 24
    expected_dtype = np.float32
    expected_label = "xct"
    expected_field = np.arange(num_points)
    expected_spacing = [1, 1, 1]
    expected_origin = [0, 0, 0]

    fromfile_mock = mocker.Mock(return_value=expected_field)
    mocker.patch(UTIL_PATH + ".np.fromfile", new=fromfile_mock)

    material = import_vgstudio(file, dimensions)

    fromfile_mock.assert_called_once_with(file, expected_dtype)
    assert_array_equal(material.dimensions, dimensions)
    assert_array_equal(material.spacing, expected_spacing)
    assert_array_equal(material.origin, expected_origin)
    assert_array_equal(material.fields[expected_label], expected_field)


def test_import_vgstudio_ranges(mocker):
    file = "file.vol"
    dimensions = [2, 3, 4]
    expected_dimensions = [1, 2, 3]
    x_range = [0, 1]
    y_range = [0, 2]
    z_range = [1, 4]
    num_points = 24
    dtype = np.float64
    label = "xray_ct"
    fromfile_field = np.arange(num_points)
    expected_field = np.array([6, 8, 12, 14, 18, 20])  # from slicing operations
    spacing = [1, 2, 3]
    origin = [4, 5, 6]

    fromfile_mock = mocker.Mock(return_value=fromfile_field)
    mocker.patch(UTIL_PATH + ".np.fromfile", new=fromfile_mock)

    material = import_vgstudio(
        file,
        dimensions,
        label=label,
        spacing=spacing,
        origin=origin,
        dtype=dtype,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
    )

    fromfile_mock.assert_called_once_with(file, dtype)
    assert_array_equal(material.dimensions, expected_dimensions)
    assert_array_equal(material.spacing, spacing)
    assert_array_equal(material.origin, origin)
    assert_array_equal(material.fields[label], expected_field)


def test_import_spparks(tmp_path, spparks_file_contents):
    filename = tmp_path / "spparks.txt"
    write_to_file(filename, spparks_file_contents)
    material = import_spparks(
        filename, skiprows=0, usecols=[1, 2, 3, 4], names=["grain", "x", "y", "z"]
    )
    expected_fields = np.array(spparks_file_contents)[:, 1:]
    fields = (
        material.get_fields()
        .sort_values(by=["z", "y", "x"])[["grain", "x", "y", "z"]]
        .to_numpy()
    )
    assert_array_equal(fields, expected_fields)


def test_import_evpfft(tmp_path, evpfft_file_contents):
    filename = tmp_path / "fields.txt"
    write_to_file(filename, evpfft_file_contents)
    material = import_evpfft(filename)
    expected_fields = pd.DataFrame(
        data=np.array(evpfft_file_contents),
        columns=[
            "euler_1",
            "euler_2",
            "euler_3",
            "x_id",
            "y_id",
            "z_id",
            "grain",
            "phase",
        ],
    ).sort_values(by=["x_id", "y_id", "z_id"])
    assert_allclose(
        material.extract("orientation").euler_angles_in_degrees,
        expected_fields[["euler_1", "euler_2", "euler_3"]].to_numpy(),
    )
    assert_array_equal(
        material.extract(["x_id", "y_id", "z_id"]) + 1,
        expected_fields[["x_id", "y_id", "z_id"]].to_numpy().astype(int),
    )
    assert_array_equal(
        material.extract(["grain", "phase"]),
        expected_fields[["grain", "phase"]].to_numpy().astype(int),
    )
