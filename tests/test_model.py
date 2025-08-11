import pytest
from materialite import Material
from materialite.models import Model


class DummyModel(Model):
    def __init__(self, attribute1):
        self.attribute1 = attribute1

    def run(self, material, arg1=10):
        return material.dimensions[0] + arg1 * self.attribute1


@pytest.fixture
def model():
    return DummyModel(5)


@pytest.mark.parametrize("arg1, expected_result", [(None, 52), (100, 502)])
def test_run(model, arg1, expected_result):
    material = Material(dimensions=[2, 3, 4])
    if arg1 is None:
        result = model(material)
    else:
        result = model(material, arg1=arg1)
    assert result == expected_result


@pytest.mark.parametrize(
    "required_fields, raises_error",
    [(["a"], False), (["a", "bc"], False), (["cb"], True), (["a", "c"], True)],
)
def test_verify_material(model, required_fields, raises_error, mocker):
    material = mocker.Mock()
    material.get_fields.return_value = ["a", "bc"]
    if raises_error:
        with pytest.raises(AttributeError):
            _ = model.verify_material(material, required_fields)
    else:
        new_material = model.verify_material(material, required_fields)
        assert new_material == material
