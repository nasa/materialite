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

import gzip
import json

import materialite.tensor
import numpy as np
import pandas as pd
from materialite import Material
from materialite.tensor import Orientation, Tensor


class MaterialEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Tensor):
            return {
                "data": obj.components.tolist(),
                "type": type(obj).__name__,
                "dims": obj.dims_str,
            }
        elif isinstance(obj, Orientation):
            return {"data": obj.rotation_matrix.tolist(), "type": type(obj).__name__}
        elif isinstance(obj, (np.number, np.ndarray)):
            return {"data": obj.tolist()}
        return super().default(obj)


def decoder_object_hook(dct):
    data = dct.get("data", None)
    if data is None:
        return dct
    data_type = dct.get("type", None)
    dims = dct.get("dims", None)
    if data_type is None:
        return data
    elif dims is None:
        tensor_type = getattr(materialite.tensor, data_type)
        return tensor_type(data)
    else:
        tensor_type = getattr(materialite.tensor, data_type)
        return tensor_type(data, dims)


def read_from_json(filename, decompress=False):
    if decompress:
        r_string = "rb"
    else:
        r_string = "r"
    with open(filename, r_string) as f:
        loaded_data = f.read()
    if decompress:
        loaded_data = gzip.decompress(loaded_data).decode("utf-8")
    loaded_data = json.loads(loaded_data, object_hook=decoder_object_hook)
    state = dict()
    fields = dict()
    regional_fields = dict()
    for k, v in loaded_data.items():
        if k == "metadata":
            origin = v["origin"]
            dimensions = v["dimensions"]
            spacing = v["spacing"]
            continue
        elif k == "state":
            state = v
        elif k == "regional_fields":
            regional_fields = v
        else:
            fields[k] = v
    if not fields:
        material = Material(dimensions=dimensions, origin=origin, spacing=spacing)
    else:
        fields = pd.DataFrame(fields)
        material = Material(
            dimensions=dimensions, origin=origin, spacing=spacing, fields=fields
        )
    material.state = state
    for k, v in regional_fields.items():
        material = material.create_regional_fields(k, pd.DataFrame(v))
    return material


def write_to_json(material, filename, compress=False):
    coord_labels = ["x", "y", "z", "x_id", "y_id", "z_id"]
    labels = list(material.fields.drop(coord_labels, axis=1))
    material_dict = {
        "metadata": {
            "origin": material.origin.tolist(),
            "dimensions": material.dimensions.tolist(),
            "spacing": material.spacing.tolist(),
        }
    }
    if material.state:
        material_dict["state"] = material.state
    if material._regional_fields:
        material_dict["regional_fields"] = dict()
        for k in material._regional_fields.keys():
            regional_labels = list(material.extract_regional_field(k))
            material_dict["regional_fields"][k] = dict()
            for l in regional_labels:
                material_dict["regional_fields"][k][l] = (
                    material.extract_regional_field(k, l)
                )
    for l in labels:
        material_dict[l] = material.extract(l)
    json_material = json.dumps(material_dict, cls=MaterialEncoder)
    if compress:
        json_material = gzip.compress(json_material.encode("utf-8"))
        w_string = "wb"
    else:
        w_string = "w"
    with open(filename, w_string) as f:
        f.write(json_material)
