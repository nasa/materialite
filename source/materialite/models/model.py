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


class Model(ABC):
    def __call__(self, material, **kwargs):
        return self.run(material, **kwargs)

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    def verify_material(self, material, required_field_labels):
        field_labels = list(material.get_fields())
        if not all([l in field_labels for l in required_field_labels]):
            raise AttributeError(
                f"{type(self)} requires the Material to have fields with labels {required_field_labels}. The actual field labels are {field_labels}."
            )
        return material
