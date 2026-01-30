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

import numpy as np
from materialite.tensor import Scalar


class TemperatureHistory:
    def __init__(self, temperatures, times):
        self.times = times
        self.temperatures = temperatures

    def temperature_increment(self, time, time_increment):
        new_time = time + time_increment
        if new_time <= self.times[0]:
            return Scalar(0.0)
        elif time >= self.times[-1]:
            return Scalar(0.0)
        else:
            T_start = self.interpolate(time)
            T_end = self.interpolate(new_time)
            return T_end - T_start

    def interpolate(self, time):
        idx = np.searchsorted(self.times, time)
        t0 = self.times[idx - 1]
        t1 = self.times[idx]
        T0 = self.temperatures[:, idx - 1]
        T1 = self.temperatures[:, idx]
        slope = (T1 - T0) / (t1 - t0)
        return T0 + slope * (time - t0)