# Materialite *(Alpha Release)* 

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Status](https://img.shields.io/badge/status-alpha-red)

**Streamlined Microstructure Modeling for Materials Science and Mechanics**

## Overview

Materialite is a Python package that aims to integrate the fragmented landscape of microstructure modeling tools into a unified framework for materials science and mechanics research.

## Purpose

Researchers in computational materials science currently rely on a patchwork of disconnected tools—MATLAB scripts, Fortran codes, C++ solvers, and various Python libraries—that are often not user-friendly and don't communicate well with each other. Entirely written in Python, Materialite provides:

- **Unified data structure** that works across different modeling approaches
- **Streamlined workflows** from manufacturing processes to microstructure formation to mechanical properties
- **Simplified interfaces** for complex operations like tensor math and crystallography
- **Data exchange** with established tools (e.g. DREAM.3D, EVP-FFT)

## Current Models

- **Process Models**: Grain coarsening, laser heating temperature fields, microstructure evolution from laser melting
- **Mechanical Solvers**: Taylor model and small-strain FFT (Fast Fourier Transform) solver for crystal elasticity and plasticity
- **Constitutive Models**: Elasticity, rate-independent plasticity, elasto-viscoplasticity with multiple hardening laws

## Documentation

Documentation, demos, and examples are available at [https://nasa.github.io/materialite](https://nasa.github.io/materialite).

## Installation

> **⚠️ ALPHA RELEASE:** This package is under active development. APIs and interfaces are subject to breaking changes without notice. Not recommended for production use.

*Clone and cd into the repository*
```bash
git clone https://github.com/nasa/materialite.git
```

```bash
cd materialite
```

*Create and activate conda environment*
```bash
conda env create -f environment.yml
```

```bash
conda activate materialite
```

*Future releases will be available on conda-forge and PyPI.*


## Simple Example

Run a basic process-structure-property simulation using `Materialite`.

Import the core `Material` object, some models, and other required objects for model inputs.

```python
from materialite import Material, Order4SymmetricTensor
from materialite.models import GrainCoarseningModel
from materialite.models.small_strain_fft import SmallStrainFFT, Elastic, LoadSchedule
```

Create a default material (generates a 16 x 16 x 16 point grid).

```python
material = Material()
```

Create a model for grain coarsening (`GrainCoarseningModel` is a Potts Monte Carlo-based model).

```python
grain_coarsening_model = GrainCoarseningModel(num_flip_attempts=10**5)
```

Run the grain coarsening model, and assign random orientations to the grains in the `Material` that is returned by the model.

```python
material = grain_coarsening_model(material)
material = material.assign_random_orientations(region_label="grain")
```

Plot the grain ID values.

```python
material.plot("grain")
```

![png](https://raw.githubusercontent.com/nasa/materialite/refs/heads/main/docs/resources/simple_example_grain.png)

Create a model that will simulate uniaxial tension (`SmallStrainFFT` is an FFT-based full-field mechanical simulation).

```python
stiffness = Order4SymmetricTensor.from_cubic_constants(C11=250, C12=150, C44=120)
constant_strain_rate = LoadSchedule.from_constant_uniaxial_strain_rate(
    magnitude=0.001, direction="x"
)
mechanical_model = SmallStrainFFT(
    load_schedule=constant_strain_rate, constitutive_model=Elastic(stiffness=stiffness)
)
```

Run the model.

```python
material = mechanical_model(material)
```

Plot the stress field in the loading direction.

```python
material.plot("stress", component=0)
```

![png](https://raw.githubusercontent.com/nasa/materialite/refs/heads/main/docs/resources/simple_example_stress_xx.png)

## Development Status

This alpha release provides basic functionality for microstructure modeling workflows. Internally at NASA, further capabilities exist in the pipeline for release. Development and release priorities depend on community needs and contributor availability.

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License

&copy; Copyright 2025 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

The Materialite platform is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Contact & Support

**Have feedback or found issues?** Please use our [GitHub issue tracker](https://github.com/nasa/materialite/issues) - we're actively monitoring and responding to user input.