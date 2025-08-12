# Materialite *(Alpha Release)*

This documentation is for the [Materialite Github Repository](https://github.com/nasa/materialite).

## Streamlined Microstructure Modeling 

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


## Installation

```{warning}
**ALPHA RELEASE - BREAKING CHANGES EXPECTED**

This is an alpha version of Materialite. The API is under active development and **subject to breaking changes without notice**. Interfaces, function signatures, and data structures may change significantly between releases. 

**Not recommended for production use.** Please pin to specific versions and expect to update your code when upgrading.
```

Currently requires cloning the repository and installing via conda:
```bash
# Clone the repository
git clone https://github.com/nasa/materialite.git
cd materialite

# Create and activate conda environment
conda env create -f environment.yml
conda activate materialite
```

*Future releases will be available on conda-forge and PyPI.*

## Getting Started

Check out a [Simple Example](demos/simple_example) and the [Material Basics Demo](demos/material_basics) to get started or browse the other demos for examples of common workflows. Additionally, runnable Jupyter notebook source code for all the demos is [here](https://github.com/nasa/materialite/edit/main/docs/_sources/demos).

## Development Status & Feedback

This alpha release provides basic functionality for microstructure modeling workflows. Internally at NASA, further capabilities exist in the pipeline for release. Development and release priorities depend on community needs and contributor availability.

**Have feedback or found issues?** Please use our [GitHub issue tracker](https://github.com/nasa/materialite/issues) - we're actively monitoring and responding to user input.

<!-- ```{tableofcontents}
``` -->

## License

&copy; Copyright 2025 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved. 

 
The Materialite platform is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0). 
 

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.