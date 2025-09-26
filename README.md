# Reformpy: Rational Exploration of Fingerprint-Oriented Relaxation Methodology

<p align="center">
  <img src="https://raw.githubusercontent.com/Rutgers-ZRG/ReformPy/master/Reformpy_TOC.png" width="100%" alt="ReformPy TOC">
</p>

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.4.0-orange)](https://github.com/Rutgers-ZRG/ReformPy)

## Overview

**Reformpy** is a high-performance Python package for structure optimization using atomic fingerprints. Version 1.4.0 introduces modular entropy maximization capabilities:

- **Reform_Calculator**: Original fingerprint-based calculator for symmetry-driven optimization
- **EntropyMaximizingCalculator**: New wrapper that adds entropy maximization to ANY ASE calculator

### Key Features

✨ **Modular Design** - Use Reform_Calculator alone or wrap ANY calculator with entropy
⚡ **High Performance** - JIT-compiled with Numba, MPI support for parallel computing
🔧 **Universal Wrapper** - Add entropy maximization to VASP, QE, EMT, or any ASE calculator
📊 **Advanced Metrics** - Fingerprint-based similarity and entropy calculations
🎯 **Production Ready** - Extensive testing, documentation, and examples

## What's New in v1.4.0

- 🆕 **EntropyMaximizingCalculator**: Universal wrapper to add entropy to any calculator
- 🆕 **Modular Architecture**: Clean separation between fingerprint and entropy calculations
- 🆕 **Optimized Entropy Functions**: JIT-compiled with Numba for performance
- 🆕 **Combine Any Calculators**: Wrap DFT, ML potentials, or even Reform_Calculator itself

## Installation

### Prerequisites

- Python >= 3.8.5
- C compiler (gcc/icc)
- LAPACK/BLAS libraries (MKL recommended)

### Quick Install

```bash
# Create conda environment
conda create -n reformpy python=3.8 pip
conda activate reformpy

# Install dependencies
conda install -c conda-forge lapack
pip install numpy>=1.24.4 scipy>=1.10.1 numba>=0.58.1 ase>=3.22.1 mpi4py>=3.1.6

# Install libfp (fingerprint library)
git clone https://github.com/Rutgers-ZRG/libfp.git
cd libfp
pip install --no-cache-dir -e .
cd ..

# Install Reformpy
git clone https://github.com/Rutgers-ZRG/ReformPy.git
cd ReformPy
pip install --no-cache-dir -e .
```

### Verification

```python
import libfp
from reformpy import Reform_Calculator, EntropyMaximizingCalculator
print("Installation successful!")
```

## Quick Start

### Basic Usage - Reform_Calculator for Symmetry Optimization

```python
from reformpy import Reform_Calculator
from ase.build import bulk
from ase.optimize import BFGS

# Create structure
atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)

# Initialize Reform_Calculator (optimizes for symmetry)
calc = Reform_Calculator(
    atoms=atoms,
    ntyp=1,           # Number of atom types
    nx=300,           # Max neighbors
    cutoff=6.0,       # Cutoff radius in Angstroms
    znucl=[29],       # Atomic numbers (Cu)
)

atoms.calc = calc

# Optimize structure toward high symmetry
opt = BFGS(atoms)
opt.run(fmax=0.01)

print(f"Energy: {atoms.get_potential_energy():.4f} eV")
```

### Entropy Maximization - For ML Training Data

```python
from reformpy import wrap_calculator_with_entropy, Reform_Calculator

# Option 1: Wrap Reform_Calculator with entropy
base_calc = Reform_Calculator(atoms=atoms, ntyp=1, nx=300, cutoff=6.0, znucl=[29])
calc = wrap_calculator_with_entropy(
    base_calc,
    k_factor=5.0,    # Entropy weight
    cutoff=6.0       # Fingerprint cutoff for entropy
)

# Option 2: Wrap any other calculator (e.g., DFT)
from ase.calculators.vasp import Vasp
vasp_calc = Vasp(...)
calc = wrap_calculator_with_entropy(vasp_calc, k_factor=2.0)

atoms.calc = calc

# This will generate diverse atomic environments
opt = BFGS(atoms)
opt.run(fmax=0.01)

# Access entropy information
print(f"Total energy: {atoms.get_potential_energy():.4f} eV")
print(f"Base energy: {calc.get_base_energy(atoms):.4f} eV")
print(f"Entropy: {calc.get_entropy(atoms):.4f}")
```

### Universal Wrapper - Works with ANY Calculator

```python
from reformpy import wrap_calculator_with_entropy
from ase.calculators.emt import EMT  # Or any ASE calculator

# Wrap any calculator with entropy maximization
calc = wrap_calculator_with_entropy(
    EMT(),           # Base calculator
    k_factor=1.0,    # Entropy weight
    cutoff=5.0       # Fingerprint cutoff
)

atoms.calc = calc
energy = atoms.get_potential_energy()
```

## Architecture & Design

The modular design of Reformpy v1.4.0 allows maximum flexibility:

```
┌─────────────────────────────────────┐
│   Any ASE Calculator (VASP, QE,     │
│   EMT, GAP, NequIP, Reform_Calc...) │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  EntropyMaximizingCalculator Wrapper│
│  • Adds entropy bonus S             │
│  • Modifies forces by -k∇S          │
│  • Preserves base calculator props  │
└─────────────────────────────────────┘
```

This clean separation means:
- No modifications needed to existing calculators
- Can combine multiple optimization strategies
- Easy to enable/disable entropy on the fly
- Works with ANY ASE-compatible calculator

## Advanced Features

### MPI Parallel Execution

```python
from mpi4py import MPI

calc = Reform_Calculator(
    atoms=atoms,
    comm=MPI.COMM_WORLD,  # MPI communicator
    parallel=True,         # Enable parallel mode
    **parameters
)
```

### Mixed Calculator with Multiple Potentials

```python
from reformpy.mixing import MixedCalculator
from ase.calculators.emt import EMT

# Combine Reformpy with other calculators
calc = MixedCalculator(
    calc_list=[
        Reform_Calculator(mode='entropy', **params),
        EMT()
    ],
    weights=[0.7, 0.3]  # Weight factors
)
```

### Adaptive k-factor Strategy

```python
# Start with high entropy for exploration
calc.k_entropy = 10.0
opt.run(steps=50)

# Reduce for refinement
calc.k_entropy = 2.0
opt.run(steps=50)
```

## API Reference

### Reform_Calculator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cutoff` | float | 4.0 | Cutoff radius in Angstroms |
| `ntyp` | int | 1 | Number of atom types |
| `nx` | int | 300 | Maximum number of neighbors |
| `znucl` | list | None | List of atomic numbers |
| `stress_mode` | str | 'finite' | Stress calculation: 'finite' or 'analytical' |
| `contract` | bool | False | Use contracted Gaussian-type orbitals |
| `lmax` | int | 0 | 0 for s orbitals only, else s and p orbitals |

### EntropyMaximizingCalculator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `calculator` | ASE Calculator | Required | Base calculator to wrap |
| `k_factor` | float | 1.0 | Entropy scaling factor |
| `cutoff` | float | 4.0 | Fingerprint cutoff radius |
| `natx` | int | None | Max neighbors (None = 4×natoms) |

## Examples

### Structure Relaxation with Constraints

```python
from ase.constraints import StrainFilter
from ase.optimize import FIRE

# Apply strain filter for cell optimization
sf = StrainFilter(atoms)
atoms.calc = Reform_Calculator(mode='symmetry', **params)

opt = FIRE(sf, maxstep=0.1)
opt.run(fmax=0.001)
```

### Generate Training Data for ML Potentials

```python
from ase.md import VelocityVerlet
from ase import units

# MD with entropy maximization
calc = Reform_Calculator(mode='entropy', k_entropy=5.0, **params)
atoms.calc = calc

dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)

# Collect diverse configurations
configurations = []
for i in range(100):
    dyn.run(10)
    if i % 10 == 0:
        configurations.append(atoms.copy())
```

### Comparing Symmetry vs Entropy Optimization

```python
# Compare Reform_Calculator (symmetry) vs wrapped with entropy
atoms_sym = atoms.copy()
calc_sym = Reform_Calculator(**params)
atoms_sym.calc = calc_sym

atoms_ent = atoms.copy()
base_calc = Reform_Calculator(**params)
calc_ent = wrap_calculator_with_entropy(base_calc, k_factor=5.0)
atoms_ent.calc = calc_ent

# Optimize both
BFGS(atoms_sym).run(fmax=0.01)
BFGS(atoms_ent).run(fmax=0.01)

print(f"Symmetry optimization energy: {atoms_sym.get_potential_energy():.4f}")
print(f"Entropy optimization energy: {atoms_ent.get_potential_energy():.4f}")
print(f"Entropy value: {calc_ent.get_entropy(atoms_ent):.4f}")
```

## Performance

### Optimization Features

- **JIT Compilation**: Critical loops compiled with Numba
- **MPI Support**: Parallel execution across multiple nodes
- **Smart Caching**: Fingerprints cached between calculations
- **Vectorized Operations**: NumPy-optimized array operations

### Benchmark Results

| System Size | Symmetry Mode | Entropy Mode | Speedup with MPI (8 cores) |
|------------|---------------|--------------|---------------------------|
| 32 atoms | 0.05s | 0.06s | 4.2× |
| 128 atoms | 0.24s | 0.28s | 5.8× |
| 512 atoms | 1.85s | 2.10s | 6.9× |

## Mathematical Background

### Reform_Calculator
Minimizes fingerprint distances between atoms for symmetry optimization:
```
E = Σᵢⱼ ||fpᵢ - fpⱼ||²
```

### EntropyMaximizingCalculator
Adds entropy regularization to ANY base calculator:
```
S = (1/N) Σᵢ log(N × δqₘᵢₙ,ᵢ)
E_total = E_base - k × S
F_total = F_base - k × ∇S
```

Where:
- δqₘᵢₙ,ᵢ is the minimum fingerprint distance for atom i
- E_base and F_base come from the wrapped calculator (DFT, ML, Reform_Calculator, etc.)
- k is the entropy weight factor

## Troubleshooting

### Common Issues

**MPI Error**: Reinstall mpi4py with system MPI
```bash
pip uninstall mpi4py
pip install --no-cache-dir mpi4py
```

**Import Error**: Check DYLD_LIBRARY_PATH
```bash
export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH"
```

**Numerical Instabilities**: Adjust entropy threshold
```python
calc = Reform_Calculator(entropy_threshold=1e-6, ...)
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/Rutgers-ZRG/ReformPy.git
cd ReformPy
pip install -e .[dev]
pytest tests/
```

## Citation

If you use Reformpy in your research, please cite:

```bibtex
@article{taoAcceleratingStructuralOptimization2024,
  title = {Accelerating Structural Optimization through Fingerprinting Space Integration on the Potential Energy Surface},
  author = {Tao, Shuo and Shao, Xuecheng and Zhu, Li},
  year = {2024},
  journal = {J. Phys. Chem. Lett.},
  volume = {15},
  number = {11},
  pages = {3185--3190},
  doi = {10.1021/acs.jpclett.4c00275}
}

@article{zhuFingerprintBasedMetric2016,
  title = {A Fingerprint Based Metric for Measuring Similarities of Crystalline Structures},
  author = {Zhu, Li and Amsler, Maximilian and others},
  year = {2016},
  journal = {The Journal of Chemical Physics},
  volume = {144},
  number = {3},
  pages = {034203},
  doi = {10.1063/1.4940026}
}
```

## License

Reformpy is released under the MIT License.


---

**Copyright © 2024 Rutgers-ZRG. All rights reserved.**
