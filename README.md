# Reformpy

**A Python package for Rational Exploration of Fingerprint-Oriented Relaxation Methodology**

### Implemented in Python3

## Dependencies
* Python >= 3.8.5
* Numpy >= 1.21.4
* Scipy >= 1.8.0
* Numba >= 0.56.2
* ASE >= 3.22.1
* libfp >= 3.2.1

## Setup
To install the C implementation of [Fingerprint Library](https://github.com/Rutgers-ZRG/fplib)  \
First, you need create a [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment:
  ```bash
  conda create -n fplibenv python=3.8 pip ; conda activate fplibenv
  python3 -m pip install -U pip setuptools wheel
  ```
Then use conda to install LAPACK:
  ```bash
  conda install conda-forge::lapack
  ```
Next, you need to download the `fplib` using `git`:
  ```bash
  git clone https://github.com/Rutgers-ZRG/fplib.git
  ```
and modify the `setup.py` in `fplib/fppy`:
  ```python
  lapack_dir=["$CONDA_PREFIX/lib"]
  lapack_lib=['openblas']
  extra_link_args = ["-Wl,-rpath,$CONDA_PREFIX/lib"]
  .
  .
  .
  include_dirs = [source_dir, "$CONDA_PREFIX/include"]
  ```
  Also set the corresponding `DYLD_LIBRARY_PATH` in your `.bashrc` file as:
  ```bash
  export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH"
  ```
  Then:
  ```bash
  cd fplib/fppy/ ; python3 -m pip install -e .
  ```


Then install the remaining Python packages through pip
  ```bash
  python3 -m pip install numpy>=1.21.4 scipy>=1.8.0 numba>=0.56.2 ase==3.22.1
  ```

## Usage
### Basic ASE style documentation
See details for [ASE calculator class](https://wiki.fysik.dtu.dk/ase/development/calculators.html)
and [ASE calculator proposal](https://wiki.fysik.dtu.dk/ase/development/proposals/calculators.html#aep1)
```
    Fingerprint Calculator interface for ASE
    
        Implemented Properties:
        
            'energy': Sum of atomic fingerprint distance (L2 norm of two atomic 
                                                          fingerprint vectors)
            
            'forces': Gradient of fingerprint energy, using Hellmann–Feynman theorem
            
            'stress': Cauchy stress tensor using finite difference method
            
        Parameters:
        
            atoms:  object
                Attach an atoms object to the calculator.
                
            contract: bool
                Calculate fingerprint vector in contracted Guassian-type orbitals or not
            
            ntype: int
                Number of different types of atoms in unit cell
            
            nx: int
                Maximum number of atoms in the sphere with cutoff radius for specific cell site
                
            lmax: int
                Integer to control whether using s orbitals only or both s and p orbitals for 
                calculating the Guassian overlap matrix (0 for s orbitals only, other integers
                will indicate that using both s and p orbitals)
                
            cutoff: float
                Cutoff radius for f_c(r) (smooth cutoff function) [amp], unit in Angstroms
                
```


### Calling fplib3 calculator from ASE API
```python
import numpy as np
import ase.io
from ase.optimize import BFGS, LBFGS, BFGSLineSearch, QuasiNewton, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import StrainFilter, UnitCellFilter
from ase.io.trajectory import Trajectory

from fplib3_api4ase import fp_GD_Calculator
# from fplib3_mixing import MixedCalculator
# from ase.calculators.mixing import MixedCalculator
# from ase.calculators.vasp import Vasp

atoms = ase.io.read('.'+'/'+'POSCAR')
ase.io.vasp.write_vasp('input.vasp', atoms, direct=True)
trajfile = 'opt.traj'

from functools import reduce

chem_nums = list(atoms.numbers)
znucl_list = reduce(lambda re, x: re+[x] if x not in re else re, chem_nums, [])
ntyp = len(znucl_list)
znucl = np.array(znucl_list, int)

calc = fp_GD_Calculator(
            cutoff = 6.0,
            contract = False,
            znucl = znucl,
            lmax = 0,
            nx = 300,
            ntyp = ntyp
            )

atoms.calc = calc

# calc.test_energy_consistency(atoms = atoms)
# calc.test_force_consistency(atoms = atoms)

print ("fp_energy:\n", atoms.get_potential_energy())
print ("fp_forces:\n", atoms.get_forces())
print ("fp_stress:\n", atoms.get_stress())

# af = atoms
# af = StrainFilter(atoms)
af = UnitCellFilter(atoms, scalar_pressure = 0.0)

############################## Relaxation method ##############################

# opt = BFGS(af, maxstep = 1.e-1, trajectory = trajfile)
opt = FIRE(af, maxstep = 1.e-1, trajectory = trajfile)
# opt = LBFGS(af, maxstep = 1.e-1, trajectory = trajfile, memory = 10, use_line_search = True)
# opt = LBFGS(af, maxstep = 1.e-1, trajectory = trajfile, memory = 10, use_line_search = False)
# opt = SciPyFminCG(af, maxstep = 1.e-1, trajectory = trajfile)
# opt = SciPyFminBFGS(af, maxstep = 1.e-1, trajectory = trajfile)

opt.run(fmax = 1.e-5)

traj = Trajectory(trajfile)
ase.io.write('opt.vasp', traj[-1], direct = True, long_format=True, vasp5 = True)
```
## Citation
If you use this Fingerprint Library (or modified version) for your research please kindly cite our paper:
```
@article{taoAcceleratingStructuralOptimization2024,
  title = {Accelerating Structural Optimization through Fingerprinting Space Integration on the Potential Energy Surface},
  author = {Tao, Shuo and Shao, Xuecheng and Zhu, Li},
  year = {2024},
  month = mar,
  journal = {J. Phys. Chem. Lett.},
  volume = {15},
  number = {11},
  pages = {3185--3190},
  doi = {10.1021/acs.jpclett.4c00275},
  url = {https://pubs.acs.org/doi/10.1021/acs.jpclett.4c00275}
}
```
If you use Fingerprint distance as a metric to measure crystal similarity please also cite the following paper:
```
@article{zhuFingerprintBasedMetric2016,
  title = {A Fingerprint Based Metric for Measuring Similarities of Crystalline Structures},
  author = {Zhu, Li and Amsler, Maximilian and Fuhrer, Tobias and Schaefer, Bastian and Faraji, Somayeh and Rostami, Samare and Ghasemi, S. Alireza and Sadeghi, Ali and Grauzinyte, Migle and Wolverton, Chris and Goedecker, Stefan},
  year = {2016},
  month = jan,
  journal = {The Journal of Chemical Physics},
  volume = {144},
  number = {3},
  pages = {034203},
  doi = {10.1063/1.4940026},
  url = {https://doi.org/10.1063/1.4940026}
}
```

