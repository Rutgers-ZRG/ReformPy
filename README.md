# SHAPE 

**Symmetry-guided Hyperspace Accelerated Potential Energy exploration**

### Implemented in Python3

## Dependencies
* Python >= 3.8.5
* Numpy >= 1.24.4
* Scipy >= 1.10.1
* Numba >= 0.58.1
* ASE >= 3.22.1
* libfp >= 3.1.2

## Setup
To install the C implementation of [Fingerprint Library](https://github.com/Rutgers-ZRG/libfp)  \
First, you need create a [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment:
  ```bash
  conda create -n shape python=3.8 pip ; conda activate shape
  python3 -m pip install -U pip setuptools wheel
  ```
Then use conda to install LAPACK:
  ```bash
  conda install conda-forge::lapack
  ```
Next, you need to download the `libfp` using `git`:
  ```bash
  git clone https://github.com/Rutgers-ZRG/libfp.git
  ```
and modify the `setup.py` in `libfp/fppy`:
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
  cd libfp/fppy/ ; python3 -m pip install -e .
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


### Calling SHAPE calculator from ASE API
```python
import numpy as np
import ase.io
from ase.optimize import BFGS, LBFGS, BFGSLineSearch, QuasiNewton, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import StrainFilter, UnitCellFilter
from ase.io.trajectory import Trajectory

from shape.calculator import SHAPE_Calculator
# from shape.mixing import MixedCalculator
# from ase.calculators.vasp import Vasp

atoms = ase.io.read('.'+'/'+'POSCAR')
ase.io.vasp.write_vasp('input.vasp', atoms, direct=True)
trajfile = 'opt.traj'

from functools import reduce

chem_nums = list(atoms.numbers)
znucl_list = reduce(lambda re, x: re+[x] if x not in re else re, chem_nums, [])
ntyp = len(znucl_list)
znucl = znucl_list

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
traj = Trajectory(trajfile, 'w', atoms=atoms, properties=['energy', 'forces', 'stress'])

############################## Relaxation method ##############################

# opt = BFGS(af, maxstep = 1.e-1)
opt = FIRE(af, maxstep = 1.e-1)
# opt = LBFGS(af, maxstep = 1.e-1, memory = 10, use_line_search = True)
# opt = LBFGS(af, maxstep = 1.e-1, memory = 10, use_line_search = False)
# opt = SciPyFminCG(af, maxstep = 1.e-1)
# opt = SciPyFminBFGS(af, maxstep = 1.e-1)

opt.attach(traj.write, interval=1)
opt.run(fmax = 1.e-3, steps = 500)
traj.close()

traj = Trajectory(trajfile, 'r')
atoms_final = traj[-1]
ase.io.write('fp_opt.vasp', atoms_final, direct = True, long_format = True, vasp5 = True)

final_cell = atoms_final.get_cell()
final_cell_par = atoms_final.cell.cellpar()
final_structure = atoms_final.get_scaled_positions()
final_energy_per_atom = float( atoms_final.get_potential_energy() / len(atoms_final) )
final_stress = atoms_final.get_stress()

print("Relaxed lattice vectors are \n{0:s}".\
      format(np.array_str(final_cell, precision=6, suppress_small=False)))
print("Relaxed cell parameters are \n{0:s}".\
     format(np.array_str(final_cell_par, precision=6, suppress_small=False)))
print("Relaxed structure in fractional coordinates is \n{0:s}".\
      format(np.array_str(final_structure, precision=6, suppress_small=False)))
print("Final energy per atom is \n{0:.6f}".format(final_energy_per_atom))
print("Final stress is \n{0:s}".\
      format(np.array_str(final_stress, precision=6, suppress_small=False)))
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

