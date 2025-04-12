import os
import sys
import numpy as np
import ase.io


atoms = ase.io.read('.'+'/'+'POSCAR')
print("Number of atoms:", len(atoms))


###################################################################################################

from reformpy.calculator import Reform_Calculator
from functools import reduce

chem_nums = list(atoms.numbers)
znucl_list = reduce(lambda re, x: re+[x] if x not in re else re, chem_nums, [])
ntyp = len(znucl_list)
znucl = znucl_list


print (atoms.get_atomic_numbers())
print (znucl)
print (atoms.get_chemical_symbols())

calc = Reform_Calculator(
            cutoff = 5.0,
            contract = False,
            znucl = znucl,
            lmax = 0,
            nx = 200,
            ntyp = ntyp
            )

atoms.calc = calc
print ("fp_energy:\n", atoms.get_potential_energy())
print ("fp_forces:\n", atoms.get_forces())
print ("fp_stress:\n", atoms.get_stress())

