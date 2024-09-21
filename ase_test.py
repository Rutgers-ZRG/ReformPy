import os
import sys
import numpy as np
# import writekp
import ase.io
from ase.optimize import BFGS, LBFGS, BFGSLineSearch, QuasiNewton, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import StrainFilter, UnitCellFilter
from ase.io.trajectory import Trajectory


atoms = ase.io.read('.'+'/'+'POSCAR')
# atoms = ase.io.read('.'+'/'+'fp_opt.vasp')
ase.io.vasp.write_vasp('input.vasp', atoms, direct = True)
# ase.io.vasp.write_vasp('fp_input.vasp', atoms, direct = True)
trajfile = 'opt.traj'
print("Number of atoms:", len(atoms))

'''
from ase.calculators.vasp import Vasp
import kp_finder

kpoints = kp_finder.get_kpoints(kgrid=0.07)
calc1 = Vasp( command = 'mpirun -n 16 /home/lz432/apps/vasp.6.3.0_intel/bin/vasp_std',
              xc = 'PBE',
              setups = 'recommended',
              txt = 'vasp.out',
              prec = 'Normal',
              # ediff = 1.0e-8,
              # ediffg = -1.0e-5,
              encut = 400.0,
              ibrion = -1, # No VASP relaxation
              nsw = 0, # Max. no of relaxation steps
              isif = 3,
              ismear = 0,
              sigma = 0.05,
              potim = 0.2,
              # lwave = False,
              # lcharge = False,
              # lplane = False,
              isym = 0,
              symprec = 1.0e-7,
              npar = 4,
              kpts = kpoints,
              gamma = True
              )

atoms.calc = calc1
print ("VASP_energy:\n", atoms.get_potential_energy())
print ("VASP_forces:\n", atoms.get_forces())
print ("VASP_stress:\n", atoms.get_stress())

###################################################################################################

from ase.calculators.espresso import Espresso
import kp_finder

kpoints = kp_finder.get_kpoints(kgrid=0.07)

pseudopotentials = {'Si': 'Si.pbe-n-rrkjus_psl.1.0.0.UPF'}
path_to_pseudopotentials="$HOME/apps/SSSP_1.3.0_PBE_efficiency"
command = 'mpirun -np 16 $HOME/apps/qe-7.2/bin pw.x -in PREFIX.pwi > PREFIX.pwo'

input_data = {
    'control': {
        'calculation': 'scf',
        'prefix': 'silicon',
        'outdir': './',
        'etot_conv_thr': 1.0e-5,
        'forc_conv_thr': 1.0e-3,
        'tstress': True,
        'tprnfor': True },
    'system': {
        'ecutwfc': 50,
        'ecutrho': 400,
        'occupations': 'smearing',
        'smearing': 'gauss',
        'degauss': 0.004,
        'nosym': True },
    'electrons': {
        'electron_maxstep': 800,
        'diagonalization': 'rmm-davidson',
        'mixing_mode': 'local-TF',
        'mixing_beta': 0.5,
        'conv_thr': 1.0e-6 }
}

calc1 = Espresso(input_data = input_data,
                 pseudopotentials = pseudopotentials,
                 kpts = tuple(kpoints))

atoms.calc = calc1
print ("QE_energy:\n", atoms.get_potential_energy())
print ("QE_forces:\n", atoms.get_forces())
print ("QE_stress:\n", atoms.get_stress())

###################################################################################################

from ase.calculators.lj import LennardJones
calc1 = LennardJones()
calc1.parameters.epsilon = 1.0
calc1.parameters.sigma = 1.0
calc1.parameters.rc = 2.5
calc1.parameters.smooth = True

atoms.calc = calc1
print ("LJ_energy:\n", atoms.get_potential_energy())
print ("LJ_forces:\n", atoms.get_forces())
print ("LJ_stress:\n", atoms.get_stress())

##################################################################################################
Sigma gives a measurement of how close two nonbonding particles can get and is thus referred to as the van der Waals radius. It is equal to one-half of the internuclear distance between nonbonding particles.
Ideally, r_min == 2**(1/6) * sigma == 2.0 * r_cov, which means van der Waals radius is approximately two times larger than covalent radius.

Reference:
https://en.wikipedia.org/wiki/Lennard-Jones_potential
https://en.wikipedia.org/wiki/Van_der_Waals_radius
https://en.wikipedia.org/wiki/Covalent_radius
##################################################################################################



##################################################################################################
# OpenKim support for emperical potential access
from ase.calculators.kim.kim import KIM

model = "SW_StillingerWeber_1985_Si__MO_405512056662_006"
calc1 = KIM(model)
atoms.calc = calc1

print ("SW_energy:\n", atoms.get_potential_energy())
print ("SW_forces:\n", atoms.get_forces())
print ("SW_stress:\n", atoms.get_stress())
##################################################################################################




from SF_LJ_api4ase import ShiftedForceLennardJones

calc1 = ShiftedForceLennardJones()
calc1.parameters.epsilon = np.array([1.00, 1.50, 0.50])
calc1.parameters.sigma = np.array([1.00, 0.80, 0.88])
calc1.parameters.rc = 2.5 * np.array([1.00, 0.80, 0.88])

atoms.calc = calc1
print ("SFLJ_energy:\n", atoms.get_potential_energy())
print ("SFLJ_forces:\n", atoms.get_forces())
print ("SFLJ_stress:\n", atoms.get_stress())




########################################## For MgAl_2O_4 ##########################################

from Buck_api4ase import Buckingham

calc1 = Buckingham()
calc1.parameters.ZZ = { 'Mg': 2, 'Al': 3, 'O': -2 }
calc1.parameters.A = np.array([1279.69, 1361.29, 9547.96])
calc1.parameters.rho = np.array([0.2997, 0.3013, 0.2240])
calc1.parameters.C = np.array([0.00, 0.00, 32.0])
calc1.parameters.rc = 10.0
calc1.parameters.smooth = False

atoms.calc = calc1
print ("Buckingham_energy:\n", atoms.get_potential_energy())
print ("Buckingham_forces:\n", atoms.get_forces())
print ("Buckingham_stress:\n", atoms.get_stress())

###################################################################################################

from gulp_api4ase import GULP, Conditions

c = Conditions(atoms)
c.min_distance_rule('O',
                    'H',
                    ifcloselabel1='O_OH',
                    ifcloselabel2='H_OH',
                    elselabel1='O_O2-')

calc1 = GULP(keywords = 'conp gradient stress_out',
             library = 'MgAlSiO.lib',
             shel=['O'])

calc1 = GULP(keywords = 'conp gradient stress_out',
             library = 'MgAlO.lib')

atoms.calc = calc1
print ("GULP_energy:\n", atoms.get_potential_energy())
print ("GULP_forces:\n", atoms.get_forces())
print ("GULP_stress:\n", atoms.get_stress())

###################################################################################################

from ase.calculators.lammpslib import LAMMPSlib

cmds = ["mass 1 24.305",
        "mass 2 26.982",
        "mass 3 15.999",
        "pair_style buck 10.0",
        "pair_coeff * * 0.0 1.0 0.0",
        "pair_coeff 1 3 1428.5 0.2945 0.0",
        "pair_coeff 2 3 1114.9 0.3118 0.0",
        "pair_coeff 3 3 2023.8 0.2674 0.0",
        "compute p_eng all pe pair bond",
        "compute k_eng all ke",
        "compute tmp all temp",
        "compute prs all pressure tmp",
        "compute strs all stress/atom NULL pair bond",
        "fix 1 all box/relax iso 1.0e+5 vmax 0.001",
        "thermo 10",
        "thermo_style custom step lx ly lz enthalpy etotal",
        "dump coord all custom 10 lammps.dump id element x y z",
        "dump_modify coord element Mg Al O",
        "min_style cg",
        "minimize 1e-25 1e-25 5000 10000"]
calc1 = LAMMPSlib(lmpcmds = cmds, log_file = 'lammps.log')

atoms.calc = calc1
print ("lmp_energy:\n", atoms.get_potential_energy())
print ("lmp_forces:\n", atoms.get_forces())
print ("lmp_stress:\n", atoms.get_stress())

###################################################################################################

from quippy.potential import Potential

calc1 = Potential(param_filename='./gp_iter6_sparse9k.xml')

atoms.calc = calc1
print ("GAP_energy:\n", atoms.get_potential_energy())
print ("GAP_forces:\n", atoms.get_forces())
print ("GAP_stress:\n", atoms.get_stress())
# fmax_1 = np.amax(np.absolute(atoms.get_forces()))

###################################################################################################

from ase.calculators.dftb import Dftb
import kp_finder

kpoints = kp_finder.get_kpoints(kgrid=0.07)
calc1 = Dftb(atoms = atoms,
             kpts = tuple(kpoints),
             label = 'dftb')
atoms.calc = calc1
print ("DFTB_energy:\n", atoms.get_potential_energy())
print ("DFTB_forces:\n", atoms.get_forces())
print ("DFTB_stress:\n", atoms.get_stress())
# fmax_1 = np.amax(np.absolute(atoms.get_forces()))

###################################################################################################

from m3gnet.models._base import Potential
from m3gnet.models._m3gnet import M3GNet
from M3GNet_api4ase import M3GNet_Calculator

calc1 = M3GNet_Calculator(Potential(M3GNet.load()),
                          compute_stress = True,
                          stress_weight = 1.0)
atoms.calc = calc1
print ("M3GNet_energy:\n", atoms.get_potential_energy())
print ("M3GNet_forces:\n", atoms.get_forces())
print ("M3GNet_stress:\n", atoms.get_stress())


'''
###################################################################################################


from fplib3_api4ase import fp_GD_Calculator
from functools import reduce

chem_nums = list(atoms.numbers)
znucl_list = reduce(lambda re, x: re+[x] if x not in re else re, chem_nums, [])
ntyp = len(znucl_list)
znucl = znucl_list

calc1 = fp_GD_Calculator(
            cutoff = 4.0,
            contract = False,
            znucl = znucl,
            lmax = 0,
            nx = 400,
            ntyp = ntyp
            )
atoms.calc = calc1

# calc1.test_energy_consistency(atoms = atoms)
# calc1.test_force_consistency(atoms = atoms)

print ("fp_energy:\n", atoms.get_potential_energy())
print ("fp_forces:\n", atoms.get_forces())
print ("fp_stress:\n", atoms.get_stress())


############################## Relaxation type ##############################
#     https ://wiki.fysik.dtu.dk/ase/ase/optimize.html#module-optimize      #
#     https ://wiki.fysik.dtu.dk/ase/ase/constraints.html                   #
#############################################################################

# af = atoms
# af = StrainFilter(atoms)
# mask = np.ones((3,3), dtype = int) - np.eye(3, dtype = int)
# mask = np.eye(3, dtype = int)
# af = UnitCellFilter(atoms, mask = mask, constant_volume = True, scalar_pressure = 0.0)
# af = UnitCellFilter(atoms, scalar_pressure = 0.062415)
af = UnitCellFilter(atoms, scalar_pressure = 0.0)

traj = Trajectory(trajfile, 'w', atoms=atoms, properties=['energy', 'forces', 'stress'])

############################## Relaxation method ##############################\

# opt = BFGS(af, maxstep = 1.e-1)
opt = FIRE(af, maxstep = 1.e-1)
# opt = LBFGS(af, maxstep = 1.e-1, memory = 10, use_line_search = True)
# opt = LBFGS(af, maxstep = 1.e-1, memory = 10, use_line_search = False)
# opt = SciPyFminCG(af, maxstep = 1.e-1)
# opt = SciPyFminBFGS(af, maxstep = 1.e-1)

opt.attach(traj.write, interval=1)
opt.run(fmax = 1.e-3, steps = 5000)

traj.close()
atoms_final = traj[-1]
ase.io.write('opt.vasp', atoms_final, direct = True, long_format=True, vasp5 = True)

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



