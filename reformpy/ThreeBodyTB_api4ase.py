""" 
This API defines a ASE Calculator for ThreeBodyTB

https://github.com/usnistgov/tb3py
https://pages.nist.gov/ThreeBodyTB.jl/

"""

import os
import sys
import numpy as np

import ase.io
from ase.atoms import Atoms
from ase.cell import Cell
from ase.units import Rydberg, Bohr
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import CalculatorSetupError, all_changes

# angst_to_bohr = 1  # 0.529177210903  # 1.88973
# const = 13.605662285137 # Ry to eV

#################################### ASE Reference ####################################
#        https://gitlab.com/ase/ase/-/blob/master/ase/calculators/calculator.py       #
#        https://wiki.fysik.dtu.dk/ase/development/proposals/calculators.html         #
#        https://wiki.fysik.dtu.dk/ase/development/calculators.html                   #
#######################################################################################

class ThreeBodyTB_Calculator(Calculator):
    
    name = 'ThreeBodyTB'
    ase_objtype = 'ThreeBodyTB_calculator'  # For JSON storage
    
    # Environment commands
    # sysimage = '$HOME/.julia/sysimages/sys_threebodytb.so'
    # Julia_COMMAND = 'Julia(runtime="julia", compiled_modules=False, sysimage=sysimage)'
    
    env_commands = ('Julia_COMMAND', 'ThreeBodyTB_COMMAND', 'ThreeBodyTB_SCRIPT')

    implemented_properties = [ 'energy', 'forces', 'stress' ]
    default_parameters = {}
    
    def __init__(self,
                 atoms = None,
                 directory = '.',
                 label = 'ThreeBodyTB',
                 command = None,
                 txt='ThreeBodyTB.out',
                 **kwargs
                ):
        self._atoms = None
        self.cell_file = 'POSCAR'
        self.results = {}
        self.default_parameters = {}
        self.restart()
        if atoms is None :
            atoms = ase.io.read(self.cell_file, format = 'vasp')
        self.atoms = atoms
        self.atoms_save = None
        self.tb3_crys = None

        # Initialize parameter dictionaries
        self._store_param_state()  # Initialize an empty parameter state
        
        # Set directory and label
        self.directory = directory
        if '/' in label:
            warn(('Specifying directory in "label" is deprecated, '
                  'use "directory" instead.'), np.VisibleDeprecationWarning)
            if self.directory != '.':
                raise ValueError('Directory redundantly specified though '
                                 'directory="{}" and label="{}".  '
                                 'Please omit "/" in label.'.format(
                                     self.directory, label))
            self.label = label
        else:
            self.prefix = label  # The label should only contain the prefix
        
        Calculator.__init__(self,
                            atoms = atoms,
                            **kwargs
                           )
        
        self.command = command
        self._txt = None
        self.txt = txt  # Set the output txt stream
        self.version = None
    
    def make_command(self, command=None):
        """Return command if one is passed, otherwise try to find
        Julia_COMMAND, ThreeBodyTB_COMMAND or ThreeBodyTB_SCRIPT.
        If none are set, a CalculatorSetupError is raised"""
        if command:
            cmd = command
        else:
            # Search for the environment commands
            for env in self.env_commands:
                if env in os.environ:
                    cmd = os.environ[env].replace('PREFIX', self.prefix)
                    if env == 'ThreeBodyTB_SCRIPT':
                        # Make the system python exe run $ThreeBodyTB_SCRIPT
                        exe = sys.executable
                        cmd = ' '.join([exe, cmd])
                    break
            else:
                msg = ('Please set either command in calculator'
                       ' or one of the following environment '
                       'variables (prioritized as follows): {}').format(
                           ', '.join(self.env_commands))
                try:
                    from julia.api import Julia
                    sysimage = os.path.join(
                        os.environ["HOME"], ".julia", "sysimages", "sys_threebodytb.so"
                    )
                    julia_cmd = "julia"
                    # print(os.environ["PATH"])
                    # print("sysimage:\n", sysimage, "path:\n", os.path.isfile(sysimage))
                    jlsession = Julia(
                        runtime=julia_cmd, compiled_modules=False, sysimage=sysimage
                    )
                    cmd = jlsession.eval("using Suppressor")  # suppress output
                except Exception:
                    print('Local system image of ThreeBodyTB cannot be found, ' \
                          'recompile the package from scratch, this could be slow.')
                    from julia.api import Julia
                    jl = Julia(compiled_modules=False)
                    cmd = jl.eval("using Suppressor; using ThreeBodyTB; ThreeBodyTB.compile()")
                    pass
                
        return cmd
    
    def set(self, **kwargs):
        """Override the set function, to test for changes in the
        ThreeBodyTB Calculator.
        """
        changed_parameters = {}

        if 'label' in kwargs:
            self.label = kwargs.pop('label')

        if 'directory' in kwargs:
            # str() call to deal with pathlib objects
            self.directory = str(kwargs.pop('directory'))

        if 'txt' in kwargs:
            self.txt = kwargs.pop('txt')

        if 'atoms' in kwargs:
            atoms = kwargs.pop('atoms')
            self.atoms = atoms  # Resets results

        if 'command' in kwargs:
            self.command = kwargs.pop('command')

        changed_parameters.update(Calculator.set(self, **kwargs))
        self.default_parameters.update(Calculator.set(self, **kwargs))
        
        if changed_parameters:
            self.clear_results()  # We don't want to clear atoms
        for key in kwargs:
            self.default_parameters[key] = kwargs[key]
            self.results.clear()

    def reset(self):
        self.atoms = None
        self.clear_results()

    def clear_results(self):
        self.results.clear()

    def restart(self):
        self._energy = None
        self._forces = None
        self._stress = None

    def check_restart(self, atoms = None):
        self.atoms = atoms
        if (self.atoms_save and atoms == self.atoms_save):
            return False
        else:
            self.atoms_save = atoms.copy()
            self.restart()
            return True
        
    def calculate(self,
                  atoms = None,
                  properties = [ 'energy', 'forces', 'stress' ],
                  system_changes = tuple(all_changes),
                 ):
        
        # Check for zero-length lattice vectors and PBC
        # and that we actually have an Atoms object.
        check_atoms(atoms)
        self.clear_results()
        # Calculator.calculate(self, atoms, properties, system_changes)
        if atoms is None:
            atoms = self.atoms
        # self.update_atoms(atoms)
        command = self.make_command(self.command)
        
        try:
            from julia import ThreeBodyTB as TB
        except Exception as exp:
            print("Julia importing error:", exp)
            pass
        
        if self.check_restart(atoms) or \
        self._energy is None or \
        self._forces is None or \
        self._stress is None:
            lattice_mat = atoms.cell[:]
            frac_coords = atoms.get_scaled_positions()
            elements = atoms.get_chemical_symbols()
            self.tb3_crys = TB.makecrys(lattice_mat, frac_coords, elements)
            self.results['energy'], self.results['forces'], self.results['stress'], tbc = \
            TB.scf_energy_force_stress(self.tb3_crys)
    
    def check_state(self, atoms, tol = 1e-15):
        """Check for system changes since last calculation."""
        def compare_dict(d1, d2):
            """Helper function to compare dictionaries"""
            # Use symmetric difference to find keys which aren't shared
            # for python 2.7 compatibility
            if set(d1.keys()) ^ set(d2.keys()):
                return False

            # Check for differences in values
            for key, value in d1.items():
                if np.any(value != d2[key]):
                    return False
            return True

        # First we check for default changes
        system_changes = Calculator.check_state(self, atoms, tol = tol)
        return system_changes
    
    def _store_param_state(self):
        """Store current parameter state"""
        self.param_state = dict(
            default_parameters = self.default_parameters.copy()
            )
        

    # Below defines some functions for faster access to certain common keywords
    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        if atoms is None:
            self._atoms = None
            self.clear_results()
        else:
            if self.check_state(atoms):
                self.clear_results()
            self._atoms = atoms.copy()

########################################################################################
####################### Helper functions for the VASP calculator #######################
########################################################################################


def check_atoms(atoms: ase.Atoms) -> None:
    """Perform checks on the atoms object, to verify that
    it can be run by VASP.
    A CalculatorSetupError error is raised if the atoms are not supported.
    """

    # Loop through all check functions
    for check in (check_atoms_type, check_cell, check_pbc):
        check(atoms)


def check_cell(atoms: ase.Atoms) -> None:
    """Check if there is a zero unit cell.
    Raises CalculatorSetupError if the cell is wrong.
    """
    if atoms.cell.rank < 3:
        raise calculator.CalculatorSetupError(
            "The lattice vectors are zero! "
            "This is the default value - please specify a "
            "unit cell.")


def check_pbc(atoms: ase.Atoms) -> None:
    """Check if any boundaries are not PBC, as VASP
    cannot handle non-PBC.
    Raises CalculatorSetupError.
    """
    if not atoms.pbc.all():
        raise calculator.CalculatorSetupError(
            "Vasp cannot handle non-periodic boundaries. "
            "Please enable all PBC, e.g. atoms.pbc=True")


def check_atoms_type(atoms: ase.Atoms) -> None:
    """Check that the passed atoms object is in fact an Atoms object.
    Raises CalculatorSetupError.
    """
    if not isinstance(atoms, ase.Atoms):
        raise calculator.CalculatorSetupError(
            ('Expected an Atoms object, '
             'instead got object of type {}'.format(type(atoms))))
