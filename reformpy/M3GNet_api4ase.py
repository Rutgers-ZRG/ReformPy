import sys
import os
import numpy as np
import ase.io
from ase import Atoms, units
from ase.stress import full_3x3_to_voigt_6_stress
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import CalculatorSetupError, all_changes
from typing import Optional
from m3gnet.models._base import Potential

#################################### ASE Reference ####################################
#        https://gitlab.com/ase/ase/-/blob/master/ase/calculators/calculator.py       #
#        https://wiki.fysik.dtu.dk/ase/development/proposals/calculators.html         #
#        https://wiki.fysik.dtu.dk/ase/development/calculators.html                   #
#######################################################################################

class M3GNet_Calculator(Calculator):
    """
    M3GNet calculator based on ase Calculator
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]
    # implemented_properties += ['energies', 'stresses'] # per-atom properties
    
    default_parameters = {}
    
    def __init__(self,
                 potential: Potential,
                 atoms = None,
                 compute_stress: bool = True,
                 stress_weight: float = 1.0,
                 **kwargs
                ):
        """
        Args:
            potential (Potential): m3gnet.models.Potential
            compute_stress (bool): whether to calculate the stress
            stress_weight (float): the stress weight.
            **kwargs:
        """
        # super().__init__(**kwargs)
        if isinstance(potential, str):
            potential = Potential(M3GNet.load(potential))
        if potential is None:
            potential = Potential(M3GNet.load())
        self.potential = potential
        self.compute_stress = compute_stress
        self.stress_weight = stress_weight
        
        self._atoms = None
        self.cell_file = 'POSCAR'
        self.results = {}
        self.default_parameters = {}
        self.restart()
        if atoms is None :
            atoms = ase.io.read(self.cell_file, format = 'vasp')
        self.atoms = atoms
        self.atoms_save = None

        # Initialize parameter dictionaries
        self._store_param_state()  # Initialize an empty parameter state
        
        Calculator.__init__(self,
                            atoms = atoms,
                            **kwargs
                           )
        
    
    def set(self, **kwargs):
        """Override the set function, to test for changes in the
        M3GNet Calculator.
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
                  atoms: Optional[Atoms] = None,
                  properties: Optional[list] = None,
                  system_changes: Optional[list] = None
                 ):
        """
        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        Returns:
        """
        
        # Check for zero-length lattice vectors and PBC
        # and that we actually have an Atoms object.
        check_atoms(atoms)
        self.clear_results()
        # Calculator.calculate(self, atoms, properties, system_changes)
        if atoms is None:
            atoms = self.atoms
        # self.update_atoms(atoms)
        
        
        # if self.check_restart(atoms) or \
        # self._energy is None or \
        # self._forces is None or \
        # self._stress is None:
        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        graph = self.potential.graph_converter(atoms)
        graph_list = graph.as_tf().as_list()
        results = self.potential.get_efs_tensor(graph_list, include_stresses=self.compute_stress)
        
        e_res = results[0].numpy().ravel()[0]
        free_res = results[0].numpy().ravel()[0]
        f_res = results[1].numpy()
        s_res = full_3x3_to_voigt_6_stress( results[2].numpy()[0] )
        
        self.results.update(
            energy = e_res,
            free_energy = free_res,
            forces = f_res
        )
        if self.compute_stress:
            self.results.update(stress = s_res * self.stress_weight)
    
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
