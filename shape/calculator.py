import numpy as np
import fplib3
import fplib
import ase.io
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import CalculatorSetupError, all_changes

#################################### ASE Reference ####################################
#        https://gitlab.com/ase/ase/-/blob/master/ase/calculators/calculator.py       #
#        https://gitlab.com/ase/ase/-/blob/master/ase/calculators/vasp/vasp.py        #
#        https://wiki.fysik.dtu.dk/ase/development/calculators.html                   #
#######################################################################################

class fp_GD_Calculator(Calculator):
    """ASE interface for fp_GD, with the Calculator interface.

        Implemented Properties:

            'energy': Sum of atomic fingerprint distance (L2 norm of two atomic
                                                          fingerprint vectors)

            'energies': Per-atom property of 'energy'

            'forces': Gradient of fingerprint energy, using Hellmann–Feynman theorem

            'stress': Cauchy stress tensor using finite difference method

            'stresses': Per-atom property of 'stress'

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

    """
    # name = 'fingerprint'
    # ase_objtype = 'fingerprint_calculator'  # For JSON storage

    implemented_properties = [ 'energy', 'forces', 'stress' ]
    # implemented_properties += ['energies', 'stresses'] # per-atom properties

    default_parameters = {
                          'contract': False,
                          'ntyp': 1,
                          'nx': 300,
                          'lmax': 0,
                          'cutoff': 4.0,
                          'znucl': None
                          }

    nolabel = True

    def __init__(self,
                 atoms = None,
                 **kwargs
                ):

        self._atoms = None
        self.cell_file = 'POSCAR'
        self.results = {}
        self.default_parameters = {}
        self.restart()
        if atoms is None :
            atoms = ase.io.read(self.cell_file)
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
        fingerprint Calculator.
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
        """Do a fingerprint calculation in the specified directory.
        This will read VASP input files (POSCAR) and then execute
        fp_GD.
        """
        # Check for zero-length lattice vectors and PBC
        # and that we actually have an Atoms object.
        check_atoms(atoms)

        self.clear_results()
        '''
        if atoms is not None:
            self.atoms = atoms.copy()

        if properties is None:
            properties = self.implemented_properties
        '''
        Calculator.calculate(self, atoms, properties, system_changes)
        if atoms is None:
            atoms = self.atoms
        # self.update_atoms(atoms)

        # natoms = len(self.atoms)
        # energies = np.ones(natoms, dtype = np.float64)
        # identity = np.eye(3, dtype = np.float64)

        # Per-atom energy has not been truely implemented yet, right now just returns average of cell energy with respect to total number of atoms in cell
        # self.results['energies'] = self.get_potential_energy(atoms) * energies / natoms
        self.results['energy'] = self.get_potential_energy(atoms)
        self.results['forces'] = self.get_forces(atoms)
        # Numerical forces, for verification
        # self.results['forces'] = self.calculate_numerical_forces(atoms)
        # Per-atom stress has not been truely implemented yet, right now just returns average of cell stress with respect to total number of atoms in cell
        # self.results['stresses'] = np.stack( [np.matmul(identity, \
        #                                                 self.get_stress(atoms))] * natoms ) / natoms
        self.results['stress'] = self.get_stress(atoms)
        # Numerical stress, for verification
        # self.results['stress'] = self.calculate_numerical_stress(atoms)


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

        '''
        # We now check if we have made any changes to the input parameters
        # XXX: Should we add these parameters to all_changes?
        for param_string, old_dict in self.param_state.items():
            param_dict = getattr(self, param_string)  # Get current param dict
            if not compare_dict(param_dict, old_dict):
                system_changes.append(param_string)
        '''

        return system_changes


    def _store_param_state(self):
        """Store current parameter state"""
        self.param_state = dict(
            default_parameters = self.default_parameters.copy()
            )

    # Below defines some functions for faster access to certain common keywords

    @property
    def contract(self):
        """Access the contract in default_parameters dict"""
        return self.default_parameters['contract']

    @contract.setter
    def contract(self, contract):
        """Set contract in default_parameters dict"""
        self.default_parameters['contract'] = contract

    @property
    def ntyp(self):
        """Access the ntyp in default_parameters dict"""
        return self.default_parameters['ntyp']

    @ntyp.setter
    def ntyp(self, ntyp):
        """Set ntyp in default_parameters dict"""
        self.default_parameters['ntyp'] = ntyp

    @property
    def nx(self):
        """Access the nx in default_parameters dict"""
        return self.default_parameters['nx']

    @nx.setter
    def nx(self, nx):
        """Set ntyp in default_parameters dict"""
        self.default_parameters['nx'] = nx

    @property
    def lmax(self):
        """Access the lmax in default_parameters dict"""
        return self.default_parameters['lmax']

    @lmax.setter
    def lmax(self, lmax):
        """Set ntyp in default_parameters dict"""
        self.default_parameters['lmax'] = lmax

    @property
    def cutoff(self):
        """Access the cutoff in default_parameters dict"""
        return self.default_parameters['cutoff']

    @cutoff.setter
    def cutoff(self, cutoff):
        """Set cutoff in default_parameters dict"""
        self.default_parameters['cutoff'] = cutoff

    @property
    def znucl(self):
        """Access the znucl array in default_parameters dict"""
        return self.default_parameters['znucl']

    @znucl.setter
    def znucl(self, znucl):
        """Direct access for setting the znucl"""
        if isinstance(znucl, (list, np.ndarray)):
            znucl = list(znucl)
        self.set(znucl = znucl)

    @property
    def types(self):
        """Direct access to the types array"""
        return fplib3.read_types(self.cell_file)

    @types.setter
    def types(self, types):
        """Direct access for setting the types array"""
        self.set(types = types)

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


    def get_potential_energy(self, atoms = None, **kwargs):
        contract = self.contract
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        cutoff = self.cutoff
        types = self.types
        znucl = self.znucl

        if self.check_restart(atoms) or self._energy is None:
            # write_vasp('input.vasp', atoms, direct=True)
            lat = atoms.cell[:]
            rxyz = atoms.get_positions()
            # print("fp_energy lat=\n", lat)
            # print("fp_energy rxyz=\n", rxyz)
            lat = np.array(lat, dtype = np.float64)
            rxyz = np.array(rxyz, dtype = np.float64)
            types = np.int32(types)
            znucl =  np.int32(znucl)
            ntyp =  np.int32(ntyp)
            nx = np.int32(nx)
            lmax = np.int32(lmax)
            cutoff = np.float64(cutoff)
            # fp, _ = fplib3.get_fp(lat, rxyz, types, znucl,
            #                       contract = contract,
            #                       ldfp = False,
            #                       ntyp = ntyp,
            #                       nx = nx,
            #                       lmax = lmax,
            #                       cutoff = cutoff)
            cell = (lat, rxyz, types, znucl)
            fp = fplib.get_lfp(cell, cutoff = cutoff, log = False, natx = nx)
            fp = np.float64(fp)
            fpe = fplib3.get_fpe(fp, ntyp = ntyp, types = types)
            self._energy = fpe
        return self._energy


    def get_forces(self, atoms = None, **kwargs):
        contract = self.contract
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        cutoff = self.cutoff
        types = self.types
        znucl = self.znucl

        if self.check_restart(atoms) or self._forces is None:
            # write_vasp('input.vasp', atoms, direct=True)
            lat = atoms.cell[:]
            rxyz = atoms.get_positions()
            # print("fp_forces lat=\n", lat)
            # print("fp_forces rxyz=\n", rxyz)
            lat = np.array(lat, dtype = np.float64)
            rxyz = np.array(rxyz, dtype = np.float64)
            types = np.int32(types)
            znucl =  np.int32(znucl)
            ntyp =  np.int32(ntyp)
            nx = np.int32(nx)
            lmax = np.int32(lmax)
            cutoff = np.float64(cutoff)
            # fp, dfp = fplib3.get_fp(lat, rxyz, types, znucl,
            #                         contract = contract,
            #                         ldfp = True,
            #                         ntyp = ntyp,
            #                         nx = nx,
            #                         lmax = lmax,
            #                         cutoff = cutoff)
            cell = (lat, rxyz, types, znucl)
            fp, dfp  = fplib.get_dfp(cell, cutoff = cutoff, log = False, natx = nx)
            fp = np.float64(fp)
            dfp = np.array(dfp, dtype = np.float64)
            fpe, fpf = fplib3.get_ef(fp, dfp, ntyp = ntyp, types = types)
            self._forces = fpf
        return self._forces


    def get_stress(self, atoms = None, **kwargs):
        contract = self.contract
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        cutoff = self.cutoff
        types = self.types
        znucl = self.znucl

        if self.check_restart(atoms) or self._stress is None:
            # write_vasp('input.vasp', atoms, direct=True)
            lat = atoms.cell[:]
            rxyz = atoms.get_positions()
            pos = atoms.get_scaled_positions()
            # print("fp_stress lat=\n", lat)
            # print("fp_stress rxyz=\n", rxyz)
            # print("fp_stress pos=\n", pos)
            lat = np.array(lat, dtype = np.float64)
            rxyz = np.array(rxyz, dtype = np.float64)
            types = np.int32(types)
            znucl =  np.int32(znucl)
            ntyp =  np.int32(ntyp)
            nx = np.int32(nx)
            lmax = np.int32(lmax)
            cutoff = np.float64(cutoff)
            # forces = self._forces
            forces=atoms.get_forces()
            stress = fplib3.get_stress(lat, rxyz, forces)
            # stress = fplib3.get_stress(lat, rxyz, types, znucl,
            #                            contract = contract,
            #                            ntyp = ntyp,
            #                            nx = nx,
            #                            lmax = lmax,
            #                            cutoff = cutoff)
            self._stress = stress
        return self._stress


    def test_energy_consistency(self, atoms = None, **kwargs):
        contract = self.contract
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        cutoff = self.cutoff
        types = self.types
        znucl = self.znucl

        # write_vasp('input.vasp', atoms, direct=True)
        lat = atoms.cell[:]
        rxyz = atoms.get_positions()
        lat = np.array(lat, dtype = np.float64)
        rxyz = np.array(rxyz, dtype = np.float64)
        types = np.int32(types)
        znucl =  np.int32(znucl)
        ntyp =  np.int32(ntyp)
        nx = np.int32(nx)
        lmax = np.int32(lmax)
        cutoff = np.float64(cutoff)
        # del_fpe, e_diff = fplib3.get_simpson_energy(lat, rxyz, types, znucl,
        #                                             contract = contract,
        #                                             ntyp = ntyp,
        #                                             nx = nx,
        #                                             lmax = lmax,
        #                                             cutoff = cutoff)
        
        
        rxyz_delta = np.zeros_like(rxyz)
        rxyz_disp = np.zeros_like(rxyz)
        rxyz_left = np.zeros_like(rxyz)
        rxyz_mid = np.zeros_like(rxyz)
        rxyz_right = np.zeros_like(rxyz)
        nat = len(rxyz)
        del_fpe = 0.0
        iter_max = 100
        step_size = 1.e-5
        rxyz_delta = step_size*( np.random.rand(nat, 3).astype(np.float64) - \
                                0.5*np.ones((nat, 3), dtype = np.float64) )
        
        for i_iter in range(iter_max):
            rxyz_disp += 2.0*rxyz_delta
            rxyz_left = rxyz.copy() + 2.0*i_iter*rxyz_delta
            rxyz_mid = rxyz.copy() + 2.0*(i_iter+1)*rxyz_delta
            rxyz_right = rxyz.copy() + 2.0*(i_iter+2)*rxyz_delta
            
            fp_left, dfp_left = fplib.get_dfp((lat, rxyz_left, types, znucl),
                                              cutoff = cutoff, log = False, natx = nx)
            fp_mid, dfp_mid = fplib.get_dfp((lat, rxyz_mid, types, znucl),
                                              cutoff = cutoff, log = False, natx = nx)
            fp_right, dfp_right = fplib.get_dfp((lat, rxyz_right, types, znucl),
                                              cutoff = cutoff, log = False, natx = nx)
            fpe_left, fpf_left = fplib3.get_ef(fp_left, dfp_left, ntyp, types)
            fpe_mid, fpf_mid = fplib3.get_ef(fp_mid, dfp_mid, ntyp, types)
            fpe_right, fpf_right = fplib3.get_ef(fp_right, dfp_right, ntyp, types)

            for i_atom in range(nat):
                del_fpe += ( -np.dot(rxyz_delta[i_atom], fpf_left[i_atom]) - \
                            4.0*np.dot(rxyz_delta[i_atom], fpf_mid[i_atom]) - \
                            np.dot(rxyz_delta[i_atom], fpf_right[i_atom]) )/3.0
        
        rxyz_final = rxyz + rxyz_disp
        fp_init = fplib.get_lfp((lat, rxyz, types, znucl),
                                cutoff = cutoff, log = False, natx = nx)
        fp_final = fplib.get_lfp((lat, rxyz_final, types, znucl),
                                cutoff = cutoff, log = False, natx = nx)
        e_init = fplib3.get_fpe(fp_init, ntyp, types)
        e_final = fplib3.get_fpe(fp_final, ntyp, types)
        e_diff = e_final - e_init
        
        print ( "Numerical integral = {0:.6e}".format(del_fpe) )
        print ( "Fingerprint energy difference = {0:.6e}".format(e_diff) )
        if np.allclose(del_fpe, e_diff, rtol=1e-6, atol=1e-6, equal_nan=False):
            print("Energy consistency test passed!")
        else:
            print("Energy consistency test failed!")


    def test_force_consistency(self, atoms = None, **kwargs):

        from ase.calculators.test import numeric_force

        indices = range(len(atoms))
        f = atoms.get_forces()[indices]
        print('{0:>16} {1:>20}'.format('eps', 'max(abs(df))'))
        for eps in np.logspace(-1, -8, 8):
            fn = np.zeros((len(indices), 3))
            for idx, i in enumerate(indices):
                for j in range(3):
                    fn[idx, j] = numeric_force(atoms, i, j, eps)
            print('{0:16.12f} {1:20.12f}'.format(eps, abs(fn - f).max()))


        print ( "Numerical forces = \n{0:s}".\
               format(np.array_str(fn, precision=6, suppress_small=False)) )
        print ( "Fingerprint forces = \n{0:s}".\
               format(np.array_str(f, precision=6, suppress_small=False)) )
        if np.allclose(f, fn, rtol=1e-6, atol=1e-6, equal_nan=False):
            print("Force consistency test passed!")
        else:
            print("Force consistency test failed!")



########################################################################################
####################### Helper functions for the VASP calculator #######################
########################################################################################

def check_atoms(atoms: Atoms) -> None:
    """Perform checks on the atoms object, to verify that
    it can be run by VASP.
    A CalculatorSetupError error is raised if the atoms are not supported.
    """

    # Loop through all check functions
    for check in (check_atoms_type, check_cell, check_pbc):
        check(atoms)


def check_cell(atoms: Atoms) -> None:
    """Check if there is a zero unit cell.
    Raises CalculatorSetupError if the cell is wrong.
    """
    if atoms.cell.rank < 3:
        raise CalculatorSetupError(
            "The lattice vectors are zero! "
            "This is the default value - please specify a "
            "unit cell.")


def check_pbc(atoms: Atoms) -> None:
    """Check if any boundaries are not PBC, as VASP
    cannot handle non-PBC.
    Raises CalculatorSetupError.
    """
    if not atoms.pbc.all():
        raise CalculatorSetupError(
            "Vasp cannot handle non-periodic boundaries. "
            "Please enable all PBC, e.g. atoms.pbc=True")


def check_atoms_type(atoms: Atoms) -> None:
    """Check that the passed atoms object is in fact an Atoms object.
    Raises CalculatorSetupError.
    """
    if not isinstance(atoms, Atoms):
        raise CalculatorSetupError(
            ('Expected an Atoms object, '
             'instead got object of type {}'.format(type(atoms))))
