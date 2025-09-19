import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import PropertyNotImplementedError
from mpi4py import MPI
from ase.parallel import world

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class LinearCombinationCalculator(Calculator):
    """LinearCombinationCalculator for weighted summation of multiple calculators.
    """

    def __init__(self, calcs, weights, atoms = None):
        """Implementation of sum of calculators.

        calcs: list
            List of an arbitrary number of :mod:`ase.calculators` objects.
        weights: list of float
            Weights for each calculator in the list.
        atoms: Atoms object
            Optional :class:`~ase.Atoms` object to which the calculator will be attached.
        """

        super().__init__(atoms = atoms)

        if len(calcs) == 0:
            raise ValueError('The value of the calcs must be a list of Calculators')

        for calc in calcs:
            if not isinstance(calc, Calculator):
                raise ValueError('All the calculators should be inherited form the \
                                  ase\'s Calculator class')

        common_properties = set.intersection(*(set(calc.implemented_properties) for calc in calcs))
        self.implemented_properties = list(common_properties)

        if not self.implemented_properties:
            raise PropertyNotImplementedError('There are no common property \
                                               implemented for the potentials!')
        if len(weights) != len(calcs):
            raise ValueError('The length of the weights must be the same as \
                              the number of calculators!')

        self.calcs = calcs
        self.weights = weights

        '''
        weights = np.ones(len(calcs)).tolist()
        pi_fmax = 1.0
        for i in range(len(calcs)):
            forces_i = calcs[i].get_property('forces', atoms)
            pi_fmax = pi_fmax*np.amax(np.absolute(forces_i))
        for j in range(len(weights)):
            forces_j = calcs[j].get_property('forces', atoms)
            weights[j] = pi_fmax / np.amax(np.absolute(forces_j))

        if len(weights) != len(calcs):
            raise ValueError('The length of the weights must be the same as \
                              the number of calculators!')

        # self.calcs = calcs
        self.weights = weights
        '''

    def calculate(self,
                  atoms = None,
                  properties = ['energy', 'forces', 'stress'],
                  system_changes = all_changes
                 ):
        """ Calculates all the specific property for each calculator
            and returns with the summed value.
        """

        super().calculate(atoms, properties, system_changes)

        if not set(properties).issubset(self.implemented_properties):
            raise PropertyNotImplementedError('Some of the requested property is not \
                                               in the ''list of supported properties \
                                               ({})'.format(self.implemented_properties))

        for w, calc in zip(self.weights, self.calcs):
            if w > 0.0:
                if calc.calculation_required(atoms, properties):
                    calc.calculate(atoms, properties, system_changes)

                for k in properties:
                    if k not in self.results:
                        self.results[k] = w * calc.results[k]
                    else:
                        self.results[k] += w * calc.results[k]

    def reset(self):
        """Clear all previous results recursively from all fo the calculators."""
        super().reset()

        for calc in self.calcs:
            calc.reset()

    def __str__(self):
        calculators = ', '.join(calc.__class__.__name__ for calc in self.calcs)
        return '{}({})'.format(self.__class__.__name__, calculators)


class MixedCalculator(LinearCombinationCalculator):
    """
    Mixing of two calculators with different weights

    H = weight1 * H1 + weight2 * H2

    Has functionality to get the energy contributions from each calculator

    Parameters
    ----------
    calc1 : ASE-calculator
    calc2 : ASE-calculator
    weight1 : float
        weight for calculator 1
    weight2 : float
        weight for calculator 2
    """

    def __init__(self, calc1, calc2, iter_max = None, comm=None):
        self.nonLinear_const = 3
        self.iter = 0
        self._last_positions = None
        if iter_max is not None:
            if iter_max <= 0:
                raise ValueError('iter_max must be a positive integer')
            self.iter_max = int(iter_max)
        else:
            self.iter_max = 300
            
        # Store MPI communicator if provided
        self.comm = comm
        if comm is not None:
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
            self.parallel = True
        else:
            self.rank = 0
            self.size = 1
            self.parallel = False
            
        self.weights = [0.0, 1.0]
        self._last_weights = tuple(self.weights)
        self._weights_changed_last_call = True
        weight1 = self.weights[0]
        weight2 = self.weights[1]
        super().__init__([calc1, calc2], [weight1, weight2])

    def set_weights(self, calc1, calc2, atoms):
        """Set weights for the two calculators based on iteration number."""
        effective_iter = min(self.iter, self.iter_max)

        if effective_iter == 0:
            # fmax_1 = np.amax(np.absolute(calc1.get_forces(atoms)))
            # fmax_2 = np.amax(np.absolute(calc2.get_forces(atoms)))
            # self.f_ratio = fmax_1 / fmax_2
            weight0 = 0.0
        elif effective_iter >= self.iter_max:
            weight0 = 1.0
        else:
            # Smoothly ramp the contribution of calc1 from 0 to 1
            progress = effective_iter / max(self.iter_max - 1, 1)
            weight0 = 0.5 * (np.sin(-np.pi * 0.5 + np.pi * 9 * progress ** 2) + 1.0)

        # Ensure numerical safety and that the two weights sum to 1.0
        weight0 = float(np.clip(weight0, 0.0, 1.0))
        weight1 = 1.0 - weight0

        prev_weights = self._last_weights
        new_weights = (weight0, weight1)

        self.weights[0] = weight0
        self.weights[1] = weight1
        self._last_weights = new_weights

        changed = not np.allclose(prev_weights, new_weights, rtol=0.0, atol=1e-12)
        self._weights_changed_last_call = changed

        # Ensure weights are synchronized if using MPI
        if self.parallel:
            self.weights = self.comm.bcast(self.weights if self.rank == 0 else None, root=0)
            changed = self.comm.bcast(changed if self.rank == 0 else None, root=0)
            self._weights_changed_last_call = bool(changed)

    def _update_iteration_counter(self, atoms):
        """Increment iteration counter only when atomic positions change."""
        positions = atoms.get_positions()

        if self._last_positions is None:
            changed = False
        else:
            changed = not np.allclose(positions, self._last_positions,
                                       rtol=0.0, atol=1e-12)

        if self.parallel:
            changed = self.comm.bcast(changed if self.rank == 0 else None, root=0)

        if self._last_positions is None or changed:
            self._last_positions = np.array(positions, copy=True)

        if changed:
            self.iter += 1

        # Synchronize iteration counter across ranks
        if self.parallel:
            self.iter = self.comm.bcast(self.iter if self.rank == 0 else None, root=0)
                
    def calculate(self,
                  atoms = None,
                  properties = ['energy', 'forces', 'stress'],
                  system_changes = all_changes
                 ):
        """ Calculates all the specific property for each calculator and returns
            with the summed value.
        """
        atoms = atoms or self.atoms
        if atoms is None:
            raise ValueError("No atoms object provided to MixedCalculator")
            
        # Make sure all ranks have the same atoms
        if self.parallel:
            # Wait for all processes to reach this point
            self.comm.Barrier()
            
        # Update iteration counter before adjusting weights
        self._update_iteration_counter(atoms)

        # Set weights based on current iteration
        self.set_weights(self.calcs[0], self.calcs[1], atoms)
        
        # Initialize dictionaries on all ranks
        self.results = {}
        
        # Calculate properties on all ranks for both calculators
        for i, (w, calc) in enumerate(zip(self.weights, self.calcs)):
            if w > 0.0:
                # if self.rank == 0:
                #     print(f"Running calculator {i+1} with weight {w}", flush=True)
                
                # Check if calculation is required
                if calc.calculation_required(atoms, properties):
                    # Calculate properties
                    try:
                        calc.calculate(atoms, properties, system_changes)
                    except Exception as e:
                        if self.rank == 0:
                            print(f"Error in calculator {i+1}: {e}", flush=True)
                        if self.parallel:
                            self.comm.Barrier()  # Make sure all processes sync
                        raise
                
                # Synchronize after each calculator
                if self.parallel:
                    self.comm.Barrier()
        
        # Collect and combine results
        for prop in properties:
            if prop in ['energy', 'forces', 'stress']:
                # Initialize results with zeros
                if prop == 'energy':
                    result = 0.0
                elif prop == 'forces':
                    result = np.zeros((len(atoms), 3), dtype=float)
                elif prop == 'stress':
                    result = np.zeros(6, dtype=float)
                
                # Add weighted contributions from each calculator
                for i, (w, calc) in enumerate(zip(self.weights, self.calcs)):
                    if w > 0.0:
                        if prop in calc.results:
                            result += w * calc.results[prop]
                
                # Store the result
                self.results[prop] = result
                
                # Store contributions for special properties
                if prop == 'energy':
                    energy_contributions = tuple(
                        calc.results.get('energy', 0.0) if w > 0.0 else 0.0 
                        for w, calc in zip(self.weights, self.calcs)
                    )
                    self.results['energy_contributions'] = energy_contributions
                elif prop == 'forces':
                    force_contributions = tuple(
                        calc.results.get('forces', np.zeros((len(atoms), 3), dtype=float)) 
                        if w > 0.0 else np.zeros((len(atoms), 3), dtype=float)
                        for w, calc in zip(self.weights, self.calcs)
                    )
                    self.results['force_contributions'] = force_contributions
                elif prop == 'stress':
                    stress_contributions = tuple(
                        calc.results.get('stress', np.zeros(6, dtype=float)) 
                        if w > 0.0 else np.zeros(6, dtype=float)
                        for w, calc in zip(self.weights, self.calcs)
                    )
                    self.results['stress_contributions'] = stress_contributions
        
        # Ensure all processes have completed the calculation
        if self.parallel:
            self.comm.Barrier()

    def reset(self):
        """Reset calculators and internal mixing state."""
        super().reset()
        self.iter = 0
        self._last_positions = None
        self.weights = [0.0, 1.0]
        self._last_weights = tuple(self.weights)
        self._weights_changed_last_call = True

    def get_energy_contributions(self, atoms = None):
        """ Return the potential energy from calc1 and calc2 respectively """
        self.calculate(properties = ['energy'], atoms = atoms)
        return self.results['energy_contributions']

    def get_force_contributions(self, atoms = None):
        """ Return the forces from calc1 and calc2 respectively """
        self.calculate(properties = ['forces'], atoms = atoms)
        return self.results['force_contributions']

    def get_stress_contributions(self, atoms = None):
        """ Return the Cauchy stress tensor from calc1 and calc2 respectively """
        self.calculate(properties = ['stress'], atoms = atoms)
        return self.results['stress_contributions']



class SumCalculator(LinearCombinationCalculator):
    """SumCalculator for combining multiple calculators.

    This calculator can be used when there are different calculators for the different chemical
    environment or for example during delta leaning. It works with a list of arbitrary calculators
    and evaluates them in sequence when it is required. The supported properties are the intersection
    of the implemented properties in each calculator.
    """

    def __init__(self, calcs, atoms = None):
        """Implementation of sum of calculators.

        calcs: list
            List of an arbitrary number of :mod:`ase.calculators` objects.
        atoms: Atoms object
            Optional :class:`~ase.Atoms` object to which the calculator will be attached.
        """

        weights = [1.] * len(calcs)
        super().__init__(calcs, weights, atoms)


class AverageCalculator(LinearCombinationCalculator):
    """AverageCalculator for equal summation of multiple calculators (for thermodynamic purposes)..
    """

    def __init__(self, calcs, atoms = None):
        """Implementation of average of calculators.

        calcs: list
            List of an arbitrary number of :mod:`ase.calculators` objects.
        atoms: Atoms object
            Optional :class:`~ase.Atoms` object to which the calculator will be attached.
        """

        n = len(calcs)

        if n == 0:
            raise ValueError('The value of the calcs must be a list of Calculators')

        weights = [1 / n] * n
        super().__init__(calcs, weights, atoms)
