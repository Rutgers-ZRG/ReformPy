import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import PropertyNotImplementedError


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

    def __init__(self, calc1, calc2):
        self.nonLinear_const = 3
        self.iter = 0
        self.iter_max = 600
        self.weights = [0.0, 1.0]
        weight1 = self.weights[0]
        weight2 = self.weights[1]
        super().__init__([calc1, calc2], [weight1, weight2])

    def set_weights(self, calc1, calc2, atoms):
        
        if self.iter == 0:
            # fmax_1 = np.amax(np.absolute(calc1.get_property('forces', atoms)))
            # fmax_2 = np.amax(np.absolute(calc2.get_property('forces', atoms)))
            # self.f_ratio = fmax_1 / fmax_2
            self.weights[0] = 0.0
            self.weights[1] = 1.0
        
        if self.iter >= self.iter_max:
            self.weights[0] = 1.0
            self.weights[1] = 0.0
        else:
            if ((self.iter+1) % 3) != 0:
                x_iter = ( 3 * (self.iter // 3) ) / (self.iter_max - 3)
                # self.weights[0] = x_iter ** self.nonLinear_const
                self.weights[0] = \
                                0.5*(np.sin(-np.pi*0.5 + np.pi*9*x_iter**2) + 1.0)
        # print("weights=", self.weights)

    def calculate(self, 
                  atoms = None, 
                  properties = ['energy', 'forces', 'stress'], 
                  system_changes = all_changes
                 ):
        """ Calculates all the specific property for each calculator and returns 
            with the summed value.
        """

        
        super().calculate(atoms, properties, system_changes)
        atoms = atoms or self.atoms
        self.set_weights(self.calcs[0], self.calcs[1], atoms)
        # print("weights=", self.weights)
        self.iter = self.iter + 1
            
        if 'energy' in properties:
            if self.weights[0] > 0.0:
                energy1 = self.calcs[0].get_property('energy', atoms)
            else:
                energy1 = 0.0
            if self.weights[1] > 0.0:
                energy2 = self.calcs[1].get_property('energy', atoms)
            else:
                energy2 = 0.0
            self.results['energy_contributions'] = (energy1, energy2)
            
        if 'forces' in properties:
            if self.weights[0] > 0.0:
                force1 = self.calcs[0].get_property('forces', atoms)
            else:
                force1 = np.zeros((len(atoms), 3), dtype = float)
            if self.weights[1] > 0.0:
                force2 = self.calcs[1].get_property('forces', atoms)
            else:
                force2 = np.zeros((len(atoms), 3), dtype = float)
            self.results['force_contributions'] = (force1, force2)
            
        if 'stress' in properties:
            if self.weights[0] > 0.0:
                stress1 = self.calcs[0].get_property('stress', atoms)
            else:
                stress1 = np.zeros(6, dtype = float)
            if self.weights[1] > 0.0:
                stress2 = self.calcs[1].get_property('stress', atoms)
            else:
                stress2 = np.zeros(6, dtype = float)
            self.results['stress_contributions'] = (stress1, stress2)
        

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

