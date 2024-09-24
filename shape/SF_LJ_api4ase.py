import numpy as np

from ase.neighborlist import neighbor_list
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress


class ShiftedForceLennardJones(Calculator):
    """
    
    Shifted-Force Lennard-Jones potential calculator

    For reference see:
    
    https://docs.lammps.org/pair_dipole.html#pair-style-lj-sf-dipole-sf-command
    
    https://aip.scitation.org/doi/pdf/10.1063/1.3512900 (eq. 19)
    
    Allen & Tildesley, Computer Simulation of Liquids, second edition, 2017 (page 190, eq. 5.11)

    The fundamental definition of this potential is a pairwise energy:

    ``u_ij = 4 epsilon ( sigma^12/r_ij^12 - sigma^6/r_ij^6 )``
    
    Now we add a shifted-force term to the original potential:
    
    uSF_ij = u_ij + uxtra_ij
    
    uxtra_ij = - ( r_ij^2 - rc^2 )/(2rc) ( d u_ij / d r_ij )|r_ij=rc ( W.R.T. uc )
    
    where uc = 4 epsilon ( sigma^12/rc^12 - sigma^6/rc^6 )
    
    ``uxtra_ij = 4 epsilon (  6 sigma^12/rc^12 - 3 sigma^6/rc^6  ) ( r_ij^2/rc^2 ) \
               + 4 epsilon ( - 6 sigma^12/rc^12 + 3 sigma^6/rc^6 ) ``

    For convenience, we'll use d_ij to refer to "distance vector" and
    ``r_ij`` to refer to "scalar distance". So, with position vectors `r_i`, `r_j`:

    ``r_ij = | r_j - r_i | = | d_ij |``

    Therefore:

    ``d r_ij / d d_ij = + d_ij / r_ij``
    ``d r_ij / d d_ji  = - d_ij / r_ij``

    The derivative of u_ij is:

    ::

        d u_ij / d r_ij
        = (-24 epsilon / r_ij) ( 2 sigma^12/r_ij^12 - sigma^6/r_ij^6 )
        
        d uxtra_ij / d r_ij
        = (8 epsilon r_ij / rc^2) ( 6 sigma^12/rc^12 - 3 sigma^6/rc^6 )
        
        d uSF_ij / d r_ij
        = d u_ij / d r_ij + d uxtra_ij / d r_ij

    We can define a "pairwise force"

    ``f_ij = d u_ij / d d_ij = ( d u_ij / d r_ij ) * ( d_ij / r_ij )``

    The terms in front of d_ij are combined into a "general derivative".

    ``du_ij = (d u_ij / d r_ij) / r_ij``

    We do this for convenience: `du_ij` is purely scalar The pairwise force is:

    ``f_ij = du_ij * d_ij``

    The total force on an atom is:

    ``f_i = sum_(j != i) f_ij``

    There is some freedom of choice in assigning atomic energies, i.e.
    choosing a way to partition the total energy into atomic contributions.

    We choose a symmetric approach (`bothways=True` in the neighbor list):

    ``u_i = 1/2 sum_(j != i) u_ij``

    The total energy of a system of atoms is then:

    ``u = sum_i u_i = 1/2 sum_(i, j != i) u_ij``

    Differentiating `u` with respect to `r_i` yields the force, indepedent of the
    choice of partitioning.

    ::

        f_i = - d u / d r_i
            = - sum_(j != i) ( d u_ij / d r_i )
            = - sum_(j != i) ( d u_ij / d r_ij ) * ( d r_ij / d r_i )
            = - sum_(j != i) ( d u_ij / d r_ij ) * ( - d_ij / r_ij )
            = sum_(j != i) du_ij d_ij
            = sum_(j != i) f_ij

    This justifies calling `f_ij` pairwise forces.

    The stress can be written as ( `(x)` denoting outer product):

    ``sigma = 1/2 sum_(i, j != i) f_ij (x) d_ij = sum_i sigma_i``
    with atomic contributions

    ``sigma_i  = 1/2 sum_(j != i) f_ij (x) d_ij``

    Another consideration is the cutoff. We have to ensure that the potential
    goes to zero smoothly as an atom moves across the cutoff threshold,
    otherwise the potential is not continuous. In cases where the cutoff is
    so large that u_ij is very small at the cutoff this is automatically
    ensured, but in general, `u_ij(rc) != 0`.

    This implementation offers two ways to deal with this:

    Either, we shift the pairwise energy

    ``u'_ij = u_ij - u_ij(rc)``

    which ensures that it is precisely zero at the cutoff. However, this means
    that the energy effectively depends on the cutoff, which might lead to
    unexpected results! If this option is chosen, the forces discontinuously
    jump to zero at the cutoff.

    An alternative is to modify the pairwise potential by multiplying
    it with a cutoff function that goes from 1 to 0 between an onset radius
    ro and the cutoff rc. If the function is chosen suitably, it can also
    smoothly push the forces down to zero, ensuring continuous forces as well.
    In order for this to work well, the onset radius has to be set suitably,
    typically around 2*sigma.

    In this case, we introduce a modified pairwise potential:

    ``u'_ij = fc * u_ij``

    The pairwise forces have to be modified accordingly:

    ``f'_ij = fc * f_ij + fc' * u_ij``

    Where `fc' = d fc / d d_ij`.

    This approach is taken from Jax-MD (https://github.com/google/jax-md), which in
    turn is inspired by HOOMD Blue (https://glotzerlab.engin.umich.edu/hoomd-blue/).

    """

    implemented_properties = ['energy', 'energies', 'forces', 'free_energy']
    implemented_properties += ['stress', 'stresses']  # bulk properties
    default_parameters = {
        'epsilon': np.array([1.00, 1.50, 0.50]),
        'sigma': np.array([1.00, 0.80, 0.88]),
        'rc': None,
        'ro': None,
        'smooth': False,
    }
    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        sigma: float
          The potential minimum is at  2**(1/6) * sigma, default np.array([1.00, 0.80, 0.88])
        epsilon: float
          The potential depth, default np.array([1.00, 1.50, 0.50])
        rc: float, None
          Cut-off for the NeighborList is set to 2.5 * sigma if None.
          The energy is upshifted to be continuous at rc.
          Default None
        ro: float, None
          Onset of cutoff function in 'smooth' mode. Defaults to 2/3 * rc.
        smooth: bool, False
          Cutoff mode. False means that the pairwise energy is simply shifted
          to be 0 at r = rc, leading to the energy going to 0 continuously,
          but the forces jumping to zero discontinuously at the cutoff.
          True means that a smooth cutoff function is multiplied to the pairwise
          energy that smoothly goes to 0 between ro and rc. Both energy and
          forces are continuous in that case.
          If smooth=True, make sure to check the tail of the forces for kinks, ro
          might have to be adjusted to avoid distorting the potential too much.

        """

        Calculator.__init__(self, **kwargs)

        if self.parameters.rc is None:
            self.parameters.rc = 2.5 * self.parameters.sigma

        if self.parameters.ro is None:
            self.parameters.ro = 0.66 * self.parameters.rc
        

    def calculate(
        self,
        atoms = None,
        properties = None,
        system_changes = all_changes
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)
        
        atoms = self.atoms
        natoms = len(atoms)
        
        sigma = self.parameters.sigma
        epsilon = self.parameters.epsilon
        rc = self.parameters.rc
        ro = self.parameters.ro
        smooth = self.parameters.smooth
        
        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))
        icount_start = 0
        icount_end = 0
        n_bin = 0
        
        ind1, ind2, disp, cell_shift = \
        neighbor_list('ijdS', atoms, {('H', 'H'): rc[0], \
                                      ('H', 'He'): rc[1], \
                                      ('He', 'He'): rc[2]})
        
        n_bin_list = np.bincount(ind1)
        
        epsilon_AA, epsilon_AB, epsilon_BB = epsilon
        sigma_AA, sigma_AB, sigma_BB = sigma
        rc_AA, rc_AB, rc_BB = rc
        ro_AA, ro_AB, ro_BB = ro
        
        for i_atom in range(natoms):
            energy = 0.0
            force = np.zeros(3)
            stress = np.zeros((3,3))
            
            AA_neighbors = []
            BB_neighbors = []
            AB_neighbors = []
            AA_offsets = []
            BB_offsets = []
            AB_offsets = []
            
            icount_start += n_bin
            icount_end = icount_start + n_bin_list[i_atom]
            n_bin = n_bin_list[i_atom]
            i_neighbors = ind2[icount_start:icount_end]
            i_offsets = cell_shift[icount_start:icount_end]
            i_offsets = i_offsets.tolist()
            # print("i_neighbors", i_neighbors)
            # print("i_offsets", i_offsets)
            for ii in range(n_bin_list[i_atom]):
                
                if atoms[i_atom].symbol == 'H' and atoms[i_neighbors[ii]].symbol == 'H':
                    AA_neighbors.append(i_neighbors[ii])
                    AA_offsets.append(i_offsets[ii])
                    
                elif atoms[i_atom].symbol == 'He' and atoms[i_neighbors[ii]].symbol == 'He':
                    BB_neighbors.append(i_neighbors[ii])
                    BB_offsets.append(i_offsets[ii])
                    
                else:
                    AB_neighbors.append(i_neighbors[ii])
                    AB_offsets.append(i_offsets[ii])
                    
            AA_offsets = np.array(AA_offsets)
            BB_offsets = np.array(BB_offsets)
            AB_offsets = np.array(AB_offsets)
            
            
            
            if len(AA_neighbors) > 0:
                e_AA, f_AA, s_AA = self.get_pairwise_efs( icenter = i_atom,
                                                          neighbors = AA_neighbors,
                                                          offsets = AA_offsets,
                                                          epsilon = epsilon_AA,
                                                          sigma = sigma_AA,
                                                          rc = rc_AA,
                                                          ro = ro_AA )
                energy += e_AA
                force += f_AA
                stress += s_AA
            
            if len(BB_neighbors) > 0:
                e_BB, f_BB, s_BB = self.get_pairwise_efs( icenter = i_atom,
                                                          neighbors = BB_neighbors,
                                                          offsets = BB_offsets,
                                                          epsilon = epsilon_BB,
                                                          sigma = sigma_BB,
                                                          rc = rc_BB,
                                                          ro = ro_BB )
                energy += e_BB
                force += f_BB
                stress += s_BB
            
            if len(AB_neighbors) > 0:
                e_AB, f_AB, s_AB = self.get_pairwise_efs( icenter = i_atom,
                                                          neighbors = AB_neighbors,
                                                          offsets = AB_offsets,
                                                          epsilon = epsilon_AB,
                                                          sigma = sigma_AB,
                                                          rc = rc_AB,
                                                          ro = ro_AB )
                energy += e_AB
                force += f_AB
                stress += s_AB
            
            energies[i_atom] += energy
            forces[i_atom] += force
            stresses[i_atom] += stress
            
        # no lattice, no stress
        if self.atoms.cell.rank == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            self.results['stress'] = stresses.sum(axis=0) / self.atoms.get_volume()
            self.results['stresses'] = stresses / self.atoms.get_volume()

        energy = energies.sum()
        self.results['energy'] = energy
        self.results['energies'] = energies
        self.results['free_energy'] = energy
        self.results['forces'] = forces
        
    def get_pairwise_efs(
        self,
        icenter = None,
        neighbors = None,
        offsets = None,
        epsilon = 1.0,
        sigma = 1.0,
        rc = 2.5,
        ro = 1.65
    ):
        
        smooth = self.parameters.smooth
        positions = self.atoms.positions
        cell = self.atoms.cell
        symbols = list(self.atoms.symbols)
        
        # pointing *towards* neighbours
        distance_vectors = positions[neighbors] - positions[icenter] + np.dot(offsets, cell)
        
        # potential value at rc
        e0 = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
        
        r2 = (distance_vectors ** 2).sum(1)
        c6 = (sigma ** 2 / r2) ** 3
        c6[r2 > rc ** 2] = 0.0
        c12 = c6 ** 2
        c06 = (sigma ** 2 / rc ** 2) ** 3
        c012 = c06 ** 2

        if smooth:
            cutoff_fn = cutoff_function(r2, rc**2, ro**2)
            d_cutoff_fn = d_cutoff_function(r2, rc**2, ro**2)

        pairwise_energies = 4 * epsilon * (c12 - c6) \
                          + 4 * epsilon * (6 * c012 - 3 * c06) * (r2 / rc ** 2) \
                          + 4 * epsilon * (- 6 * c012 + 3 * c06)
        pairwise_forces = - 24 * epsilon * (2 * c12 - c6) / r2 \
                          + 8 * epsilon * (6 * c012 - 3 * c06) / (rc ** 2) # du_ij

        if smooth:
            # order matters, otherwise the pairwise energy is already modified
            pairwise_forces = (
                cutoff_fn * pairwise_forces + 2 * d_cutoff_fn * pairwise_energies
            )
            pairwise_energies *= cutoff_fn
        else:
            pairwise_energies -= e0 * (c6 != 0.0)
        
        
        pairwise_forces = pairwise_forces[:, np.newaxis] * distance_vectors
        
        energy = 0.5 * pairwise_energies.sum()  # atomic energies
        forces = pairwise_forces.sum(axis=0)
        stress = 0.5 * np.dot( pairwise_forces.T, distance_vectors )  # equivalent to outer product
        
        return energy, forces, stress


def cutoff_function(r, rc, ro):
    """Smooth cutoff function.

    Goes from 1 to 0 between ro and rc, ensuring
    that u(r) = lj(r) * cutoff_function(r) is C^1.

    Defined as 1 below ro, 0 above rc.

    Note that r, rc, ro are all expected to be squared,
    i.e. `r = r_ij^2`, etc.

    Taken from https://github.com/google/jax-md.

    """

    return np.where(
        r < ro,
        1.0,
        np.where(r < rc, (rc - r) ** 2 * (rc + 2 * r - 3 * ro) / (rc - ro) ** 3, 0.0),
    )


def d_cutoff_function(r, rc, ro):
    """Derivative of smooth cutoff function wrt r.

    Note that `r = r_ij^2`, so for the derivative wrt to `r_ij`,
    we need to multiply `2*r_ij`. This gives rise to the factor 2
    above, the `r_ij` is cancelled out by the remaining derivative
    `d r_ij / d d_ij`, i.e. going from scalar distance to distance vector.
    """

    return np.where(
        r < ro,
        0.0,
        np.where(r < rc, 6 * (rc - r) * (ro - r) / (rc - ro) ** 3, 0.0),
    )

