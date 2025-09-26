"""
Entropy-maximizing calculator wrapper for combining with external calculators.
This module provides a wrapper calculator that adds entropy regularization
to any ASE-compatible calculator for diverse structure generation.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.atoms import Atoms
from reformpy.entropy import (
    calculate_entropy,
    calculate_entropy_gradient,
    calculate_entropy_stress
)

try:
    import libfp
except:
    from reformpy import libfppy as libfp
    print("Warning: Using Python implementation of libfp for entropy calculations")


class EntropyMaximizingCalculator(Calculator):
    """
    ASE Calculator wrapper that adds entropy maximization to any base calculator.

    This calculator modifies the energy and forces from a base calculator by adding
    an entropy term based on atomic fingerprint diversity:

    E_total = E_base - k * S(fingerprints)
    F_total = F_base - k * ∇S(fingerprints)

    Parameters
    ----------
    calculator : ASE Calculator
        Base calculator to provide energy/forces/stress
    k_factor : float
        Scaling factor for entropy contribution (default: 1.0)
    cutoff : float
        Cutoff radius for fingerprint calculations (default: 4.0)
    natx : int or None
        Maximum number of neighbors. If None, uses 4*natoms
    entropy_threshold : float
        Minimum distance threshold to avoid numerical issues (default: 1e-8)
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self,
                 calculator,
                 k_factor=1.0,
                 cutoff=4.0,
                 natx=None,
                 entropy_threshold=1e-8,
                 **kwargs):
        """
        Initialize the EntropyMaximizingCalculator.

        Parameters
        ----------
        calculator : ASE Calculator
            The base calculator to wrap
        k_factor : float
            Entropy scaling factor
        cutoff : float
            Fingerprint cutoff radius
        natx : int or None
            Maximum neighbors for fingerprints
        entropy_threshold : float
            Minimum distance threshold
        """
        Calculator.__init__(self, **kwargs)

        self.base_calculator = calculator
        self.k_factor = k_factor
        self.fp_cutoff = cutoff
        self.natx = natx
        self.entropy_threshold = entropy_threshold

        # Track if we need to recalculate fingerprints
        self._fp_cache = {}
        self._last_positions = None
        self._last_cell = None

    def calculate(self,
                  atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        """
        Calculate properties with entropy modification.
        """
        if atoms is None:
            atoms = self.atoms

        # Check if calculation is required
        if not self.calculation_required(atoms, properties):
            return

        # Clone atoms for base calculator
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.base_calculator

        # Initialize results
        self.results = {}

        # Determine what we need
        need_energy = 'energy' in properties
        need_forces = 'forces' in properties
        need_stress = 'stress' in properties

        # Get fingerprints and derivatives if needed
        if need_energy or need_forces or need_stress:
            # We need derivatives for forces, but not for energy-only calculations
            fp_data = self._get_fingerprint_data(atoms,
                                                  need_derivatives=(need_forces or need_stress),
                                                  need_stress=need_stress)

        # Calculate energy
        if need_energy:
            # Get base energy
            base_energy = atoms_copy.get_potential_energy()

            # Calculate entropy
            entropy = calculate_entropy(fp_data['fp'], self.entropy_threshold)

            # Combine: E = E_base - k * S
            total_energy = base_energy - self.k_factor * entropy

            self.results['energy'] = total_energy
            self.results['base_energy'] = base_energy
            self.results['entropy'] = entropy

        # Calculate forces
        if need_forces:
            # Get base forces
            base_forces = atoms_copy.get_forces()

            # Calculate entropy gradient
            entropy_grad = calculate_entropy_gradient(
                fp_data['fp'], fp_data['dfp'], self.entropy_threshold
            )

            # Combine: F = F_base - k * ∇S
            total_forces = base_forces - self.k_factor * entropy_grad

            self.results['forces'] = total_forces

        # Calculate stress
        if need_stress:
            # Get base stress
            base_stress = atoms_copy.get_stress()

            if 'dfpe' in fp_data and fp_data['dfpe'] is not None:
                # Calculate entropy stress contribution
                entropy_stress = calculate_entropy_stress(
                    fp_data['fp'], fp_data['dfpe'], self.entropy_threshold
                )

                # Scale entropy stress by volume to match ASE convention
                volume = atoms.get_volume()
                entropy_stress_scaled = entropy_stress / volume

                # Combine stresses
                total_stress = base_stress - self.k_factor * entropy_stress_scaled
            else:
                # No entropy contribution to stress if derivatives not available
                total_stress = base_stress

            self.results['stress'] = total_stress

    def _get_fingerprint_data(self, atoms, need_derivatives=False, need_stress=False):
        """
        Get fingerprint data, using cache if available.
        """
        # Check if we can use cached data
        positions = atoms.get_positions()
        cell = atoms.get_cell()

        positions_changed = (self._last_positions is None or
                            not np.allclose(positions, self._last_positions))
        cell_changed = (self._last_cell is None or
                       not np.allclose(cell, self._last_cell))

        # Check if we need to recalculate due to missing derivatives
        need_recalc = (positions_changed or cell_changed or not self._fp_cache or
                      (need_derivatives and self._fp_cache.get('dfp') is None) or
                      (need_stress and self._fp_cache.get('dfpe') is None))

        if need_recalc:
            # Need to recalculate
            self._fp_cache = self._calculate_fingerprints(
                atoms, need_derivatives, need_stress
            )
            self._last_positions = positions.copy()
            self._last_cell = cell.copy()

        return self._fp_cache

    def _calculate_fingerprints(self, atoms, need_derivatives=False, need_stress=False):
        """
        Calculate fingerprints and optionally their derivatives.
        """
        # Prepare atom data
        lat = np.array(atoms.cell[:], dtype=np.float64)
        rxyz = np.array(atoms.get_positions(), dtype=np.float64)

        # Get types from atomic numbers
        atomic_numbers = atoms.get_atomic_numbers()
        unique_z = sorted(set(atomic_numbers))
        z_to_type = {z: i+1 for i, z in enumerate(unique_z)}
        types = np.array([z_to_type[z] for z in atomic_numbers], dtype=np.int32)
        znucl = np.array(unique_z, dtype=np.int32)

        # Set natx if not specified
        natx = self.natx if self.natx is not None else 4 * len(atoms)

        # Prepare cell data
        cell_data = (lat, rxyz, types, znucl)

        # Calculate fingerprints
        if need_derivatives:
            result = libfp.get_dfp(
                cell_data,
                cutoff=self.fp_cutoff,
                natx=natx,
                include_stress=need_stress,
                log=False
            )

            if need_stress:
                fp, dfp, dfpe = result
            else:
                fp, dfp = result
                dfpe = None
        else:
            fp = libfp.get_lfp(
                cell_data,
                cutoff=self.fp_cutoff,
                natx=natx,
                log=False
            )
            dfp = None
            dfpe = None

        return {
            'fp': np.array(fp, dtype=np.float64),
            'dfp': np.array(dfp, dtype=np.float64) if dfp is not None else None,
            'dfpe': np.array(dfpe, dtype=np.float64) if dfpe is not None else None,
            'types': types,
            'znucl': znucl
        }

    def set_k_factor(self, k_factor):
        """Set the entropy scaling factor."""
        self.k_factor = float(k_factor)
        self.results = {}  # Clear cached results

    def get_k_factor(self):
        """Get the current entropy scaling factor."""
        return self.k_factor

    def get_entropy(self, atoms=None):
        """
        Calculate just the entropy for the current configuration.

        Parameters
        ----------
        atoms : Atoms or None
            The atoms object. If None, uses self.atoms

        Returns
        -------
        float
            The entropy value
        """
        if atoms is None:
            atoms = self.atoms

        fp_data = self._get_fingerprint_data(atoms, need_derivatives=False)
        return calculate_entropy(fp_data['fp'], self.entropy_threshold)

    def get_base_energy(self, atoms=None):
        """
        Get the energy from the base calculator without entropy contribution.

        Parameters
        ----------
        atoms : Atoms or None
            The atoms object. If None, uses self.atoms

        Returns
        -------
        float
            The base calculator energy
        """
        if atoms is None:
            atoms = self.atoms

        atoms_copy = atoms.copy()
        atoms_copy.calc = self.base_calculator
        return atoms_copy.get_potential_energy()


# Convenience function
def wrap_calculator_with_entropy(calculator, k_factor=1.0, cutoff=4.0, **kwargs):
    """
    Convenience function to wrap any calculator with entropy maximization.

    Parameters
    ----------
    calculator : ASE Calculator
        The base calculator to wrap
    k_factor : float
        Entropy scaling factor
    cutoff : float
        Fingerprint cutoff radius
    **kwargs
        Additional arguments passed to EntropyMaximizingCalculator

    Returns
    -------
    EntropyMaximizingCalculator
        The wrapped calculator
    """
    return EntropyMaximizingCalculator(
        calculator=calculator,
        k_factor=k_factor,
        cutoff=cutoff,
        **kwargs
    )