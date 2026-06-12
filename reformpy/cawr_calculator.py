"""ASE Calculator for CAWR reform (mirrors entropy_calculator.py pattern).

E = L_CAWR (within-cluster fingerprint variance); pure potential-free
reform — no base calculator. Labels come from a ClusterState and are
FROZEN during a drive phase: calculate() never re-clusters. Call
refresh_labels(atoms) at round boundaries (cawr_reform does this).
"""
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from reformpy.cawr import (ClusterState, compute_fp, resolve_backend,
                           get_ef_clustered, _import_libfp, _cell_tuple_np)


class CAWRCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, cutoff=4.0, nx=300, backend='auto',
                 stability_M=3, state=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.cutoff = float(cutoff)
        self.nx = int(nx)
        self.backend = backend
        self.stability_M = int(stability_M)
        self.state = state            # ClusterState or None (lazy K=1 init)

    # -- label management ------------------------------------------------

    def _ensure_state(self, atoms):
        if self.state is None:
            self.state = ClusterState(atoms.get_atomic_numbers(),
                                      stability_M=self.stability_M)
        return self.state

    @property
    def labels(self):
        if self.state is None:
            raise RuntimeError("no labels yet: run a calculation or "
                               "refresh_labels() first")
        return self.state.labels

    def refresh_labels(self, atoms):
        """Engine evaluation — call at reform-round boundaries ONLY."""
        state = self._ensure_state(atoms)
        fp = compute_fp(atoms, backend=self.backend,
                        cutoff=self.cutoff, nx=self.nx)
        return state.evaluate(fp)

    # -- ASE interface ---------------------------------------------------

    def calculate(self, atoms=None, properties=('energy',),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms
        state = self._ensure_state(atoms)
        labels = state.labels  # committed labels; NEVER re-cluster here
        need_stress = 'stress' in properties
        if need_stress and self.backend == 'libfp':
            raise NotImplementedError(
                "stress requires backend='torch' (libfp dfpe is "
                "unreliable off-diagonal; see design doc)")
        backend = resolve_backend(self.backend, variable_cell=need_stress)

        if backend == 'torch':
            from reformpy.cawr_torch import cawr_efs_torch
            energy, forces, stress = cawr_efs_torch(
                atoms, labels, cutoff=self.cutoff, nx=self.nx,
                compute_stress=need_stress)
            self.results['energy'] = energy
            self.results['forces'] = forces
            if need_stress:
                self.results['stress'] = stress
        else:
            if need_stress:
                raise NotImplementedError(
                    "stress requires backend='torch' (libfp dfpe is "
                    "unreliable off-diagonal; see design doc)")
            libfp = _import_libfp()
            cell = _cell_tuple_np(atoms)
            fp, dfp = libfp.get_dfp(cell, cutoff=self.cutoff, log=False,
                                    natx=self.nx)
            energy, forces = get_ef_clustered(np.asarray(fp),
                                              np.asarray(dfp), labels)
            self.results['energy'] = energy
            self.results['forces'] = forces
