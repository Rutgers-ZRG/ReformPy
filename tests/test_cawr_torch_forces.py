# tests/test_cawr_torch_forces.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import make_rocksalt

CUTOFF, NX = 4.0, 64


def _labels_k1(atoms):
    """K=1 per element labels (Na->0, Cl->1)."""
    z = atoms.get_atomic_numbers()
    uniq = sorted(set(z))
    return np.array([uniq.index(zz) for zz in z])


def test_torch_forces_match_fd():
    from reformpy.cawr_torch import cawr_efs_torch
    atoms = make_rocksalt(rattle=0.05)
    labels = _labels_k1(atoms)
    energy, forces, _ = cawr_efs_torch(atoms, labels, cutoff=CUTOFF, nx=NX)
    assert energy > 0.0
    h = 1e-5
    pos0 = atoms.get_positions()
    for (i, k) in [(0, 0), (3, 1), (5, 2), (7, 0)]:  # spot-check 4 components
        ap = atoms.copy(); p = pos0.copy(); p[i, k] += h; ap.set_positions(p)
        am = atoms.copy(); p = pos0.copy(); p[i, k] -= h; am.set_positions(p)
        ep, _, _ = cawr_efs_torch(ap, labels, cutoff=CUTOFF, nx=NX)
        em, _, _ = cawr_efs_torch(am, labels, cutoff=CUTOFF, nx=NX)
        fd_force = -(ep - em) / (2 * h)
        assert forces[i, k] == pytest.approx(fd_force, abs=1e-6)


def test_torch_forces_zero_for_all_singletons():
    from reformpy.cawr_torch import cawr_efs_torch
    atoms = make_rocksalt(rattle=0.05)
    labels = np.arange(len(atoms))  # every atom its own cluster
    energy, forces, _ = cawr_efs_torch(atoms, labels, cutoff=CUTOFF, nx=NX)
    assert energy == 0.0
    assert np.allclose(forces, 0.0)
