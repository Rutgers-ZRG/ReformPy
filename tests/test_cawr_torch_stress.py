# tests/test_cawr_torch_stress.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import make_rocksalt, make_sheared_rocksalt

CUTOFF, NX = 4.0, 64
VOIGT_IDX = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def _strained(atoms, a, b, h):
    """Apply a SINGLE strain component eps[a,b]=h via cell' = cell @ (I+eps),
    positions transformed consistently (r' = r @ (I+eps))."""
    out = atoms.copy()
    eps = np.zeros((3, 3)); eps[a, b] = h
    out.set_cell(atoms.cell[:] @ (np.eye(3) + eps), scale_atoms=False)
    out.set_positions(atoms.get_positions() @ (np.eye(3) + eps))
    return out


def _fd_stress(atoms, energy_fn, h=1e-6):
    V = atoms.get_volume()
    s = np.zeros(6)
    for v, (a, b) in enumerate(VOIGT_IDX):
        ep = energy_fn(_strained(atoms, a, b, +h))
        em = energy_fn(_strained(atoms, a, b, -h))
        s[v] = (ep - em) / (2 * h) / V
    return s


def test_lj_witness_pins_fd_construction_to_ase_convention():
    """Our single-component FD construction must reproduce ASE's stress
    for a known calculator (LJ). Pins lat' = lat0 @ (I+eps) + sign."""
    from ase.calculators.lj import LennardJones
    atoms = make_sheared_rocksalt()

    def e(a):
        a = a.copy(); a.calc = LennardJones(sigma=2.0, epsilon=0.1, rc=5.0)
        return a.get_potential_energy()

    atoms.calc = LennardJones(sigma=2.0, epsilon=0.1, rc=5.0)
    sigma_ase = atoms.get_stress(voigt=True)
    sigma_fd = _fd_stress(atoms, e, h=1e-6)
    assert np.max(np.abs(sigma_ase - sigma_fd)) < 1e-6


def test_cawr_autograd_stress_matches_fd_on_sheared_cell():
    """All 6 Voigt components, single-component FD, sheared cell —
    the exact regime where libfp dfpe failed (57-1625% off-diagonal)."""
    from reformpy.cawr_torch import cawr_efs_torch
    atoms = make_sheared_rocksalt()
    z = atoms.get_atomic_numbers(); uniq = sorted(set(z))
    labels = np.array([uniq.index(zz) for zz in z])

    def e(a):
        return cawr_efs_torch(a, labels, cutoff=CUTOFF, nx=NX)[0]

    _, _, sigma_auto = cawr_efs_torch(atoms, labels, cutoff=CUTOFF, nx=NX,
                                      compute_stress=True)
    sigma_fd = _fd_stress(atoms, e, h=1e-6)
    assert sigma_auto is not None
    assert np.max(np.abs(sigma_auto - sigma_fd)) < 1e-9
    # sanity: the shear made off-diagonal components genuinely non-zero
    assert np.max(np.abs(sigma_fd[3:])) > 1e-10
