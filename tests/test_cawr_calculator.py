# tests/test_cawr_calculator.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import make_rocksalt

CUTOFF, NX = 4.0, 64


def test_calculator_matches_backend_function():
    from reformpy.cawr_calculator import CAWRCalculator
    from reformpy.cawr_torch import cawr_efs_torch
    atoms = make_rocksalt(rattle=0.05)
    calc = CAWRCalculator(cutoff=CUTOFF, nx=NX, backend='torch')
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    z = atoms.get_atomic_numbers(); uniq = sorted(set(z))
    labels = np.array([uniq.index(zz) for zz in z])
    e_ref, f_ref, _ = cawr_efs_torch(atoms, labels, cutoff=CUTOFF, nx=NX)
    assert e == pytest.approx(e_ref)
    assert np.allclose(f, f_ref)


def test_labels_frozen_during_drive():
    """calculate() must never re-cluster; only refresh_labels() may."""
    from reformpy.cawr_calculator import CAWRCalculator
    atoms = make_rocksalt(rattle=0.05)
    calc = CAWRCalculator(cutoff=CUTOFF, nx=NX, backend='torch')
    atoms.calc = calc
    atoms.get_potential_energy()
    labels_before = calc.labels.copy()
    atoms.set_positions(atoms.get_positions() + 0.01)
    atoms.get_forces()
    assert np.array_equal(calc.labels, labels_before)


def test_fire_relaxation_reduces_cawr_energy():
    from ase.optimize import FIRE
    from reformpy.cawr_calculator import CAWRCalculator
    atoms = make_rocksalt(rattle=0.08, seed=4)
    calc = CAWRCalculator(cutoff=CUTOFF, nx=NX, backend='torch')
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    FIRE(atoms, logfile=None).run(fmax=0.005, steps=40)
    e1 = atoms.get_potential_energy()
    assert e1 < e0


def test_libfp_backend_stress_raises():
    from conftest import import_libfp_or_skip
    import_libfp_or_skip()
    from reformpy.cawr_calculator import CAWRCalculator
    atoms = make_rocksalt(rattle=0.03)
    atoms.calc = CAWRCalculator(cutoff=CUTOFF, nx=NX, backend='libfp')
    atoms.get_potential_energy()  # fine
    with pytest.raises(NotImplementedError):
        atoms.get_stress()


def test_label_commit_invalidates_cache():
    """refresh_labels() committing a label change must invalidate ASE's
    results cache even though the atoms did not move."""
    from reformpy.cawr import ClusterState
    from reformpy.cawr_calculator import CAWRCalculator
    atoms = make_rocksalt(rattle=0.02)
    z = atoms.get_atomic_numbers()
    state = ClusterState(z, stability_M=1)
    # Pre-split Na into two artificial clusters; the actual fp environment
    # is one tight blob, so the next engine evaluation merges them (a
    # committed label change without any atomic motion).
    na = np.where(z == 11)[0]
    state.labels = state.labels.copy()
    state.labels[na[2:]] = state.labels.max() + 1
    calc = CAWRCalculator(cutoff=CUTOFF, nx=NX, backend='torch', state=state)
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    labels0 = calc.labels.copy()
    calc.refresh_labels(atoms)
    assert not np.array_equal(calc.labels, labels0)  # the merge committed
    e1 = atoms.get_potential_energy()                # atoms unchanged
    assert e1 != pytest.approx(e0)
