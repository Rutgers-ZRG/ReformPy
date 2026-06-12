# tests/test_cawr_snap.py
import numpy as np

from conftest import make_rocksalt, import_libfp_or_skip

CUTOFF, NX = 4.0, 64


def test_snap_reduces_cawr_loss():
    import_libfp_or_skip()
    from reformpy.cawr import cawr_snap, cawr_loss_grad, compute_fp
    atoms = make_rocksalt(rattle=0.08, seed=2)
    z = atoms.get_atomic_numbers(); uniq = sorted(set(z))
    labels = np.array([uniq.index(zz) for zz in z])  # K=1 per element

    fp0 = compute_fp(atoms, backend='libfp', cutoff=CUTOFF, nx=NX)
    L0, _ = cawr_loss_grad(fp0, labels)
    snapped = cawr_snap(atoms, labels, cutoff=CUTOFF, nx=NX, n_iter=3)
    fp1 = compute_fp(snapped, backend='libfp', cutoff=CUTOFF, nx=NX)
    L1, _ = cawr_loss_grad(fp1, labels)

    assert L0 > 0.0
    assert L1 < 0.02 * L0          # measured 0.0084 deterministic; gate at 2.4x margin
    assert len(snapped) == len(atoms)
    # original atoms untouched (snap works on a copy)
    assert not np.allclose(snapped.get_positions(), atoms.get_positions())


def test_resolve_backend_rules():
    from reformpy.cawr import resolve_backend
    import pytest
    assert resolve_backend('torch', variable_cell=False) == 'torch'
    assert resolve_backend('torch', variable_cell=True) == 'torch'
    assert resolve_backend('auto', variable_cell=True) == 'torch'
    with pytest.raises(ValueError):
        resolve_backend('libfp', variable_cell=True)
    with pytest.raises(ValueError):
        resolve_backend('numpy', variable_cell=False)
