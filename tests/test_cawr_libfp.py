# tests/test_cawr_libfp.py
import numpy as np
import pytest

from conftest import make_rocksalt, import_libfp_or_skip
from reformpy.cawr import _cell_tuple_np

CUTOFF, NX = 4.0, 64


def test_libfp_chain_rule_forces_match_fd():
    libfp = import_libfp_or_skip()
    from reformpy.cawr import get_ef_clustered, get_fpe_clustered
    atoms = make_rocksalt(rattle=0.05)
    z = atoms.get_atomic_numbers(); uniq = sorted(set(z))
    labels = np.array([uniq.index(zz) for zz in z])

    def fp_of(a):
        cell = _cell_tuple_np(a)
        return np.asarray(libfp.get_lfp(cell, cutoff=CUTOFF, log=False, natx=NX),
                          dtype=np.float64)

    cell = _cell_tuple_np(atoms)
    fp, dfp = libfp.get_dfp(cell, cutoff=CUTOFF, log=False, natx=NX)
    _, forces = get_ef_clustered(np.asarray(fp), np.asarray(dfp), labels)

    h = 1e-5
    pos0 = atoms.get_positions()
    for (i, k) in [(0, 0), (3, 1), (4, 1), (7, 2)]:
        ap = atoms.copy(); p = pos0.copy(); p[i, k] += h; ap.set_positions(p)
        am = atoms.copy(); p = pos0.copy(); p[i, k] -= h; am.set_positions(p)
        ep = get_fpe_clustered(fp_of(ap), labels)
        em = get_fpe_clustered(fp_of(am), labels)
        fd_force = -(ep - em) / (2 * h)
        assert forces[i, k] == pytest.approx(fd_force, abs=1e-9)


def test_libfp_and_torch_fplib_fingerprints_agree():
    libfp = import_libfp_or_skip()
    pytest.importorskip("torch")
    from reformpy import torch_fplib
    atoms = make_rocksalt(rattle=0.03)
    lat, pos, types, znucl = _cell_tuple_np(atoms)
    fp_c = np.asarray(libfp.get_lfp((lat, pos, types, znucl),
                                    cutoff=CUTOFF, log=False, natx=NX))
    fp_t = torch_fplib.get_lfp((lat, pos, list(types), list(znucl)),
                               cutoff=CUTOFF, natx=NX).numpy()
    assert fp_c.shape == fp_t.shape
    assert np.max(np.abs(fp_c - fp_t)) < 1e-6
