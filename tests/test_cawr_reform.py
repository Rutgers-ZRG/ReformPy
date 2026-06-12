# tests/test_cawr_reform.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import make_rocksalt, import_libfp_or_skip

CUTOFF, NX = 4.0, 64


def test_reform_fire_torch_reduces_loss_and_returns_result():
    from reformpy.cawr import cawr_reform
    atoms = make_rocksalt(rattle=0.08, seed=6)
    res = cawr_reform(atoms, cutoff=CUTOFF, nx=NX, driver='fire',
                      backend='torch', max_rounds=3, inner_steps=25)
    assert len(res.atoms) == len(atoms)
    assert res.history[-1]['L_after'] < res.history[0]['L_before']
    assert set(res.K_per_element.keys()) == {11, 17}  # Na, Cl
    assert len(res.labels) == len(atoms)
    # input untouched
    assert np.allclose(atoms.get_positions(),
                       make_rocksalt(rattle=0.08, seed=6).get_positions())


def test_reform_snap_libfp_reduces_loss():
    import_libfp_or_skip()
    from reformpy.cawr import cawr_reform
    atoms = make_rocksalt(rattle=0.08, seed=2)
    res = cawr_reform(atoms, cutoff=CUTOFF, nx=NX, driver='snap',
                      backend='auto', max_rounds=3)
    assert res.history[-1]['L_after'] < 0.1 * res.history[0]['L_before']


def test_variable_cell_with_snap_raises():
    from reformpy.cawr import cawr_reform
    atoms = make_rocksalt()
    with pytest.raises(ValueError):
        cawr_reform(atoms, driver='snap', variable_cell=True)


def test_snap_with_torch_backend_raises():
    from reformpy.cawr import cawr_reform
    atoms = make_rocksalt()
    with pytest.raises(ValueError):
        cawr_reform(atoms, driver='snap', backend='torch')


def test_unknown_driver_raises():
    from reformpy.cawr import cawr_reform
    with pytest.raises(ValueError):
        cawr_reform(make_rocksalt(), driver='newton')
