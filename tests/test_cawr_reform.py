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


def test_commit_is_driven_in_same_round(monkeypatch):
    """A committed label change must be driven in the round it commits:
    history must show the commit AND an L reduction in that same round."""
    import reformpy.cawr as cawr_mod
    from reformpy.cawr import ClusterState

    class ForcedSplitState(ClusterState):
        """Commits a fixed Na split on the first evaluation."""
        def __init__(self, types, **kw):
            super().__init__(types, **kw)
            self._forced = False

        def evaluate(self, fp):
            fp = np.asarray(fp, dtype=np.float64)
            self.last_committed = False
            if not self._forced:
                na = np.where(self.types == 11)[0]
                new_label = int(self.labels.max()) + 1
                self.labels[na[len(na) // 2:]] = new_label
                self.last_committed = True
                self._forced = True
            from reformpy.cawr import cawr_loss_grad
            loss, _ = cawr_loss_grad(fp, self.labels)
            self.history.append({'L': loss, 'K': self.K_per_element(),
                                 'committed': self.last_committed,
                                 'labels': self.labels.copy()})
            return self.labels

    monkeypatch.setattr(cawr_mod, 'ClusterState', ForcedSplitState)
    from reformpy.cawr import cawr_reform
    atoms = make_rocksalt(rattle=0.05, seed=3)
    res = cawr_reform(atoms, cutoff=CUTOFF, nx=NX, driver='fire',
                      backend='torch', max_rounds=2, inner_steps=20)
    # round 0: commit recorded AND the new labels were driven (L dropped)
    assert res.history[0]['committed'] is True
    assert res.history[0]['K'][11] == 2
    assert res.history[0]['L_after'] < res.history[0]['L_before']
    # the result carries the committed (split) labels
    assert res.K_per_element[11] == 2


def test_pending_proposal_blocks_drive_until_commit(monkeypatch):
    """Rounds with a pending proposal must not drive (static maturation);
    the commit then happens and driving resumes with the new labels."""
    import reformpy.cawr as cawr_mod
    from reformpy.cawr import ClusterState

    class SlowSplitState(ClusterState):
        """Proposes the same Na split every evaluation; commits on the 3rd."""
        def _best_proposal(self, fp):
            na = np.where(self.types == 11)[0]
            if len(np.unique(self.labels[na])) > 1:
                return None  # already split
            child = frozenset(int(i) for i in na[len(na) // 2:])
            key = ('split', int(self.labels[na[0]]), child)

            def apply(child=child):
                new_label = int(self.labels.max()) + 1
                for i in child:
                    self.labels[i] = new_label

            return (key, apply)

    monkeypatch.setattr(cawr_mod, 'ClusterState', SlowSplitState)
    from reformpy.cawr import cawr_reform
    atoms = make_rocksalt(rattle=0.05, seed=3)
    res = cawr_reform(atoms, cutoff=CUTOFF, nx=NX, driver='fire',
                      backend='torch', max_rounds=5, inner_steps=15,
                      stability_M=3)
    h = res.history
    # rounds 0 and 1: pending, NOT driven (L unchanged), no commit
    assert h[0]['pending'] and h[1]['pending']
    assert h[0]['L_after'] == h[0]['L_before']
    assert h[1]['L_after'] == h[1]['L_before']
    assert not h[0]['committed'] and not h[1]['committed']
    # round 2: the commit lands and THAT round drives the new labels
    assert h[2]['committed'] is True
    assert not h[2]['pending']
    assert h[2]['L_after'] < h[2]['L_before']
    assert res.K_per_element[11] == 2
