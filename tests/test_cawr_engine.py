# tests/test_cawr_engine.py
import numpy as np


def _fp_two_envs(n_per=4, sep=10.0, noise=0.1, seed=0):
    """Single-element fp set containing two distinct environments."""
    rng = np.random.default_rng(seed)
    a = rng.normal(scale=noise, size=(n_per, 5))
    b = rng.normal(scale=noise, size=(n_per, 5)) + sep
    return np.concatenate([a, b])


def test_initial_state_is_k1_per_element():
    from reformpy.cawr import ClusterState
    types = np.array([1, 1, 1, 2, 2])
    st = ClusterState(types)
    assert st.K_per_element() == {1: 1, 2: 1}
    # element blocking: labels never shared across elements
    l1 = set(st.labels[types == 1]); l2 = set(st.labels[types == 2])
    assert l1.isdisjoint(l2)


def test_split_commits_only_after_stability_M_rounds():
    from reformpy.cawr import ClusterState
    types = np.ones(8, dtype=int)
    fp = _fp_two_envs()
    st = ClusterState(types, stability_M=3)
    k = []
    for _ in range(4):
        st.evaluate(fp)
        k.append(st.K_per_element()[1])
    # evaluations 1 and 2: proposal pending, K still 1; commits on the 3rd
    assert k == [1, 1, 2, 2]


def test_flickering_proposal_never_commits():
    from reformpy.cawr import ClusterState
    types = np.ones(8, dtype=int)
    fp_a = _fp_two_envs(seed=0)
    rng = np.random.default_rng(9)
    fp_b = rng.normal(size=(8, 5))  # single blob: no split proposal
    st = ClusterState(types, stability_M=3)
    for fp in (fp_a, fp_b, fp_a, fp_b, fp_a, fp_b):
        st.evaluate(fp)
    assert st.K_per_element()[1] == 1


def test_merge_reduces_K():
    from reformpy.cawr import ClusterState
    types = np.ones(8, dtype=int)
    st = ClusterState(types, stability_M=1)
    # force a 2-cluster committed state, then feed identical environments
    st.labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    rng = np.random.default_rng(5)
    fp_same = rng.normal(scale=0.05, size=(8, 5))  # one tight blob
    st.evaluate(fp_same)
    assert st.K_per_element()[1] == 1


def test_element_blocking_in_proposals():
    from reformpy.cawr import ClusterState
    types = np.array([1] * 8 + [2] * 4)
    fp = np.vstack([_fp_two_envs(), np.random.default_rng(7).normal(size=(4, 5))])
    st = ClusterState(types, stability_M=1)
    st.evaluate(fp)
    l1 = set(st.labels[types == 1]); l2 = set(st.labels[types == 2])
    assert l1.isdisjoint(l2)
    assert st.K_per_element()[2] == 1


def test_history_records_loss_K_and_commits():
    from reformpy.cawr import ClusterState
    types = np.ones(8, dtype=int)
    st = ClusterState(types, stability_M=1)
    st.evaluate(_fp_two_envs())
    assert len(st.history) == 1
    h = st.history[0]
    assert set(h.keys()) >= {'L', 'K', 'committed', 'labels'}
    assert h['committed'] is True and h['K'][1] == 2
