# tests/test_cawr_split.py
import numpy as np


def _two_blobs(n=12, sep=8.0, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n // 2, 4))
    b = rng.normal(size=(n // 2, 4)) + sep
    return np.concatenate([a, b]), np.array([0] * (n // 2) + [1] * (n // 2))


def test_bisect_2means_separates_blobs():
    from reformpy.cawr import bisect_2means
    X, truth = _two_blobs()
    labels = bisect_2means(X)
    # agreement up to label permutation
    agree = max((labels == truth).mean(), (labels == 1 - truth).mean())
    assert agree == 1.0


def test_bisect_2means_deterministic():
    from reformpy.cawr import bisect_2means
    X, _ = _two_blobs(seed=3)
    assert np.array_equal(bisect_2means(X), bisect_2means(X))


def test_bic_prefers_two_components_for_blobs():
    from reformpy.cawr import bic_spherical, bisect_2means
    X, _ = _two_blobs()
    sub = bisect_2means(X)
    one = np.zeros(len(X), dtype=int)
    assert bic_spherical(X, one) - bic_spherical(X, sub) > 10.0


def test_bic_prefers_one_component_for_single_blob():
    from reformpy.cawr import bic_spherical, bisect_2means
    rng = np.random.default_rng(4)
    X = rng.normal(size=(12, 4))
    sub = bisect_2means(X)
    one = np.zeros(len(X), dtype=int)
    assert bic_spherical(X, one) <= bic_spherical(X, sub) + 10.0
