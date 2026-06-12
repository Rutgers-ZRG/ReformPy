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


def test_split_is_justified_rejects_single_blob_across_seeds():
    """Rate test: a single Gaussian blob must essentially never justify a
    split (guards the adversarial 2-means over-split bias)."""
    from reformpy.cawr import bisect_2means, split_is_justified
    false_splits = 0
    trials = 0
    for n in (6, 12, 20, 50):
        for seed in range(50):
            X = np.random.default_rng(seed * 997 + n).normal(size=(n, 4))
            sub = bisect_2means(X)
            if sub.min() != sub.max() and split_is_justified(X, sub):
                false_splits += 1
            trials += 1
    assert false_splits / trials < 0.03


def test_split_is_justified_accepts_separated_blobs_n12_plus():
    """Power: clearly separated blobs (8 sigma) must be accepted for
    clusters of n >= 12."""
    from reformpy.cawr import bisect_2means, split_is_justified
    accepted = 0
    trials = 0
    for n in (12, 20, 50):
        for seed in range(25):
            rng = np.random.default_rng(seed)
            a = rng.normal(size=(n // 2, 4))
            b = rng.normal(size=(n // 2, 4)) + 8.0
            X = np.concatenate([a, b])
            sub = bisect_2means(X)
            if split_is_justified(X, sub):
                accepted += 1
            trials += 1
    assert accepted / trials > 0.9


def test_split_is_justified_small_n_needs_extreme_separation():
    """Documented small-sample conservatism: n=6 splits at sep=20, and the
    gate is deterministic (same input -> same verdict)."""
    from reformpy.cawr import bisect_2means, split_is_justified
    accepted = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=(3, 4))
        b = rng.normal(size=(3, 4)) + 20.0
        X = np.concatenate([a, b])
        sub = bisect_2means(X)
        if split_is_justified(X, sub):
            accepted += 1
    assert accepted / 20 > 0.9
    # determinism of the gate itself
    rng = np.random.default_rng(7)
    X = np.concatenate([rng.normal(size=(3, 4)),
                        rng.normal(size=(3, 4)) + 20.0])
    sub = bisect_2means(X)
    assert split_is_justified(X, sub) == split_is_justified(X, sub)


def test_bisect_2means_degenerate_spectrum_is_deterministic():
    """Symmetric ring: top singular pair degenerate; fallback must be
    deterministic and not crash."""
    from reformpy.cawr import bisect_2means
    th = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    X = np.stack([np.cos(th), np.sin(th),
                  np.zeros_like(th), np.zeros_like(th)], axis=1)
    l1 = bisect_2means(X)
    l2 = bisect_2means(X)
    assert np.array_equal(l1, l2)
