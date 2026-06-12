"""CAWR: Cluster-Aware Within-structure Reform.

FP-native annealed-K reform: discover Wyckoff-like groups of atoms by
their fingerprint environments (starting from K=1 per element, splitting
or merging only when statistically justified and stable), and drive
same-cluster atoms toward identical environments.

Design doc: docs/superpowers/specs/2026-06-11-cawr-annealed-k-design.md
This module is numpy-only (no torch, no mpi4py). The torch backend lives
in reformpy.cawr_torch; the ASE wrapper in reformpy.cawr_calculator.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Loss and gradient
# ---------------------------------------------------------------------------

def cawr_loss_grad(fp, labels):
    """CAWR loss and its exact fingerprint-space gradient.

    L = Σ_c Σ_{i∈c} ||fp_i − μ_c||²,   dL/dfp_i = 2 (fp_i − μ_c)

    The μ_c-dependence cancels exactly (Σ_{i∈c}(fp_i − μ_c) = 0), so no
    (1 − 1/n_c) or n_c factor appears. Singleton clusters contribute zero
    loss and zero gradient.

    Parameters
    ----------
    fp : ndarray, shape (nat, fp_dim)
    labels : ndarray, shape (nat,)

    Returns
    -------
    loss : float
    grad : ndarray, shape (nat, fp_dim)
    """
    fp = np.asarray(fp, dtype=np.float64)
    labels = np.asarray(labels)
    nat, fp_dim = fp.shape
    loss = 0.0
    grad = np.zeros((nat, fp_dim), dtype=np.float64)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        diff = fp[idx] - fp[idx].mean(axis=0)
        loss += float((diff ** 2).sum())
        grad[idx] = 2.0 * diff
    return loss, grad


def get_fpe_clustered(fp, labels):
    """CAWR energy only (see cawr_loss_grad)."""
    return cawr_loss_grad(fp, labels)[0]


def get_ef_clustered(fp, dfp, labels):
    """CAWR energy and Cartesian forces via the libfp Jacobian.

    dfp[i, j, k, m] = ∂fp[i, m]/∂r[j, k]  (libfp get_dfp layout).
    """
    fp = np.ascontiguousarray(np.asarray(fp, dtype=np.float64))
    dfp = np.ascontiguousarray(np.asarray(dfp, dtype=np.float64))
    energy, dL_dfp = cawr_loss_grad(fp, labels)
    forces = -np.einsum("im,ijkm->jk", dL_dfp, dfp, optimize=True)
    forces -= forces.mean(axis=0)  # translational invariance
    return energy, forces


# ---------------------------------------------------------------------------
# Deterministic 2-means and BIC (no sklearn dependency)
# ---------------------------------------------------------------------------

def bisect_2means(X, max_iter=100):
    """Deterministic 2-means: init by sign of projection onto the first
    principal component (sign fixed), then Lloyd iterations.

    Returns labels in {0, 1}. Fully reproducible: no RNG. When the top
    singular pair is (near-)degenerate — where vt[0] would be
    LAPACK-build-dependent — falls back to a deterministic direction
    (centroid to farthest point, lowest index on ties).
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    if len(s) > 1 and s[0] > 0 and (s[0] - s[1]) / s[0] < 1e-6:
        # Degenerate top singular pair: vt[0] is LAPACK-build-dependent.
        # Deterministic fallback: direction from centroid to the farthest
        # point (lowest index on ties).
        dist = np.linalg.norm(Xc, axis=1)
        far = int(np.argmax(dist))
        if dist[far] < 1e-12:
            return np.zeros(len(X), dtype=int)
        d = Xc[far] / dist[far]
    else:
        d = vt[0]
    if d[int(np.argmax(np.abs(d)))] < 0:
        d = -d
    proj = Xc @ d
    labels = (proj > 0).astype(int)
    if labels.min() == labels.max():
        labels = (proj > np.median(proj)).astype(int)
    for _ in range(max_iter):
        if labels.min() == labels.max():
            break
        c0 = X[labels == 0].mean(axis=0)
        c1 = X[labels == 1].mean(axis=0)
        new = (np.linalg.norm(X - c1, axis=1)
               < np.linalg.norm(X - c0, axis=1)).astype(int)
        if new.min() == new.max() or np.array_equal(new, labels):
            break
        labels = new
    return labels


def bic_spherical(X, labels):
    """BIC of a hard-assignment spherical Gaussian mixture (shared σ²).

    Lower is better. Used to gate cluster splits (2-component BIC must
    beat 1-component by a margin) and merges (the reverse).
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    uniq = np.unique(labels)
    k = len(uniq)
    rss = 0.0
    for c in uniq:
        xc = X[np.asarray(labels) == c]
        rss += float(((xc - xc.mean(axis=0)) ** 2).sum())
    sigma2 = max(rss / (n * d), 1e-12)
    loglik = -0.5 * n * d * (np.log(2 * np.pi * sigma2) + 1.0)
    n_params = k * d + 1 + (k - 1)  # means + shared variance + mixing weights
    return -2.0 * loglik + n_params * np.log(n)


# Expected within-RSS fraction when hard-splitting a single 1-D Gaussian at
# its mean: 1 - 2/pi. Documented for reference; the actual gate calibrates
# the null per-cluster by Monte Carlo (see split_is_justified).
NULL_RSS_RATIO = 1.0 - 2.0 / np.pi


def _projected_rss_ratio(X, labels2):
    """Within-RSS ratio on the 1-D projection along the child-mean
    separation direction. 1.0 signals a degenerate / uninformative split."""
    mu0 = X[labels2 == 0].mean(axis=0)
    mu1 = X[labels2 == 1].mean(axis=0)
    dvec = mu1 - mu0
    norm = np.linalg.norm(dvec)
    if norm < 1e-12:
        return 1.0
    t = X @ (dvec / norm)
    rss1 = float(((t - t.mean()) ** 2).sum())
    if rss1 < 1e-24:
        return 1.0
    rss2 = sum(float(((t[labels2 == c] - t[labels2 == c].mean()) ** 2).sum())
               for c in (0, 1))
    return rss2 / rss1


def split_is_justified(X, labels2, bic_margin=10.0, n_null=199, q=0.005,
                       seed=0):
    """Dual gate for accepting a 2-way split of cluster data X.

    (1) BIC pre-filter: 2-component spherical BIC must beat 1-component
        by bic_margin.
    (2) Matched-null Monte-Carlo test: the observed projected-RSS ratio
        (see _projected_rss_ratio) must fall below the q-quantile of the
        same statistic computed on n_null Gaussian replicates drawn with
        this cluster's own empirical covariance, run through the same
        bisect_2means procedure. This calibrates the null per (n, d,
        covariance shape) — a fixed threshold cannot (the null 1% quantile
        spans 0.016-0.214 over n=6-50 alone), because 2-means selects the
        split direction adversarially.

    Deterministic: the null replicates use a fixed RNG seed. False-split
    rate is exactly floor(q*(n_null+1))/(n_null+1) per test under the
    Gaussian null (rank test; q=0.005, n_null=199 -> 1/200) across all
    (n, d); small clusters (n<~8)
    split only under extreme separation — intentional asymmetry, since a
    false split poisons the reform drive while a missed split merely keeps
    a coarser symmetrization target.
    """
    X = np.asarray(X, dtype=np.float64)
    labels2 = np.asarray(labels2)
    uniq = np.unique(labels2)
    if len(uniq) != 2:
        return False
    labels2 = (labels2 == uniq[1]).astype(int)
    one = np.zeros(len(X), dtype=int)
    if bic_spherical(X, one) - bic_spherical(X, labels2) <= bic_margin:
        return False
    obs = _projected_rss_ratio(X, labels2)
    if obs >= 1.0:
        return False
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / max(len(X) - 1, 1)
    w, V = np.linalg.eigh(cov)
    A = V * np.sqrt(np.clip(w, 0.0, None))
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_null)
    for r in range(n_null):
        Z = rng.standard_normal((len(X), X.shape[1])) @ A.T
        sub = bisect_2means(Z)
        if sub.min() == sub.max():
            nulls[r] = 1.0
        else:
            nulls[r] = _projected_rss_ratio(Z, sub)
    return obs < np.quantile(nulls, q, method='lower')
