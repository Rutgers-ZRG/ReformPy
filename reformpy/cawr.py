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

    Returns labels in {0, 1}. Fully reproducible: no RNG.
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
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
