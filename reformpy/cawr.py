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
