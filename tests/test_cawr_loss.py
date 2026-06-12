# tests/test_cawr_loss.py
import numpy as np
import pytest


def _fd_grad(fp, labels, loss_fn, eps=1e-6):
    g = np.zeros_like(fp)
    for i in range(fp.shape[0]):
        for m in range(fp.shape[1]):
            fp_p = fp.copy(); fp_p[i, m] += eps
            fp_m = fp.copy(); fp_m[i, m] -= eps
            g[i, m] = (loss_fn(fp_p, labels) - loss_fn(fp_m, labels)) / (2 * eps)
    return g


def test_cawr_loss_grad_matches_fd():
    from reformpy.cawr import cawr_loss_grad
    rng = np.random.default_rng(0)
    fp = rng.normal(size=(6, 5))
    labels = np.array([0, 0, 0, 1, 1, 2])  # sizes 3, 2, and a singleton
    _, grad = cawr_loss_grad(fp, labels)
    fd = _fd_grad(fp, labels, lambda f, l: cawr_loss_grad(f, l)[0])
    assert np.max(np.abs(grad - fd)) < 1e-7


def test_singleton_cluster_zero_loss_and_grad():
    from reformpy.cawr import cawr_loss_grad
    fp = np.random.default_rng(1).normal(size=(3, 4))
    labels = np.array([0, 1, 2])  # all singletons
    loss, grad = cawr_loss_grad(fp, labels)
    assert loss == 0.0
    assert np.all(grad == 0.0)


def test_old_one_minus_inv_n_form_fails_fd():
    """Regression doc: the (1 - 1/n_c) form from the old auto_clusters code
    is NOT the gradient of the loss."""
    from reformpy.cawr import cawr_loss_grad
    rng = np.random.default_rng(2)
    fp = rng.normal(size=(5, 4))
    labels = np.array([0, 0, 0, 0, 0])  # one cluster, n_c = 5
    _, exact = cawr_loss_grad(fp, labels)
    mu = fp.mean(axis=0)
    old = 2.0 * (1.0 - 1.0 / 5) * (fp - mu)
    fd = _fd_grad(fp, labels, lambda f, l: cawr_loss_grad(f, l)[0])
    assert np.max(np.abs(exact - fd)) < 1e-7        # exact form passes
    assert np.max(np.abs(old - fd)) > 1e-3          # old form fails


def test_get_ef_clustered_projects_with_jacobian():
    """Forces = -einsum('im,ijkm->jk', dL/dfp, dfp), net force removed."""
    from reformpy.cawr import cawr_loss_grad, get_ef_clustered
    rng = np.random.default_rng(3)
    nat, fpd = 4, 6
    fp = rng.normal(size=(nat, fpd))
    dfp = rng.normal(size=(nat, nat, 3, fpd))
    labels = np.array([0, 0, 1, 1])
    energy, forces = get_ef_clustered(fp, dfp, labels)
    loss, dL = cawr_loss_grad(fp, labels)
    expected = -np.einsum("im,ijkm->jk", dL, dfp)
    expected -= expected.mean(axis=0)
    assert energy == pytest.approx(loss)
    assert np.allclose(forces, expected)
    assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-12)
