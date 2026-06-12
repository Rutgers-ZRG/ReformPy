"""Torch backend for CAWR: autograd energy, forces, and stress.

Uses the vendored reformpy.torch_fplib. Conventions (CRISP-validated):
row-vector cells, lat' = lat0 @ (I + eps), sigma_v = (1/V) dL/deps[a, b]
extracted per single Voigt component — dL/deps is NOT symmetric; never
symmetrize or double off-diagonal strains (factor-of-4 bug).

The autograd path differentiates through mu_c automatically, so it is
immune to the hand-derived-gradient bug class by construction.
"""
import numpy as np
import torch

from reformpy import torch_fplib

VOIGT_IDX = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def cell_tuple(atoms):
    """ASE Atoms -> (lat, pos, types, znucl) in libfp/torch_fplib format
    (types 1-indexed per element, znucl sorted unique atomic numbers)."""
    z = atoms.get_atomic_numbers()
    uniq = sorted(set(int(zz) for zz in z))
    z2t = {zz: i + 1 for i, zz in enumerate(uniq)}
    types = [z2t[int(zz)] for zz in z]
    lat = np.array(atoms.cell[:], dtype=np.float64)
    pos = np.array(atoms.get_positions(), dtype=np.float64)
    return lat, pos, types, uniq


def _cawr_loss_torch(fp, labels):
    """Within-cluster variance loss in torch; labels are constants."""
    labels = np.asarray(labels)
    L = fp.new_zeros(())  # seed from fp (tensor, requires_grad=False) — the all-singleton early return in cawr_efs_torch relies on this staying a constant
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        sub = fp[idx]
        L = L + ((sub - sub.mean(dim=0)) ** 2).sum()
    return L


def cawr_efs_torch(atoms, labels, cutoff=4.0, nx=300, compute_stress=False):
    """CAWR energy, forces, and (optionally) stress via torch autograd.

    Returns (energy: float, forces: (nat,3) ndarray, stress: (6,) ndarray
    in ASE Voigt order divided by volume, or None).
    """
    lat_np, _, types, znucl = cell_tuple(atoms)
    lat0 = torch.tensor(lat_np, dtype=torch.float64)
    spos = torch.tensor(atoms.get_scaled_positions(), dtype=torch.float64,
                        requires_grad=True)
    strain = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    lat = lat0 @ (torch.eye(3, dtype=torch.float64) + strain)
    pos = spos @ lat
    fp = torch_fplib.get_lfp((lat, pos, types, znucl), cutoff=cutoff, natx=nx)
    L = _cawr_loss_torch(fp, labels)

    if not L.requires_grad:  # all clusters singletons: loss is constant 0
        nat = len(atoms)
        stress = np.zeros(6) if compute_stress else None
        return 0.0, np.zeros((nat, 3)), stress

    wanted = [spos, strain] if compute_stress else [spos]
    grads = torch.autograd.grad(L, wanted)
    g_spos = grads[0]
    # r = s @ lat  =>  grad_s = grad_r @ lat^T  =>  grad_r = grad_s @ inv(lat0)^T
    forces = -(g_spos @ torch.linalg.inv(lat0).T).detach().numpy()
    energy = float(L.detach())
    stress = None
    if compute_stress:
        ge = grads[1].detach().numpy()
        volume = float(abs(np.linalg.det(lat_np)))
        stress = np.array([ge[a, b] for a, b in VOIGT_IDX]) / volume
    return energy, forces, stress
