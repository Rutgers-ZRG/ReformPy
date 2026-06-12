"""Shared fixtures for ReformPy tests."""
import numpy as np
import pytest
from ase import Atoms


def make_rocksalt(a=5.6, rattle=0.0, seed=0):
    """8-atom NaCl-like conventional cubic cell (4 Na + 4 Cl)."""
    frac = np.array([
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],  # Na
        [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.5, 0.5, 0.5],  # Cl
    ])
    atoms = Atoms('Na4Cl4', scaled_positions=frac, cell=np.eye(3) * a, pbc=True)
    if rattle > 0:
        rng = np.random.default_rng(seed)
        atoms.set_positions(atoms.get_positions()
                            + rng.normal(scale=rattle, size=(len(atoms), 3)))
    return atoms


def make_sheared_rocksalt(a=5.6, shear=0.06, rattle=0.03, seed=1):
    """Rocksalt cell with all-nonzero Voigt stress components (sheared)."""
    atoms = make_rocksalt(a=a, rattle=rattle, seed=seed)
    cell = atoms.cell[:].copy()
    eps = np.array([[0.00,  0.04, -0.03],
                    [0.02,  0.01,  shear],
                    [-0.05, 0.03,  0.02]])
    atoms.set_cell(cell @ (np.eye(3) + eps), scale_atoms=True)
    return atoms


@pytest.fixture
def rocksalt_rattled():
    return make_rocksalt(rattle=0.05)


@pytest.fixture
def rocksalt_sheared():
    return make_sheared_rocksalt()


def import_libfp_or_skip():
    """Return a libfp-compatible module (C libfp or libfppy), or skip."""
    try:
        import libfp
        return libfp
    except ImportError:
        pass
    try:
        from reformpy import libfppy
        return libfppy
    except ImportError:
        pytest.skip("neither libfp nor libfppy (numba) available")
