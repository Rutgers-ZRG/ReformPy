#!/usr/bin/env python3
"""Smoke test for the vendored ``reformpy.torch_fplib`` subpackage.

Confirms the vendored library imports under the ``reformpy`` namespace and
that ``get_lfp`` produces a correctly-shaped fingerprint. Skips cleanly when
``torch`` is not installed (it is an optional extra: ``pip install reformpy[torch]``).

Can also be run directly without pytest::

    python tests/test_torch_fplib_vendored.py
"""
import numpy as np

try:
    import pytest
except ImportError:  # allow standalone `python` execution without pytest
    pytest = None

if pytest is not None:
    torch = pytest.importorskip("torch")
else:
    import torch  # noqa: F401


def _tiny_cell():
    """Minimal 2-atom cell in torch_fplib's ``(lat, rxyz, types, znucl)`` format."""
    lat = np.eye(3, dtype=np.float64) * 5.43
    rxyz = np.array([[0.0, 0.0, 0.0],
                     [1.3575, 1.3575, 1.3575]], dtype=np.float64)
    types = [1, 1]   # 1-indexed atom types
    znucl = [14]     # Si
    return (lat, rxyz, types, znucl)


def test_vendored_import_and_get_lfp():
    from reformpy import torch_fplib

    natx = 64
    fp = torch_fplib.get_lfp(_tiny_cell(), cutoff=4.0, natx=natx)
    # (nat, natx) eigenvalue fingerprint, descending order
    assert tuple(fp.shape) == (2, natx)


if __name__ == "__main__":
    test_vendored_import_and_get_lfp()
    print("OK: reformpy.torch_fplib vendored import + get_lfp works")
