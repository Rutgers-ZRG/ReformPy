"""
Entropy-based fingerprint calculations for diversity-driven optimization.
This module provides entropy calculations and derivatives for maximizing
structural diversity in atomic configurations.
"""

import numpy as np
from numba import jit, float64, int32
import warnings

try:
    from numba import jit, float64, int32
    use_numba = True
except ImportError:
    use_numba = False
    def jit(*args, **kwargs):
        return lambda func: func
    float64 = int32 = lambda: None


@jit(nopython=True, cache=True)
def find_min_fp_indices_jit(fp, min_threshold=1e-8):
    """
    Find the index of the nearest neighbor for each atom based on fingerprints.

    Parameters:
    -----------
    fp : np.ndarray
        Fingerprint array of shape (nat, fp_dim)
    min_threshold : float
        Minimum distance threshold to avoid numerical issues

    Returns:
    --------
    min_indices : np.ndarray
        Array of nearest neighbor indices for each atom
    """
    nat = fp.shape[0]
    min_indices = np.zeros(nat, dtype=np.int32)

    for i in range(nat):
        min_dist = 1e12
        min_idx = 0

        for j in range(nat):
            if i == j:
                continue

            # Calculate L2 distance
            dist = 0.0
            for k in range(fp.shape[1]):
                diff = fp[i, k] - fp[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)

            if dist < min_dist:
                min_dist = dist
                min_idx = j

        min_indices[i] = min_idx

    return min_indices


@jit(nopython=True, cache=True)
def find_min_fp_distances_jit(fp, min_threshold=1e-8):
    """
    Find the minimum fingerprint distance for each atom.

    Parameters:
    -----------
    fp : np.ndarray
        Fingerprint array of shape (nat, fp_dim)
    min_threshold : float
        Minimum distance threshold to avoid numerical issues

    Returns:
    --------
    min_distances : np.ndarray
        Array of minimum distances for each atom
    """
    nat = fp.shape[0]
    min_distances = np.zeros(nat, dtype=np.float64)

    for i in range(nat):
        min_dist = 1e12

        for j in range(nat):
            if i == j:
                continue

            # Calculate L2 distance
            dist = 0.0
            for k in range(fp.shape[1]):
                diff = fp[i, k] - fp[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)

            if dist < min_dist:
                min_dist = dist

        # Apply threshold to avoid log(0)
        if min_dist < min_threshold:
            min_dist = min_threshold

        min_distances[i] = min_dist

    return min_distances


@jit(nopython=True, cache=True)
def calculate_entropy_jit(fp, min_threshold=1e-8):
    """
    Calculate the entropy based on minimum fingerprint distances.

    S = (1/N) * Σ_i log(N * δq_min,i)

    where δq_min,i is the minimum fingerprint distance for atom i.

    Parameters:
    -----------
    fp : np.ndarray
        Fingerprint array of shape (nat, fp_dim)
    min_threshold : float
        Minimum distance threshold to avoid numerical issues

    Returns:
    --------
    entropy : float
        The calculated entropy value
    """
    nat = fp.shape[0]
    min_distances = find_min_fp_distances_jit(fp, min_threshold)

    entropy = 0.0
    for i in range(nat):
        entropy += np.log(nat * min_distances[i])

    entropy /= nat
    return entropy


@jit(nopython=True, cache=True)
def calculate_entropy_gradient_jit(fp, dfp, min_threshold=1e-8):
    """
    Calculate the gradient of entropy with respect to atomic positions.

    Parameters:
    -----------
    fp : np.ndarray
        Fingerprint array of shape (nat, fp_dim)
    dfp : np.ndarray
        Fingerprint derivatives of shape (nat, nat, 3, fp_dim)
    min_threshold : float
        Minimum distance threshold to avoid numerical issues

    Returns:
    --------
    entropy_grad : np.ndarray
        Entropy gradient of shape (nat, 3)
    """
    nat = fp.shape[0]
    fp_dim = fp.shape[1]
    entropy_grad = np.zeros((nat, 3), dtype=np.float64)

    # Find nearest neighbors
    min_indices = find_min_fp_indices_jit(fp, min_threshold)

    # Calculate gradient contributions
    for j in range(nat):
        l_j = min_indices[j]

        # Calculate fp difference and distance
        deltaq = 0.0
        fpdiff = np.zeros(fp_dim, dtype=np.float64)
        for m in range(fp_dim):
            diff = fp[j, m] - fp[l_j, m]
            fpdiff[m] = diff
            deltaq += diff * diff
        deltaq = np.sqrt(deltaq)

        # Skip if distance is too small
        if deltaq < min_threshold:
            deltaq = min_threshold

        # Calculate gradient for each atom i
        for i in range(nat):
            for k in range(3):
                # Derivative difference
                deriv_dot = 0.0
                for m in range(fp_dim):
                    deriv_diff = dfp[j, i, k, m] - dfp[l_j, i, k, m]
                    deriv_dot += deriv_diff * fpdiff[m]

                # Add contribution to gradient
                entropy_grad[i, k] += (1.0 / nat) * deriv_dot / (deltaq * deltaq)

    return entropy_grad


@jit(nopython=True, cache=True)
def calculate_entropy_stress_jit(fp, dfpe, min_threshold=1e-8):
    """
    Calculate the stress contribution from entropy.

    Parameters:
    -----------
    fp : np.ndarray
        Fingerprint array of shape (nat, fp_dim)
    dfpe : np.ndarray
        Fingerprint strain derivatives of shape (nat, 6, fp_dim)
    min_threshold : float
        Minimum distance threshold to avoid numerical issues

    Returns:
    --------
    entropy_stress : np.ndarray
        Entropy stress contribution in Voigt notation (6,)
    """
    nat = fp.shape[0]
    fp_dim = fp.shape[1]
    entropy_stress = np.zeros(6, dtype=np.float64)

    # Find nearest neighbors
    min_indices = find_min_fp_indices_jit(fp, min_threshold)

    # Calculate stress contributions
    for j in range(nat):
        l_j = min_indices[j]

        # Calculate fp difference and distance
        deltaq = 0.0
        fpdiff = np.zeros(fp_dim, dtype=np.float64)
        for m in range(fp_dim):
            diff = fp[j, m] - fp[l_j, m]
            fpdiff[m] = diff
            deltaq += diff * diff
        deltaq = np.sqrt(deltaq)

        # Skip if distance is too small
        if deltaq < min_threshold:
            deltaq = min_threshold

        # Calculate stress for each component
        for comp in range(6):
            # Derivative difference for this stress component
            deriv_dot = 0.0
            for m in range(fp_dim):
                deriv_diff = dfpe[j, comp, m] - dfpe[l_j, comp, m]
                deriv_dot += deriv_diff * fpdiff[m]

            # Add contribution to stress
            entropy_stress[comp] -= (1.0 / nat) * deriv_dot / (deltaq * deltaq)

    return entropy_stress


# Non-JIT versions for fallback
def calculate_entropy(fp, min_threshold=1e-8):
    """
    Calculate entropy (non-JIT version).
    """
    if use_numba:
        return calculate_entropy_jit(fp, min_threshold)

    nat = len(fp)
    entropy = 0.0

    for i in range(nat):
        min_dist = float('inf')
        for j in range(nat):
            if i != j:
                dist = np.linalg.norm(fp[i] - fp[j])
                if dist < min_dist:
                    min_dist = dist

        if min_dist < min_threshold:
            min_dist = min_threshold

        entropy += np.log(nat * min_dist)

    return entropy / nat


def calculate_entropy_gradient(fp, dfp, min_threshold=1e-8):
    """
    Calculate entropy gradient (non-JIT version).
    """
    if use_numba:
        return calculate_entropy_gradient_jit(fp, dfp, min_threshold)

    nat = len(fp)
    entropy_grad = np.zeros((nat, 3))

    # Find nearest neighbors
    min_indices = []
    for i in range(nat):
        min_dist = float('inf')
        min_idx = 0
        for j in range(nat):
            if i != j:
                dist = np.linalg.norm(fp[i] - fp[j])
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
        min_indices.append(min_idx)

    # Calculate gradient
    for j in range(nat):
        l_j = min_indices[j]
        fpdiff = fp[j] - fp[l_j]
        deltaq = np.linalg.norm(fpdiff)

        if deltaq < min_threshold:
            deltaq = min_threshold

        for i in range(nat):
            for k in range(3):
                deriv_diff = dfp[j, i, k] - dfp[l_j, i, k]
                numerator = np.dot(deriv_diff, fpdiff) / nat
                denominator = deltaq * deltaq
                entropy_grad[i, k] += numerator / denominator

    return entropy_grad


def calculate_entropy_stress(fp, dfpe, min_threshold=1e-8):
    """
    Calculate entropy stress contribution (non-JIT version).
    """
    if use_numba:
        return calculate_entropy_stress_jit(fp, dfpe, min_threshold)

    nat = len(fp)
    entropy_stress = np.zeros(6)

    # Find nearest neighbors
    min_indices = []
    for i in range(nat):
        min_dist = float('inf')
        min_idx = 0
        for j in range(nat):
            if i != j:
                dist = np.linalg.norm(fp[i] - fp[j])
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
        min_indices.append(min_idx)

    # Calculate stress
    for j in range(nat):
        l_j = min_indices[j]
        fpdiff = fp[j] - fp[l_j]
        deltaq = np.linalg.norm(fpdiff)

        if deltaq < min_threshold:
            deltaq = min_threshold

        for comp in range(6):
            deriv_diff = dfpe[j, comp] - dfpe[l_j, comp]
            numerator = -np.dot(deriv_diff, fpdiff) / nat
            denominator = deltaq * deltaq
            entropy_stress[comp] += numerator / denominator

    return entropy_stress