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


# ---------------------------------------------------------------------------
# Cluster engine
# ---------------------------------------------------------------------------

class ClusterState:
    """Annealed-K cluster assignment engine.

    Owns all "which atoms belong together" decisions. Element-blocked,
    starts at K=1 per element. Splits and merges both route through the
    single statistical criterion `split_is_justified` (a 2-cluster state
    is justified iff the union passes the split test), and any proposal
    must persist for `stability_M` consecutive evaluate() calls
    (identified by canonical atom-index sets) before it commits. Gradient
    consumers must use `labels` (committed) only; evaluate() is called at
    reform-round boundaries, never inside a drive phase.

    Split-persistence semantics: a split proposal is identified by the
    EXACT set of child atom indices and must reproduce identically for
    stability_M consecutive evaluations. Under a moving structure this
    means unambiguous environment groups commit readily, while borderline
    groups (separation ~ within-spread, membership flickering by an atom)
    may never commit — intentional conservatism: a false split poisons the
    reform drive, a missed split merely keeps a coarser target.

    Cost note: merge proposals run the Monte-Carlo gate per close pair —
    O(K^2) worst case per element per evaluation (~15-40 ms per gate
    call). Fine at the expected K <= 8; the rms-overlap prefilter skips
    well-separated pairs before the gate.
    """

    def __init__(self, types, stability_M=3, min_cluster=2, bic_margin=10.0,
                 var_threshold=1e-4):
        self.types = np.asarray(types)
        self.stability_M = int(stability_M)
        self.min_cluster = int(min_cluster)
        self.bic_margin = float(bic_margin)
        self.var_threshold = float(var_threshold)
        # committed labels: one cluster per element (K=1 each)
        self.labels = np.zeros(len(self.types), dtype=np.int64)
        for i, t in enumerate(np.unique(self.types)):
            self.labels[self.types == t] = i
        self._pending_key = None
        self._pending_count = 0
        self._pending_apply = None
        self.last_committed = False
        self.history = []

    def K_per_element(self):
        return {int(t): int(len(np.unique(self.labels[self.types == t])))
                for t in np.unique(self.types)}

    @property
    def has_pending(self):
        """True while a split/merge proposal is accumulating persistence."""
        return self._pending_key is not None

    # -- proposal machinery -------------------------------------------------

    def _merge_proposal(self, fp):
        """Weakest-separated close pair whose union FAILS the split
        criterion, or None. Returns (key, apply_fn)."""
        best = None  # (ratio, key, apply)
        labels = self.labels
        for t in np.unique(self.types):
            cs = np.unique(labels[self.types == t])
            for ai in range(len(cs)):
                for bi in range(ai + 1, len(cs)):
                    ia = np.where(labels == cs[ai])[0]
                    ib = np.where(labels == cs[bi])[0]
                    mua, mub = fp[ia].mean(0), fp[ib].mean(0)
                    rmsa = np.sqrt(((fp[ia] - mua) ** 2).sum(1).mean())
                    rmsb = np.sqrt(((fp[ib] - mub) ** 2).sum(1).mean())
                    if np.linalg.norm(mua - mub) >= rmsa + rmsb:
                        continue
                    X = np.concatenate([fp[ia], fp[ib]])
                    two = np.array([0] * len(ia) + [1] * len(ib))
                    if split_is_justified(X, two, bic_margin=self.bic_margin):
                        continue  # the 2-cluster state holds; no merge
                    ratio = _projected_rss_ratio(X, two)
                    key = ('merge', int(cs[ai]), int(cs[bi]))
                    a_label, b_label = int(cs[ai]), int(cs[bi])

                    def apply(a=a_label, b=b_label):
                        self.labels[self.labels == b] = a

                    if best is None or ratio > best[0]:
                        best = (ratio, key, apply)
        return (best[1], best[2]) if best is not None else None

    def _split_proposal(self, fp):
        """Strongest-separated justified split, or None.
        Returns (key, apply_fn)."""
        best = None  # (ratio, key, apply)
        labels = self.labels
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) < 2 * self.min_cluster:
                continue
            # per-dimension mean variance, so var_threshold is fp_dim-independent
            var_c = float(((fp[idx] - fp[idx].mean(0)) ** 2).sum()) / (len(idx) * fp.shape[1])
            if var_c <= self.var_threshold:
                continue
            sub = bisect_2means(fp[idx])
            if sub.min() == sub.max():
                continue
            if min(int((sub == 0).sum()), int((sub == 1).sum())) < self.min_cluster:
                continue
            if not split_is_justified(fp[idx], sub, bic_margin=self.bic_margin):
                continue
            ratio = _projected_rss_ratio(fp[idx], sub)
            child = frozenset(int(i) for i in idx[sub == 1])
            key = ('split', int(c), child)

            def apply(child=child):
                new_label = int(self.labels.max()) + 1
                for i in child:
                    self.labels[i] = new_label

            if best is None or ratio < best[0]:
                best = (ratio, key, apply)
        return (best[1], best[2]) if best is not None else None

    def _best_proposal(self, fp):
        """Merges first (undoing an unjustified split is more conservative
        than making a new one), then splits."""
        return self._merge_proposal(fp) or self._split_proposal(fp)

    def evaluate(self, fp):
        """One engine evaluation (call at round boundaries only).

        Updates the pending-proposal persistence counter; commits a
        proposal that has persisted `stability_M` consecutive evaluations.
        Returns committed labels (always safe for gradient use).
        """
        fp = np.asarray(fp, dtype=np.float64)
        self.last_committed = False
        prop = self._best_proposal(fp)
        if prop is None:
            self._pending_key, self._pending_count, self._pending_apply = None, 0, None
        elif prop[0] == self._pending_key:
            self._pending_count += 1
            self._pending_apply = prop[1]
        else:
            self._pending_key, self._pending_count, self._pending_apply = prop[0], 1, prop[1]
        if self._pending_count >= self.stability_M and self._pending_apply is not None:
            self._pending_apply()
            self.last_committed = True
            self._pending_key, self._pending_count, self._pending_apply = None, 0, None
        loss, _ = cawr_loss_grad(fp, self.labels)
        self.history.append({'L': loss, 'K': self.K_per_element(),
                             'committed': self.last_committed,
                             'labels': self.labels.copy()})
        return self.labels


# ---------------------------------------------------------------------------
# FP backends (module-level helpers) and snap driver
# ---------------------------------------------------------------------------

def _import_libfp():
    try:
        import libfp
        return libfp
    except ImportError:
        from reformpy import libfppy as libfp  # needs numba
        return libfp


def _cell_tuple_np(atoms):
    """ASE Atoms -> (lat, rxyz, types, znucl) with libfp dtypes."""
    z = atoms.get_atomic_numbers()
    uniq = sorted(set(int(zz) for zz in z))
    types = np.array([uniq.index(int(zz)) + 1 for zz in z], dtype=np.int32)
    znucl = np.array(uniq, dtype=np.int32)
    lat = np.array(atoms.cell[:], dtype=np.float64)
    rxyz = np.array(atoms.get_positions(), dtype=np.float64)
    return lat, rxyz, types, znucl


def resolve_backend(backend, variable_cell):
    """'auto' -> libfp if importable and fixed-cell, else torch.
    Backends are never mixed within a run."""
    if backend not in ('auto', 'libfp', 'torch'):
        raise ValueError(f"backend must be auto|libfp|torch, got {backend!r}")
    if variable_cell:
        if backend == 'libfp':
            raise ValueError("variable_cell requires backend='torch' "
                             "(libfp backend offers no stress)")
        return 'torch'
    if backend != 'auto':
        return backend
    try:
        _import_libfp()
        return 'libfp'
    except ImportError:
        return 'torch'


def compute_fp(atoms, backend='auto', cutoff=4.0, nx=300):
    """Per-atom fingerprints as a (nat, fp_dim) ndarray."""
    backend = resolve_backend(backend, variable_cell=False)
    if backend == 'libfp':
        libfp = _import_libfp()
        fp = libfp.get_lfp(_cell_tuple_np(atoms), cutoff=cutoff, log=False, natx=nx)
        return np.asarray(fp, dtype=np.float64)
    from reformpy.cawr_torch import cell_tuple
    from reformpy import torch_fplib
    lat, pos, types, znucl = cell_tuple(atoms)
    return torch_fplib.get_lfp((lat, pos, types, znucl),
                               cutoff=cutoff, natx=nx).numpy()


def cawr_snap(atoms, labels, cutoff=4.0, nx=300, n_iter=3, max_step=0.25):
    """Least-squares Newton step in fingerprint space (positions only).

    Solves the stacked linear system J . dr = (mu_c - fp_i) over all
    non-singleton atoms, clips per-atom |dr| to max_step (Angstrom), and
    iterates n_iter times recomputing fp and J. Zero potential calls.
    Returns a modified COPY of atoms.
    """
    libfp = _import_libfp()
    atoms = atoms.copy()
    labels = np.asarray(labels)
    prev_positions = None
    prev_loss = np.inf
    for _ in range(int(n_iter)):
        cell = _cell_tuple_np(atoms)
        fp, dfp = libfp.get_dfp(cell, cutoff=cutoff, log=False, natx=nx)
        fp = np.asarray(fp, dtype=np.float64)
        dfp = np.asarray(dfp, dtype=np.float64)  # (nat, nat, 3, fp_dim)
        # Monotonicity guard: if the previous step increased the loss
        # (Gauss-Newton overshoot of the moving mu target), revert it and
        # stop. With the default max_step clip this is rare — the clip is
        # the primary stability mechanism; this guard makes larger steps
        # safe too. The loss after the FINAL step is the caller's to check
        # (cawr_reform records L_after).
        loss, _ = cawr_loss_grad(fp, labels)
        if loss > prev_loss:
            atoms.set_positions(prev_positions)
            break
        prev_positions = atoms.get_positions()
        prev_loss = loss
        nat, fpd = fp.shape
        resid = np.zeros_like(fp)
        sel = np.zeros(nat, dtype=bool)
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) < 2:
                continue
            resid[idx] = fp[idx].mean(axis=0) - fp[idx]
            sel[idx] = True
        if not sel.any():
            break
        # rows (i, m), cols (j, k):  A[(i,m),(j,k)] = dfp[i, j, k, m]
        ns = int(sel.sum())
        A = dfp[sel].transpose(0, 3, 1, 2).reshape(ns * fpd, nat * 3)
        b = resid[sel].reshape(-1)
        dx, *_ = np.linalg.lstsq(A, b, rcond=None)
        dx = dx.reshape(nat, 3)
        norms = np.linalg.norm(dx, axis=1)
        scale = np.minimum(1.0, max_step / np.maximum(norms, 1e-12))
        atoms.set_positions(atoms.get_positions() + dx * scale[:, None])
    return atoms


# ---------------------------------------------------------------------------
# Outer reform loop
# ---------------------------------------------------------------------------

class CAWRResult:
    """Result of cawr_reform: reformed atoms + diagnostics."""

    def __init__(self, atoms, labels, K_per_element, history):
        self.atoms = atoms
        self.labels = labels
        self.K_per_element = K_per_element
        self.history = history


def cawr_reform(atoms, cutoff=4.0, nx=300, driver='snap', variable_cell=False,
                backend='auto', max_rounds=10, stability_M=3,
                inner_steps=30, fmax=0.005, max_step=0.25, snap_iters=3,
                tol_plateau=0.02):
    """Annealed-K CAWR reform (potential-free).

    Per round: compute FPs -> engine evaluation (gated labels) -> drive
    (snap or FIRE) -> terminate when a round commits no proposal AND the
    loss is flat. The engine evaluates ONLY at round boundaries — this
    loop is the single owner of engine evaluations (the per-round
    CAWRCalculator is given the shared state but its refresh_labels is
    never called); labels are frozen during drives.

    Note the fmax default (0.005, not ASE's customary 0.05): CAWR forces
    are O(0.01) on small cells — a larger fmax makes FIRE exit at 0 steps
    and the drive silently no-ops.

    Commit headroom: each committed split/merge costs at least stability_M
    rounds of persistence, so the maximum number of commits in one call is
    ~ max_rounds // stability_M. Structures needing K >= 4 distinct
    environments for one element need max_rounds raised accordingly.

    Rounds with a pending proposal do not drive: the proposal matures on a
    static structure (committing after stability_M evaluations), because
    driving with the not-yet-split labels would homogenize environments and
    destroy the evidence the proposal needs to persist. Pending rounds are
    cheap (one fingerprint evaluation, no drive).
    """
    if driver not in ('snap', 'fire'):
        raise ValueError(f"driver must be 'snap' or 'fire', got {driver!r}")
    if variable_cell and driver == 'snap':
        raise ValueError("snap is positions-only: variable_cell=True "
                         "requires driver='fire'")
    backend = resolve_backend(backend, variable_cell)
    if driver == 'snap' and backend == 'torch':
        raise ValueError("driver='snap' uses the libfp analytic Jacobian "
                         "(backend='libfp'); use driver='fire' with the "
                         "torch backend")

    atoms = atoms.copy()
    state = ClusterState(atoms.get_atomic_numbers(), stability_M=stability_M)
    history = []

    for rnd in range(int(max_rounds)):
        fp = compute_fp(atoms, backend=backend, cutoff=cutoff, nx=nx)
        labels = state.evaluate(fp)
        L_before, _ = cawr_loss_grad(fp, labels)

        pending = state.has_pending
        if pending:
            # A proposal is maturing: drive nothing this round. Driving
            # with the current (coarser) labels homogenizes environments
            # and destroys the very evidence the pending proposal must
            # reproduce for stability_M consecutive evaluations
            # (the "discovery deadlock" found by the b28 diagnostic).
            L_after = L_before
        elif driver == 'snap':
            atoms = cawr_snap(atoms, labels, cutoff=cutoff, nx=nx,
                              n_iter=snap_iters, max_step=max_step)
        else:
            from ase.optimize import FIRE
            from reformpy.cawr_calculator import CAWRCalculator
            calc = CAWRCalculator(cutoff=cutoff, nx=nx, backend=backend,
                                  state=state)
            atoms.calc = calc
            if variable_cell:
                try:
                    from ase.filters import FrechetCellFilter as CellFilter
                except ImportError:
                    from ase.constraints import ExpCellFilter as CellFilter
                target = CellFilter(atoms)
            else:
                target = atoms
            FIRE(target, logfile=None).run(fmax=fmax, steps=int(inner_steps))
            atoms.calc = None

        if not pending:
            fp_after = compute_fp(atoms, backend=backend, cutoff=cutoff, nx=nx)
            L_after, _ = cawr_loss_grad(fp_after, labels)
        history.append({'round': rnd, 'L_before': L_before, 'L_after': L_after,
                        'K': state.K_per_element(),
                        'committed': state.last_committed,
                        'pending': pending})

        flat = (L_before <= 0.0
                or abs(L_after - L_before) / max(L_before, 1e-12) < tol_plateau)
        if rnd > 0 and not state.last_committed and not pending and flat:
            break

    return CAWRResult(atoms, state.labels.copy(), state.K_per_element(),
                      history)
