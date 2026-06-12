#!/usr/bin/env python3
"""K-discovery diagnostic for annealed-K CAWR (spec gate 4).

Perturb a known multi-orbit structure, run cawr_reform, and score:
  - K-accuracy: discovered K per element vs spglib orbit count of the ideal
  - label agreement: adjusted Rand index vs ideal crystallographic orbits
  - oracle arm: drive with TRUE orbit labels (bounds what perfect
    assignment buys; separates assignment quality from driving quality)

Intermediate diagnostics only — NOT the headline audit metric (which is
spglib SG recovery at matched budget; follow-up plan).

Usage:
  python benchmark/cawr_kdiag.py --ref benchmark/gamma_b28_ref.xyz \
      --sigma 0.05 0.10 --seeds 3 --driver fire --backend torch
"""
import argparse
import numpy as np
import ase.io
import spglib
from sklearn.metrics import adjusted_rand_score

from reformpy.cawr import cawr_reform, cawr_loss_grad, compute_fp, cawr_snap


def _dataset_field(ds, name):
    """spglib >=2.5 returns an object; older versions a dict."""
    if ds is None:
        raise RuntimeError("spglib found no symmetry dataset")
    try:
        return getattr(ds, name)
    except AttributeError:
        return ds[name]


def ideal_orbits(atoms, symprec=1e-3):
    """Crystallographic orbit label per atom from spglib."""
    cell = (atoms.cell[:], atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    orbits = np.asarray(_dataset_field(ds, 'crystallographic_orbits'))
    sg = _dataset_field(ds, 'international')
    return orbits, sg


def orbit_K_per_element(atoms, orbits):
    z = atoms.get_atomic_numbers()
    return {int(t): len(set(orbits[z == t])) for t in sorted(set(z))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', default='benchmark/gamma_b28_ref.xyz')
    p.add_argument('--sigma', type=float, nargs='+', default=[0.05, 0.10])
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--driver', default='fire', choices=['fire', 'snap'])
    p.add_argument('--backend', default='auto')
    p.add_argument('--cutoff', type=float, default=4.0)
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--max-rounds', type=int, default=15)
    args = p.parse_args()

    ideal = ase.io.read(args.ref)
    orbits, sg = ideal_orbits(ideal)
    K_true = orbit_K_per_element(ideal, orbits)
    print(f"Reference: {args.ref}  SG={sg}  true K per element: {K_true}")

    rows = []
    for sigma in args.sigma:
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            pert = ideal.copy()
            pert.set_positions(pert.get_positions()
                               + rng.normal(scale=sigma, size=(len(pert), 3)))

            res = cawr_reform(pert, cutoff=args.cutoff, nx=args.nx,
                              driver=args.driver, backend=args.backend,
                              max_rounds=args.max_rounds)
            ari = adjusted_rand_score(orbits, res.labels)
            k_ok = res.K_per_element == K_true

            # oracle arm: drive the same perturbed input with TRUE labels
            if args.driver == 'snap':
                oracle = cawr_snap(pert, orbits, cutoff=args.cutoff,
                                   nx=args.nx, n_iter=3 * args.max_rounds)
            else:
                from ase.optimize import FIRE
                from reformpy.cawr import ClusterState
                from reformpy.cawr_calculator import CAWRCalculator
                oracle = pert.copy()
                st = ClusterState(oracle.get_atomic_numbers())
                st.labels = np.asarray(orbits, dtype=np.int64).copy()
                oracle.calc = CAWRCalculator(cutoff=args.cutoff, nx=args.nx,
                                             backend=args.backend, state=st)
                FIRE(oracle, logfile=None).run(fmax=0.005,
                                               steps=30 * args.max_rounds)
            fp_o = compute_fp(oracle, backend=args.backend,
                              cutoff=args.cutoff, nx=args.nx)
            L_oracle, _ = cawr_loss_grad(fp_o, np.asarray(orbits))

            rows.append((sigma, seed, res.K_per_element, k_ok, ari,
                         res.history[-1]['L_after'], L_oracle))
            print(f"sigma={sigma:5.2f} seed={seed}  K={res.K_per_element} "
                  f"K_ok={k_ok}  ARI={ari:6.3f}  "
                  f"L_final={res.history[-1]['L_after']:.4e}  "
                  f"L_oracle={L_oracle:.4e}")

    n_ok = sum(r[3] for r in rows)
    print(f"\nK-accuracy: {n_ok}/{len(rows)}   "
          f"mean ARI: {np.mean([r[4] for r in rows]):.3f}")


if __name__ == '__main__':
    main()
