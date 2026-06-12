#!/usr/bin/env python3
"""CSP pilot: can CAWR-biased MatterSim relaxation find gamma-B28?

Protocol (matched budget, per the CRISP audit discipline):
  1. Generate N random 28-atom boron structures with pyxtal (random
     space groups, fixed seed).
  2. Relax each under P = 50 GPa (gamma-B28 stability field) with TWO
     arms at the SAME 200-step budget:
       control   — plain MatterSim-1M, FIRE + FrechetCellFilter
       cawr-bias — MixedCalculator(mode='bias'): E/stress from MatterSim,
                   F = F_ms + lambda(t)*F_cawr, lambda annealed to 0 over
                   the 200 steps (adaptive eta=0.3); CAWR labels refresh
                   with static exhaustion every 10 steps.
  3. Score: spglib SG (symprec sweep), enthalpy/atom vs the gamma-B28
     reference relaxed under the same potential+pressure, Hungarian fp
     distance to the relaxed reference.

Success per structure: SG = Pnnm (#58) at any swept symprec AND
enthalpy within 5 meV/atom of the reference.

Usage:
  python benchmark/cawr_mattersim_csp.py --n 50 --steps 200 --out benchmark/csp_results.json
"""
import argparse
import json
import time

import numpy as np
import ase.io
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from ase.units import GPa
import spglib
from scipy.optimize import linear_sum_assignment

P_GPA = 50.0
REF_POSCAR = '/Users/li/dev/RA/struct-predict/POSCAR_gamma_b28_unrelaxed.vasp'
SYMPREC_SWEEP = (1e-3, 1e-2, 3e-2, 5e-2, 1e-1)
H_TOL = 0.005           # eV/atom window around the reference enthalpy
FP_CUTOFF, FP_NX = 4.0, 128


# ── helpers ──────────────────────────────────────────────────────────

def make_mattersim():
    from mattersim.forcefield import MatterSimCalculator
    return MatterSimCalculator(device="cpu")


def gen_structures(n, seed):
    """N pyxtal random crystals: 28 B atoms, random space group."""
    from pyxtal import pyxtal
    rng = np.random.default_rng(seed)
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        sg = int(rng.integers(2, 231))
        x = pyxtal()
        try:
            x.from_random(3, sg, ['B'], [28], factor=0.85,
                          random_state=int(rng.integers(0, 2 ** 31)))
            atoms = x.to_ase()
        except Exception:
            continue
        if len(atoms) == 28:
            atoms.set_pbc(True)
            out.append({'gen_sg': sg, 'atoms': atoms})
    return out


def sg_sweep(atoms):
    """Spacegroup across the symprec sweep; returns {symprec: 'SG (#)'}"""
    res = {}
    cell = (atoms.cell[:], atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    for sp in SYMPREC_SWEEP:
        try:
            res[sp] = spglib.get_spacegroup(cell, symprec=sp) or 'P1 (1)'
        except Exception:
            res[sp] = 'ERR'
    return res


def enthalpy_per_atom(atoms, energy):
    return (energy + P_GPA * GPa * atoms.get_volume()) / len(atoms)


def fp_dist(atoms1, atoms2):
    """Hungarian-matched per-atom fingerprint distance (drive metric,
    not an identity metric)."""
    from reformpy.cawr import compute_fp
    fp1 = compute_fp(atoms1, backend='torch', cutoff=FP_CUTOFF, nx=FP_NX)
    fp2 = compute_fp(atoms2, backend='torch', cutoff=FP_CUTOFF, nx=FP_NX)
    nat = len(fp1)
    cost = np.linalg.norm(fp1[:, None, :] - fp2[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].sum() / nat)


# ── relaxation arms (matched budget) ─────────────────────────────────

def relax_control(atoms, steps, fmax=0.02):
    a = atoms.copy()
    a.calc = make_mattersim()
    opt = FIRE(FrechetCellFilter(a, scalar_pressure=P_GPA * GPa),
               logfile=None)
    opt.run(fmax=fmax, steps=steps)
    a.calc = make_mattersim()  # fresh, to read converged E cleanly
    return a, opt.nsteps


def relax_cawr_bias(atoms, steps, fmax=0.02, refresh_every=10):
    from reformpy.mixing import MixedCalculator
    from reformpy.cawr_calculator import CAWRCalculator
    a = atoms.copy()
    cawr = CAWRCalculator(cutoff=FP_CUTOFF, nx=FP_NX, backend='torch')
    mixed = MixedCalculator(make_mattersim(), cawr, iter_max=steps,
                            scheme='cosine', mode='bias',
                            adaptive_lambda=True, eta=0.3)
    a.calc = mixed
    opt = FIRE(FrechetCellFilter(a, scalar_pressure=P_GPA * GPa),
               logfile=None)

    def refresh():
        # round boundary every `refresh_every` steps: exhaust statically
        # justified proposals (mirrors cawr_reform's discovery loop)
        if opt.nsteps == 0 or opt.nsteps % refresh_every:
            return
        for _ in range(50):
            cawr.refresh_labels(a)
            st = cawr.state
            if not (st.has_pending or st.last_committed):
                break

    opt.attach(refresh)
    opt.run(fmax=fmax, steps=steps)
    K = cawr.state.K_per_element() if cawr.state is not None else {}
    a.calc = make_mattersim()
    return a, opt.nsteps, K


# ── scoring ──────────────────────────────────────────────────────────

def score(atoms, h_ref, ref_relaxed):
    e = atoms.get_potential_energy()
    h = enthalpy_per_atom(atoms, e)
    sgs = sg_sweep(atoms)
    found_pnnm = any('Pnnm' in s for s in sgs.values())
    d_fp = fp_dist(atoms, ref_relaxed)
    success = found_pnnm and abs(h - h_ref) < H_TOL
    return {'H_per_atom': h, 'dH_meV': (h - h_ref) * 1000.0,
            'sg_sweep': {str(k): v for k, v in sgs.items()},
            'pnnm': bool(found_pnnm), 'fp_dist_to_ref': d_fp,
            'success': bool(success)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='benchmark/csp_results.json')
    p.add_argument('--structures-out', default='benchmark/csp_structures.xyz')
    args = p.parse_args()

    t_start = time.time()

    # Reference: relax gamma-B28 under the same potential + pressure
    # (double budget so the reference itself is well converged).
    ref = ase.io.read(REF_POSCAR)
    ref_relaxed, _ = relax_control(ref, steps=2 * args.steps, fmax=0.01)
    h_ref = enthalpy_per_atom(ref_relaxed, ref_relaxed.get_potential_energy())
    print(f"gamma-B28 reference @ {P_GPA} GPa: H = {h_ref:.4f} eV/at, "
          f"SG sweep: {sg_sweep(ref_relaxed)}", flush=True)

    print(f"Generating {args.n} pyxtal structures (seed {args.seed})...",
          flush=True)
    pool = gen_structures(args.n, args.seed)
    print(f"  got {len(pool)}", flush=True)

    results = []
    for i, entry in enumerate(pool):
        row = {'idx': i, 'gen_sg': entry['gen_sg']}
        t0 = time.time()

        ctrl, n1 = relax_control(entry['atoms'], args.steps)
        row['control'] = score(ctrl, h_ref, ref_relaxed)
        row['control']['nsteps'] = int(n1)

        bias, n2, K = relax_cawr_bias(entry['atoms'], args.steps)
        row['cawr_bias'] = score(bias, h_ref, ref_relaxed)
        row['cawr_bias']['nsteps'] = int(n2)
        row['cawr_bias']['K_final'] = {str(k): v for k, v in K.items()}

        row['wall_s'] = round(time.time() - t0, 1)
        results.append(row)

        ctrl.info.update(idx=i, arm='control')
        bias.info.update(idx=i, arm='cawr_bias')
        ase.io.write(args.structures_out, [ctrl, bias], append=(i > 0))

        print(f"[{i + 1}/{len(pool)}] gen_sg={entry['gen_sg']:3d}  "
              f"ctrl: dH={row['control']['dH_meV']:8.1f} meV "
              f"Pnnm={row['control']['pnnm']} S={row['control']['success']}  |  "
              f"bias: dH={row['cawr_bias']['dH_meV']:8.1f} meV "
              f"Pnnm={row['cawr_bias']['pnnm']} S={row['cawr_bias']['success']} "
              f"K={row['cawr_bias']['K_final']}  ({row['wall_s']}s)", flush=True)

        with open(args.out, 'w') as f:  # incremental save
            json.dump({'h_ref': h_ref, 'P_GPa': P_GPA, 'steps': args.steps,
                       'seed': args.seed, 'results': results}, f, indent=1)

    n_ok_c = sum(r['control']['success'] for r in results)
    n_ok_b = sum(r['cawr_bias']['success'] for r in results)
    n_p_c = sum(r['control']['pnnm'] for r in results)
    n_p_b = sum(r['cawr_bias']['pnnm'] for r in results)
    print(f"\n=== SUMMARY (N={len(results)}, {args.steps} steps, "
          f"{P_GPA} GPa, {time.time() - t_start:.0f}s) ===")
    print(f"control  : success {n_ok_c}/{len(results)}  Pnnm {n_p_c}")
    print(f"cawr-bias: success {n_ok_b}/{len(results)}  Pnnm {n_p_b}")


if __name__ == '__main__':
    main()
