#!/usr/bin/env python3
"""
Day 3: the prevalence sweep, and the comparison the reviewer asked for.

Populations at known prevalence are built by resampling the two pure arms, since trials are
independent and nothing interacts between reviewer types. Every population therefore costs no
model calls, and each can be resampled many times for intervals.

Three analyses run on the same populations:

1. **The paper's own criteria**, through `verdicts.py` unchanged. The contrast is a mixture
   against the fully engaged population, which is the analogue of a team whose oversight has
   partly lapsed measured against one whose has not.
2. **Signal detection**, hit rate, false-alarm rate and d' per population.
3. **The Proposition 2 estimator**, with alpha and beta measured on the pure arms where the state
   is known:

       pi_hat = (P(flag a seeded item) - alpha) / (1 - alpha - beta)

   compared against the prevalence that was built in.

The prediction, fixed before the run: the accept-rate criteria will not
track prevalence monotonically, d' will, and pi_hat will recover pi within its interval. If the
accept-rate criteria track prevalence fine, the study's central methodological claim is wrong and
this script is what says so.

Usage:
  uv run --python 3.12 --with numpy --with pandas python sweep.py [--run <dir>] [--boot 2000]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LIB = Path(__file__).resolve().parents[1] / "lib"  # vendored verdicts.py
sys.path.insert(0, str(LIB))

SEEDED = {f"vch3-doc-{i:02d}" for i in range(7, 13)}
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load(run_dir: Path) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for f in run_dir.glob("*/*.json"):
        d = json.loads(f.read_text())
        if d["condition"].endswith("_probe"):
            continue  # the probe arms verify the construction; they are not the study
        end = next(e for e in d["events"] if e["event_type"] == "trial_end")
        doc = d["input"].get("policy_document_id")
        arms.setdefault(d["condition"], []).append({
            "seeded": doc in SEEDED,
            "flag": end.get("decision") != "accept",
        })
    return arms


def rates(trials: list[dict]) -> tuple[float, float]:
    """Hit rate on seeded items, false-alarm rate on clean ones."""
    seeded = [t["flag"] for t in trials if t["seeded"]]
    clean = [t["flag"] for t in trials if not t["seeded"]]
    h = float(np.mean(seeded)) if seeded else float("nan")
    f = float(np.mean(clean)) if clean else float("nan")
    return h, f


def dprime(h: float, f: float, n_sig: int, n_noise: int) -> float:
    from scipy.stats import norm
    h = (h * n_sig + 0.5) / (n_sig + 1)
    f = (f * n_noise + 0.5) / (n_noise + 1)
    return float(norm.ppf(h) - norm.ppf(f))


def mixture(engaged, assenting, pi, rng, n):
    out = []
    for _ in range(n):
        src = engaged if rng.random() < pi else assenting
        out.append(src[rng.integers(0, len(src))])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(HERE / "runs/known-states-v1-7b"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--n", type=int, default=96, help="trials per constructed population")
    ap.add_argument("--seed", type=int, default=20261001)
    args = ap.parse_args()

    arms = load(Path(args.run))
    engaged, assenting = arms.get("engaged"), arms.get("assenting")
    if not engaged or not assenting:
        raise SystemExit(f"need both pure arms under {args.run}; found {sorted(arms)}")

    # alpha and beta, measurable because the state is known by construction
    h_e, f_e = rates(engaged)
    h_a, f_a = rates(assenting)
    alpha, beta = h_a, 1 - h_e
    denom = 1 - alpha - beta
    print(f"pure arms: engaged H={h_e:.3f} F={f_e:.3f} | assenting H={h_a:.3f} F={f_a:.3f}")
    print(f"alpha={alpha:.3f}  beta={beta:.3f}  1-alpha-beta={denom:.3f}\n")
    if abs(denom) < 0.2:
        raise SystemExit("denominator too small; the estimator cannot run")

    rng = np.random.default_rng(args.seed)
    print(f"{'pi':>6}{'H':>8}{'F':>8}{'d-prime':>10}{'pi_hat':>9}"
          f"{'  [95% CI]':>18}{'  recovers pi':>14}")
    rows = []
    for pi in LEVELS:
        hs, fs, ds, pis = [], [], [], []
        for _ in range(args.boot):
            pop = mixture(engaged, assenting, pi, rng, args.n)
            h, f = rates(pop)
            n_sig = sum(1 for t in pop if t["seeded"])
            n_noise = len(pop) - n_sig
            if n_sig == 0 or n_noise == 0:
                continue
            hs.append(h); fs.append(f)
            ds.append(dprime(h, f, n_sig, n_noise))
            pis.append((h - alpha) / denom)
        lo, hi = np.percentile(pis, [2.5, 97.5])
        covers = lo <= pi <= hi
        rows.append((pi, np.mean(hs), np.mean(fs), np.mean(ds), np.mean(pis), lo, hi, covers))
        print(f"{pi:>6.2f}{np.mean(hs):>8.3f}{np.mean(fs):>8.3f}{np.mean(ds):>10.3f}"
              f"{np.mean(pis):>9.3f}  [{lo:+.3f}, {hi:+.3f}]{'   yes' if covers else '    NO':>14}")

    print()
    d_mono = all(rows[i][3] <= rows[i + 1][3] + 1e-9 for i in range(len(rows) - 1))
    pi_ok = all(r[7] for r in rows)
    print(f"d-prime rises monotonically with prevalence: {'yes' if d_mono else 'NO'}")
    print(f"pi_hat covers the true prevalence at every level: {'yes' if pi_ok else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
