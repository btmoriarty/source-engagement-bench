"""
verdicts.py
Reproduction verdicts for the four validity-check baselines (SV1 to SV4) defined
in `02_operations/scoping/sim_validity_check_design.md`, with nonparametric
bootstrap 95% confidence intervals.

Each baseline carries a pre-registered threshold on a tight-vs-unconstrained
contrast. The verdict is three-way:

  REPRODUCED      the whole 95% CI clears the threshold in the predicted direction
  NOT REPRODUCED  the whole 95% CI misses the threshold (or points the wrong way)
  INCONCLUSIVE    the CI straddles the threshold (common on a small pilot)

The design doc's reporting plan asks for the observed effect with a CI and the
outcome at the same depth for null results, which is what this module returns.

Usage
-----
    from loader import load_trial_archive
    from verdicts import run_all, format_report

    df = load_trial_archive("runs/validity-check-3x3-v0-local")
    print(format_report(run_all(df)))

or as a script:

    python verdicts.py /path/to/run_root
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_NBOOT = 5000
DEFAULT_SEED = 20260703
TIGHT = "tight"
UNCONSTRAINED = "unconstrained"


@dataclass
class Verdict:
    baseline: str
    metric: str
    estimate: float
    ci: tuple[float, float]
    threshold: float
    direction: str            # ">=" or "<="
    verdict: str
    notes: list[str] = field(default_factory=list)


def _mean(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.mean()) if len(s) else float("nan")


def _cond(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    return df[df["condition"] == condition]


def _bootstrap_ci(fn, *frames: pd.DataFrame, nboot: int, rng: np.random.Generator):
    """
    fn takes the resampled frames (in the same order) and returns a scalar.
    Returns (point_estimate, ci_low, ci_high) over `nboot` trial-level resamples,
    stratified within each frame.
    """
    est = fn(*frames)
    vals = []
    for _ in range(nboot):
        resampled = [
            f.iloc[rng.integers(0, len(f), len(f))] for f in frames
        ]
        v = fn(*resampled)
        if v == v:  # drop NaN draws
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return est, float(lo), float(hi)


def _decide(est: float, lo: float, hi: float, threshold: float, direction: str) -> str:
    if direction == ">=":
        if lo >= threshold:
            return "REPRODUCED"
        if hi < threshold:
            return "NOT REPRODUCED"
        return "INCONCLUSIVE"
    # direction "<=": a drop of at least |threshold|, i.e. value <= threshold
    if hi <= threshold:
        return "REPRODUCED"
    if lo > threshold:
        return "NOT REPRODUCED"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# The four baselines
# ---------------------------------------------------------------------------

def sv1_bainbridge(df, *, nboot=DEFAULT_NBOOT, rng=None) -> Verdict:
    """SV1: corrections drop >= 30% in tight vs unconstrained, WHILE accept rate
    stays within 10%. The paired accept-stability guard is what separates the
    Bainbridge pattern from a uniform slowdown."""
    rng = rng or np.random.default_rng(DEFAULT_SEED)
    unc, tight = _cond(df, UNCONSTRAINED), _cond(df, TIGHT)

    def pct_change(u, t):
        cu, ct = _mean(u["corrections_count"]), _mean(t["corrections_count"])
        return (ct - cu) / cu if cu else float("nan")

    def accept_gap(u, t):
        return _mean(t["accept"]) - _mean(u["accept"])

    e, lo, hi = _bootstrap_ci(pct_change, unc, tight, nboot=nboot, rng=rng)
    ga, glo, ghi = _bootstrap_ci(accept_gap, unc, tight, nboot=nboot, rng=rng)

    corr_verdict = _decide(e, lo, hi, -0.30, "<=")
    stable = abs(ga) <= 0.10 and abs(glo) <= 0.10 and abs(ghi) <= 0.10
    if corr_verdict == "REPRODUCED" and stable:
        v = "REPRODUCED"
    elif corr_verdict == "INCONCLUSIVE":
        v = "INCONCLUSIVE"
    else:
        v = "NOT REPRODUCED"

    return Verdict(
        "SV1", "corrections %change (tight vs unconstrained)",
        e, (lo, hi), -0.30, "<=", v,
        notes=[f"accept gap {ga:+.3f} [{glo:+.3f}, {ghi:+.3f}], "
               f"{'stable' if stable else 'NOT stable'} (need within 0.10)"],
    )


def sv2_parasuraman(df, *, nboot=DEFAULT_NBOOT, rng=None) -> Verdict:
    """SV2: accept-without-revise rate rises >= 20% AND rationale length drops
    >= 25%. Timing sub-criterion (response time down >= 50%) reported now that the
    attention block gives real durations."""
    rng = rng or np.random.default_rng(DEFAULT_SEED)
    unc, tight = _cond(df, UNCONSTRAINED), _cond(df, TIGHT)

    def accept_pct(u, t):
        au, at = _mean(u["accept"]), _mean(t["accept"])
        return (at - au) / au if au else float("nan")

    def rat_pct(u, t):
        ru, rt = _mean(u["rationale_text_length_chars"]), _mean(t["rationale_text_length_chars"])
        return (rt - ru) / ru if ru else float("nan")

    def dur_pct(u, t):
        du, dt = _mean(u["total_duration_ms"]), _mean(t["total_duration_ms"])
        return (dt - du) / du if du else float("nan")

    ea, loa, hia = _bootstrap_ci(accept_pct, unc, tight, nboot=nboot, rng=rng)
    er, lor, hir = _bootstrap_ci(rat_pct, unc, tight, nboot=nboot, rng=rng)
    ed, lod, hid = _bootstrap_ci(dur_pct, unc, tight, nboot=nboot, rng=rng)

    va = _decide(ea, loa, hia, 0.20, ">=")
    vr = _decide(er, lor, hir, -0.25, "<=")
    if va == "REPRODUCED" and vr == "REPRODUCED":
        v = "REPRODUCED"
    elif "INCONCLUSIVE" in (va, vr):
        v = "INCONCLUSIVE"
    else:
        v = "NOT REPRODUCED"

    return Verdict(
        "SV2", "accept-rate %rise (tight vs unconstrained)",
        ea, (loa, hia), 0.20, ">=", v,
        notes=[f"rationale %change {er:+.3f} [{lor:+.3f}, {hir:+.3f}] (need <= -0.25)",
               f"duration %change {ed:+.3f} [{lod:+.3f}, {hid:+.3f}] (need <= -0.50)"],
    )


def sv3_lee_see(df, *, nboot=DEFAULT_NBOOT, rng=None) -> Verdict:
    """SV3: the clean-vs-seeded accept-rate gap narrows >= 40% from unconstrained
    to tight (discrimination collapses under pressure)."""
    rng = rng or np.random.default_rng(DEFAULT_SEED)
    unc, tight = _cond(df, UNCONSTRAINED), _cond(df, TIGHT)

    def narrowing(u, t):
        def gap(pool):
            c = _mean(pool[pool["seeded_error"] == False]["accept"])   # noqa: E712
            s = _mean(pool[pool["seeded_error"] == True]["accept"])    # noqa: E712
            return c - s
        gu, gt = gap(u), gap(t)
        return (gu - gt) / gu if gu and gu == gu else float("nan")

    e, lo, hi = _bootstrap_ci(narrowing, unc, tight, nboot=nboot, rng=rng)
    v = _decide(e, lo, hi, 0.40, ">=")
    return Verdict("SV3", "clean-vs-seeded gap narrowing", e, (lo, hi), 0.40, ">=", v)


def sv4_schemmer(df, *, nboot=DEFAULT_NBOOT, rng=None) -> Verdict:
    """SV4: relative self-reliance drops >= 0.15. RSR = (override rate on seeded)
    minus (override rate on clean), per the design-doc appendix."""
    rng = rng or np.random.default_rng(DEFAULT_SEED)
    unc, tight = _cond(df, UNCONSTRAINED), _cond(df, TIGHT)

    def rsr(pool):
        appropriate = _mean(pool[pool["seeded_error"] == True]["override"])   # noqa: E712
        over_self = _mean(pool[pool["seeded_error"] == False]["override"])    # noqa: E712
        return appropriate - over_self

    def drop(u, t):
        return rsr(u) - rsr(t)

    e, lo, hi = _bootstrap_ci(drop, unc, tight, nboot=nboot, rng=rng)
    v = _decide(e, lo, hi, 0.15, ">=")
    return Verdict("SV4", "RSR drop (unconstrained minus tight)", e, (lo, hi), 0.15, ">=", v,
                   notes=[f"RSR unconstrained {rsr(unc):+.3f}, RSR tight {rsr(tight):+.3f}"])


def run_all(df: pd.DataFrame, *, nboot=DEFAULT_NBOOT, seed=DEFAULT_SEED) -> dict[str, Verdict]:
    rng = np.random.default_rng(seed)
    return {
        "SV1": sv1_bainbridge(df, nboot=nboot, rng=rng),
        "SV2": sv2_parasuraman(df, nboot=nboot, rng=rng),
        "SV3": sv3_lee_see(df, nboot=nboot, rng=rng),
        "SV4": sv4_schemmer(df, nboot=nboot, rng=rng),
    }


def format_report(results: dict[str, Verdict]) -> str:
    lines = ["Reproduction verdicts (tight vs unconstrained, bootstrap 95% CI)", "=" * 62]
    for key, r in results.items():
        arrow = "need <=" if r.direction == "<=" else "need >="
        lines.append(
            f"{r.baseline}  {r.metric}\n"
            f"    {r.estimate:+.3f} [{r.ci[0]:+.3f}, {r.ci[1]:+.3f}]  "
            f"({arrow} {r.threshold:+.2f})  ->  {r.verdict}"
        )
        for n in r.notes:
            lines.append(f"    {n}")
    return "\n".join(lines)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: python verdicts.py <archive_run_dir>", file=sys.stderr)
        return 2
    # Import here so the module is usable without the loader on the path.
    from loader import load_trial_archive
    df = load_trial_archive(Path(argv[0]))
    print(f"loaded {len(df)} trials from {argv[0]}")
    for c in (UNCONSTRAINED, "moderate", TIGHT):
        print(f"  {c}: n={len(_cond(df, c))}")
    print()
    print(format_report(run_all(df)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
