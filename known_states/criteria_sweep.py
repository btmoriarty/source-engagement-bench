#!/usr/bin/env python3
"""
What the four pre-registered criteria conclude about known states.

`sweep.py` showed that signal detection tracks prevalence and that the Proposition 2 estimator
recovers it. That is only half the claim. The other half is that the existing evaluation reaches
the wrong conclusion on the same populations, and this runs it to find out.

For each prevalence level, a mixture population is built by resampling the two pure arms and
scored by `verdicts.py` unchanged, against the fully engaged population as the baseline. In the
scorer's vocabulary the mixture takes the `tight` slot and the engaged population the
`unconstrained` one, which is the analogue of a team whose oversight has partly lapsed measured
against one whose has not.

The question each row answers: at this prevalence, does the original check report that something
is wrong?

Usage:
  uv run --python 3.12 --with numpy --with pandas python criteria_sweep.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LIB = Path(__file__).resolve().parents[1] / "lib"  # vendored verdicts.py
sys.path.insert(0, str(LIB))
import verdicts  # noqa: E402

SEEDED = {f"vch3-doc-{i:02d}" for i in range(7, 13)}
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load(run_dir: Path) -> dict[str, pd.DataFrame]:
    rows: dict[str, list[dict]] = {}
    for f in run_dir.glob("*/*.json"):
        d = json.loads(f.read_text())
        if d["condition"].endswith("_probe"):
            continue
        end = next(e for e in d["events"] if e["event_type"] == "trial_end")
        resp = next(e for e in d["events"] if e["event_type"] == "agent_response_received")
        doc = d["input"].get("policy_document_id")
        accept = 1.0 if end.get("decision") == "accept" else 0.0
        rows.setdefault(d["condition"], []).append({
            "condition": d["condition"],
            "seeded_error": doc in SEEDED,
            "decision": end.get("decision"),
            "accept": accept,
            "override": 1.0 - accept,
            "corrections_count": float(end.get("corrections_count") or 0),
            "rationale_text_length_chars": float(resp.get("rationale_text_length_chars") or 0),
            "total_duration_ms": float(d["attention"].get("duration_ms") or 0),
        })
    return {k: pd.DataFrame(v) for k, v in rows.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(HERE / "runs/known-states-v1-7b"))
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20261001)
    args = ap.parse_args()

    arms = load(Path(args.run))
    engaged, assenting = arms.get("engaged"), arms.get("assenting")
    if engaged is None or assenting is None:
        raise SystemExit(f"need both pure arms; found {sorted(arms)}")

    rng = np.random.default_rng(args.seed)
    print("The paper's four criteria, mixture against the fully engaged population.\n")
    print(f"{'pi':>6}  {'SV1':<16}{'SV2':<16}{'SV3':<16}{'SV4':<16}{'any alarm?':>12}")

    for pi in LEVELS:
        take = rng.random(args.n) < pi
        pool = pd.concat([
            engaged.iloc[rng.integers(0, len(engaged), int(take.sum()))],
            assenting.iloc[rng.integers(0, len(assenting), int((~take).sum()))],
        ])
        pool = pool.assign(condition="tight")
        base = engaged.iloc[rng.integers(0, len(engaged), args.n)].assign(
            condition="unconstrained")
        df = pd.concat([base, pool], ignore_index=True)

        res = verdicts.run_all(df)
        verds = {k: v.verdict for k, v in res.items()}
        alarm = any(v == "REPRODUCED" for v in verds.values())
        print(f"{pi:>6.2f}  " + "".join(f"{verds[k]:<16}" for k in ("SV1", "SV2", "SV3", "SV4"))
              + f"{'yes' if alarm else 'no':>12}")

    print("\nA row reading 'no' is the original check failing to report a population in which "
          f"{100 - int(LEVELS[0] * 100)}% to {100 - int(LEVELS[-2] * 100)}% of the reviewers "
          "never saw the source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
