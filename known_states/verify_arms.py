#!/usr/bin/env python3
"""
Day 2: does the construction hold?

Three questions, in the order that matters. A failure on any of them stops the study rather than
licensing five prevalence levels on an arm that is not what it claims to be.

1. **Does the engaged arm respond to the probe?** The probe alters one numeral in the source only,
   so a clean document becomes one whose summary contradicts its source. An engaged reviewer must
   flag more of those than it flags unprobed clean documents. If it does not, the arm is not
   reading the source, the construction failed, and nothing downstream is worth running.

2. **Is the assenting arm actually assenting?** It receives no source, and the risk named before
   the run is that it flags items as unverifiable instead of approving them, which is abstention
   rather than assent. Its flag rate on clean documents has to be low, and the probe must make no
   difference to it, since it cannot see what was altered.

3. **What are alpha and beta?** With the state known by construction they are measurable rather
   than assumed, which is the whole argument for building the populations:

       alpha = P(flag a seeded document | S = 0), from the assenting arm
       beta  = P(miss a seeded document | S = 1), from the engaged arm

   Proposition 2 needs alpha + beta != 1, and the estimator's precision degrades as that sum
   approaches 1, so these two numbers decide whether the study's third analysis can run at all.

Usage:
  uv run --python 3.12 --with numpy --with pandas --with scipy python verify_arms.py [--run <dir>]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
SEEDED = {f"vch3-doc-{i:02d}" for i in range(7, 13)}


def load(run_dir: Path) -> list[dict]:
    rows = []
    for f in run_dir.glob("*/*.json"):
        d = json.loads(f.read_text())
        end = next(e for e in d["events"] if e["event_type"] == "trial_end")
        doc = d["input"].get("policy_document_id")
        rows.append({
            "condition": d["condition"],
            "doc": doc,
            "seeded": doc in SEEDED,
            "decision": end.get("decision"),
            "flag": end.get("decision") != "accept",
        })
    return rows


def rate(rows, **sel) -> tuple[float, int, int]:
    sub = [r for r in rows if all(r[k] == v for k, v in sel.items())]
    if not sub:
        return float("nan"), 0, 0
    k = sum(1 for r in sub if r["flag"])
    return k / len(sub), k, len(sub)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(HERE / "runs/known-states-v1-7b"))
    args = ap.parse_args()
    run = Path(args.run)
    rows = load(run)
    if not rows:
        raise SystemExit(f"no trials under {run}")

    print(f"{len(rows)} trials from {run.name}\n")
    print(f"{'arm':<18}{'cell':<10}{'flag rate':>12}{'  (k/n)'}")
    for cond in ("engaged", "engaged_probe", "assenting", "assenting_probe"):
        for seeded in (False, True):
            r, k, n = rate(rows, condition=cond, seeded=seeded)
            if n:
                print(f"{cond:<18}{'seeded' if seeded else 'clean':<10}{r:>12.3f}  ({k}/{n})")
    print()

    # 1. the engaged arm must respond to the probe, on clean documents
    base, bk, bn = rate(rows, condition="engaged", seeded=False)
    probed, pk, pn = rate(rows, condition="engaged_probe", seeded=False)
    lift = probed - base
    print(f"1. engaged responds to the probe: clean flag rate {base:.3f} ({bk}/{bn}) "
          f"-> {probed:.3f} ({pk}/{pn}), lift {lift:+.3f}")
    responds = lift > 0.15
    print(f"   {'PASS' if responds else 'FAIL'}: the arm "
          f"{'reads the source' if responds else 'does not appear to read the source'}\n")

    # 2. the assenting arm must assent, and must not respond
    a_clean, ak, an = rate(rows, condition="assenting", seeded=False)
    a_probe, apk, apn = rate(rows, condition="assenting_probe", seeded=False)
    assents = a_clean < 0.25
    # The probed and unprobed prompts are identical for this arm by construction, so any
    # difference is sampling noise unless it is large enough to indicate otherwise. An absolute
    # threshold on n of this size tests nothing; Fisher's exact test does.
    _, p_blind = fisher_exact([[ak, an - ak], [apk, apn - apk]])
    blind = p_blind > 0.05
    print(f"2. assenting assents: clean flag rate {a_clean:.3f} ({ak}/{an}) "
          f"-> {'PASS' if assents else 'FAIL, it is abstaining rather than assenting'}")
    print(f"   and cannot see the probe: {a_clean:.3f} vs {a_probe:.3f} ({apk}/{apn}), "
          f"Fisher p = {p_blind:.3f} "
          f"-> {'PASS' if blind else 'FAIL, something is leaking'}\n")

    # 3. alpha and beta, measurable because the state is known
    alpha, alk, aln = rate(rows, condition="assenting", seeded=True)
    beta_catch, bck, bcn = rate(rows, condition="engaged", seeded=True)
    beta = 1 - beta_catch
    print(f"3. alpha = P(flag seeded | S=0) = {alpha:.3f} ({alk}/{aln})")
    print(f"   beta  = P(miss seeded | S=1) = {beta:.3f} ({bcn - bck}/{bcn})")
    denom = 1 - alpha - beta
    print(f"   1 - alpha - beta = {denom:.3f}")
    usable = abs(denom) > 0.2
    print(f"   {'PASS' if usable else 'FAIL'}: the Proposition 2 estimator is "
          f"{'usable' if usable else 'too ill-conditioned to run'}\n")

    ok = responds and assents and blind and usable
    print("VERDICT:", "construction holds, proceed to the prevalence sweep" if ok
          else "construction does not hold, stop and fix the arm that failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
