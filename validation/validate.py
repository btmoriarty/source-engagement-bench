#!/usr/bin/env python3
"""
The validation study, analysis only.

Calibration: alpha and beta from the earlier known-states runs, seeded calibration documents
(07 to 09) only. Evaluation: the new validation runs, on documents 04 to 06 and 10 to 12, with
every response text captured. A trial is verified engaged when its response contains an
eight-word span found in the source excerpt and not in the summary; on point when that span comes
from the key sentence the seeded error contradicts.

Usage:
  uv run --python 3.12 --with numpy python validate.py --model 7b
  uv run --python 3.12 --with numpy python validate.py --model 32b
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np

REV = Path(__file__).resolve().parents[1]  # repository root
CORPUS = REV / "corpus_hard" / "corpus_hard_v3"
CAL_RUN = {"7b": REV / "known_states/runs/known-states-v1-7b",
           "32b": REV / "known_states/runs/known-states-v2-32b"}
EVAL_RUN = {"7b": REV / "validation/runs/validation-v1-7b",
            "32b": REV / "validation/runs/validation-v1-32b"}
RESPONSES = {"7b": REV / "validation/runs/validation-v1-7b-responses.jsonl",
             "32b": REV / "validation/runs/validation-v1-32b-responses.jsonl"}

SEEDED = {f"vch3-doc-{i:02d}" for i in range(7, 13)}
CAL_DOCS = {f"vch3-doc-{i:02d}" for i in (1, 2, 3, 7, 8, 9)}
EVAL_DOCS = {f"vch3-doc-{i:02d}" for i in (4, 5, 6, 10, 11, 12)}
# Fixed before any response existed.
KEY_OPENINGS = {
    "vch3-doc-10": "Author order is determined by the contributing authors together, and this policy sets no default",
    "vch3-doc-11": "A programme seeking approval for fully online delivery demonstrates readiness against each of the four readiness indicators",
    "vch3-doc-12": "Syllabi for courses taught under a non-disclosure agreement with an industry partner are filed with the registrar but not published",
}
N = 8
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).split()


def grams(ws: list[str]) -> set[tuple[str, ...]]:
    return {tuple(ws[i:i + N]) for i in range(len(ws) - N + 1)}


def doc_parts(doc: str) -> tuple[str, str]:
    t = json.loads((CORPUS / f"{doc}.json").read_text())["text"]
    i, j = t.find("SOURCE EXCERPT:"), t.find("AI-DRAFTED SUMMARY:")
    return t[i + len("SOURCE EXCERPT:"):j], t[j + len("AI-DRAFTED SUMMARY:"):]


def key_sentence(doc: str) -> str:
    source, _ = doc_parts(doc)
    flat = " ".join(source.split())
    start = flat.find(KEY_OPENINGS[doc])
    assert start >= 0, f"key sentence not found in {doc}"
    end = flat.find(". ", start)
    return flat[start:] if end < 0 else flat[start:end + 1]


def label(doc: str, text: str) -> tuple[bool, bool | None]:
    source, summary = doc_parts(doc)
    only = grams(words(source)) - grams(words(summary))
    resp = grams(words(text))
    verified = bool(resp & only)
    on_point = None
    if doc in KEY_OPENINGS:
        on_point = bool(resp & (grams(words(key_sentence(doc))) - grams(words(summary))))
    return verified, on_point


def archives(run: Path, docs: set[str]) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for f in run.glob("*/*.json"):
        d = json.loads(f.read_text())
        if d["condition"].endswith("_probe"):
            continue
        doc = d["input"].get("policy_document_id")
        if doc not in docs:
            continue
        end = next(e for e in d["events"] if e["event_type"] == "trial_end")
        arms.setdefault(d["condition"], []).append({
            "doc": doc, "seeded": doc in SEEDED, "flag": end.get("decision") != "accept",
            "trial_index": d["trial_index"]})
    return arms


def attach_text(arms: dict[str, list[dict]], path: Path) -> int:
    text = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        text[(r["condition"], r["trial_index"])] = r["text"]  # last response per trial
    missing = 0
    for cond, trials in arms.items():
        for t in trials:
            tx = text.get((cond, t["trial_index"]))
            if tx is None:
                missing += 1
                t["verified"], t["on_point"] = False, None
            else:
                t["verified"], t["on_point"] = label(t["doc"], tx)
    return missing


def _mix(s1, s0, pi, rng, n):
    out = []
    for _ in range(n):
        src = s1 if rng.random() < pi else s0
        out.append(src[rng.integers(0, len(src))])
    return out


def rate(xs) -> float:
    return float(np.mean(xs)) if len(xs) else float("nan")


def sweep(s1, s0, cal_a, cal_e, rng, boot, n, propagate: bool):
    """The estimator targets the verified-engaged share among probed (seeded) items, which is not
    the nominal mixing share when the two pools hold different proportions of seeded trials.
    Corrected 2026-10-01 after the comparison against the nominal share was found to be wrong;
    both targets are reported."""
    alpha0 = rate([t["flag"] for t in cal_a])
    beta0 = 1 - rate([t["flag"] for t in cal_e])
    rows = []
    for pi in LEVELS:
        est, tgt = [], []
        for _ in range(boot):
            if propagate:
                a = [cal_a[i]["flag"] for i in rng.integers(0, len(cal_a), len(cal_a))]
                e = [cal_e[i]["flag"] for i in rng.integers(0, len(cal_e), len(cal_e))]
                alpha, beta = rate(a), 1 - rate(e)
            else:
                alpha, beta = alpha0, beta0
            pop = _mix(s1, s0, pi, rng, n)
            p = rate([t["flag"] for t in pop if t["seeded"]])
            den = 1 - alpha - beta
            if np.isnan(p) or abs(den) < 1e-9:
                continue
            est.append((p - alpha) / den)
            tgt.append(rate([bool(t.get("verified")) for t in pop if t["seeded"]]))
        est = np.array(est)
        lo, hi = np.percentile(est, [2.5, 97.5])
        target = float(np.mean(tgt))
        rows.append((pi, float(est.mean()), float(lo), float(hi), lo <= target <= hi, target,
                     lo <= pi <= hi))
    return alpha0, beta0, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("7b", "32b"), required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20261011)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if not CAL_RUN[args.model].is_dir():
        raise SystemExit(f"{CAL_RUN[args.model]} is missing; for the 32B, run "
                         "known_states/analyze_32b_v2.sh first to rebuild it")
    cal = archives(CAL_RUN[args.model], CAL_DOCS)
    cal_a = [t for t in cal["assenting"] if t["seeded"]]
    cal_e = [t for t in cal["engaged"] if t["seeded"]]
    ev = archives(EVAL_RUN[args.model], EVAL_DOCS)
    missing = attach_text(ev, RESPONSES[args.model])
    eng, asn = ev["engaged"], ev["assenting"]

    print(f"model {args.model}; calibration {CAL_RUN[args.model].name}, seeded docs 07-09: "
          f"{len(cal_a)} assenting, {len(cal_e)} engaged")
    print(f"evaluation {EVAL_RUN[args.model].name}: {len(eng)} engaged, {len(asn)} assenting; "
          f"{missing} trials without captured text\n")

    v_e = rate([t["verified"] for t in eng])
    v_a = rate([t["verified"] for t in asn])
    seeded_e = [t for t in eng if t["seeded"]]
    op = rate([t["on_point"] for t in seeded_e if t["on_point"] is not None])
    print("1. engagement established from the text")
    print(f"   engaged arm verified:   {v_e:.3f} ({sum(t['verified'] for t in eng)}/{len(eng)})")
    print(f"   assenting arm verified: {v_a:.3f} ({sum(t['verified'] for t in asn)}/{len(asn)})")
    print(f"   engaged seeded on point: {op:.3f} ({sum(bool(t['on_point']) for t in seeded_e)}/{len(seeded_e)})")
    ver = [t for t in eng if t["verified"]]
    unver = [t for t in eng if not t["verified"]]
    print(f"   hit rate, verified engaged:   {rate([t['flag'] for t in ver if t['seeded']]):.3f}")
    print(f"   hit rate, unverified engaged: {rate([t['flag'] for t in unver if t['seeded']]):.3f} "
          f"(n seeded = {sum(t['seeded'] for t in unver)})")
    p1 = v_e >= 0.80 and v_a < 0.05
    print(f"   prediction 1 (engaged >= 0.80 verified, assenting < 0.05): {'HELD' if p1 else 'FAILED'}\n")

    alpha, beta, prop = sweep(ver, asn, cal_a, cal_e, rng, args.boot, args.n, propagate=True)
    _, _, fixed = sweep(ver, asn, cal_a, cal_e, rng, args.boot, args.n, propagate=False)
    print(f"2. populations of verified-engaged and assenting trials; calibration alpha {alpha:.3f}, "
          f"beta {beta:.3f}")
    print("   target = verified-engaged share among seeded items in the population")
    print(f"   {'pi':>5} {'target':>7} {'pi_hat':>8} {'propagated 2.5-97.5':>22} {'covers':>7} "
          f"{'fixed 2.5-97.5':>18} {'covers':>7} {'covers nominal':>15}")
    for (pi, m, lo, hi, c, tg, cn), (_, _, flo, fhi, fc, _, _) in zip(prop, fixed):
        print(f"   {pi:5.2f} {tg:7.3f} {m:8.3f}   [{lo:+.3f}, {hi:+.3f}] {('yes' if c else 'no'):>7}"
              f"   [{flo:+.3f}, {fhi:+.3f}] {('yes' if fc else 'no'):>7} {('yes' if cn else 'no'):>15}")
    p2 = all(r[4] for r in prop)
    p3 = all((r[3] - r[2]) > (f[3] - f[2]) for r, f in zip(prop, fixed) if r[0] >= 0.5)
    print(f"   prediction 2 (propagated interval covers pi at every level): {'HELD' if p2 else 'FAILED'}")
    print(f"   prediction 3 (propagated wider than fixed at pi >= 0.5): {'HELD' if p3 else 'FAILED'}\n")

    print("3. secondary: populations built by arm label, verified or not")
    _, _, arm = sweep(eng, asn, cal_a, cal_e, rng, args.boot, args.n, propagate=True)
    print("   verified share = verified-engaged share among seeded items in the population")
    print(f"   {'pi arm':>6} {'verified share':>15} {'pi_hat':>8} {'2.5-97.5':>20}")
    for pi, m, lo, hi, _, tg, _ in arm:
        print(f"   {pi:6.2f} {tg:15.3f} {m:8.3f}   [{lo:+.3f}, {hi:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
