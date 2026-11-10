# source-engagement-bench

A controlled benchmark for checking whether an oversight instrument can tell an engaged review from an assenting one.

An approval record looks the same whether or not the reviewer read the evidence. Any instrument that claims to tell the two apart has to be tested against reviews whose state is already known, and no operational process supplies those. This benchmark builds them. Model reviewers check summaries of documents with planted errors. Engaged reviewers receive the source and must quote it; assenting reviewers never receive it. Each trial's state is read from the reviewer's own text, a span that only the source contains, rather than from its verdict.

The benchmark measures observable engagement with the source. It does not measure human judgment, and it has not been run with human reviewers.

## What it found

The full write-up, with the corrections, is in [RESULTS.md](RESULTS.md).

- About four engaged trials in five quoted text found only in the source (7B 78/96, 32B 79/96), and no assenting trial did (0/96 on each model).
- A randomized probe, using the Rogan-Gladen correction for an imperfect test, was calibrated on seeded documents 07 to 09 and evaluated on documents 10 to 12. On Qwen2.5-32B the estimate equaled its target at every level, with no errors in the 108 seeded-document evaluation trials. On Qwen2.5-7B it overestimated by a steady factor of about 1.36, because the 7B's miss rate on the calibration documents (5 of 19) did not carry over. Calibration did not transfer across item sets.
- Four criteria, pre-registered from the automation-complacency literature, missed partial lapses. With a quarter of reviewers never seeing the source, 9 of 10 resampled seeds (five per model) raised no alarm; with half, 7 of 10 did not.

## Layout

| Path | Contents |
|---|---|
| `corpus_hard/` | Twelve hand-written documents of fictional university policy (01 to 06 clean, 07 to 12 seeded), the answer key, and the generator |
| `known_states/` | Trial runner, arm checks, prevalence sweep, criteria sweep, manifests, and the archived runs |
| `validation/` | The transfer study: `validate.py`, manifests, evaluation corpus, archived runs, full reviewer text, and the original and corrected outputs |
| `sim/` | The trial harness and the two reviewer prompts, vendored from an MIT-licensed platform |
| `lib/verdicts.py` | The four criteria |

## Reproduce the analysis from the archived runs

No model is needed. From the repository root:

```bash
bash known_states/analyze_32b_v2.sh
uv run --no-project --python 3.12 --with numpy==2.5.4 python validation/validate.py --model 7b
uv run --no-project --python 3.12 --with numpy==2.5.4 python validation/validate.py --model 32b
```

The first command rebuilds `known_states/runs/known-states-v2-32b/` from the v1 engaged arms and the v2 assenting arms, then reruns the arm check, the sweep and the criteria over five seeds, with uv's messages going to a separate log. Its outputs should match `known_states/runs/known-states-v2-32b-outputs/`. The `validate.py` outputs should match `validation/runs/validate-*-corrected.txt`.

## Rerun the trials

Serve `qwen2.5:7b` and `qwen2.5:32b-instruct` with ollama at `http://localhost:11434/v1`, then run `known_states/run_known_states.py --manifest <manifest>` from the repository root. Manifest seeds are recorded (known states 20260703, validation 20261011).

## Known gaps

- The 7B known-states run has its trial archives, but no captured reviewer text or sweep output, and its five criteria seeds were not recorded.
- Only the ollama tags were recorded, not model digests, so a rerun may use different weights.
- Dependencies were unpinned when the trials ran; `pyproject.toml` pins the versions this analysis was checked with on 2026-11-10.
- The 7B used the first version of the assenting instruction and the 32B used the second.
- Each trial archive records the SHA-256 of the manifest it ran under. The manifests here differ from those originals: corpus paths are now relative, and description and notes fields carried over from an earlier experiment were rewritten. The recorded hashes will not match these files.
- An earlier version of the results reported the 7B overestimate as 1.4 to 1.5 times, measured against the nominal mixing share. The estimator targets the engaged share among seeded items, and against that target the factor is about 1.36. Both outputs are kept.

## Licence

MIT, in [LICENSE](LICENSE). The harness in `sim/` comes from the oversight-platform repository of hosl-research, also MIT; its notice is in [LICENSE.harness-original](LICENSE.harness-original).

## Author

Brian Moriarty, Stevens Institute of Technology. Related papers: https://hosl-research.github.io/manuscripts/
