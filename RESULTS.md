# Results

Trials ran in 2026 on `qwen2.5:7b` and `qwen2.5:32b-instruct`, served locally with ollama. Every figure below comes from the archived runs in this repository and can be recomputed with the commands in the README.

## 1. Engagement read from the text

Validation runs: 192 trials per model, 96 engaged and 96 assenting, on evaluation documents 04 to 06 and 10 to 12. A trial counts as verified engaged when the reviewer's text contains an 8-word span that appears in the source and not in the summary.

| | 7B | 32B |
|---|---|---|
| engaged trials verified | 0.812 (78/96) | 0.823 (79/96) |
| assenting trials verified | 0.000 (0/96) | 0.000 (0/96) |
| engaged seeded trials quoting the sentence the error contradicts | 0.796 (43/54) | 0.833 (45/54) |

## 2. Does a calibrated probe transfer across documents?

The probe's estimator is the Rogan-Gladen correction for an imperfect test. Its false-alarm and miss rates come from the known-states runs on seeded documents 07 to 09, and it is evaluated on documents 10 to 12. The target is the engaged share among seeded items, after filtering the engaged pool to verified trials.

| Model | target at 0.25 / 0.5 / 0.75 / 1 | estimate | propagated interval covers |
|---|---|---|---|
| 7B | 0.275 / 0.533 / 0.775 / 1.000 | 0.380 / 0.742 / 1.069 / 1.378 | yes / yes / no / no |
| 32B | 0.267 / 0.520 / 0.765 / 1.000 | 0.267 / 0.520 / 0.765 / 1.000 | yes at every level |

- On the 7B the engaged arm missed 5 of 19 seeded errors on the calibration documents and none on the evaluation documents. The estimate overestimates by a steady factor, which is the miss-rate shift alone: 1/(1 - 0.263) = 1.36. Calibration did not transfer.
- On the 32B both error rates were zero on both halves, 0/54 missed by engaged reviewers and 0/54 flagged by assenting reviewers in evaluation, so the estimate equals its target. The task is at ceiling for this model, which makes the result uninformative about transfer.
- A percentile bootstrap adds no width when calibration rates are 0/19 and 19/19, so the 32B intervals are degenerate. A boundary-aware interval, such as Wilson or a beta posterior, was not used.

## 3. Known states and the pre-registered criteria

The known-states runs build populations at engaged shares from 0 to 1 by mixing the two arms. On the 7B, with error rates measured from the pure arms of the same run, the estimate covered the true share at every level (0.252, 0.499, 0.746 and 1.001 at 0.25, 0.5, 0.75 and 1). The first 32B run did not hold its construction under the first assenting instruction; a second instruction fixed it, and the 32B analysis uses that run (`known_states/analyze_32b_v2.sh`).

Four criteria from the automation-complacency literature (`lib/verdicts.py`) were scored against each mixture over five seeds per model.

- On the 7B neither accept-rate criterion fired at any share, including a population where no reviewer saw the source. On the 32B one of them caught that case in every seed.
- The two discriminability criteria fired on both models when no reviewer saw the source, and missed partial lapses. With a quarter of reviewers never seeing the source, an alarm fired in 1 of 5 seeds on the 7B and 0 of 5 on the 32B; with half, 3 of 5 and 0 of 5.
- The 7B criteria seeds were not recorded, so the 7B counts cannot be rerun seed for seed.

## Limits

- The reviewers are language models, not people. The benchmark says nothing about whether a model's constructed assent resembles a person's.
- Engagement here means observable engagement with the source, defined by quotation. It is not a measure of judgment.
- The criteria were written by the author from the literature, not taken from a published evaluation.
- The 7B used the first version of the assenting instruction and the 32B the second.

## Correction

An earlier analysis compared the estimate with the nominal mixing share and reported the 7B overestimate as 1.4 to 1.5 times. Filtering the engaged pool to verified trials changed its share of seeded documents (7B 50/78, 32B 48/79, against 54/96 in each assenting pool), so the correct target is the engaged share among seeded items. Both outputs are kept: `validation/runs/validate-*.txt` and `validation/runs/validate-*-corrected.txt`.
