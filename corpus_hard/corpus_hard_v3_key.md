# validity-check-hard-v3: the error key

Written by build_corpus_v3.py. Keep it out of the corpus directory: the loader parses
every .json in there as a document.

6 clean documents, 6 seeded (50%).

Designed against the 32B, which caught 52 of 52 on v2 and v2.1 and flags about 62 percent
of faithful summaries. An error it will miss cannot be one a close reading catches, so
each of these makes a close reading agree: the wrong version is the commoner shape, the
obvious computation, the everyday sense of a defined term, the usual convention, or the
neighbouring rule.

## vch3-doc-07 (factual)

Mechanism: logical_connective: a conjunctive requirement restated as disjunctive

The source requires all three conditions together. The summary joins them with 'or', which turns a conjunction into three independent routes to eligibility. The disjunctive shape is the commoner one in policy prose, so the sentence reads naturally, and the rest of the summary is exact.

## vch3-doc-08 (factual)

Mechanism: derived_value: a figure the source never states, computed the obvious wrong way

Forty-five days appears nowhere in the source, and the arithmetic that produces it is wrong: the fifteen days run from the notice, which the office issues at some point after the window closes, not from the quarter's close. The summary presents a derived total as though the source stated it, and the derivation is the one a reader performs automatically.

## vch3-doc-09 (paraphrase_drift)

Mechanism: defined_term: a term the source defines, restated in its everyday sense

The source defines a business day and both deadlines run in them. The summary drops the qualifier and states plain days, which shortens a ten-business-day appeal window by four days or more across a holiday. The definition sits in the opening sentence, away from the deadlines it governs.

## vch3-doc-10 (factual)

Mechanism: prior_agreement: the summary states the usual convention, the source departs from it

The source explicitly refuses the senior-last convention and forbids departments from imposing it. The summary asserts the convention that holds across most of the sciences. A reader whose prior matches the summary has no reason to go back, and the refusal sits mid-paragraph rather than in a heading.

## vch3-doc-11 (factual)

Mechanism: quantifier: a requirement to satisfy each item restated as any one of them

'Each of the four' becomes 'any one of four'. The summary then keeps the falling-short-on-one clause, which reads consistently with either version and makes the weakened requirement look deliberate.

## vch3-doc-12 (omission)

Mechanism: attachment: a real exception moved onto the neighbouring rule

The non-disclosure exception belongs to syllabus publication, and the summary attaches it to the examination-schedule rule instead. Both rules and the exception all appear, so a check that asks whether each fact is present passes; only a check on which rule the exception governs fails.

## Clean documents

vch3-doc-01, vch3-doc-02, vch3-doc-03, vch3-doc-04, vch3-doc-05, vch3-doc-06
