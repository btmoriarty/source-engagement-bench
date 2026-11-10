#!/usr/bin/env python3
"""
Build `validity-check-hard-v1`, a corpus whose seeded errors need the source to catch.

Why this exists: the interim corpus pairs a two-sentence source with a summary that contradicts it
in the same breath ("seven years" above, "five years" below), so the error is visible without any
checking. The 2026-09-29 positive control confirmed the consequence: an agent instructed to skip
verification still caught every seeded error, catch rate 1.000, which pins SV3 and SV4 at
zero-width intervals no matter the sample size.

The design rules here follow from that failure. In every seeded document:

1. The source is long enough that checking costs effort, and the contradicted fact sits away from
   the opening, never adjacent to where the summary states it.
2. The wrong value is plausible from the source itself. Each seeded number appears somewhere in
   the source attached to something else, so a reader who skims for "does this number occur"
   accepts it.
3. The summary is otherwise accurate and fluent. Nothing signals the error except the conflict.
4. Clean documents carry the same furniture, several numbers, two or three named bodies, and
   nested qualifiers, so complexity does not mark a document as faulty.

Error kinds stay in the vocabulary the analysis already uses (factual, omission,
paraphrase_drift), so the scorer and the seeded_error split work unchanged.

Usage:
  python3 build_corpus.py            # writes ./corpus_hard_v1/*.json and the key beside it
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus_hard_v1"

NOTE = "Hand-authored for validity-check-hard-v1; errors require checking the source."


def doc(text_source: str, text_summary: str) -> str:
    return f"SOURCE EXCERPT:\n{text_source.strip()}\n\nAI-DRAFTED SUMMARY:\n{text_summary.strip()}"


# ---------------------------------------------------------------------------
# Clean documents. Same furniture as the seeded ones: several numbers, more than
# one approving body, and at least one exception clause.
# ---------------------------------------------------------------------------

CLEAN = [
    (
        "Records of human-subjects research are held by the sponsoring department for the "
        "duration of the study and for a defined period after closure. For studies that collect "
        "identifiable health information, the retention period is ten years from the date of "
        "closure. For all other human-subjects studies the period is seven years. Departments may "
        "not destroy records before the applicable period ends without written approval from the "
        "Institutional Review Board, and the board will not consider a destruction request while "
        "any related complaint is open. Separately, the Office of Research Integrity holds a copy "
        "of the approved protocol for three years after closure for audit purposes; that copy is "
        "not a substitute for the departmental record. Where a sponsor's agreement sets a longer "
        "period than this policy, the sponsor's period governs, and the department must record "
        "the governing period in the study file at the time the agreement is signed.",
        "Sponsoring departments keep human-subjects research records for ten years after closure "
        "when the study collected identifiable health information, and seven years otherwise. "
        "Early destruction requires written IRB approval, which will not be considered while a "
        "related complaint is open. The Office of Research Integrity separately keeps the approved "
        "protocol for three years for audit, which does not replace the departmental record. A "
        "sponsor agreement with a longer period overrides the policy and must be noted in the "
        "study file when signed.",
    ),
    (
        "Faculty may apply for a course release in any semester in which they hold an active "
        "externally funded award. A full release from one course requires that the award cover at "
        "least twenty-five percent of the applicant's academic-year salary; a release from two "
        "courses requires fifty percent. Applications go to the department chair, who forwards "
        "endorsed applications to the dean by the fifteenth of the month preceding registration. "
        "The dean decides within thirty days. A denial may be appealed once to the Provost's "
        "office, whose decision is final. Releases do not accumulate: an unused release in one "
        "semester does not carry into the next. Faculty on a phased retirement agreement are not "
        "eligible for a course release, regardless of award coverage.",
        "Faculty with an active externally funded award can apply for a course release. One course "
        "requires the award to cover at least 25 percent of academic-year salary and two courses "
        "require 50 percent. The chair endorses applications and forwards them to the dean by the "
        "fifteenth of the month before registration, and the dean decides within thirty days. A "
        "denial can be appealed once to the Provost, whose decision is final. Unused releases do "
        "not carry over, and faculty in phased retirement are ineligible whatever the coverage.",
    ),
    (
        "The university's incident-reporting standard requires that a suspected breach of "
        "restricted data be reported to the Information Security Office within twenty-four hours "
        "of discovery. The office triages every report within one business day and assigns a "
        "severity of one through four. Severity one and two incidents are escalated to the Data "
        "Governance Board, which must convene within seventy-two hours of escalation. Severity "
        "three and four incidents are handled by the Information Security Office alone and are "
        "reported to the board in a monthly summary. Regardless of severity, any incident "
        "involving student education records is also reported to the Registrar, who determines "
        "whether notification obligations apply. The twenty-four-hour clock runs from discovery, "
        "not from confirmation, and a report may not be delayed to gather further detail.",
        "A suspected breach of restricted data must reach the Information Security Office within "
        "24 hours of discovery, and the clock runs from discovery rather than confirmation, so a "
        "report cannot be held back for more detail. The office triages within one business day "
        "and assigns severity one to four. Severity one and two go to the Data Governance Board, "
        "which convenes within 72 hours; severity three and four stay with the security office and "
        "appear in a monthly summary to the board. Any incident touching student education records "
        "also goes to the Registrar, who decides whether notification is required.",
    ),
    (
        "Graduate assistantship appointments are made for one academic year and are renewable. A "
        "renewal requires that the student remain in good academic standing, defined as a grade "
        "point average of at least 3.0 and no more than one incomplete grade outstanding at the "
        "time of review. Appointments carry a tuition remission of nine credits per semester for "
        "doctoral students and six credits per semester for master's students. The stipend is set "
        "annually by the Graduate School and is not negotiable at the department level. A student "
        "who falls below good standing is given one probationary semester, during which the "
        "appointment continues at full stipend, before the appointment is ended. Time spent on an "
        "approved medical leave does not count against the probationary semester.",
        "Graduate assistantships run for an academic year and can be renewed if the student stays "
        "in good standing, meaning a GPA of 3.0 or higher and at most one outstanding incomplete "
        "at review. Tuition remission is nine credits a semester for doctoral students and six for "
        "master's students, and the Graduate School sets the stipend annually with no departmental "
        "negotiation. Falling below good standing brings one probationary semester at full stipend "
        "before the appointment ends, and approved medical leave does not count against that "
        "semester.",
    ),
    (
        "Equipment purchased on a federal award belongs to the university unless the award terms "
        "state otherwise. Items with an acquisition cost of five thousand dollars or more are "
        "tagged and inventoried by Property Management, which conducts a physical inventory every "
        "two years. Items below that threshold are expensed and are not tracked individually. When "
        "a principal investigator leaves the university, equipment may be transferred to the "
        "receiving institution with the approval of both the dean and the sponsoring agency; the "
        "agency's approval is required first. Transfers are recorded at net book value. Equipment "
        "acquired in the final ninety days of an award requires a written justification from the "
        "principal investigator explaining why the purchase could not have been made earlier.",
        "Equipment bought on a federal award is university property unless the award says "
        "otherwise. Property Management tags and inventories items costing $5,000 or more, with a "
        "physical inventory every two years; cheaper items are expensed and not tracked "
        "individually. A departing principal investigator can transfer equipment to the receiving "
        "institution with approval from both the dean and the sponsoring agency, the agency first, "
        "recorded at net book value. Purchases in the last 90 days of an award need a written "
        "justification explaining why they were not made earlier.",
    ),
    (
        "A student may request a review of a final course grade within thirty calendar days of the "
        "grade being posted. The request goes first to the instructor of record, who responds "
        "within ten business days. If the student is not satisfied, the request may be taken to "
        "the department chair within a further ten business days, and then, once, to the college's "
        "academic standing committee. The committee reviews only whether the grade was assigned "
        "according to the criteria stated in the syllabus; it does not re-evaluate the academic "
        "merit of the work. A grade may be changed only by the instructor of record, or, where the "
        "instructor is unavailable, by the chair acting on the committee's written finding.",
        "A student has thirty calendar days from posting to request review of a final grade. It "
        "goes first to the instructor of record, who answers within ten business days, then to the "
        "chair within another ten business days, and then once to the college academic standing "
        "committee. The committee only considers whether the grade followed the syllabus criteria "
        "rather than re-judging the work. Only the instructor of record can change a grade, or the "
        "chair acting on the committee's written finding if the instructor is unavailable.",
    ),
    (
        "Effort certification is required twice a year for all personnel paid from sponsored "
        "awards. The certification covers the preceding six-month period and must be completed "
        "within forty-five days of the period closing. Certifications for faculty are completed by "
        "the faculty member; certifications for students and staff may be completed by the "
        "principal investigator, who is expected to have first-hand knowledge of the work "
        "performed. A certification that is late by more than thirty days is reported to the dean, "
        "and awards with two consecutive late certifications are placed on an internal hold that "
        "suspends new spending until the record is corrected. Cost transfers made after a "
        "certification is filed require a separate justification approved by the Office of "
        "Sponsored Programs.",
        "Everyone paid from sponsored awards certifies effort twice a year, covering the previous "
        "six months, within 45 days of the period closing. Faculty certify for themselves, while "
        "the principal investigator may certify for students and staff and is expected to have "
        "first-hand knowledge of the work. Certifications more than thirty days late are reported "
        "to the dean, and two consecutive late certifications put the award on an internal hold "
        "that suspends new spending until corrected. Cost transfers after a certification need "
        "separate justification approved by the Office of Sponsored Programs.",
    ),
    (
        "Emeritus status may be conferred on a retiring faculty member who has completed at least "
        "ten years of full-time service and holds the rank of associate professor or professor at "
        "the time of retirement. Nominations originate with the department and require a majority "
        "vote of the tenured faculty, the endorsement of the dean, and approval by the Provost. "
        "Emeritus faculty retain library and email privileges, may apply for internal research "
        "funds on the same terms as active faculty, and may supervise doctoral students already "
        "under their direction, but may not take on new doctoral advisees. Office space is not "
        "guaranteed and is allocated by the department where available.",
        "A retiring faculty member with at least ten years of full-time service, holding associate "
        "professor or professor rank at retirement, may be given emeritus status. The department "
        "nominates by majority vote of tenured faculty, with the dean's endorsement and the "
        "Provost's approval. Emeritus faculty keep library and email access, can apply for "
        "internal research funds on the same terms as active faculty, and may continue supervising "
        "current doctoral students but may not take new ones. Office space depends on "
        "departmental availability.",
    ),
]

# ---------------------------------------------------------------------------
# Seeded documents. Each carries one error, the wrong value plausible from the
# source, stated away from where the source settles it.
# ---------------------------------------------------------------------------

SEEDED = [
    {
        "kind": "factual",
        "mechanism": "number transplanted from another category in the same source",
        "source": (
            "The university retains records of sponsored-project financial transactions for seven "
            "years from the date of final financial report submission. Where an award is subject "
            "to audit, the period runs from the resolution of the audit rather than from the "
            "report. Procurement records unrelated to sponsored projects are retained for five "
            "years, and general ledger records are retained permanently. Personnel records for "
            "employees paid from sponsored awards are retained for three years after separation. "
            "Departments may not destroy records covered by this policy while an audit, claim, or "
            "litigation hold is open, and the Office of Sponsored Programs must confirm in writing "
            "that no hold applies before any destruction proceeds."
        ),
        "summary": (
            "Sponsored-project financial transaction records are kept for five years from final "
            "financial report submission, or from audit resolution where the award is audited. "
            "General ledger records are kept permanently, and personnel records for staff paid "
            "from awards are kept three years after separation. Nothing covered by the policy may "
            "be destroyed while an audit, claim, or litigation hold is open, and the Office of "
            "Sponsored Programs must confirm in writing that no hold applies first."
        ),
        "error": (
            "The summary gives five years for sponsored-project financial records. The source says "
            "seven; five is the period for unrelated procurement records, which the summary drops. "
            "A reviewer who checks only that '5 years' appears in the source finds it."
        ),
    },
    {
        "kind": "factual",
        "mechanism": "approving bodies swapped between two decisions",
        "source": (
            "Requests to place restricted data on a personally owned device are reviewed by the "
            "Information Security Office, which may approve, approve with conditions, or deny. "
            "Requests to share restricted data with an external collaborator are reviewed by the "
            "Data Governance Board, which meets monthly and may not delegate the decision. Both "
            "kinds of request are logged in the same register, and the register is audited "
            "annually by Internal Audit. Where a single project needs both a device exception and "
            "an external-sharing agreement, the two reviews proceed independently and neither "
            "review may rely on the other's finding. An approval lapses after one year and must be "
            "renewed on the same terms as an original request."
        ),
        "summary": (
            "The Data Governance Board reviews requests to hold restricted data on a personally "
            "owned device and may approve, approve with conditions, or deny. The Information "
            "Security Office reviews requests to share restricted data with an external "
            "collaborator, meeting monthly and unable to delegate. Both go in one register, audited "
            "annually by Internal Audit. A project needing both runs the two reviews "
            "independently, with neither relying on the other, and approvals lapse after a year "
            "and must be renewed like a new request."
        ),
        "error": (
            "The two reviewing bodies are swapped. Both names appear in the source and in the "
            "summary, and every other detail is correct, so only matching each body to its own "
            "decision catches it."
        ),
    },
    {
        "kind": "omission",
        "mechanism": "scope qualifier dropped, turning a limited rule into a general one",
        "source": (
            "Investigators must register a clinical trial in a public registry before enrolling "
            "the first participant where the trial is funded by a federal agency or is intended to "
            "support an application to a regulatory authority. Registration is not required for "
            "trials that are neither federally funded nor regulatory in purpose, although "
            "investigators are encouraged to register them. Results must be posted within one year "
            "of the primary completion date for trials subject to the requirement. The Institutional "
            "Review Board verifies at continuing review that a registration exists where one is "
            "required, and a missing registration is grounds for suspending enrollment."
        ),
        "summary": (
            "Investigators must register a clinical trial in a public registry before the first "
            "participant is enrolled, and must post results within one year of the primary "
            "completion date. The Institutional Review Board checks at continuing review that the "
            "registration exists, and a missing registration can suspend enrollment."
        ),
        "error": (
            "The source limits the requirement to federally funded or regulatory trials and makes "
            "registration voluntary otherwise. The summary states the duty without the limit, so "
            "it reads as universal. Nothing in the summary is false on its face; the error is what "
            "it leaves out, which only the source reveals."
        ),
    },
    {
        "kind": "paraphrase_drift",
        "mechanism": "exception removed, so a conditional rule reads as absolute",
        "source": (
            "A department may not appoint an instructor of record who holds fewer than eighteen "
            "graduate credit hours in the teaching discipline. Where a candidate holds fewer than "
            "eighteen hours but has demonstrable professional qualification in the discipline, the "
            "dean may grant an exception on the written recommendation of the chair, and the "
            "exception is recorded in the accreditation file with the evidence relied on. "
            "Exceptions are granted per appointment and do not carry to a later semester. The "
            "Provost's office reports the number of exceptions granted to the accrediting body "
            "each year, and a department whose exceptions exceed ten percent of its instructional "
            "appointments is subject to review."
        ),
        "summary": (
            "An instructor of record must hold at least eighteen graduate credit hours in the "
            "teaching discipline, and a department may not appoint anyone below that line. The "
            "Provost's office reports annually to the accrediting body, and a department whose "
            "exceptions pass ten percent of instructional appointments is reviewed."
        ),
        "error": (
            "The source lets the dean grant an exception for demonstrable professional "
            "qualification; the summary states the credit-hour rule as absolute. The summary "
            "mentions exceptions in its last sentence, which makes the omission easy to read past "
            "without going back to the source."
        ),
    },
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.json"):
        f.unlink()

    key = []
    n = 0
    for src, summ in CLEAN:
        n += 1
        doc_id = f"vch1-doc-{n:02d}"
        (OUT / f"{doc_id}.json").write_text(json.dumps({
            "id": doc_id,
            "text": doc(src, summ),
            "seeded_error": False,
            "error_kind": None,
            "source_note": NOTE,
        }, indent=2) + "\n", encoding="utf-8")
        key.append((doc_id, "clean", "", ""))

    for s in SEEDED:
        n += 1
        doc_id = f"vch1-doc-{n:02d}"
        (OUT / f"{doc_id}.json").write_text(json.dumps({
            "id": doc_id,
            "text": doc(s["source"], s["summary"]),
            "seeded_error": True,
            "error_kind": s["kind"],
            "source_note": NOTE,
        }, indent=2) + "\n", encoding="utf-8")
        key.append((doc_id, s["kind"], s["mechanism"], s["error"]))

    lines = ["# validity-check-hard-v1: the error key", "",
             "Written by build_corpus.py. Keep it out of the corpus directory: the loader parses",
             "every .json in there as a document.", "",
             f"{len(CLEAN)} clean documents, {len(SEEDED)} seeded, "
             f"{len(SEEDED) / (len(CLEAN) + len(SEEDED)):.0%} seeded.", ""]
    for doc_id, kind, mech, err in key:
        if kind == "clean":
            continue
        lines += [f"## {doc_id} ({kind})", "", f"Mechanism: {mech}", "", err, ""]
    lines += ["## Clean documents", "",
              ", ".join(d for d, k, _, _ in key if k == "clean"), ""]
    (HERE / "corpus_hard_v1_key.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {n} documents to {OUT}")
    print(f"  clean {len(CLEAN)}, seeded {len(SEEDED)}")
    print(f"key written to {HERE / 'corpus_hard_v1_key.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
