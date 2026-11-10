#!/usr/bin/env python3
"""
Build `validity-check-hard-v3`, aimed at the model that has caught everything so far.

The 32B caught 52 of 52 seeded errors on v2 and v2.1, including the two omissions the 7B never
found. So there is no observed 32B miss to generalise from, and v3 has to be designed from what
the 32B does rather than from what it failed at.

What it does: it reads both texts closely and flags aggressively, returning a non-accept on about
62 percent of faithful summaries with rationales that misdescribe them
(`probe_v2_1_rationales.txt`). It is not skimming. So an error it will miss cannot be one that a
close reading catches; it has to be one where a close reading produces confident agreement.

Six mechanisms, one per seeded document, chosen so the summary reads as canonical:

1. `logical_connective`  the source makes eligibility conjunctive, the summary makes it
   disjunctive. Two characters of difference, opposite meaning, and the disjunctive version is the
   more common shape in policy prose.
2. `derived_value`       the stated figure is not in the source; it has to be computed from two
   facts in different paragraphs, and the wrong value is what the more obvious computation gives.
3. `defined_term`        the source defines a term in one paragraph and uses it in another; the
   summary silently substitutes the everyday meaning.
4. `prior_agreement`     the summary states the convention that holds nearly everywhere, and this
   source departs from it. A reader whose prior matches the summary has no reason to check.
5. `quantifier`          "any one of" against "each of", which changes a requirement into an
   option.
6. `attachment`          an exception real in the source, attached in the summary to the
   neighbouring rule rather than its own.

The clean documents are new and written to the same length and density, so nothing about size,
numeric load, or the presence of defined terms marks a document as seeded.

Usage:
  python3 build_corpus_v3.py        # writes ./corpus_hard_v3/*.json and the key beside it
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_corpus import doc  # noqa: E402

OUT = HERE / "corpus_hard_v3"
NOTE = "Hand-authored for validity-check-hard-v3; errors survive a close reading."

CLEAN = [
    (
        "For the purposes of this policy, a working day is any day the university's "
        "administrative offices are open, which excludes weekends, public holidays, and the "
        "winter closure period published each September. A research misconduct allegation is "
        "assessed in three stages. In the inquiry stage, which concludes within sixty working "
        "days of receipt, a panel of three decides whether the allegation has enough substance to "
        "warrant investigation. If it does, the investigation stage runs for up to one hundred and "
        "twenty working days and may be extended once by the Provost, in writing, for a further "
        "sixty. The adjudication stage follows, in which the Provost accepts, rejects, or remands "
        "the investigation's findings within thirty working days of receiving them. Throughout, "
        "the respondent may be accompanied by an adviser of their choosing, who may not be a "
        "member of the panel's department. Records of an allegation that is dismissed at inquiry "
        "are sealed after three years; records of an allegation that proceeds to investigation are "
        "retained for seven years from adjudication, whatever the outcome. Where the research was "
        "federally funded, the university notifies the funding agency at the point the "
        "investigation stage opens, not at the allegation."
    ,
        "A working day under this policy excludes weekends, public holidays, and the published "
        "winter closure. A misconduct allegation runs through inquiry, within sixty working days, "
        "where a panel of three decides whether there is enough substance to investigate; "
        "investigation, up to one hundred and twenty working days with one written extension of "
        "sixty available from the Provost; and adjudication, in which the Provost accepts, "
        "rejects, or remands the findings within thirty working days of receiving them. The "
        "respondent may bring an adviser, who may not come from the panel's department. Records "
        "are sealed after three years if the allegation is dismissed at inquiry, and kept seven "
        "years from adjudication if it goes to investigation, whatever the outcome. For federally "
        "funded research the agency is notified when the investigation stage opens rather than "
        "when the allegation is made."
    ),
    (
        "Indirect costs are recovered on sponsored awards at the university's federally negotiated "
        "rate, currently fifty-four percent of modified total direct costs for on-campus research "
        "and twenty-six percent for off-campus. An activity is off-campus when more than half of "
        "the work is performed in space not owned or leased by the university; a single award may "
        "not apply both rates. Modified total direct costs exclude equipment over five thousand "
        "dollars, tuition remission, and the portion of any subaward above the first twenty-five "
        "thousand dollars. Where a sponsor publishes a rate limit below the negotiated rate, the "
        "published limit applies and the difference is recorded as cost sharing only if the "
        "sponsor requires it. Waivers of the negotiated rate are granted by the Vice Provost for "
        "Research and require a written case showing strategic benefit to the university; the need "
        "to be competitive on price is not by itself a sufficient case. Waivers are recorded "
        "centrally and reported to the Faculty Senate annually in aggregate."
    ,
        "Indirect costs are recovered at the federally negotiated rate, fifty-four percent of "
        "modified total direct costs on campus and twenty-six percent off campus, where off campus "
        "means more than half the work happens in space the university neither owns nor leases. "
        "One award uses one rate. Modified total direct costs leave out equipment above five "
        "thousand dollars, tuition remission, and subaward amounts beyond the first twenty-five "
        "thousand. A sponsor's published rate limit below the negotiated rate governs, and the "
        "difference counts as cost sharing only where the sponsor requires it. The Vice Provost "
        "for Research grants waivers on a written case of strategic benefit, and price "
        "competitiveness alone does not qualify. Waivers are recorded centrally and reported to "
        "the Faculty Senate annually in aggregate."
    ),
    (
        "A student on academic probation is limited to twelve credit hours in the following "
        "semester and must meet an adviser before registration opens. Probation is triggered by a "
        "semester grade point average below 2.0 or by a cumulative average below 2.0; a student "
        "who triggers it on the semester average alone while holding a cumulative average of 2.5 "
        "or higher is placed on notice instead, which carries the adviser meeting but no credit "
        "limit. Probation is cleared when the cumulative average reaches 2.0 at the end of a "
        "semester in which the student completed at least nine credit hours. A second consecutive "
        "semester on probation results in academic suspension for one calendar year, and a "
        "suspended student may petition for early reinstatement only after two full semesters have "
        "passed. Grades earned elsewhere during a suspension do not transfer. Courses dropped "
        "after the published withdrawal deadline count as attempted hours for the completion "
        "requirement."
    ,
        "A student on academic probation may take twelve credit hours in the next semester and "
        "must see an adviser before registration opens. Probation follows a semester or a "
        "cumulative average below 2.0, except that a student whose semester average alone falls "
        "short while the cumulative average is 2.5 or higher goes on notice, which requires the "
        "adviser meeting without the credit limit. It clears when the cumulative average reaches "
        "2.0 at the end of a semester with at least nine completed credit hours. Two consecutive "
        "probation semesters bring a one-year suspension, with early reinstatement petitionable "
        "only after two full semesters, and grades earned elsewhere during suspension do not "
        "transfer. Courses dropped after the withdrawal deadline still count as attempted hours."
    ),
    (
        "Laboratory space is assigned to a principal investigator for the duration of an active "
        "award and reviewed every three years by the space committee, which weighs research "
        "expenditures per square foot, graduate student headcount, and the presence of shared "
        "instrumentation. A principal investigator whose expenditures fall below one hundred "
        "dollars per square foot for two consecutive review cycles may have space reassigned, "
        "though the committee may not reassign space that houses instrumentation purchased on a "
        "federal award without the agency's consent. Requests for additional space are considered "
        "annually in March and must be accompanied by a funding plan covering renovation costs, "
        "which are not borne centrally. Space vacated mid-cycle returns to the dean's pool rather "
        "than the department's. Emeritus faculty may retain bench space for two years after "
        "retirement where an active award continues, and thereafter only by the dean's annual "
        "renewal."
    ,
        "Laboratory space goes to a principal investigator for the life of an active award and is "
        "reviewed every three years against research expenditures per square foot, graduate "
        "headcount, and shared instrumentation. Below one hundred dollars per square foot for two "
        "consecutive cycles, space may be reassigned, though not where it houses instrumentation "
        "bought on a federal award unless the agency consents. Additional space is considered each "
        "March with a funding plan for renovation, which is not centrally funded. Space vacated "
        "mid-cycle goes to the dean's pool rather than the department's. Emeritus faculty keep "
        "bench space for two years after retirement while an award is active, and after that only "
        "on the dean's annual renewal."
    ),
    (
        "Export-controlled information may be stored only on university-managed systems approved "
        "for that purpose and listed in the controlled-systems register. A project becomes subject "
        "to these requirements when the award terms restrict publication or foreign-national "
        "participation, or when the technology appears on the Commerce Control List; the "
        "determination is made by the Export Compliance Officer before the award is accepted, not "
        "by the principal investigator. Foreign nationals may participate in a controlled project "
        "only under a licence or a documented exemption, and the licence covers named individuals "
        "rather than roles, so a personnel change requires an amendment. Travel to a sanctioned "
        "country with university equipment requires a separate review regardless of the project's "
        "control status. Violations are reported to the federal agency within five days of "
        "discovery by the Export Compliance Officer, and the report is made whether or not the "
        "principal investigator agrees with the finding."
    ,
        "Export-controlled information lives only on approved university-managed systems in the "
        "controlled-systems register. A project falls under the requirements when the award "
        "restricts publication or foreign-national participation, or when the technology is on the "
        "Commerce Control List, and the Export Compliance Officer makes that determination before "
        "the award is accepted rather than the principal investigator. Foreign nationals "
        "participate only under a licence or documented exemption, and because a licence names "
        "individuals rather than roles, a personnel change needs an amendment. Travelling to a "
        "sanctioned country with university equipment triggers a separate review whatever the "
        "project's status. The Export Compliance Officer reports violations to the federal agency "
        "within five days of discovery, with or without the principal investigator's agreement."
    ),
    (
        "The university's conflict-of-interest disclosure is filed annually by every investigator "
        "on a sponsored award and updated within thirty days of any new interest. A significant "
        "financial interest means anything of value received from a single entity exceeding five "
        "thousand dollars in the preceding twelve months, or any equity in a non-publicly-traded "
        "entity regardless of value. Income from seminars or teaching sponsored by a federal or "
        "state agency, or by an institution of higher education, is excluded from the threshold. "
        "The Conflict Review Committee determines whether a disclosed interest relates to the "
        "investigator's research, and only a related interest requires a management plan. "
        "Management plans are reviewed annually and made available to the sponsoring agency on "
        "request. An investigator who fails to disclose is removed from the award until the "
        "committee has reviewed the interest, and the department chair is notified in writing."
    ,
        "Every investigator on a sponsored award files a conflict-of-interest disclosure each year "
        "and updates it within thirty days of a new interest. A significant financial interest is "
        "more than five thousand dollars from a single entity over the preceding twelve months, or "
        "any equity in a non-publicly-traded entity whatever its value, excluding income from "
        "seminars or teaching sponsored by a federal or state agency or a higher-education "
        "institution. The Conflict Review Committee decides whether a disclosed interest relates "
        "to the research, and only a related interest needs a management plan. Plans are reviewed "
        "annually and shared with the sponsoring agency on request. An investigator who does not "
        "disclose is removed from the award until the committee reviews the interest, and the "
        "chair is notified in writing."
    ),
]

SEEDED = [
    {
        "kind": "factual",
        "mechanism": "logical_connective: a conjunctive requirement restated as disjunctive",
        "source": (
            "A graduate student is eligible to sit the qualifying examination when the student has "
            "completed the four core courses with no grade below B, has a cumulative average of "
            "3.3 or higher, and has an approved adviser of record on file with the Graduate "
            "School. A student who meets the coursework requirement but not the average may "
            "petition the department's graduate committee, which may permit the examination once "
            "where the shortfall is attributable to a single semester and the student has since "
            "returned above 3.3. The examination is offered twice a year, in January and in "
            "August, and a student may attempt it twice. A second failure ends the student's "
            "candidacy in the doctoral programme, though the student may apply the earned "
            "coursework to the terminal master's degree. The examination is set and graded by a "
            "committee of four, at least one of whom is from outside the student's department, and "
            "the outside member's vote is recorded separately in the file."
        ),
        "summary": (
            "A graduate student may sit the qualifying examination after completing the four core "
            "courses with no grade below B, or holding a cumulative average of 3.3 or higher, or "
            "having an approved adviser of record on file. A student short on the average may "
            "petition the graduate committee, which can permit one attempt where a single semester "
            "caused the shortfall and the student has since recovered above 3.3. The examination "
            "runs in January and August with two attempts allowed; a second failure ends doctoral "
            "candidacy while the coursework may count toward a terminal master's. A committee of "
            "four sets and grades it, at least one member from outside the department, whose vote "
            "is recorded separately."
        ),
        "error": (
            "The source requires all three conditions together. The summary joins them with 'or', "
            "which turns a conjunction into three independent routes to eligibility. The "
            "disjunctive shape is the commoner one in policy prose, so the sentence reads "
            "naturally, and the rest of the summary is exact."
        ),
    },
    {
        "kind": "factual",
        "mechanism": "derived_value: a figure the source never states, computed the obvious wrong way",
        "source": (
            "Effort on a sponsored award is certified each quarter. The certification window opens "
            "on the first working day after the quarter closes and remains open for thirty "
            "calendar days. Where a certification is not filed in that window, the Office of "
            "Sponsored Programs issues a first notice, and the investigator has a further fifteen "
            "calendar days from the notice to file. A certification still outstanding after the "
            "notice period is escalated to the dean, and the award is placed on spending hold at "
            "escalation rather than at the first notice. Separately, cost transfers affecting a "
            "certified quarter must be requested within ninety days of the original charge, and a "
            "transfer request that would alter an already-certified quarter requires the "
            "investigator to re-certify that quarter within the same fifteen-day period that "
            "applies to a first notice."
        ),
        "summary": (
            "Effort on a sponsored award is certified each quarter, in a window that opens on the "
            "first working day after the quarter closes and stays open for thirty calendar days. "
            "Where nothing is filed in that window, the Office of Sponsored Programs issues a "
            "first notice and the investigator has fifteen more calendar days, so an investigator "
            "has forty-five days from the quarter's close before the matter is escalated to the "
            "dean. The spending hold attaches at escalation rather than at the first notice. Cost "
            "transfers affecting a certified quarter must be requested within ninety days of the "
            "original charge, and a transfer that would alter an already-certified quarter "
            "requires the investigator to re-certify that quarter within the same fifteen-day "
            "period that applies to a first notice."
        ),
        "error": (
            "Forty-five days appears nowhere in the source, and the arithmetic that produces it is "
            "wrong: the fifteen days run from the notice, which the office issues at some point "
            "after the window closes, not from the quarter's close. The summary presents a derived "
            "total as though the source stated it, and the derivation is the one a reader performs "
            "automatically."
        ),
    },
    {
        "kind": "paraphrase_drift",
        "mechanism": "defined_term: a term the source defines, restated in its everyday sense",
        "source": (
            "In this procedure, a business day means a day on which the university's "
            "administrative offices are open, which excludes weekends, public holidays, and any "
            "day within the winter closure. A student who receives a sanction from the academic "
            "integrity board may appeal within ten business days of the written decision. The "
            "appeal is heard by a panel that has not previously considered the matter, and the "
            "panel issues its decision within twenty business days of receiving the appeal. A "
            "sanction of suspension or expulsion is stayed during the appeal; a sanction of a "
            "failing grade in the course is not. Where the sanction is upheld, the record is "
            "annotated and the annotation is removed three years after graduation unless the "
            "sanction was expulsion. Where the sanction is overturned, the record carries no "
            "annotation and the original decision is removed from the file."
        ),
        "summary": (
            "A student who receives a sanction from the academic integrity board may appeal within "
            "ten days of the written decision. The appeal goes to a panel that has not previously "
            "considered the matter, and that panel issues its decision within twenty days of "
            "receiving the appeal. A sanction of suspension or expulsion is stayed while the "
            "appeal is pending, while a failing grade in the course is not. Where the sanction is "
            "upheld, the record is annotated, and the annotation is removed three years after "
            "graduation unless the sanction was expulsion. Where it is overturned, the record "
            "carries no annotation and the original decision is removed from the file."
        ),
        "error": (
            "The source defines a business day and both deadlines run in them. The summary drops "
            "the qualifier and states plain days, which shortens a ten-business-day appeal window "
            "by four days or more across a holiday. The definition sits in the opening sentence, "
            "away from the deadlines it governs."
        ),
    },
    {
        "kind": "factual",
        "mechanism": "prior_agreement: the summary states the usual convention, the source departs from it",
        "source": (
            "Authorship on publications arising from university research follows the contribution "
            "standard set out below. A person qualifies as an author when they have made a "
            "substantial contribution to the design, acquisition, analysis, or interpretation of "
            "the work, and have approved the final manuscript. Acquisition of funding alone does "
            "not qualify a person for authorship, and neither does supervision of the laboratory "
            "in which the work was performed. Author order is determined by the contributing "
            "authors together, and this policy sets no default: the last position carries no "
            "special standing, and departments may not impose a convention that assigns it to the "
            "senior investigator. Disputes are mediated by the department chair and, failing that, "
            "by the Vice Provost for Research, whose determination is final and is recorded in "
            "writing to all authors."
        ),
        "summary": (
            "The policy covers publications arising from university research. A person qualifies "
            "as an author by making a substantial contribution to the design, acquisition, "
            "analysis, or interpretation of the work and by approving the final manuscript. "
            "Acquiring the funding does not by itself qualify someone, and neither does "
            "supervising the laboratory where the work was performed. Author order is settled "
            "among the contributing authors, with the senior investigator taking the last position "
            "by convention. Where the authors cannot agree, the department chair mediates, and "
            "failing that the Vice Provost for Research determines the matter. That determination "
            "is final and is recorded in writing to all authors."
        ),
        "error": (
            "The source explicitly refuses the senior-last convention and forbids departments from "
            "imposing it. The summary asserts the convention that holds across most of the "
            "sciences. A reader whose prior matches the summary has no reason to go back, and the "
            "refusal sits mid-paragraph rather than in a heading."
        ),
    },
    {
        "kind": "factual",
        "mechanism": "quantifier: a requirement to satisfy each item restated as any one of them",
        "source": (
            "A programme seeking approval for fully online delivery demonstrates readiness against "
            "each of the four readiness indicators: an instructional-design review completed "
            "within the previous two years, a student-support plan covering advising and technical "
            "help at the hours the programme runs, an assessment plan with at least one "
            "proctored or otherwise identity-verified component per course, and a faculty "
            "development record showing every instructor certified for online teaching. The "
            "readiness review is conducted by the Office of Digital Learning and forwarded to the "
            "University Curriculum Committee with a recommendation. A programme that falls short "
            "on one indicator may be approved provisionally for one year, during which the "
            "shortfall must be corrected; provisional approval is not renewable. Programmes "
            "approved before this policy took effect are reviewed at their next scheduled "
            "programme review rather than immediately."
        ),
        "summary": (
            "A programme seeking approval for fully online delivery demonstrates readiness against "
            "any one of four readiness indicators: an instructional-design review completed within "
            "the previous two years, a student-support plan covering advising and technical help "
            "at the hours the programme runs, an assessment plan with at least one proctored or "
            "otherwise identity-verified component per course, and a faculty development record "
            "showing every instructor certified for online teaching. The Office of Digital "
            "Learning conducts the readiness review and forwards it to the University Curriculum "
            "Committee with a recommendation. A programme falling short on one indicator may be "
            "approved provisionally for one year, during which the shortfall must be corrected, "
            "and provisional approval is not renewable. Programmes approved before the policy took "
            "effect are reviewed at their next scheduled programme review."
        ),
        "error": (
            "'Each of the four' becomes 'any one of four'. The summary then keeps the "
            "falling-short-on-one clause, which reads consistently with either version and makes "
            "the weakened requirement look deliberate."
        ),
    },
    {
        "kind": "omission",
        "mechanism": "attachment: a real exception moved onto the neighbouring rule",
        "source": (
            "Course syllabi are filed with the registrar by the end of the first week of classes "
            "and are published in the course catalogue system, where they remain available to "
            "students for the duration of the term. Syllabi for courses taught under a "
            "non-disclosure agreement with an industry partner are filed with the registrar but "
            "not published, and the department records the agreement's reference number in the "
            "file. Examination schedules are published in the same system at the start of the "
            "term and may not be changed after the eighth week except by the dean's approval on "
            "the written request of the instructor. Grade distributions by course are published at "
            "the end of each term without instructor names. A course with fewer than five enrolled "
            "students is omitted from the distribution report."
        ),
        "summary": (
            "Course syllabi are filed with the registrar by the end of the first week of classes "
            "and published in the course catalogue system, where they remain available to students "
            "for the duration of the term. Examination schedules are published in the same system "
            "at the start of the term and may not be changed after the eighth week except with the "
            "dean's approval on the written request of the instructor, unless the course is taught "
            "under a non-disclosure agreement with an industry partner, in which case the "
            "department records the agreement's reference number in the file. Grade distributions "
            "by course are published at the end of each term without instructor names, and a "
            "course with fewer than five enrolled students is omitted from the distribution "
            "report."
        ),
        "error": (
            "The non-disclosure exception belongs to syllabus publication, and the summary attaches "
            "it to the examination-schedule rule instead. Both rules and the exception all appear, "
            "so a check that asks whether each fact is present passes; only a check on which rule "
            "the exception governs fails."
        ),
    },
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.json"):
        f.unlink()

    key: list[tuple[str, str, str, str]] = []
    n = 0
    for src, summ in CLEAN:
        n += 1
        doc_id = f"vch3-doc-{n:02d}"
        (OUT / f"{doc_id}.json").write_text(json.dumps({
            "id": doc_id, "text": doc(src, summ), "seeded_error": False,
            "error_kind": None, "source_note": NOTE,
        }, indent=2) + "\n", encoding="utf-8")
        key.append((doc_id, "clean", "", ""))

    for s in SEEDED:
        n += 1
        doc_id = f"vch3-doc-{n:02d}"
        (OUT / f"{doc_id}.json").write_text(json.dumps({
            "id": doc_id, "text": doc(s["source"], s["summary"]), "seeded_error": True,
            "error_kind": s["kind"], "source_note": NOTE,
        }, indent=2) + "\n", encoding="utf-8")
        key.append((doc_id, s["kind"], s["mechanism"], s["error"]))

    seeded = [k for k in key if k[1] != "clean"]
    lines = [
        "# validity-check-hard-v3: the error key", "",
        "Written by build_corpus_v3.py. Keep it out of the corpus directory: the loader parses",
        "every .json in there as a document.", "",
        f"{len(CLEAN)} clean documents, {len(seeded)} seeded ({len(seeded) / len(key):.0%}).", "",
        "Designed against the 32B, which caught 52 of 52 on v2 and v2.1 and flags about 62 percent",
        "of faithful summaries. An error it will miss cannot be one a close reading catches, so",
        "each of these makes a close reading agree: the wrong version is the commoner shape, the",
        "obvious computation, the everyday sense of a defined term, the usual convention, or the",
        "neighbouring rule.", "",
    ]
    for doc_id, kind, mech, err in seeded:
        lines += [f"## {doc_id} ({kind})", "", f"Mechanism: {mech}", "", err, ""]
    lines += ["## Clean documents", "",
              ", ".join(d for d, k, _, _ in key if k == "clean"), ""]
    (HERE / "corpus_hard_v3_key.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {n} documents to {OUT}")
    print(f"  clean {len(CLEAN)}, seeded {len(seeded)}")
    print(f"key written to {HERE / 'corpus_hard_v3_key.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
