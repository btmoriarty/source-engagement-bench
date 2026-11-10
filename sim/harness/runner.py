"""
sim/harness/runner.py

Trial runner for the sim harness.

Iterates over conditions and trials in a parsed Manifest, builds prompts from the
manifest prompt refs, dispatches each trial to an LLM client (real or mock),
emits the four telemetry events, and returns a TrialResult per trial. Replaces
the standalone mock_pilot_driver.

CONTRACT CHANGE: the runner stub typed llm_client as Callable[[str, str], str].
This widens it to Callable[[str, str], LLMResult] (see llm_client.py). The runner
reads result.text for outcome extraction and result token fields for cost.

Outcome extraction (Step 5) is delegated to sim.harness.outcomes.extract_outcome,
which parses the response per the documented agent-output contract and returns
honest nulls on an unparseable response rather than fabricating values.

Step 6: run_all honors manifest.sample.max_parallel_trials as a concurrency
ceiling (ThreadPoolExecutor) and enforces manifest.budget. A _BudgetTracker stops
dispatch when spend reaches cap_usd under the manifest halt_policy, and on_halt
decides whether already-dispatched trials finish or are cancelled. Per-trial USD
cost comes from sim.harness.pricing.compute_cost; it stays null when the client
returns no token counts (the bare mock), and a null-cost run never halts.

RECONCILIATION (2026-07-02): main carried a directly-pushed runner.py that used
the str contract and added an exponential-backoff retry wrapper, but had no
outcome extraction, cost, or budget halt. This version unifies the two: it keeps
that retry/backoff wrapper (_call_llm_with_retry, adapted to the LLMResult
contract) and keeps the LLMResult contract, outcome extraction, cost, and budget
halt from the harness stack (PRs #13/#14/#16/#17). The attention-telemetry hook
(Park, PR #21) compiles a per-trial payload and carries it on the TrialResult
(AttentionTelemetry), so it persists to the archive (#20).

INTERIM PROMPT + CORPUS WIRING (2026-07-07): run_trial accepts an optional
prompt_provider (see sim/harness/interim_prompts.py). When it is None the runner
keeps the exact placeholder behavior (registry_id strings via _build_prompts);
when it is provided, the runner resolves prompts from the prompt registry and
injects a per-trial policy_document_text from the corpus.

PERSONA ITERATION (2026-07-09): run_all iterates the manifest's personas x
conditions x trials (the validity-check 3x3). run_trial takes an optional persona
(defaults to personas[0], so single-persona callers and tests are unchanged), and
the persona is part of the trial id so archive paths stay unique across personas.
The provider resolves each persona's system prompt from the registry.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import Callable, Optional

from sim.harness.parser import ConditionModel, Manifest, PersonaRef
from sim.harness.llm_client import LLMResult
from sim.harness.outcomes import extract_outcome
from sim.harness.pricing import compute_cost
from sim.harness.telemetry import SimAttentionTelemetry
from sim.harness.models import (
    AttentionTelemetry,
    AgentRequestDispatchedEvent,
    AgentResponseReceivedEvent,
    CostRecord,
    ManifestVersion,
    ModelSnapshot,
    PromptSnapshot,
    PromptsSnapshot,
    TrialEndEvent,
    TrialInput,
    TrialResult,
    TrialStartEvent,
)

logger = logging.getLogger(__name__)

# Typed loosely on purpose: any callable matching the LLMResult-returning
# signature is accepted, so the smoke test can inject MockLLMClient.
LLMClient = Callable[[str, str], LLMResult]


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


class _BudgetTracker:
    """
    Tracks accumulated USD spend and decides when to stop dispatching trials.

    halt_policy (from the manifest budget block):
      actual_spend_reaches_cap    : dispatch while accumulated spend < cap. The
                                    trial that crosses the cap has already run, so
                                    actual spend may overshoot.
      projected_spend_reaches_cap : dispatch while accumulated + a per-trial
                                    estimate < cap. Halts one trial earlier, so it
                                    avoids overshooting. The safer default.

    A cost of None (the bare mock, or an unpriced model) counts as 0 and never
    moves the tracker, so a mock run never halts on budget.
    """

    def __init__(self, cap_usd: float, halt_policy: str) -> None:
        self.cap_usd = cap_usd
        self.halt_policy = halt_policy
        self.accumulated = 0.0
        self._n_costed = 0

    def record(self, cost: Optional[float]) -> None:
        if cost:
            self.accumulated += cost
            self._n_costed += 1

    @property
    def _per_trial_estimate(self) -> float:
        # Running mean of the costed trials so far. Zero before any cost is seen,
        # so projected behaves like actual until the first priced trial completes.
        if self._n_costed == 0:
            return 0.0
        return self.accumulated / self._n_costed

    def should_dispatch(self) -> bool:
        if self.halt_policy == "projected_spend_reaches_cap":
            return (self.accumulated + self._per_trial_estimate) < self.cap_usd
        # actual_spend_reaches_cap, and any unrecognized policy, uses actual spend.
        return self.accumulated < self.cap_usd


class TrialRunner:
    """
    Runs all trials defined in a parsed manifest.

    Parameters
    ----------
    manifest : Manifest
        Parsed manifest from ManifestParser.parse().
    llm_client : Callable[[str, str], LLMResult]
        Callable accepting (system_prompt, user_prompt) and returning an
        LLMResult. Injected so tests can pass a mock.
    prompt_provider : optional
        Prompt/corpus provider (see sim/harness/interim_prompts.py). When None
        (the default) the runner uses the placeholder registry_id strings and the
        historical TrialInput shape, so existing callers and tests are unaffected.
        When provided, the runner resolves real prompts from the registry and
        injects a per-trial policy_document_text from the corpus.
    """

    def __init__(
        self,
        manifest: Manifest,
        llm_client: LLMClient,
        prompt_provider: Optional[object] = None,
        manifest_sha256: str = "stub-replace-with-real-sha256-in-wk3",
    ) -> None:
        self.manifest = manifest
        self.llm_client = llm_client
        self.prompt_provider = prompt_provider
        self.manifest_sha256 = manifest_sha256

    def _build_prompts(self, condition: ConditionModel) -> tuple[str, str]:
        """
        Build system and user prompts from the manifest prompt refs.

        This placeholder path is used only when no prompt_provider is injected: the
        registry_id stands in as a placeholder string so the lifecycle and archive
        shape can be exercised. With a prompt_provider, prompts come from the
        registry (see sim/harness/prompt_registry.py).
        """
        prompts = self.manifest.prompts
        system_prompt = prompts.agent_system.registry_id  # placeholder (no provider)
        signal = self._time_budget_signal(condition)
        user_prompt = prompts.agent_user_template.registry_id  # placeholder (no provider)
        if signal:
            user_prompt = f"{user_prompt}\n\n{signal}"
        return system_prompt, user_prompt

    @staticmethod
    def _time_budget_signal(condition: ConditionModel) -> str | None:
        if condition.time_pressure_ms is None:
            return None
        seconds = condition.time_pressure_ms // 1000
        return f"You have {seconds} seconds to complete your review."

    @staticmethod
    def _time_budget_seconds(condition: ConditionModel) -> Optional[int]:
        if condition.time_pressure_ms is None:
            return None
        return condition.time_pressure_ms // 1000

    def _call_llm_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        trial_index: int,
        condition_name: str,
        max_retries: int = 3,
    ) -> LLMResult:
        """
        Call llm_client with exponential-backoff retry on transient errors.

        Reconciled from the retry wrapper that was on main. Retries only on
        exceptions raised by the client (1 s, 2 s, 4 s backoff). A result that
        is not an LLMResult is a contract violation, not a transient error, so
        it raises immediately without retry.
        """
        for attempt in range(max_retries):
            try:
                result = self.llm_client(system_prompt, user_prompt)
            except Exception as exc:  # noqa: BLE001
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"LLM client failed after {max_retries} attempts on trial "
                        f"{trial_index} (condition={condition_name!r}): {exc}"
                    ) from exc
                time.sleep(2 ** attempt)  # 1 s, 2 s, 4 s
                continue
            if not isinstance(result, LLMResult):
                raise RuntimeError(
                    f"LLM client returned {type(result).__name__}, expected LLMResult"
                )
            return result
        raise AssertionError("unreachable")

    def run_trial(
        self,
        trial_index: int,
        condition: ConditionModel,
        persona: Optional[PersonaRef] = None,
    ) -> TrialResult:
        """
        Run a single trial; returns a TrialResult (consumed by the archiver).

        persona defaults to personas[0] so existing single-persona callers are
        unchanged; run_all passes the iterated persona explicitly for the 3x3.
        """
        exp = self.manifest.experiment
        model = self.manifest.model
        persona = persona if persona is not None else self.manifest.personas[0]

        # Park (#21): attention telemetry tracker for this trial.
        telemetry_tracker = SimAttentionTelemetry()
        telemetry_tracker.record_trial_start()

        #added get_source
        def get_source(source_id: str, collapse: bool = False) -> str:
            if collapse:
                telemetry_tracker.record_evidence_collapsed(source_id)
            else:
                telemetry_tracker.record_evidence_expanded(source_id)
            return f"Source content for {source_id} retrieved successfully."

        
        # Step 1: trial_start
        t_trial_start = time.monotonic()
        start_event = TrialStartEvent(timestamp=_now_iso())

        # Per-trial time-budget values, computed once and reused for the prompt
        # signal and the archived TrialInput.
        signal = self._time_budget_signal(condition)
        seconds = self._time_budget_seconds(condition)

        # Step 2: build prompts and assemble the per-trial input. Two paths:
        #   - prompt_provider present: resolve the persona's prompts from the
        #     registry and inject a per-trial policy_document_text from the corpus.
        #   - prompt_provider absent: the historical placeholder path, so
        #     existing callers and tests behave exactly as before.
        if self.prompt_provider is not None:
            bundle = self.prompt_provider.build_trial(
                manifest=self.manifest,
                condition=condition,
                trial_index=trial_index,
                persona=persona,
                time_budget_signal=signal,
            )
            system_prompt = bundle.system_prompt
            user_prompt = bundle.user_prompt
            trial_input = TrialInput(
                policy_document_id=bundle.policy_document_id,
                time_budget_seconds=seconds,
                time_budget_signal=signal,
                trial_context=bundle.trial_context,
            )
        else:
            system_prompt, user_prompt = self._build_prompts(condition)
            trial_input = TrialInput(
                time_budget_seconds=seconds,
                time_budget_signal=signal,
            )

        # Step 3: agent_request_dispatched. Timestamp captured now; the
        # prompt-token count and dispatch latency are filled after the call.
        dispatch_ts = _now_iso()
        dispatch_latency_ms = int(round((time.monotonic() - t_trial_start) * 1000))

        # Step 4: call the LLM client, with backoff retry on transient errors
        # (retry wrapper reconciled from main).
        t_call_start = time.monotonic()
        import os as _os
        if _os.environ.get("HOSL_REASONING_BUDGET") and getattr(condition, "token_budget", None) is not None:
            # Reasoning-budget experiment: call A produces reasoning under the
            # per-condition token cap; call B makes the decision from whatever
            # reasoning survived, uncapped, so the decision always parses.
            if hasattr(self.llm_client, "max_tokens_override"):
                self.llm_client.max_tokens_override = condition.token_budget
            _reasoning = self._call_llm_with_retry(
                system_prompt,
                user_prompt + "\n\nThink step by step about this review. Write ONLY your reasoning; do not give a final decision yet.",
                trial_index, condition.name,
            )
            if hasattr(self.llm_client, "max_tokens_override"):
                self.llm_client.max_tokens_override = None
            result = self._call_llm_with_retry(
                system_prompt,
                user_prompt + "\n\nYour step-by-step reasoning so far:\n" + (_reasoning.text or "") + "\n\nBased on that reasoning, now give your final decision in the required format.",
                trial_index, condition.name,
            )
        else:
            result = self._call_llm_with_retry(
                system_prompt, user_prompt, trial_index, condition.name
            )
        response_latency_ms = int(round((time.monotonic() - t_call_start) * 1000))

        dispatch_event = AgentRequestDispatchedEvent(
            timestamp=dispatch_ts,
            request_tokens_prompt=result.input_tokens,
            request_dispatch_latency_ms=dispatch_latency_ms,
        )

        # Step 5: parse the response into structured outcomes. Tolerant: an
        # unparseable response yields honest nulls and parse_ok=False, never a
        # fabricated decision.
        outcome = extract_outcome(result.text)

        # Step 6: agent_response_received.
        # rationale_text_length_chars comes from the parsed outcome.
        # TODO(hassler/experiment-design): evidence_expansion_requests and
        # planning_trace_depth are interaction-model signals. They depend on an
        # experiment interaction design (how a persona requests more evidence,
        # what a planning trace looks like) that is not designed yet. Do not
        # fabricate them with sleeps or hardcoded values; leave them null until
        # the persona/experiment interaction spec defines how they are produced.
        response_event = AgentResponseReceivedEvent(
            timestamp=_now_iso(),
            response_tokens=result.output_tokens,
            response_latency_ms=response_latency_ms,
            rationale_text_length_chars=outcome.rationale_text_length_chars,
            evidence_expansion_requests=None,   # TODO(hassler/experiment-design)
            planning_trace_depth=None,          # TODO(hassler/experiment-design)
        )

        # Cost: token counts come straight from the LLMResult when present (None
        # for the mock). USD fields come from the pricing map; they stay None when
        # token counts are absent or the model is unpriced. See pricing.py.
        trial_cost = compute_cost(model.name, result)
        cost = CostRecord(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            input_cost_usd=trial_cost.input_cost_usd,
            output_cost_usd=trial_cost.output_cost_usd,
            total_cost_usd=trial_cost.total_cost_usd,
        )

        # Persona is part of the id so the 3x3 crossing produces unique trial ids
        # (and archive paths) across personas within a condition.
        trial_id = f"{exp.id}__{persona.id}__{condition.name}__{trial_index:04d}"

        # Step 8: trial_end. decision and corrections_count come from the parsed
        # outcome. total_duration_ms stays None until timing lands in Wk 3.
        end_event = TrialEndEvent(
            timestamp=_now_iso(),
            decision=outcome.decision,
            total_duration_ms=int(round((time.monotonic() - t_trial_start) * 1000)),
            corrections_count=outcome.corrections_count,
        )

        # Park (#21/#20): compile the attention payload and carry it on the
        # TrialResult so it persists to the archive (replaces the in-memory hack).
        # duration_ms is milliseconds per the #19 decision; evidence_expansion_count
        # and swift_collapse_detected stay at defaults until the interaction layer
        # calls the record_evidence_* hooks.
        _att = telemetry_tracker.compile_final_payload(
            rationale_text=result.text, hidden_trace=[]
        )
        attention = AttentionTelemetry(
            duration_ms=int(_att["duration_seconds"] * 1000),
            evidence_expansion_count=_att["evidence_expansion_count"],
            swift_collapse_detected=_att["swift_collapse_detected"],
            trace_token_length=_att["trace_token_length"],
        )

        # Step 9: assemble TrialResult
        trial_result = TrialResult(
            trial_id=trial_id,
            trial_index=trial_index,
            experiment_id=exp.id,
            experiment_lifecycle_stage=exp.lifecycle_stage,
            condition=condition.name,
            condition_label=condition.label,
            persona_id=persona.id,
            manifest_version=ManifestVersion(
                manifest_id=exp.id,
                manifest_sha256=self.manifest_sha256,
            ),
            model=ModelSnapshot(
                provider=model.provider,
                name=model.name,
                version_returned_by_api=result.model_version,
                max_tokens=model.max_tokens,
                temperature=model.temperature,
                top_p=model.top_p,
            ),
            prompts=PromptsSnapshot(
                agent_system=PromptSnapshot(
                    registry_id=self.manifest.prompts.agent_system.registry_id,
                    version=self.manifest.prompts.agent_system.version,
                ),
                agent_user_template=PromptSnapshot(
                    registry_id=self.manifest.prompts.agent_user_template.registry_id,
                    version=self.manifest.prompts.agent_user_template.version,
                ),
            ),
            input=trial_input,
            events=[start_event, dispatch_event, response_event, end_event],
            cost=cost,
            attention=attention,
            seed=self.manifest.random.seed,
        )
        return trial_result

    
    def run_all(self) -> list[TrialResult]:
        """
        Run all trials across all personas and conditions.

        Honors manifest.sample.max_parallel_trials as a concurrency ceiling and
        enforces manifest.budget: dispatch stops when the _BudgetTracker says the
        cap is reached under the manifest halt_policy, and on_halt decides whether
        already-dispatched trials finish (finish_inflight_trials) or are cancelled
        where not yet started (cancel_inflight_trials). Output order is always the
        deterministic persona-major, condition, trial-index order regardless of
        concurrency.
        """
        budget = self.manifest.budget
        tracker = _BudgetTracker(budget.cap_usd, budget.halt_policy)
        alert_threshold = budget.cap_usd * (budget.alert_at_percent / 100.0)
        alerted = False

        # Persona-major, then condition, then trial index. With a single-persona
        # manifest this is the historical order; with the 3x3 it produces every
        # persona x condition cell. Output order stays deterministic.
        n = self.manifest.sample.trials_per_condition
        flat = [
            (persona, cond, ti)
            for persona in self.manifest.personas
            for cond in self.manifest.conditions
            for ti in range(n)
        ]
        tasks = [
            (pos, persona, ti, cond)
            for pos, (persona, cond, ti) in enumerate(flat)
        ]

        max_workers = self.manifest.sample.max_parallel_trials or 1
        if max_workers < 1:
            max_workers = 1

        collected: dict[int, TrialResult] = {}

        def _maybe_alert() -> None:
            nonlocal alerted
            if (not alerted) and alert_threshold > 0 and tracker.accumulated >= alert_threshold:
                alerted = True
                logger.warning(
                    "sim budget alert: $%.4f reached %d%% of cap $%.2f",
                    tracker.accumulated,
                    budget.alert_at_percent,
                    budget.cap_usd,
                )

        # Sequential path. Deterministic and the common case under a unit cap.
        if max_workers == 1:
            for pos, persona, ti, cond in tasks:
                if not tracker.should_dispatch():
                    break
                result = self.run_trial(ti, cond, persona)
                collected[pos] = result
                tracker.record(result.cost.total_cost_usd)
                _maybe_alert()
            return [collected[p] for p in sorted(collected)]

        # Concurrent path. In-flight futures are bounded to max_workers; the
        # budget is checked before each submit; on_halt is honored once the cap
        # is reached. Note: a running thread cannot be interrupted in Python, so
        # cancel_inflight_trials cancels only futures the pool has not started.
        task_iter = iter(tasks)
        inflight: dict = {}
        halted = False

        def _fill(executor: ThreadPoolExecutor) -> None:
            nonlocal halted
            while len(inflight) < max_workers:
                if not tracker.should_dispatch():
                    halted = True
                    return
                try:
                    pos, persona, ti, cond = next(task_iter)
                except StopIteration:
                    return
                fut = executor.submit(self.run_trial, ti, cond, persona)
                inflight[fut] = pos

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            _fill(executor)
            while inflight:
                done, _pending = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    pos = inflight.pop(fut)
                    result = fut.result()
                    collected[pos] = result
                    tracker.record(result.cost.total_cost_usd)
                    _maybe_alert()
                if halted:
                    if budget.on_halt == "cancel_inflight_trials":
                        for fut in list(inflight):
                            if fut.cancel():
                                inflight.pop(fut)
                else:
                    _fill(executor)

        return [collected[p] for p in sorted(collected)]
