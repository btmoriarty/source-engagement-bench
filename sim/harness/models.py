"""
sim/harness/models.py

Shared data models for the sim harness.

Responsibilities:
    - Define TrialResult: the output of a single trial run
    - Define TelemetryEvent: one event emitted by the runner during a trial
    - Define CostRecord: cost breakdown for a single trial

These models are the contract between the runner and the archiver.
The runner produces them; the archiver serializes them to JSON.
The JSON shape mirrors trial_archive_stub.json exactly.

Out of scope:
    - Serialization logic (archiver handles this)
    - Validation logic (fields are typed but no validators yet)
    - Cost calculation (runner fills cost fields from API response in Wk 3)

Dependencies:
    - pydantic v2
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, Field


__all__ = [
    "TrialStartEvent",
    "AgentRequestDispatchedEvent",
    "AgentResponseReceivedEvent",
    "TrialEndEvent",
    "TelemetryEvent",
    "CostRecord",
    "AttentionTelemetry",
    "ManifestVersion",
    "ModelSnapshot",
    "PromptSnapshot",
    "PromptsSnapshot",
    "TrialInput",
    "TrialResult",
]


# ---------------------------------------------------------------------------
# Telemetry event models
# One TelemetryEvent per hook point in the runner.
# Field names mirror the events array in trial_archive_stub.json.
# ---------------------------------------------------------------------------

class TrialStartEvent(BaseModel):
    """
    Emitted by the runner at the start of a trial.

    Fields
    ------
    event_type : Literal["trial_start"]
        Always "trial_start". Used as the discriminator key.
    timestamp : str
        ISO 8601 datetime string with microseconds and UTC offset.
    """
    event_type: Literal["trial_start"] = "trial_start"
    timestamp: str


class AgentRequestDispatchedEvent(BaseModel):
    """
    Emitted by the runner immediately before the LLM client is called.

    Fields
    ------
    event_type : Literal["agent_request_dispatched"]
        Always "agent_request_dispatched". Used as the discriminator key.
    timestamp : str
        ISO 8601 datetime string.
    request_tokens_prompt : Optional[int]
        Token count of the prompt sent to the model.
        None until Wk 3 when the real API returns token counts.
    request_dispatch_latency_ms : Optional[int]
        Time in ms between trial_start and this event.
        None until Wk 3.
    """
    event_type: Literal["agent_request_dispatched"] = "agent_request_dispatched"
    timestamp: str
    request_tokens_prompt: Optional[int] = None
    request_dispatch_latency_ms: Optional[int] = None


class AgentResponseReceivedEvent(BaseModel):
    """
    Emitted by the runner immediately after the LLM client returns.

    Fields
    ------
    event_type : Literal["agent_response_received"]
        Always "agent_response_received". Used as the discriminator key.
    timestamp : str
        ISO 8601 datetime string.
    response_tokens : Optional[int]
        Token count of the model's response. None until Wk 3.
    response_latency_ms : Optional[int]
        Time in ms between request dispatched and response received.
        None until Wk 3.
    evidence_expansion_requests : Optional[int]
        Count of evidence expansion requests in the response.
        None until outcome extraction is implemented in Wk 3.
    rationale_text_length_chars : Optional[int]
        Character count of the rationale text in the response.
        None until Wk 3.
    planning_trace_depth : Optional[int]
        Depth of the planning trace if present. None until Wk 3.
    """
    event_type: Literal["agent_response_received"] = "agent_response_received"
    timestamp: str
    response_tokens: Optional[int] = None
    response_latency_ms: Optional[int] = None
    evidence_expansion_requests: Optional[int] = None
    rationale_text_length_chars: Optional[int] = None
    planning_trace_depth: Optional[int] = None


class TrialEndEvent(BaseModel):
    """
    Emitted by the runner at the end of a trial, after the response
    is received and the TrialResult is assembled.

    Fields
    ------
    event_type : Literal["trial_end"]
        Always "trial_end". Used as the discriminator key.
    timestamp : str
        ISO 8601 datetime string.
    decision : Optional[str]
        The agent's decision extracted from the response.
        One of: accept, reject, revise. None until Wk 3.
    total_duration_ms : Optional[int]
        Total trial duration in ms from trial_start to trial_end.
        None until Wk 3.
    corrections_count : Optional[int]
        Count of corrections in the response. None until Wk 3.
    """
    event_type: Literal["trial_end"] = "trial_end"
    timestamp: str
    decision: Optional[str] = None
    total_duration_ms: Optional[int] = None
    corrections_count: Optional[int] = None


# Discriminated union over event_type so pydantic deserializes the correct
# subclass when loading a TrialResult from JSON.
TelemetryEvent = Annotated[
    Union[
        TrialStartEvent,
        AgentRequestDispatchedEvent,
        AgentResponseReceivedEvent,
        TrialEndEvent,
    ],
    Field(discriminator="event_type"),
]


# ---------------------------------------------------------------------------
# Cost record
# Filled from the API response in Wk 3. All fields None for mock LLM runs.
# ---------------------------------------------------------------------------

class CostRecord(BaseModel):
    """
    Per-trial cost breakdown.

    All fields are None when the mock LLM is in use.
    Filled from the Anthropic API response in Wk 3.
    The archiver writes this block as-is into the trial JSON.

    Fields mirror the cost block in trial_archive_stub.json.
    """
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    input_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    total_cost_usd: Optional[float] = None


# ---------------------------------------------------------------------------
# Attention telemetry
# The sim attention/engagement signals (Park, PR #21), persisted per trial.
# ---------------------------------------------------------------------------

class AttentionTelemetry(BaseModel):
    """
    Attention/engagement signals from the sim, compiled per trial by
    SimAttentionTelemetry (see telemetry.py) and carried on the TrialResult so
    they persist to the archive.

    evidence_expansion_count and swift_collapse_detected hold placeholder
    defaults (0 / False) until the experiment interaction layer calls the
    record_evidence_* hooks; read them as "not yet instrumented," not measured
    zeros. trace_token_length is the sim-only planning-trace token length.
    duration_ms is the tracker's wall-clock span for the trial, in milliseconds.
    """
    duration_ms: Optional[int] = None
    evidence_expansion_count: int = 0
    swift_collapse_detected: bool = False
    trace_token_length: int = 0


# ---------------------------------------------------------------------------
# Manifest provenance
# Recorded in every trial archive so the archive is self-contained.
# ---------------------------------------------------------------------------

class ManifestVersion(BaseModel):
    """
    Provenance record identifying which manifest produced this trial.

    Fields
    ------
    manifest_id : str
        The experiment.id from the manifest.
    manifest_sha256 : str
        SHA-256 hash of the manifest file at run time.
        Stub uses a placeholder string; real hash computed in Wk 3.
    """
    manifest_id: str
    manifest_sha256: str


# ---------------------------------------------------------------------------
# Model snapshot
# Records the exact model configuration used for this trial.
# ---------------------------------------------------------------------------

class ModelSnapshot(BaseModel):
    """
    Snapshot of the model configuration used for this trial.
    Copied from the manifest at run time so the archive is self-contained.

    Fields mirror the model block in trial_archive_stub.json.
    """
    provider: str
    name: str
    version_returned_by_api: Optional[str] = None   # None for mock LLM
    max_tokens: int
    temperature: float
    top_p: float


# ---------------------------------------------------------------------------
# Prompt snapshot
# Records which prompt registry entries were used for this trial.
# ---------------------------------------------------------------------------

class PromptSnapshot(BaseModel):
    """Registry id and version for a single prompt used in this trial."""
    registry_id: str
    version: str


class PromptsSnapshot(BaseModel):
    """Snapshot of both prompt references used for this trial."""
    agent_system: PromptSnapshot
    agent_user_template: PromptSnapshot


# ---------------------------------------------------------------------------
# Trial input
# Records the per-trial input variables filled into the prompt template.
# ---------------------------------------------------------------------------

class TrialInput(BaseModel):
    """
    Per-trial input variables.

    Fields are the runtime values substituted into the prompt template.
    The field list here matches the required_variables declared in the
    manifest's prompts.agent_user_template block.

    Additional input fields may be added as new prompt templates are
    introduced. This covers the faculty-reviewer-baseline template.
    """
    policy_document_id: Optional[str] = None
    time_budget_seconds: Optional[int] = None       # None for baseline condition
    time_budget_signal: Optional[str] = None        # e.g. "You have 60 seconds to complete your review."
    trial_context: Optional[str] = None             # required_variable in agent_user_template


# ---------------------------------------------------------------------------
# TrialResult
# The primary output of TrialRunner.run_trial().
# The archiver serializes this to one JSON file per trial.
# ---------------------------------------------------------------------------

class TrialResult(BaseModel):
    """
    Complete record of a single trial run.

    Produced by TrialRunner.run_trial(). Consumed by the archiver,
    which serializes it to JSON at the path defined in the manifest's
    archive.base_path template.

    The JSON shape mirrors trial_archive_stub.json exactly.
    Field names and nesting are stable; adding fields is non-breaking,
    removing or renaming fields bumps archive_schema_version.

    Fields
    ------
    trial_id : str
        Stable unique identifier for this trial.
        Format: {experiment_id}__{condition_name}__{trial_index:04d}
        Example: time-budget-baseline-pilot__moderate__0042
    trial_index : int
        Zero-based index of this trial within its condition.
    experiment_id : str
        Copied from manifest.experiment.id.
    experiment_lifecycle_stage : str
        Copied from manifest.experiment.lifecycle_stage.
        One of: pilot, calibration, full_run, replication.
        Lets analysis queries filter pilot archives from full runs
        without re-reading the manifest.
    condition : str
        Name of the condition this trial ran under.
    condition_label : Optional[str]
        Human-readable label for the condition, copied from
        manifest.conditions[n].label.
        Example: "Moderate time pressure"
        None if the manifest condition has no label field.
    persona_id : str
        ID of the persona used for this trial.
    manifest_version : ManifestVersion
        Provenance of the manifest that produced this trial.
    model : ModelSnapshot
        Model configuration used for this trial.
    prompts : PromptsSnapshot
        Prompt registry references used for this trial.
    input : TrialInput
        Per-trial input variables filled into the prompt template.
    events : list[TelemetryEvent]
        Ordered list of telemetry events emitted by the runner.
        Always four events: trial_start, agent_request_dispatched,
        agent_response_received, trial_end.
    cost : CostRecord
        Per-trial cost breakdown. All None for mock LLM runs.
    attention : Optional[AttentionTelemetry]
        Sim attention/engagement signals (Park, PR #21). None when no tracker
        ran; otherwise the compiled per-trial payload.
    seed : int
        Random seed copied from the manifest.
    archive_schema_version : str
        Version of the archive JSON schema.
        Bump when fields are added, removed, or renamed.
    archived_at : str
        ISO 8601 datetime string written by the archiver at write time.
    """
    trial_id: str
    trial_index: int
    experiment_id: str
    experiment_lifecycle_stage: str
    condition: str
    condition_label: Optional[str] = None
    persona_id: str
    manifest_version: ManifestVersion
    model: ModelSnapshot
    prompts: PromptsSnapshot
    input: TrialInput
    events: list[TelemetryEvent]
    cost: CostRecord
    attention: Optional[AttentionTelemetry] = None
    seed: int
    archive_schema_version: str = "1.0.1"
    archived_at: Optional[str] = None   # filled by the archiver at write time
