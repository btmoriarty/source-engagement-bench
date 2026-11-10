"""
sim/harness/parser.py

Author: Taylor Brooks

Manifest parser for the sim harness.

Responsibilities:
    - Load a YAML manifest file from disk
    - Validate it against the Manifest Pydantic model
    - Return a typed Manifest object the runner and archiver can consume

Out of scope for this stub:
    - Pydantic validators (field-level logic beyond type checking)
    - CI checks (unpinned prompt versions, floating model aliases)
    - JSON Schema validation (manifest.schema.json, once built)

Dependencies:
    - pydantic v2
    - pyyaml

Usage:
    parser = ManifestParser()
    manifest = parser.parse(Path("experiments/my_experiment.yaml"))
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ValidationError


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ManifestValidationError(Exception):
    """
    Raised when a manifest file fails validation.

    The message lists every failing field so the experimenter can
    self-debug without reading the harness internals.

    Example message:
        ManifestValidationError: 2 validation errors in manifest:
          - experiment.status: must be one of [draft, ready, running, complete]; got 'wip'
          - budget.cap_usd: must be a positive float; got -5.0
    """
    pass


# ---------------------------------------------------------------------------
# Pydantic models
# Field names and nesting mirror manifest_schema.yaml exactly.
# Validators are intentionally absent in this stub; they land in Wk 3.
# ---------------------------------------------------------------------------

class ExperimentModel(BaseModel):
    """Top-level experiment metadata block."""
    id: str
    name: str
    description: str
    owner: str
    project_line: str
    hypothesis_id: Optional[str] = None
    lifecycle_stage: str
    status: str
    run_date: date                        # ISO 8601 date string or date object


class LLMConfig(BaseModel):
    """Model configuration block."""
    provider: str
    endpoint_profile: str
    name: str
    version_policy: str
    max_tokens: int
    temperature: float
    top_p: float


class PromptRef(BaseModel):
    """Single prompt reference (system or user template)."""
    registry_id: str
    version: str
    required_variables: Optional[list[str]] = None


class PromptsModel(BaseModel):
    """Prompt references block."""
    agent_system: PromptRef
    agent_user_template: PromptRef


class PersonaRef(BaseModel):
    """Single persona reference."""
    id: str
    version: str
    description: Optional[str] = None


class ConditionModel(BaseModel):
    """Single experimental condition."""
    name: str
    label: Optional[str] = None
    token_budget: Optional[int] = None
    time_pressure_ms: Optional[int] = None
    explanation_visible: bool = True


class SampleModel(BaseModel):
    """Trial sample and dispatch configuration."""
    trials_per_condition: int
    max_parallel_trials: Optional[int] = None   # optional; absent = 1 (sequential)


class OutcomesModel(BaseModel):
    """Pre-registered outcome variables."""
    primary: list[str]
    secondary: Optional[list[str]] = None


class BudgetModel(BaseModel):
    """Budget cap and halt policy."""
    cap_usd: float
    currency: str
    alert_at_percent: int
    halt_policy: str
    on_halt: str


class RandomModel(BaseModel):
    """Random seed and determinism scope."""
    seed: int
    determinism_scope: list[str]


class TelemetryModel(BaseModel):
    """Telemetry hooks list."""
    hooks: list[str]


class ArchiveModel(BaseModel):
    """Archive path and write behavior."""
    base_path: str
    atomic: bool


class CorpusModel(BaseModel):
    """
    Optional per-trial policy-document corpus block.

    Interim wiring (see sim/harness/interim_prompts.py): points the runner at a
    directory of policy-document review items so each trial injects a real
    ``policy_document_text`` instead of a placeholder string. Optional with a
    default of None, so every manifest that predates the corpus block still
    validates unchanged and the mock/registry-only path is untouched.

    Fields
    ------
    fixture_id : str
        Stable id for the corpus fixture (recorded in the run for provenance).
    path : str
        Path to the corpus directory, relative to the repo root, holding one
        JSON document per file.
    seeded_error_flag : bool
        True when the corpus documents carry a per-document ``seeded_error``
        flag (required for the SV3 Lee-and-See calibration split).
    notes : Optional[str]
        Free-text provenance note.
    """
    fixture_id: str
    path: str
    seeded_error_flag: bool = False
    notes: Optional[str] = None


class Manifest(BaseModel):
    """
    Fully parsed and validated sim experiment manifest.

    This is the object the runner and archiver consume. All fields
    mirror the top-level blocks in manifest_schema.yaml. Nested blocks
    are typed sub-models, not raw dicts.

    Field names use the same keys as the YAML manifest so that
    manifest.experiment.id reads the same way it appears in the file.
    """
    schema_version: str
    experiment: ExperimentModel
    model: LLMConfig
    prompts: PromptsModel
    personas: list[PersonaRef]
    conditions: list[ConditionModel]
    sample: SampleModel
    outcomes: OutcomesModel
    budget: BudgetModel
    random: RandomModel
    telemetry: TelemetryModel
    archive: ArchiveModel
    corpus: Optional[CorpusModel] = None    # optional; absent = registry-id placeholder path
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ManifestParser:
    """
    Loads and validates a sim experiment manifest from a YAML file.

    Methods
    -------
    load(path)
        Read the YAML file at path and return the raw dict.
        Raises FileNotFoundError if the path does not exist.
        Raises ValueError if the file is not valid YAML.

    validate(raw)
        Validate a raw dict against the Manifest Pydantic model.
        Raises ManifestValidationError listing every failing field
        if validation fails.
        Returns a Manifest on success.

    parse(path)
        Composes load() and validate().
        The primary entry point for the runner and archiver.
        Returns a Manifest on success.
    """

    def load(self, path: Path) -> dict:
        """
        Read and parse a YAML manifest file from disk.

        Parameters
        ----------
        path : Path
            Absolute or relative path to the manifest YAML file.

        Returns
        -------
        dict
            Raw manifest contents as a Python dict.

        Raises
        ------
        FileNotFoundError
            If the file does not exist at the given path.
        ValueError
            If the file exists but is not valid YAML.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise FileNotFoundError(f"Unable to read manifest file: {path}") from exc

        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in manifest {path}: {exc}") from exc

        if raw is None:
            raise ValueError(f"Manifest {path} is empty")
        if not isinstance(raw, dict):
            raise ValueError(f"Manifest {path} must contain a top-level mapping")

        return raw

    def validate(self, raw: dict) -> Manifest:
        """
        Validate a raw manifest dict against the Manifest Pydantic model.

        Collects all validation errors before raising so the experimenter
        sees every problem at once rather than fixing one field at a time.

        Parameters
        ----------
        raw : dict
            Raw manifest dict, typically the output of load().

        Returns
        -------
        Manifest
            Fully validated and typed manifest object.

        Raises
        ------
        ManifestValidationError
            If any field fails validation. The message lists every
            failing field and the reason.
        """
        try:
            return Manifest.model_validate(raw)
        except ValidationError as exc:
            errors = []
            for error in exc.errors():
                loc = ".".join(str(part) for part in error.get("loc", []))
                msg = error.get("msg", "validation error")
                errors.append(f"- {loc}: {msg}")
            formatted = "\n  ".join(errors)
            raise ManifestValidationError(
                f"{len(errors)} validation error(s) in manifest:\n  {formatted}"
            ) from exc

    def parse(self, path: Path) -> Manifest:
        """
        Load and validate a manifest file in one call.

        This is the primary entry point. The runner and archiver call
        this method; they do not call load() or validate() directly.

        Parameters
        ----------
        path : Path
            Path to the manifest YAML file.

        Returns
        -------
        Manifest
            Fully validated and typed manifest object ready for
            the runner and archiver to consume.

        Raises
        ------
        FileNotFoundError
            If the manifest file does not exist.
        ValueError
            If the file is not valid YAML.
        ManifestValidationError
            If the manifest fails schema validation.
        """
        raw = self.load(path)
        return self.validate(raw)
