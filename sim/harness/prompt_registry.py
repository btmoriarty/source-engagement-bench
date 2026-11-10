"""
sim/harness/prompt_registry.py

Versioned prompt registry for the sim.

Formalizes the interim inline prompts (PR #27's interim_prompts.py) into
file-backed, semver-versioned records under sim/prompts/. Every sim manifest
references a persona by id + pinned version, and the registry resolves that to a
real prompt record, so trials are reproducible and persona is a controlled factor
(not an ad hoc string).

Layout (matches the validity-check design's `sim/prompts/personas/` and the
example_prompt_record.yaml schema):

    sim/prompts/personas/<persona-id>/<version>.yaml      kind: persona
    sim/prompts/templates/<template-id>/<version>.yaml    kind: user_template

A persona record carries a `system_prompt` plus persona metadata. A user_template
record carries a `user_template` plus the `required_placeholders` it fills. The
manifest's `prompts.agent_user_template` names the shared template; the manifest's
`personas` list names the persona records the runner iterates (the 3x3 crossing).

CI checks (see check_registry and check_manifest_prompts):
  - version strings are concrete semver (vMAJOR.MINOR.PATCH), never floating.
  - a user_template's {placeholders} exactly match its required_placeholders.
  - a persona record has a non-empty system_prompt; a template has a user_template.
  - change_log has at least one entry.
  - every persona and template a manifest references resolves at its pinned version.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel

from sim.harness.parser import Manifest


_SEMVER_RE = re.compile(r"^v\d+\.\d+\.\d+$")
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class PromptResolutionError(KeyError):
    """Raised when a manifest reference has no matching registry record."""


class RegistryValidationError(ValueError):
    """Raised when a registry record fails a CI validation rule."""


class ChangeLogEntry(BaseModel):
    version: str
    date: Optional[str] = None
    author: Optional[str] = None
    note: Optional[str] = None


class PersonaMeta(BaseModel):
    role: Optional[str] = None
    domain_familiarity: Optional[str] = None
    prior_ai_experience: Optional[str] = None
    disposition: Optional[str] = None
    pre_registered_core: bool = False
    description: Optional[str] = None


class PromptRecord(BaseModel):
    """One registry record: a persona (system prompt) or a user template."""
    schema_version: str
    id: str
    version: str
    kind: Literal["persona", "user_template"]
    persona: Optional[PersonaMeta] = None
    system_prompt: Optional[str] = None
    user_template: Optional[str] = None
    required_placeholders: Optional[list[str]] = None
    license: Optional[str] = None
    change_log: list[ChangeLogEntry] = []
    notes: Optional[str] = None

    def key(self) -> tuple[str, str, str]:
        return (self.kind, self.id, self.version)


def _template_placeholders(text: str) -> set[str]:
    # The verdict template embeds a literal JSON example ({"decision": ...}); its
    # tokens are not simple {identifier} placeholders, so the identifier-only
    # regex skips them and matches only true substitution slots.
    return set(_PLACEHOLDER_RE.findall(text))


def validate_record(rec: PromptRecord) -> list[str]:
    """Return a list of CI-rule violations for one record; empty means it passes."""
    v: list[str] = []
    if not _SEMVER_RE.match(rec.version):
        v.append(f"{rec.id}: version {rec.version!r} is not concrete semver (vMAJOR.MINOR.PATCH).")
    if not rec.change_log:
        v.append(f"{rec.id}: change_log is empty; every record needs at least one entry.")
    if rec.kind == "persona":
        if not (rec.system_prompt and rec.system_prompt.strip()):
            v.append(f"{rec.id}: persona record has no system_prompt.")
    else:  # user_template
        if not (rec.user_template and rec.user_template.strip()):
            v.append(f"{rec.id}: user_template record has no user_template.")
        else:
            found = _template_placeholders(rec.user_template)
            declared = set(rec.required_placeholders or [])
            missing = found - declared
            extra = declared - found
            if missing:
                v.append(f"{rec.id}: template uses undocumented placeholders {sorted(missing)}.")
            if extra:
                v.append(f"{rec.id}: required_placeholders lists {sorted(extra)} not present in the template.")
    return v


class PromptRegistry:
    """In-memory index of prompt records, loaded from the sim/prompts tree."""

    def __init__(self, records: list[PromptRecord]) -> None:
        self._by_key: dict[tuple[str, str, str], PromptRecord] = {}
        for rec in records:
            self._by_key[rec.key()] = rec

    @classmethod
    def from_directory(cls, root: Path) -> "PromptRegistry":
        root = Path(root)
        if not root.is_dir():
            raise PromptResolutionError(f"registry directory not found: {root}")
        records: list[PromptRecord] = []
        for f in sorted(root.rglob("*.yaml")):
            raw = yaml.safe_load(f.read_text(encoding="utf-8"))
            records.append(PromptRecord.model_validate(raw))
        if not records:
            raise PromptResolutionError(f"registry directory has no records: {root}")
        return cls(records)

    def resolve_persona(self, persona_id: str, version: str) -> PromptRecord:
        key = ("persona", persona_id, version)
        if key not in self._by_key:
            raise PromptResolutionError(
                f"no persona record for ({persona_id}, {version}); "
                f"known personas: {sorted(self.persona_ids())}"
            )
        return self._by_key[key]

    def resolve_template(self, template_id: str, version: str) -> PromptRecord:
        key = ("user_template", template_id, version)
        if key not in self._by_key:
            raise PromptResolutionError(
                f"no user_template record for ({template_id}, {version}); "
                f"known templates: {sorted(self.template_ids())}"
            )
        return self._by_key[key]

    def persona_ids(self) -> list[str]:
        return [rid for (kind, rid, _v) in self._by_key if kind == "persona"]

    def template_ids(self) -> list[str]:
        return [rid for (kind, rid, _v) in self._by_key if kind == "user_template"]

    def all_records(self) -> list[PromptRecord]:
        return list(self._by_key.values())


def check_registry(registry: PromptRegistry) -> list[str]:
    """Validate every record in the registry; empty list means the registry passes."""
    violations: list[str] = []
    for rec in registry.all_records():
        violations.extend(validate_record(rec))
    return violations


def check_manifest_prompts(manifest: Manifest, registry: PromptRegistry) -> list[str]:
    """
    CI check: every persona and template the manifest references resolves in the
    registry at its pinned version, versions are concrete semver, and the shared
    template's required_placeholders match the manifest's declared
    required_variables. Returns a list of violations; empty means it passes.
    """
    violations: list[str] = []

    for persona in manifest.personas:
        if not _SEMVER_RE.match(persona.version):
            violations.append(
                f"personas[{persona.id}].version {persona.version!r} is not a pinned semver."
            )
            continue
        try:
            registry.resolve_persona(persona.id, persona.version)
        except PromptResolutionError as exc:
            violations.append(str(exc))

    tref = manifest.prompts.agent_user_template
    if not _SEMVER_RE.match(tref.version):
        violations.append(
            f"prompts.agent_user_template.version {tref.version!r} is not a pinned semver."
        )
    else:
        try:
            template = registry.resolve_template(tref.registry_id, tref.version)
        except PromptResolutionError as exc:
            violations.append(str(exc))
        else:
            declared = set(tref.required_variables or [])
            needed = set(template.required_placeholders or [])
            if declared and declared != needed:
                violations.append(
                    f"manifest required_variables {sorted(declared)} do not match template "
                    f"{template.id} required_placeholders {sorted(needed)}."
                )

    return violations


def assert_manifest_prompts_ok(manifest: Manifest, registry: PromptRegistry) -> None:
    """Raise RegistryValidationError listing every violation, or return None if clean."""
    violations = check_manifest_prompts(manifest, registry)
    if violations:
        raise RegistryValidationError(
            "manifest prompt references failed registry checks:\n  - " + "\n  - ".join(violations)
        )
