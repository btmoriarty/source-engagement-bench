"""
sim/harness/interim_prompts.py

Per-trial corpus injection and the trial prompt provider for the sim.

History: PR #27 introduced this module with an inline prompt registry (persona
system prompts and a user template hardcoded as Python dicts) plus a corpus
loader. This revision moves the prompts out to the file-backed, versioned
registry (sim/harness/prompt_registry.py, records under sim/prompts/), and makes
the provider resolve prompts per persona so the runner can iterate personas (the
3x3 crossing). What stays here is the interim corpus (a synthetic stopgap until
the cluster_g_v1 fixture) and the provider that wires a persona + document into a
trial's prompts.

The provider resolves, per trial:
  - the persona's system prompt, from the registry persona record named by the
    manifest persona being run;
  - the shared user template, from the registry template record named by the
    manifest's prompts.agent_user_template ref;
then fills the template's two placeholders (time_budget_signal, policy_document_text)
by literal token replacement (not str.format, because the template embeds a JSON
example whose braces must survive).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from sim.harness.parser import ConditionModel, Manifest, PersonaRef
from sim.harness.prompt_registry import PromptRegistry, PromptResolutionError


# Placeholders the shared user template fills. Kept explicit so build_trial can
# check the manifest's declared required_variables against what is fillable.
_FILLABLE_VARIABLES = ("time_budget_signal", "policy_document_text")


# ---------------------------------------------------------------------------
# Corpus (interim synthetic stopgap; see corpus_interim/README.md)
# ---------------------------------------------------------------------------

class PolicyDocument(BaseModel):
    """
    One policy-document review item. `text` is injected as policy_document_text;
    it holds the source excerpt paired with the AI-drafted summary to review.
    `seeded_error` marks whether the summary was deliberately made faulty (used to
    split accept behavior for the SV3 calibration test).
    """
    id: str
    text: str
    seeded_error: bool = False
    error_kind: Optional[str] = None      # factual | omission | paraphrase_drift | None
    source_note: Optional[str] = None


class CorpusError(RuntimeError):
    """Raised when a corpus directory is missing or empty."""


class Corpus:
    """
    A directory of policy-document review items, one JSON file per document.

    Documents are sorted by id for a stable order, then sampled per trial by a
    deterministic RNG seeded on (manifest seed, trial_index). Selection does not
    depend on the condition or the persona, so a trial_index draws the same
    document under every time-budget condition and every persona (paired
    within-document across the whole crossing).
    """

    def __init__(self, documents: list[PolicyDocument], fixture_id: str) -> None:
        if not documents:
            raise CorpusError("corpus has no documents")
        self.documents = sorted(documents, key=lambda d: d.id)
        self.fixture_id = fixture_id

    @classmethod
    def from_directory(cls, path: Path, fixture_id: str) -> "Corpus":
        path = Path(path)
        if not path.is_dir():
            raise CorpusError(f"corpus directory not found: {path}")
        docs: list[PolicyDocument] = []
        for f in sorted(path.glob("*.json")):
            raw = json.loads(f.read_text(encoding="utf-8"))
            docs.append(PolicyDocument.model_validate(raw))
        if not docs:
            raise CorpusError(f"corpus directory has no *.json documents: {path}")
        return cls(docs, fixture_id=fixture_id)

    def sample(self, seed: int, trial_index: int) -> PolicyDocument:
        rng = random.Random(f"{seed}:{trial_index}")
        return self.documents[rng.randrange(len(self.documents))]

    @property
    def seeded_error_count(self) -> int:
        return sum(1 for d in self.documents if d.seeded_error)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

@dataclass
class TrialPromptBundle:
    """What build_trial returns to the runner for one trial."""
    system_prompt: str
    user_prompt: str
    persona_id: str
    persona_version: str
    template_id: str
    template_version: str
    policy_document_id: str
    trial_context: str


class InterimPromptProvider:
    """
    The runner's optional prompt_provider. Wraps the file-backed prompt registry
    and the interim corpus and, per trial, resolves the running persona's system
    prompt and the shared user template, samples a document, and fills the two
    required_variables.
    """

    def __init__(self, registry: PromptRegistry, corpus: Corpus) -> None:
        self.registry = registry
        self.corpus = corpus

    def _fill_user_template(
        self, template: str, *, policy_document_text: str, time_budget_signal: Optional[str]
    ) -> str:
        if "{policy_document_text}" not in template:
            raise PromptResolutionError(
                "user template does not contain the {policy_document_text} placeholder"
            )
        filled = template.replace("{policy_document_text}", policy_document_text)
        # An absent time signal (unconstrained condition) fills as empty text.
        filled = filled.replace("{time_budget_signal}", time_budget_signal or "")
        return filled

    def build_trial(
        self,
        *,
        manifest: Manifest,
        condition: ConditionModel,
        trial_index: int,
        persona: PersonaRef,
        time_budget_signal: Optional[str],
    ) -> TrialPromptBundle:
        # Light guard: every variable the manifest declares required must be one
        # this template can fill.
        declared = manifest.prompts.agent_user_template.required_variables or []
        unknown = [v for v in declared if v not in _FILLABLE_VARIABLES]
        if unknown:
            raise PromptResolutionError(
                f"manifest declares required_variables {unknown} the template does "
                f"not fill; fillable: {list(_FILLABLE_VARIABLES)}"
            )

        persona_rec = self.registry.resolve_persona(persona.id, persona.version)
        tref = manifest.prompts.agent_user_template
        template_rec = self.registry.resolve_template(tref.registry_id, tref.version)

        doc = self.corpus.sample(manifest.random.seed, trial_index)
        user_prompt = self._fill_user_template(
            template_rec.user_template,
            policy_document_text=doc.text,
            time_budget_signal=time_budget_signal,
        )

        trial_context = (
            f"corpus={self.corpus.fixture_id}; document={doc.id}; "
            f"seeded_error={doc.seeded_error}; error_kind={doc.error_kind or 'none'}"
        )
        return TrialPromptBundle(
            system_prompt=persona_rec.system_prompt,
            user_prompt=user_prompt,
            persona_id=persona_rec.id,
            persona_version=persona_rec.version,
            template_id=template_rec.id,
            template_version=template_rec.version,
            policy_document_id=doc.id,
            trial_context=trial_context,
        )


def default_registry_dir(repo_root: Path) -> Path:
    return Path(repo_root) / "sim" / "prompts"


def load_provider_from_manifest(
    manifest: Manifest, repo_root: Path
) -> Optional[InterimPromptProvider]:
    """
    Build an InterimPromptProvider from the manifest's optional corpus block and
    the prompt registry under sim/prompts/. Returns None when the manifest
    declares no corpus, so the caller falls back to the placeholder path.
    """
    if manifest.corpus is None:
        return None
    registry = PromptRegistry.from_directory(default_registry_dir(repo_root))
    corpus_dir = (Path(repo_root) / manifest.corpus.path).resolve()
    corpus = Corpus.from_directory(corpus_dir, fixture_id=manifest.corpus.fixture_id)
    return InterimPromptProvider(registry=registry, corpus=corpus)
