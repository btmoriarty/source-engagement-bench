"""
sim/harness/manifest_checks.py

CI-time validation rules for a parsed Manifest, beyond what the Pydantic schema
already enforces. These are the checks the manifest schema doc calls out under
its CI section:

  - model.name must be a pinned identifier, not a floating alias ("latest",
    "sonnet", etc.).
  - model.version_policy must not be "floating" on a production run.
  - every prompt version must be pinned (not empty, not a floating alias).

check_manifest returns a list of human-readable violation strings; an empty list
means the manifest passes. assert_manifest_ok raises with all violations at once.

ASSUMPTION (confirm with the manifest-schema owner): a "production run" is
lifecycle_stage in {"full_run", "replication"}. pilot and calibration are treated
as non-production, so a floating version_policy there is allowed. The schema doc
says "CI rejects floating in production runs" without naming the stages; this is
the working interpretation.
"""

from __future__ import annotations

from sim.harness.parser import Manifest

# Model-name aliases that must never appear on a manifest; a real run pins an
# exact dated identifier (e.g. claude-sonnet-4-6), not a moving target.
_MODEL_ALIASES = {"latest", "sonnet", "opus", "haiku", "claude-latest"}

# Version strings that are not actually pinned.
_FLOATING_VERSIONS = {"", "latest", "head", "floating"}

# Lifecycle stages treated as production for the floating-policy check.
_PRODUCTION_STAGES = {"full_run", "replication"}


def _is_alias_model_name(name: str) -> bool:
    n = name.strip().lower()
    return n in _MODEL_ALIASES or n.endswith("-latest") or n.endswith(":latest")


def _is_unpinned_version(version: str) -> bool:
    return version.strip().lower() in _FLOATING_VERSIONS


def check_manifest(manifest: Manifest) -> list[str]:
    """Return a list of CI-rule violations; empty means the manifest passes."""
    violations: list[str] = []

    name = manifest.model.name
    if _is_alias_model_name(name):
        violations.append(
            f"model.name {name!r} is a floating alias; pin an exact model identifier."
        )

    stage = manifest.experiment.lifecycle_stage
    if manifest.model.version_policy == "floating" and stage in _PRODUCTION_STAGES:
        violations.append(
            f"model.version_policy is 'floating' on a production run "
            f"(lifecycle_stage={stage!r}); pin the version for production."
        )

    for label, ref in (
        ("agent_system", manifest.prompts.agent_system),
        ("agent_user_template", manifest.prompts.agent_user_template),
    ):
        if _is_unpinned_version(ref.version):
            violations.append(
                f"prompts.{label}.version {ref.version!r} is unpinned; pin a concrete version."
            )

    return violations


class ManifestCheckError(ValueError):
    """Raised by assert_manifest_ok when a manifest fails a CI rule."""


def assert_manifest_ok(manifest: Manifest) -> None:
    """Raise ManifestCheckError listing every violation, or return None if clean."""
    violations = check_manifest(manifest)
    if violations:
        raise ManifestCheckError(
            "manifest failed CI checks:\n  - " + "\n  - ".join(violations)
        )
