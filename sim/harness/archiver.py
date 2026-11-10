"""
sim/harness/archiver.py

Trial archiver for the sim harness.

Responsibilities:
    - Accept a completed TrialResult from the runner
    - Set archived_at on the TrialResult
    - Serialize the TrialResult to JSON
    - Write the JSON atomically to the archive path defined in the manifest
    - Verify no orphaned .tmp files remain after a run

Atomic write contract:
    Writes to a .tmp file first, then renames to the final path on success.
    This guarantees the archive never contains a partially written file.
    archive.atomic must be True; if False the archiver raises at init time.

Usage:
    archiver = Archiver(archive_config=manifest.archive, experiment_id=manifest.experiment.id)
    path = archiver.write(trial_result)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from sim.harness.models import TrialResult
from sim.harness.parser import ArchiveModel


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Archiver:
    """
    Writes completed trial results to the archive as JSON files.

    One Archiver instance is created per experiment run and reused
    across all trials. The archive path for each trial is derived
    from the manifest's archive.base_path template.

    Parameters
    ----------
    archive_config : ArchiveModel
        The archive block from the parsed manifest.
        Provides base_path template and atomic flag.
    experiment_id : str
        The experiment.id from the manifest.
        Used to resolve the {experiment.id} placeholder in base_path.
    """

    def __init__(self, archive_config: ArchiveModel, experiment_id: str):
        if not archive_config.atomic:
            raise ValueError(
                "Non-atomic writes are not supported; "
                "set archive.atomic: true in the manifest."
            )
        self.archive_config = archive_config
        self.experiment_id = experiment_id

    def resolve_path(self, trial_result: TrialResult) -> Path:
        """
        Resolve the archive file path for a given trial result.

        Substitutes template variables in archive.base_path:
            {experiment.id}  -> trial_result.experiment_id
            {condition.id}   -> trial_result.condition
            {trial.id}       -> trial_result.trial_id
        """
        path_str = self.archive_config.base_path
        path_str = path_str.replace("{experiment.id}", trial_result.experiment_id)
        path_str = path_str.replace("{condition.id}", trial_result.condition)
        path_str = path_str.replace("{trial.id}", trial_result.trial_id)
        return Path(path_str)

    def write(self, trial_result: TrialResult) -> Path:
        """
        Write a completed trial result to the archive.

        Sequence:
            1. Set trial_result.archived_at to the current UTC timestamp
            2. Resolve the final archive path via resolve_path()
            3. Create parent directories if they do not exist
            4. Serialize trial_result to JSON
            5. Write JSON to <final_path>.tmp
            6. Rename <final_path>.tmp to <final_path> (atomic)
            7. Return the final path

        Raises
        ------
        FileExistsError
            If a file already exists at the resolved archive path.
            Prevents accidental overwrites of existing trial archives.
        OSError
            If the write or rename fails for any filesystem reason.
        """
        trial_result.archived_at = _utcnow()
        final_path = self.resolve_path(trial_result)

        if final_path.exists():
            raise FileExistsError(
                f"Archive file already exists, will not overwrite: {final_path}"
            )

        final_path.parent.mkdir(parents=True, exist_ok=True)

        json_bytes = trial_result.model_dump_json(indent=2).encode("utf-8")
        tmp_path = final_path.with_suffix(".tmp")

        try:
            tmp_path.write_bytes(json_bytes)
            os.replace(tmp_path, final_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        return final_path

    def verify_no_orphaned_tmp(self, base_path: Path) -> None:
        """
        Verify no .tmp files remain under base_path after a run.

        Called after run_all() completes to confirm every atomic write
        finished cleanly. A .tmp file left behind indicates a write was
        interrupted before the rename step.

        Raises
        ------
        RuntimeError
            If any .tmp files are found. Lists every orphaned path.
        """
        orphans = list(Path(base_path).rglob("*.tmp"))
        if orphans:
            listed = "\n  ".join(str(p) for p in orphans)
            raise RuntimeError(
                f"{len(orphans)} orphaned .tmp file(s) found under {base_path}:\n  {listed}"
            )
