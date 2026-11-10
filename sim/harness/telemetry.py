"""
sim/harness/telemetry.py

Attention/engagement telemetry for the sim harness. Author: Jong In Park (PR #21),
reapplied onto the reconciled runner 2026-07-02.

The evidence-expand/collapse hooks fire only once the experiment interaction layer
calls them; until then those fields hold their defaults (0 / False) and should be
read as "not yet instrumented," not as measured zeros. The compiled payload is picked
up by the runner (see runner.py), converted into AttentionTelemetry, and persisted
to the archive; this was resolved in #20 following the #19 data-model reconciliation.
"""

import time


class SimAttentionTelemetry:
    def __init__(self):
        self.current_event = {
            "duration_seconds": 0.0,
            "evidence_expansion_count": 0,
            "swift_collapse_detected": False,
            "rationale": "",
            "trace_token_length": 0,
        }
        self._review_started_at = None
        self._panel_opened_at = None

    def record_trial_start(self):
        """Called at the start of a trial loop step."""
        self._review_started_at = time.time()

    def record_evidence_expanded(self):
        """Called when an agent opens an evidence sub-array."""
        self.current_event["evidence_expansion_count"] += 1
        self._panel_opened_at = time.time()

    def record_evidence_collapsed(self):
        """Flags rubber-stamping if closed in under 1.5 seconds."""
        if self._panel_opened_at:
            elapsed = time.time() - self._panel_opened_at
            if elapsed < 1.5:
                self.current_event["swift_collapse_detected"] = True

    def compile_final_payload(self, rationale_text: str, hidden_trace: "list | str") -> dict:
        """Packages up the metrics at the end of the trial."""
        if self._review_started_at:
            self.current_event["duration_seconds"] = time.time() - self._review_started_at
        self.current_event["rationale"] = rationale_text
        self.current_event["trace_token_length"] = len(hidden_trace)
        return self.current_event
