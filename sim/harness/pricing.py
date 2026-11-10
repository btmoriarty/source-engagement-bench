"""
sim/harness/pricing.py

Per-model token pricing and per-trial cost computation for the sim harness.

Step 5 left the cost.*_usd fields null because no price map existed. This module
adds a small, explicit price map and a function that turns an LLMResult's token
counts into input_cost_usd, output_cost_usd, and total_cost_usd. When token counts
are absent (the bare mock returns None), cost stays null. The harness never invents
a cost from a missing token count.

================================================================================
RATES CONFIRMED 2026-07-20 - RECONFIRM BEFORE ANY PAID RUN IF THIS DATE IS STALE
================================================================================
Pricing was confirmed July 20, 2026 via Anthropic's official pricing page.
https://platform.claude.com/docs/en/about-claude/pricing

The numbers below are entered as USD per million tokens, the unit providers
publish. Two items remain genuinely open even though the rates are confirmed:

  - the manifest's endpoint_profile (standard vs zdr) may carry a different rate
    than the standard tier, and that delta is not reflected here yet;
  - the trial_archive_schema open question (Rishab) on float vs Decimal precision
    for budget reconciliation is unresolved; this module uses float for now.

Budget enforcement in the runner consumes these numbers, so a stale rate produces
a wrong halt point. Reconfirm against the price sheet if it's been a while since
2026-07-20.

The map keys on model.name (the pinned identifier from the manifest model block).
A model not in the map computes no cost (returns null fields) rather than guessing.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from sim.harness.llm_client import LLMResult


class TokenPrice(BaseModel):
    """
    Per-million-token rates for one model. USD.

    input_usd_per_mtok  : price per 1,000,000 input (prompt) tokens.
    output_usd_per_mtok : price per 1,000,000 output (response) tokens.
    """
    input_usd_per_mtok: float
    output_usd_per_mtok: float


# Rates confirmed 2026-07-20 against the provider's published price sheet, keyed
# on the exact pinned model name. Endpoint-tier (standard vs zdr) delta not yet
# reflected; see module docstring.
_PRICE_MAP: dict[str, TokenPrice] = {
    # Confirmed 2026-07-20. zdr-tier delta not yet reflected (see docstring).
    "claude-sonnet-4-6": TokenPrice(
        input_usd_per_mtok=3.00,    # as of July 20, 2026
        output_usd_per_mtok=15.00,  # as of July 20, 2026
    ),
    # Confirmed 2026-07-20. Key must match the exact model.name pinned in the
    # manifest, or compute_cost misses.
    "claude-haiku-4-5": TokenPrice(
        input_usd_per_mtok=1.00,    # as of July 20, 2026
        output_usd_per_mtok=5.00,   # as of July 20, 2026
    ),
}

_USD_PER_MTOK = 1_000_000.0


class TrialCost(BaseModel):
    """Computed per-trial cost. Fields are None when token counts are absent."""
    input_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    total_cost_usd: Optional[float] = None


def has_price(model_name: str) -> bool:
    """True if the price map has a rate for this model name."""
    return model_name in _PRICE_MAP


def compute_cost(model_name: str, result: LLMResult) -> TrialCost:
    """
    Compute per-trial USD cost from an LLMResult's token counts.

    Returns all-null TrialCost when:
      - the model is not in the price map (no rate to apply), or
      - either token count is None (the bare mock leaves them None).

    Otherwise returns input_cost_usd, output_cost_usd, and their sum. Costs are
    computed from confirmed rates; see the module docstring.
    """
    price = _PRICE_MAP.get(model_name)
    if price is None:
        return TrialCost()
    if result.input_tokens is None or result.output_tokens is None:
        return TrialCost()
    input_cost = result.input_tokens * price.input_usd_per_mtok / _USD_PER_MTOK
    output_cost = result.output_tokens * price.output_usd_per_mtok / _USD_PER_MTOK
    return TrialCost(
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + output_cost,
    )
